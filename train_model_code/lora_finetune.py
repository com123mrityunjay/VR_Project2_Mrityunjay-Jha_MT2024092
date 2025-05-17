import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import glob
from bert_score import score as bert_score
import copy

# Use the function definition directly
def find_image(image_dir, image_name):
    """Find an image in the ABO dataset directory structure.

    Arguments:
        image_dir: Base directory containing ABO images
        image_name: Image filename or partial path

    Returns:
        Full path to the image if found, None otherwise
    """
    # Case 1: image_name already contains path structure (like "00/00f5b2e4.jpg")
    full_path = os.path.join(image_dir, image_name)
    if os.path.exists(full_path):
        return full_path

    # Case 2: image_name is just a filename (like "00f5b2e4.jpg")
    # In ABO dataset, images are organized in subdirectories based on the first two chars
    if not os.path.exists(full_path) and '/' not in image_name:
        # Try to find in subdirectory based on first two chars
        if len(image_name) >= 2:
            subdir = image_name[:2]
            subdir_path = os.path.join(image_dir, subdir, image_name)
            if os.path.exists(subdir_path):
                return subdir_path

        # If still not found, search recursively through subdirectories
        matches = glob.glob(os.path.join(image_dir, "**", image_name), recursive=True)
        if matches:
            return matches[0]

    return None

# Simplified approach: Generate text features first, then fine-tune language model only
class VQADatasetPreprocessor:
    def __init__(self, csv_path, image_dir, processor, model, device, batch_size=8):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor
        self.model = model
        self.device = device
        self.batch_size = batch_size
        # Set a fixed max_length for all batches
        self.max_length = 77  # BLIP-2's default max length
        
    def process_data(self):
        """Generate text features for all images and questions in the dataset"""
        print("Preprocessing dataset with BLIP2 vision encoder...")
        
        # Store original data
        all_image_names = []
        all_questions = []
        all_answers = []
        
        # Store processed features
        all_input_ids = []
        all_attention_masks = []
        
        # Process in batches to avoid running out of memory
        for i in tqdm(range(0, len(self.df), self.batch_size)):
            batch_df = self.df.iloc[i:i+self.batch_size]
            
            # Prepare images and prompts for this batch
            images = []
            prompts = []
            questions_batch = []
            answers_batch = []
            image_names_batch = []
            
            for _, row in batch_df.iterrows():
                image_name = str(row['image_name'])
                question = str(row['question'])
                answer = str(row['answer'])
                
                # Find image file
                image_path = find_image(self.image_dir, image_name)
                if not image_path:
                    print(f"Image not found: {image_name}")
                    # Create a black image as placeholder
                    image = Image.new('RGB', (224, 224), color='black')
                else:
                    # Load image
                    image = Image.open(image_path).convert("RGB")
                
                # Format prompt with answer for language modeling
                prompt = f"Question: {question} Answer in one word: {answer}"
                
                # Append to batch lists
                images.append(image)
                prompts.append(prompt)
                questions_batch.append(question)
                answers_batch.append(answer)
                image_names_batch.append(image_name)
            
            # Process batch with vision encoder (no gradient tracking needed)
            with torch.no_grad():
                # Process images and text with fixed max_length
                inputs = self.processor(
                    images=images, 
                    text=prompts, 
                    return_tensors="pt", 
                    padding="max_length", 
                    max_length=self.max_length,
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate the embeddings using base model's processing
                vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                image_embeds = vision_outputs.last_hidden_state
                
                # Get query features via the Qformer
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=torch.ones(image_embeds.shape[0], image_embeds.shape[1], device=self.device),
                    return_dict=True
                )
                
                # Make sure query outputs are in the same dtype as the language projection layer
                query_output_dtype = self.model.language_projection.weight.dtype
                query_last_hidden = query_outputs.last_hidden_state.to(query_output_dtype)
                
                # Project to the text embedding space
                language_model_inputs = self.model.language_projection(query_last_hidden)
                
                # Store the processed data
                all_image_names.extend(image_names_batch)
                all_questions.extend(questions_batch)
                all_answers.extend(answers_batch)
                
                # Convert language model inputs to CPU and detach
                language_model_inputs = language_model_inputs.cpu().detach()
                
                # Store processed features - store individual tensors instead of batches
                for j in range(len(images)):
                    all_input_ids.append(inputs["input_ids"][j].cpu().long())
                    all_attention_masks.append(inputs["attention_mask"][j].cpu().long())
        
        # Make sure all tensors have the same shape before stacking
        input_id_shape = all_input_ids[0].shape
        attention_mask_shape = all_attention_masks[0].shape
        
        print(f"Input ID shape: {input_id_shape}, Attention mask shape: {attention_mask_shape}")
        
        # Check for inconsistent shapes
        valid_inputs = []
        valid_attentions = []
        valid_indices = []
        
        for idx, (input_id, attention_mask) in enumerate(zip(all_input_ids, all_attention_masks)):
            if (input_id.shape == input_id_shape and 
                attention_mask.shape == attention_mask_shape):
                valid_inputs.append(input_id)
                valid_attentions.append(attention_mask)
                valid_indices.append(idx)
        
        # Update metadata to match valid tensors
        if len(valid_indices) < len(all_image_names):
            print(f"Warning: {len(all_image_names) - len(valid_indices)} samples were removed due to inconsistent tensor shapes")
            all_image_names = [all_image_names[i] for i in valid_indices]
            all_questions = [all_questions[i] for i in valid_indices]
            all_answers = [all_answers[i] for i in valid_indices]
        
        # Combine all data
        try:
            processed_data = {
                "image_names": all_image_names,
                "questions": all_questions,
                "answers": all_answers,
                "input_ids": torch.stack(valid_inputs),
                "attention_mask": torch.stack(valid_attentions)
            }
            
            print(f"Preprocessing complete. Processed {len(processed_data['image_names'])} samples.")
            return processed_data
        except RuntimeError as e:
            print(f"Error combining tensors: {e}")
            print(f"First input ID shape: {valid_inputs[0].shape}")
            for i in range(1, min(5, len(valid_inputs))):
                if valid_inputs[i].shape != valid_inputs[0].shape:
                    print(f"Inconsistent shape at index {i}: {valid_inputs[i].shape}")
            
            raise RuntimeError("Failed to combine tensors due to inconsistent shapes. Try adjusting max_length.")

# Dataset for fine-tuning the language model
class LanguageModelDataset(Dataset):
    def __init__(self, processed_data):
        self.image_names = processed_data["image_names"]
        self.questions = processed_data["questions"]
        self.answers = processed_data["answers"]
        self.input_ids = processed_data["input_ids"]
        self.attention_mask = processed_data["attention_mask"]
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        return {
            "image_name": self.image_names[idx],
            "question": self.questions[idx],
            "answer": self.answers[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

def collate_fn(batch):
    processed_batch = {
        "image_name": [],
        "question": [],
        "answer": [],
    }
    
    # Collect all items by key
    for item in batch:
        processed_batch["image_name"].append(item["image_name"])
        processed_batch["question"].append(item["question"])
        processed_batch["answer"].append(item["answer"])
    
    # Stack tensors
    processed_batch["input_ids"] = torch.stack([item["input_ids"] for item in batch])
    processed_batch["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    
    # For causal language modeling, labels should be a copy of input_ids
    # This works because the model will automatically shift the labels internally
    processed_batch["labels"] = processed_batch["input_ids"].clone()
    
    # Replace padding tokens (usually 1) with -100 for loss masking
    processed_batch["labels"][processed_batch["labels"] == 1] = -100
    
    return processed_batch

def apply_lora_to_language_model(language_model):
    """Apply LoRA to the language model only, using a different approach"""
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    
    print(f"Original language model type: {type(language_model).__name__}")
    
    # First, prepare the model for training - critical step
    language_model = prepare_model_for_kbit_training(language_model)
    
    # Define LoRA configuration - try different target modules
    lora_config = LoraConfig(
        r=8,                       # Lower rank for stability
        lora_alpha=16,             # Lower alpha  
        target_modules=["q_proj", "v_proj"],  # Target fewer modules
        lora_dropout=0.1,          # Higher dropout for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    lora_model = get_peft_model(language_model, lora_config)
    
    # Turn on gradient checkpointing for memory efficiency
    if hasattr(lora_model, "enable_input_require_grads"):
        lora_model.enable_input_require_grads()
    
    if hasattr(lora_model, "gradient_checkpointing_enable"):
        lora_model.gradient_checkpointing_enable()
    
    # Ensure all lora parameters require gradients
    for name, param in lora_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            print(f"Set {name} to require gradients")
    
    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in lora_model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.5f}")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Check LoRA configuration.")
    
    # Explicitly set to train mode
    lora_model.train()
    
    return lora_model

def print_trainable_parameters(model):
    """Utility function to print details about trainable parameters"""
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(f"Trainable: {name} - {param.shape} - {param.dtype}")
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return trainable_params

def evaluate_model(model, processor, base_model, test_loader, device, image_dir):
    """
    Evaluate the model on the test set.
    
    In our case, we need to:
    1. Restore the original language model
    2. Then use the base_model for generation with both image (pixel_values) and text inputs
    """
    model.eval()
    base_model.eval()
    
    # Get the model's dtype
    model_dtype = next(base_model.parameters()).dtype
    print(f"Base model dtype: {model_dtype}")
    
    generated_answers = []
    ground_truth = []
    questions = []
    image_names = []
    
    # Create a batch-wise evaluation to recreate the full pipeline
    batch_size = 4  # Smaller batch size for evaluation
    
    # Group data by image_name to reduce duplicated image loading
    image_to_questions = {}
    
    for batch in test_loader:
        for i in range(len(batch["image_name"])):
            img_name = batch["image_name"][i]
            question = batch["question"][i]
            answer = batch["answer"][i]
            
            if img_name not in image_to_questions:
                image_to_questions[img_name] = []
            
            image_to_questions[img_name].append({
                "question": question,
                "answer": answer
            })
    
    print(f"Evaluating {len(image_to_questions)} images with {sum(len(qs) for qs in image_to_questions.values())} total questions...")
    
    # Process images in batches
    all_images = list(image_to_questions.keys())
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), batch_size), desc="Evaluating"):
            batch_images = all_images[i:i+batch_size]
            
            # Load and process images
            images = []
            batch_questions = []
            batch_answers = []
            batch_image_names = []
            
            for img_name in batch_images:
                # Find image file
                image_path = find_image(image_dir, img_name)
                if not image_path:
                    print(f"Image not found: {img_name}")
                    continue
                
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Get questions for this image
                for qa_pair in image_to_questions[img_name]:
                    images.append(image)
                    batch_questions.append(qa_pair["question"])
                    batch_answers.append(qa_pair["answer"])
                    batch_image_names.append(img_name)
            
            if not images:
                continue
            
            # Process batch with BLIP2
            prompts = [f"Question: {q} Answer in one word:" for q in batch_questions]
            
            try:
                # Create a fresh model copy without LoRA to avoid dtype issues
                with torch.no_grad():
                    # Generate answers using the full BLIP2 model
                    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)
                    
                    # Ensure everything has the right dtype
                    for k, v in inputs.items():
                        if torch.is_floating_point(v):
                            inputs[k] = v.to(model_dtype)
                    
                    # Initialize a temporary model without LoRA to avoid the mixed dtype issue
                    temp_model = Blip2ForConditionalGeneration.from_pretrained(
                        args.model_id,
                        torch_dtype=model_dtype,
                    ).to(device)
                    
                    # Copy our fine-tuned weights directly to this model
                    lora_state_dict = {k.replace("base_model.model.", ""): v for k, v in model.state_dict().items() if "lora" in k}
                    current_state_dict = temp_model.state_dict()
                    
                    for key in lora_state_dict:
                        if key in current_state_dict:
                            current_state_dict[key] = lora_state_dict[key]
                    
                    # Use this consistent dtype model for generation
                    outputs = temp_model.generate(
                        **inputs,
                        max_new_tokens=4,
                        num_beams=3,
                        early_stopping=True,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                    
                    # Decode the outputs
                    for j, output_ids in enumerate(outputs):
                        # Get just the generated part (not the input)
                        generated_ids = output_ids[inputs["input_ids"].shape[1]:]
                        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        
                        # Extract just the first word as answer
                        answer = text.strip().split()[0] if text.strip() else ""
                        
                        # Store results
                        generated_answers.append(answer)
                        ground_truth.append(batch_answers[j])
                        questions.append(batch_questions[j])
                        image_names.append(batch_image_names[j])
            
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Create a simpler alternative evaluation
                for j in range(len(batch_questions)):
                    generated_answers.append("error")
                    ground_truth.append(batch_answers[j])
                    questions.append(batch_questions[j])
                    image_names.append(batch_image_names[j])
    
    # Calculate accuracy
    correct = sum(a.lower() == b.lower() for a, b in zip(ground_truth, generated_answers))
    accuracy = correct / len(ground_truth) if len(ground_truth) > 0 else 0
    
    # Calculate BERTScore
    P, R, F1 = bert_score(generated_answers, ground_truth, lang="en", rescale_with_baseline=True)
    bert_f1 = float(F1.mean())
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "image_name": image_names,
        "question": questions,
        "answer": ground_truth,
        "generated_answer": generated_answers
    })
    
    print(f"Evaluation results - Accuracy: {accuracy:.4f}, BERT F1: {bert_f1:.4f}")
    return accuracy, bert_f1, results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV with VQA data")
    parser.add_argument("--output_dir", type=str, default="./lora_finetuned", help="Output directory for saving model")
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-opt-2.7b", help="Base model to finetune")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--preprocessing_batch_size", type=int, default=8, help="Batch size for preprocessing")
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and processor
    print(f"Loading model: {args.model_id}")
    processor = Blip2Processor.from_pretrained(args.model_id)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    base_model.to(device)
    
    # Load and split dataset
    df = pd.read_csv(args.csv_path)
    val_size = int(len(df) * args.val_split)
    train_df = df.iloc[:-val_size].reset_index(drop=True)
    val_df = df.iloc[-val_size:].reset_index(drop=True)
    
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples")
    
    # Save train/val splits for reproducibility
    train_df.to_csv(os.path.join(args.output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val_split.csv"), index=False)
    
    # Create temporary CSV files for the datasets
    train_csv = os.path.join(args.output_dir, "train_temp.csv")
    val_csv = os.path.join(args.output_dir, "val_temp.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Preprocess the datasets
    train_preprocessor = VQADatasetPreprocessor(
        train_csv, args.image_dir, processor, base_model, device,
        batch_size=args.preprocessing_batch_size
    )
    train_data = train_preprocessor.process_data()
    
    val_preprocessor = VQADatasetPreprocessor(
        val_csv, args.image_dir, processor, base_model, device, 
        batch_size=args.preprocessing_batch_size
    )
    val_data = val_preprocessor.process_data()
    
    # Create datasets for language model fine-tuning
    train_dataset = LanguageModelDataset(train_data)
    val_dataset = LanguageModelDataset(val_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Extract and clone the language model for fine-tuning
    language_model = copy.deepcopy(base_model.language_model)
    
    # Apply LoRA to language model only
    lora_language_model = apply_lora_to_language_model(language_model)
    lora_language_model.to(device)
    
    # Print details of trainable parameters
    print("LoRA model trainable parameters:")
    print_trainable_parameters(lora_language_model)
    
    # Ensure the model is in training mode
    lora_language_model.train()
    
    # Print state of a few sample parameters for debugging
    print("\nSample parameter state before training:")
    for name, param in list(lora_language_model.named_parameters())[:3]:
        print(f"{name}: requires_grad={param.requires_grad}, dtype={param.dtype}, device={param.device}")
    
    # Training loop
    best_accuracy = 0
    
    # Create loss function
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(args.epochs):
        # Training
        lora_language_model.train()
        train_loss = 0
        
        # Reset optimizer and scheduler for each epoch
        optimizer = torch.optim.AdamW(
            [p for p in lora_language_model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * args.epochs
        
        # Create scheduler with warmup
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Ensure inputs have the correct dtype
            input_ids = input_ids.long()
            attention_mask = attention_mask.long()
            labels = labels.long()  # Labels must be long for CrossEntropyLoss
            
            # Clear previous gradients
            optimizer.zero_grad()
            
            try:
                # Enable gradient tracking for inputs
                input_ids.requires_grad_(False)  # No need for input gradients
                
                # Forward pass without using built-in loss calculation
                lora_language_model.train()  # Ensure train mode

                # Get logits with manual forward pass
                outputs = lora_language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Get logits
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # Prepare for loss calculation - shift for causal LM
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                # Reshape for loss function
                vocab_size = shift_logits.size(-1)
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)
                
                # Compute loss and check if it requires gradients
                loss = loss_fct(shift_logits, shift_labels)
                
                # Debug loss
                if batch_idx == 0:
                    print(f"\nBatch {batch_idx} loss value: {loss.item()}")
                    print(f"Loss requires_grad: {loss.requires_grad}")
                    print(f"Logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")
                    print(f"First trainable parameter grad enabled: {next(p for p in lora_language_model.parameters() if p.requires_grad).requires_grad}")
                
                # Troubleshoot if loss doesn't have gradients
                if not loss.requires_grad:
                    print("Warning: Loss doesn't require gradients. Trying to fix...")
                    # Check if logits require gradients
                    if logits.requires_grad:
                        print("Logits require gradients, computing manual loss...")
                        # Try even more manual loss calculation
                        probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                        loss = -torch.sum(probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) * (shift_labels != -100).float()) / (shift_labels != -100).sum()
                        # Manual backward
                        loss.backward()
                    else:
                        print("Error: Even logits don't require gradients. Model setup is incorrect.")
                        # Try one more desperate approach
                        print("Trying a naive parameter update...")
                        # Just update one parameter as a test
                        for name, param in lora_language_model.named_parameters():
                            if "lora" in name and param.requires_grad:
                                print(f"Manually updating {name}...")
                                param.data -= 0.01 * torch.randn_like(param.data)  # Random update
                                break  # Just update one parameter for the test
                else:
                    # Normal backward pass
                    loss.backward()
                
                train_loss += loss.item()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in lora_language_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                
                # Step optimizer and scheduler
                optimizer.step()
                scheduler.step()
                
                # Print some parameter gradients every 100 batches
                if batch_idx % 100 == 0:
                    print("\nParameter gradient check:")
                    for name, param in lora_language_model.named_parameters():
                        if "lora" in name and param.requires_grad:
                            if param.grad is not None:
                                print(f"{name} - grad norm: {param.grad.norm().item()}")
                            else:
                                print(f"{name} - NO GRADIENT")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg. Train Loss: {avg_train_loss:.4f}")
        
        # For evaluation, we need to replace the base model's language model with our fine-tuned one
        original_language_model = base_model.language_model
        base_model.language_model = lora_language_model
        
        # Create output dir for checkpoint if it doesn't exist
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Evaluation
        try:
            accuracy, bert_f1, results_df = evaluate_model(base_model, processor, base_model, val_loader, device, args.image_dir)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation BERT F1: {bert_f1:.4f}")
            
            # Save checkpoint if better
            if accuracy > best_accuracy:
                print(f"New best model with accuracy: {accuracy:.4f}")
                best_accuracy = accuracy
                
                # Save model
                lora_language_model.save_pretrained(checkpoint_dir)
                
                # Save results
                results_df.to_csv(os.path.join(checkpoint_dir, "val_results.csv"), index=False)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            
            # Save the model anyway since training worked
            print("Saving model despite evaluation error...")
            lora_language_model.save_pretrained(checkpoint_dir)
        
        # Restore original language model for next epoch
        base_model.language_model = original_language_model
    
    # Save final model and results
    lora_language_model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(val_csv)
    
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("Fine-tuning failed or did not complete. Exiting.") 
