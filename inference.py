import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV with VQA data")
    parser.add_argument("--output_path", type=str, default="results.csv", help="Path to save results")
    parser.add_argument("--base_model_id", type=str, default="Salesforce/blip2-opt-2.7b", 
                        help="Base model ID")
    parser.add_argument("--lora_model_path", type=str, default="mrityunjayjha/lora-blip2-vqa-model", 
                        help="Path to the LoRA fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base model and processor
    print(f"Loading base model: {args.base_model_id}")
    processor = Blip2Processor.from_pretrained(args.base_model_id, use_fast=True)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load LoRA fine-tuned model
    print(f"Loading LoRA model from: {args.lora_model_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    model.to(device)
    model.eval()

    generated_answers = []
    failed_images = 0

    # Run inference
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = str(row['image_name'])
        question = str(row['question'])

        # Direct image path - no need for complex search 
        image_path = os.path.join(args.image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            answer = "error"
            failed_images += 1
            generated_answers.append(answer)
            continue

        try:
            # Load and process the image
            image = Image.open(image_path).convert("RGB")

            # Format prompt for VQA
            prompt = f"Question: {question} Answer in one word:"

            # Process inputs
            inputs = processor(image, prompt, return_tensors="pt").to(device)

            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=4,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=1,
                )

                # Decode the response
                full = processor.decode(generated_ids[0], skip_special_tokens=True)
                answer = full.split("Answer in one word:")[-1].strip().split()[0]

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            answer = "error"
            failed_images += 1

        # Store the answer
        generated_answers.append(answer)

        # Log progress periodically
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} images. Failed: {failed_images}")

    # Add generated answers to dataframe
    df["generated_answer"] = generated_answers

    # Save results to CSV
    df.to_csv(args.output_path, index=False)
    print(f"Saved results to: {args.output_path}")

if __name__ == "__main__":
    main() 