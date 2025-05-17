# LoRA Fine-Tuned BLIP2 for Visual Question Answering

This repository contains code for fine-tuning the BLIP-2 model on Visual Question Answering (VQA) tasks using Low-Rank Adaptation (LoRA) and running inference with the fine-tuned model.

## Project Overview

The BLIP-2 baseline model achieves:
- BERT Score: 0.66

After LoRA fine-tuning, the model achieves:
- BERT Score: 0.70

This improvement demonstrates the effectiveness of our fine-tuning approach while keeping the parameter count low and training efficient.

## Repository Structure

```
.
├── finetuned_model/
│   └── lora_finetune.py      # LoRA fine-tuning script
├── vqa_generator/
│   └── abo_vqa_generator.py  # VQA data generation utility
├── results/                  # Directory for storing results
├── inference.py              # Inference script for the fine-tuned model
├── requirements.txt          # Required dependencies
├── metadata.csv              # Dataset metadata
└── README.md                 # This file
```

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Upload your fine-tuned LoRA model to Hugging Face (optional):

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Log in to Hugging Face
huggingface-cli login

# Create a repository for your model
python -c "from huggingface_hub import HfApi; api = HfApi(); api.create_repo(repo_id='YOUR_USERNAME/lora-blip2-vqa-model', private=False)"

# Push your local model to Hugging Face
python -c "from huggingface_hub import upload_folder; upload_folder(folder_path='path/to/your/lora/model', repo_id='YOUR_USERNAME/lora-blip2-vqa-model')"
```

## Fine-Tuning

To fine-tune the BLIP-2 model with LoRA:

```bash
python finetuned_model/lora_finetune.py \
  --image_dir /path/to/images \
  --csv_path /path/to/dataset.csv \
  --output_dir ./output_model \
  --base_model_id Salesforce/blip2-opt-2.7b \
  --batch_size 4 \
  --epochs 3 \
  --learning_rate 1e-4
```

The script includes:
- Efficient preprocessing of the VQA dataset
- LoRA configuration targeting specific modules of the BLIP-2 model
- Training with mixed precision for faster processing
- BERT Score evaluation of generated answers

## Running Inference

To run inference with the fine-tuned model:

```bash
python inference.py \
  --image_dir /path/to/images \
  --csv_path /path/to/metadata.csv \
  --output_path results.csv \
  --base_model_id Salesforce/blip2-opt-2.7b \
  --lora_model_path YOUR_USERNAME/lora-blip2-vqa-model
```

### Inference Options:

- `--image_dir`: Path to the directory containing images
- `--csv_path`: Path to the CSV file with columns: image_name, question
- `--output_path`: Path to save the results CSV (default: results.csv)
- `--base_model_id`: Base BLIP-2 model ID (default: Salesforce/blip2-opt-2.7b)
- `--lora_model_path`: Path or HF repo ID of the LoRA fine-tuned model (default: mrityunjayjha/lora-blip2-vqa-model)
- `--batch_size`: Batch size for inference (default: 1)

The results will be saved to the specified output path.

## Model Details

### LoRA Configuration

The LoRA configuration targets the following modules in the BLIP-2 model:
- Query, key, value and output projection layers in attention blocks
- Up and down projection layers in MLP blocks

Parameters:
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05

### Available BLIP-2 Models

You can use different BLIP-2 models based on your hardware requirements:

- `Salesforce/blip2-opt-2.7b` (default, ~5GB VRAM)
- `Salesforce/blip2-opt-6.7b` (larger model, ~12GB VRAM)
- `Salesforce/blip2-flan-t5-xl` (alternative architecture, ~5GB VRAM)

## Notes

- For optimal training performance, use a GPU with at least 8GB of VRAM
- The inference script uses torch half-precision (float16) if CUDA is available
- The model is optimized to answer questions with concise, one-word responses

# BLIP-2 LoRA Fine-Tuning for VQA

This repository contains scripts for fine-tuning the BLIP-2 model using LoRA (Low-Rank Adaptation) on a Visual Question Answering (VQA) dataset. 

## Overview

The current BLIP-2 model achieves the following metrics on the VQA dataset:
- Accuracy: 0.27
- BERT Mean F1 score: 0.66

By applying LoRA fine-tuning, we aim to improve these metrics while keeping the parameter count low and training efficient.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The expected dataset format is a CSV file with the following columns:
- `image_name`: The image filename or path
- `question`: The question about the image
- `answer`: The ground truth answer (typically one word or a short phrase)

## Scripts

### 1. LoRA Fine-Tuning

```bash
python lora_finetune.py \
  --image_dir /path/to/images \
  --csv_path /path/to/dataset.csv \
  --output_dir ./lora_finetuned \
  --model_id Salesforce/blip2-opt-2.7b \
  --batch_size 4 \
  --epochs 3 \
  --learning_rate 1e-4
```

#### Arguments

- `--image_dir`: Path to the base directory containing images
- `--csv_path`: Path to the CSV file with columns: image_name, question, answer
- `--output_dir`: Output directory for saving checkpoints (default: ./lora_finetuned)
- `--model_id`: Base BLIP-2 model ID (default: Salesforce/blip2-opt-2.7b)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of epochs to train for (default: 3)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_steps`: Warmup steps (default: 50)
- `--val_split`: Validation split ratio (default: 0.2)
- `--seed`: Random seed (default: 42)

### 2. Inference with Fine-Tuned Model

```bash
python lora_inference.py \
  --image_dir /path/to/images \
  --csv_path /path/to/test_dataset.csv \
  --output_path lora_results.csv \
  --base_model_id Salesforce/blip2-opt-2.7b \
  --lora_model_path ./lora_finetuned/final_model
```

#### Arguments

- `--image_dir`: Path to the base directory containing images
- `--csv_path`: Path to the CSV file with columns: image_name, question, [answer]
- `--output_path`: Path to save the results CSV (default: lora_results.csv)
- `--base_model_id`: Base BLIP-2 model ID (default: Salesforce/blip2-opt-2.7b)
- `--lora_model_path`: Path to the LoRA fine-tuned model
- `--batch_size`: Batch size for inference (default: 1)

## LoRA Parameters

The LoRA configuration uses the following parameters:
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"

These parameters can be modified in the `apply_lora` function within the `lora_finetune.py` script.

## Available BLIP-2 Models

You can use different BLIP-2 models based on your GPU memory and performance requirements:

- `Salesforce/blip2-opt-2.7b` (default, ~5GB VRAM)
- `Salesforce/blip2-opt-6.7b` (larger model, ~12GB VRAM)
- `Salesforce/blip2-flan-t5-xl` (alternative architecture, ~5GB VRAM)
- `Salesforce/blip2-flan-t5-xxl` (largest model, ~20GB VRAM)

## Notes

- For optimal training performance, use a GPU with at least 8GB of VRAM.
- The model saves checkpoints after each epoch if the validation accuracy improves.
- The script processes images efficiently by loading them on-demand.
- LoRA significantly reduces the number of trainable parameters while still achieving good performance. 
