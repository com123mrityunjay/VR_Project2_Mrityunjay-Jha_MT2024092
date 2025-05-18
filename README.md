# BLIP-2 with LoRA Fine-Tuning for Visual Question Answering

This repository contains the code and resources for fine-tuning BLIP-2 models on Visual Question Answering (VQA) tasks using Low-Rank Adaptation (LoRA).

## Project Overview

This project demonstrates how to improve BLIP-2's Visual Question Answering capabilities by using LoRA fine-tuning. LoRA is an efficient fine-tuning technique that significantly reduces the number of trainable parameters while maintaining high performance.

### Performance Metrics

| Model | BERT Score |
|-------|------------|
| BLIP-2 Baseline | 0.66 |
| BLIP-2 with LoRA | 0.70 |

The improvement in BERT Score demonstrates the effectiveness of our fine-tuning approach while keeping the parameter count low and training efficient.

## Repository Structure

```
.
├── train_model_code/
│   └── lora_finetune.py       # LoRA fine-tuning implementation
├── vqa_generator/
│   └── abo_vqa_generator.py   # VQA data generation from Amazon Berkeley Objects
├── results/
│   ├── baseline_results.csv   # Results from base BLIP-2 model
│   └── results.csv            # Results from LoRA fine-tuned model
├── inference.py               # Inference script for the fine-tuned model
├── requirements.txt           # Required dependencies
├── metadata.csv               # Dataset with image_name, question, answer
└── README.md                  # This file
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/blip2-lora-vqa.git
cd blip2-lora-vqa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The project uses VQA data from the Amazon Berkeley Objects (ABO) dataset. The `vqa_generator` module helps create VQA pairs for product images using the Gemini API.

To generate your own VQA data:

```bash
python vqa_generator/abo_vqa_generator.py \
  --api_key YOUR_GEMINI_API_KEY \
  --abo_dir /path/to/abo_dataset \
  --output_csv vqa_data.csv \
  --num_images 1000
```

The expected data format is a CSV file with the following columns:
- `image_name`: Image filename
- `question`: Question about the image
- `answer`: Ground-truth answer (typically one word)

## Fine-Tuning with LoRA

To fine-tune the BLIP-2 model with LoRA:

```bash
python train_model_code/lora_finetune.py \
  --image_dir /path/to/images \
  --csv_path metadata.csv \
  --output_dir ./output_model \
  --base_model_id Salesforce/blip2-opt-2.7b \
  --batch_size 4 \
  --epochs 3 \
  --learning_rate 1e-4
```

### LoRA Configuration

The LoRA configuration targets specific modules in BLIP-2:

```python
lora_config = LoraConfig(
    r=16,                       # Rank dimension
    lora_alpha=32,              # Scaling factor
    lora_dropout=0.05,          # Dropout probability
    target_modules=[            # Target specific layers
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
        "gate_proj", "down_proj", "up_proj"      # MLP layers
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

## Running Inference

To run inference with the fine-tuned model:

```bash
python inference.py \
  --image_dir /path/to/images \
  --csv_path test_metadata.csv \
```

### Inference Arguments

- `--image_dir`: Path to the directory containing images
- `--csv_path`: Path to CSV with image_name and question columns
- `--output_path`: Path to save results (default: results.csv)
- `--base_model_id`: Base BLIP-2 model ID (default: Salesforce/blip2-opt-2.7b)
- `--lora_model_path`: Path to the LoRA fine-tuned model
- `--batch_size`: Batch size for inference (default: 1)

## Available BLIP-2 Models

You can use different BLIP-2 models based on your hardware capabilities:

- `Salesforce/blip2-opt-2.7b` (default, ~5GB VRAM)
- `Salesforce/blip2-opt-6.7b` (larger model, ~12GB VRAM)
- `Salesforce/blip2-flan-t5-xl` (alternative architecture, ~5GB VRAM)

## Technical Notes

- The model is optimized for one-word answers (as configured in the VQA generator)
- For optimal training, a GPU with at least 8GB VRAM is recommended
- The inference script uses torch half-precision (float16) if CUDA is available


## Requirements

Key dependencies include:
- transformers
- peft
- torch
- torchvision
- PIL
- bert-score
- pandas
- tqdm
- google.generativeai (for VQA data generation)




