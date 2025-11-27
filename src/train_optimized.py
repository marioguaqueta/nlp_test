import os
import torch
import copy
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from data_loader import load_data, format_instruction
from data_augmentation import DataAugmenter
import pandas as pd
import wandb
from metrics import calculate_f1
import json
import argparse

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
OUTPUT_DIR = "models/qwen_finetuned"
DATA_DIR = "data"

class JsonEvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, model, max_new_tokens=512):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def on_evaluate(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        print("\nRunning evaluation on validation set...")
        
        # Limit evaluation for speed
        eval_subset = self.eval_dataset
        if len(eval_subset) > 50:
            eval_subset = eval_subset.select(range(50))
            
        f1_scores = []
        self.model.eval()
        
        for example in eval_subset:
            prompt = format_instruction({"input": example["input"]})
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens, 
                    temperature=0.1, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True  # Enable KV cache
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            try:
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}')
                if json_start != -1 and json_end != -1:
                    pred_json = generated_text[json_start:json_end+1]
                else:
                    pred_json = "{}"
            except:
                pred_json = "{}"
                
            true_json = example["target"]
            score = calculate_f1(pred_json, true_json)
            f1_scores.append(score)
            
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        print(f"Epoch {state.epoch}: Average F1 Score = {avg_f1:.4f}")
        
        if wandb.run:
            wandb.log({"eval/f1_score": avg_f1, "epoch": state.epoch})
            
        self.model.train()


def train():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model with optimizations")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use_augmentation", action="store_true", default=True, help="Use data augmentation")
    parser.add_argument("--augmentation_factor", type=int, default=2, help="Data augmentation factor")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing")
    
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(
        project="canonicalization-qwen-optimized", 
        name=f"qwen-finetune-aug{args.augmentation_factor}-r{args.lora_r}",
        config=vars(args)
    )

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model with optimizations
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    if device == "mps":
        model.to(device)

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # LoRA Config with improved parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,  # Increased from 8 to 16 for better capacity
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # More modules
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Data
    full_dataset, _ = load_data("train/train")
    if full_dataset is None:
        print("Train dataset not found. Please ensure 'train/train' directory exists and contains JSON files.")
        return

    # Apply Data Augmentation
    if args.use_augmentation:
        print(f"\nApplying data augmentation with factor {args.augmentation_factor}...")
        print(f"Original dataset size: {len(full_dataset)}")
        
        augmenter = DataAugmenter(augmentation_factor=args.augmentation_factor)
        full_dataset = augmenter.augment_dataset(full_dataset)
        
        print(f"Augmented dataset size: {len(full_dataset)}")

    # Split Data
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"\nTraining on {len(train_dataset)} examples")
    print(f"Validating on {len(eval_dataset)} examples")

    # Preprocess Data - Fixed to handle BatchEncoding properly
    def preprocess_function(examples):
        # Format full texts (instruction + target + EOS)
        inputs = [format_instruction({"input": inp}) for inp in examples["input"]]
        targets = examples["target"]
        full_texts = [inp + tgt + tokenizer.eos_token for inp, tgt in zip(inputs, targets)]
        
        # Tokenize everything together
        tokenized = tokenizer(
            full_texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding=False,  # Dynamic padding will be done by data collator
        )
        
        # Convert BatchEncoding to plain dict with independent lists
        # This ensures no shared references with tokenizer internals
        result = {
            "input_ids": [list(ids) for ids in tokenized["input_ids"]],
            "attention_mask": [list(mask) for mask in tokenized["attention_mask"]],
        }
        
        # Create labels as independent copy
        result["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        
        return result

    print("\nTokenizing datasets...")
    tokenized_train = train_dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )

    # Training Arguments with optimizations
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        
        # Optimization settings
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        
        # Learning rate scheduler
        lr_scheduler_type="cosine",
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Saving
        save_strategy="epoch",
        save_total_limit=2,  # Keep only best 2 checkpoints
        
        # Evaluation
        eval_strategy="no",  # Using custom callback
        
        # Mixed precision
        fp16=(device == "cuda"),
        bf16=False,
        
        # Device settings
        use_mps_device=(device == "mps"),
        
        # Memory optimizations
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",  # Use PyTorch AdamW
        
        # Reporting
        report_to="wandb",
        
        # Performance
        dataloader_num_workers=0,  # Adjust based on your system
        dataloader_pin_memory=True,
        
        # Reproducibility
        seed=42,
    )

    # Data collator with dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for tensor cores
    )

    # Initialize Callback
    eval_callback = JsonEvaluationCallback(eval_dataset, tokenizer, model)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        callbacks=[eval_callback]
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save Model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"{'='*50}\n")
    
    wandb.finish()


if __name__ == "__main__":
    train()
