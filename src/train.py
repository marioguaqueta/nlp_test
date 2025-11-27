import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from data_loader import load_data, format_instruction
import pandas as pd
import wandb
from metrics import calculate_f1
import json

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base" 
MODEL_ID = "Qwen/Qwen3-0.6B-Base" 
TARGET_MODEL = "Qwen/Qwen3-0.6B-Base" 
# I will assume the user might have made a typo or has access to a private model. 
# I will use "Qwen/Qwen2.5-0.5B" which is very close to 0.6B parameters.
# Wait, Qwen2.5-0.5B is 0.5B. Qwen1.5-0.5B is 0.5B.
# Let's use "Qwen/Qwen2.5-0.5B" as the default but allow override.

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
        # This is called when trainer.evaluate() is called
        # But we need to do generation, which standard evaluate doesn't do for CausalLM easily
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        # Run evaluation at the end of each epoch
        print("\nRunning evaluation on validation set...")
        
        # Select a subset for speed if needed, but let's try full val set if small
        eval_subset = self.eval_dataset
        if len(eval_subset) > 50:
            eval_subset = eval_subset.select(range(50)) # Limit to 50 examples for speed
            
        f1_scores = []
        
        # Ensure model is in eval mode
        self.model.eval()
        
        for example in eval_subset:
            # Prepare input
            # We need to strip the target from the input for generation
            # The 'input' field in dataset contains the natural language request
            # We format it as instruction
            prompt = format_instruction({"input": example["input"]})
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens, 
                    temperature=0.1, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON
            # The generated text includes the prompt. We need to extract the part after "JSON:\n"
            # Or just find the first { and last }
            try:
                # Heuristic extraction
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}')
                if json_start != -1 and json_end != -1:
                    pred_json = generated_text[json_start:json_end+1]
                else:
                    pred_json = "{}"
            except:
                pred_json = "{}"
                
            # Ground Truth
            true_json = example["target"]
            
            # Calculate F1
            score = calculate_f1(pred_json, true_json)
            f1_scores.append(score)
            
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        print(f"Epoch {state.epoch}: Average F1 Score = {avg_f1:.4f}")
        
        # Log to WandB
        if wandb.run:
            wandb.log({"eval/f1_score": avg_f1, "epoch": state.epoch})
            
        # Set model back to train mode
        self.model.train()

def train():
    # Initialize WandB
    wandb.init(project="canonicalization-qwen", name="qwen-finetune-run")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None # MPS usually handles device placement manually or via .to()
    )
    if device == "mps":
        model.to(device)

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"] # Common for attention layers
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Data
    # The user has data in 'train/train' directory containing JSON files
    full_dataset, _ = load_data("train/train")
    if full_dataset is None:
        print("Train dataset not found. Please ensure 'train/train' directory exists and contains JSON files.")
        return

    # Split Data
    # 90% train, 10% validation
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Training on {len(train_dataset)} examples, Validating on {len(eval_dataset)} examples")

    # Preprocess Data
    def preprocess_function(examples):
        inputs = [format_instruction({"input": inp}) for inp in examples["input"]]
        targets = examples["target"]
        
        # Combine input and target
        full_texts = [i + t + tokenizer.eos_token for i, t in zip(inputs, targets)]
        
        model_inputs = tokenizer(full_texts, max_length=512, truncation=True, padding="max_length")
        
        # Create labels (same as input_ids for Causal LM)
        # Ideally we mask the instruction part, but for simplicity we train on all
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # Optional: Mask the input part in labels so we don't train on it
        # This requires finding the length of the input tokens.
        # Skipping for simplicity in this baseline.
        
        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    # We don't necessarily need to tokenize eval for the callback since we generate from raw input,
    # but standard evaluation loop might need it. We skip standard eval loop for custom generation.

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no", # We use custom callback
        fp16=(device == "cuda"), # MPS doesn't always support fp16 training well, better to use default (float32) or bf16 if supported
        use_mps_device=(device == "mps"),
        report_to="wandb"
    )

    # Initialize Callback
    eval_callback = JsonEvaluationCallback(eval_dataset, tokenizer, model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[eval_callback]
    )

    trainer.train()
    
    # Save Model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete. Model saved.")
    wandb.finish()

if __name__ == "__main__":
    train()
