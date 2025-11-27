import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import load_data, format_instruction
import pandas as pd
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

MODEL_BASE = "Qwen/Qwen3-0.6B-Base"
ADAPTER_PATH = "models/qwen_finetuned"
DATA_PATH = "eval.json"
OUTPUT_PATH = "output/submission.csv"

# OPTIMIZATION PARAMETERS
BATCH_SIZE = 8  # Process multiple examples at once
MAX_NEW_TOKENS = 512
USE_CACHE = True  # Enable KV cache for faster generation
NUM_BEAMS = 1  # Greedy decoding is faster than beam search

def extract_json(text):
    """
    Extracts the JSON part from the generated text.
    """
    try:
        start = text.find('{')
        if start == -1:
            return "{}"
        
        end = text.rfind('}')
        if end == -1:
            return "{}"
            
        json_candidate = text[start:end+1]
        
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            return json_candidate
    except:
        return "{}"

def collate_fn(batch):
    """Custom collate function for batching"""
    return batch

def inference_batch(model, tokenizer, batch_examples, device):
    """Process a batch of examples"""
    prompts = [format_instruction(example) for example in batch_examples]
    
    # Tokenize with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        # Use optimized generation parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=False,
            num_beams=NUM_BEAMS,
            use_cache=USE_CACHE,
            pad_token_id=tokenizer.eos_token_id,
            # Early stopping when EOS is generated
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode all outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract responses (remove prompts)
    responses = []
    for i, generated_text in enumerate(generated_texts):
        response = generated_text[len(prompts[i]):]
        responses.append(response)
    
    return responses

def inference():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading model from {ADAPTER_PATH}...")
    
    # Load Base Model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,  # Reduce CPU memory usage
    )
    
    if device == "mps":
        model.to(device)
        
    # Load Adapter
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    # Enable inference optimizations
    if hasattr(model, 'merge_and_unload'):
        print("Merging adapter weights for faster inference...")
        model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Data
    _, test_dataset = load_data(None, test_path=DATA_PATH)
    if test_dataset is None:
        print("Test dataset not found.")
        return

    predictions = []
    ids = []

    print(f"Generating predictions for {len(test_dataset)} examples...")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Create batches
    num_batches = (len(test_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(test_dataset), BATCH_SIZE), total=num_batches):
        batch_examples = test_dataset[i:min(i + BATCH_SIZE, len(test_dataset))]
        
        # Handle single example vs batch
        if not isinstance(batch_examples, list):
            batch_examples = [batch_examples]
        
        # Process batch
        responses = inference_batch(model, tokenizer, batch_examples, device)
        
        # Extract JSON from responses
        for j, response in enumerate(responses):
            json_pred = extract_json(response)
            predictions.append(json_pred)
            ids.append(batch_examples[j]['id'])

    # Create Submission
    df_sub = pd.DataFrame({'id': ids, 'prediction': predictions})
    df_sub.to_csv(OUTPUT_PATH, index=False, quoting=1)
    print(f"Submission saved to {OUTPUT_PATH}")
    
    # Print statistics
    print(f"\nInference Statistics:")
    print(f"Total examples: {len(test_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {device}")

if __name__ == "__main__":
    inference()
