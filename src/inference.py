import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import load_data, format_instruction
import pandas as pd
import json
import argparse
import os
from datetime import datetime
import re
from tqdm import tqdm

MODEL_BASE = "Qwen/Qwen3-0.6B-Base" 
ADAPTER_PATH = "models/qwen_finetuned"
DATA_PATH = "eval.json"
OUTPUT_DIR = "output"

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

def inference():
    parser = argparse.ArgumentParser(description="Run inference on test data")
    
    parser.add_argument("--model_base", type=str, default=MODEL_BASE,
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH,
                        help="Path to fine-tuned adapter")
    parser.add_argument("--data_path", type=str, default=DATA_PATH,
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for submission file")
    
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    
    parser.add_argument("--no_timestamp", action="store_true",
                        help="Don't add timestamp to output filename")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Custom suffix for output filename")
    
    args = parser.parse_args()
    
    sdevice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("="*60)
    print("INFERENCE CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model base: {args.model_base}")
    print(f"Adapter path: {args.adapter_path}")
    print(f"Data path: {args.data_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Num beams: {args.num_beams}")
    print("="*60)
    print()
    
    print(f"Loading model from {args.adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "mps":
        model.to(device)
        
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nLoading test data from {args.data_path}...")
    _, test_dataset = load_data(None, test_path=args.data_path)
    if test_dataset is None:
        print(f"Error: Test dataset not found at {args.data_path}")
        return

    predictions = []
    ids = []

    print(f"\nGenerating predictions for {len(test_dataset)} examples...")
    for example in tqdm(test_dataset, desc="Inference"):
        prompt = format_instruction(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature, 
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]
        
        json_pred = extract_json(response)
        
        predictions.append(json_pred)
        ids.append(example['id'])

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.no_timestamp:
        if args.output_suffix:
            output_filename = f"submission{args.output_suffix}.csv"
        else:
            output_filename = "submission.csv"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_suffix:
            output_filename = f"submission_{timestamp}{args.output_suffix}.csv"
        else:
            output_filename = f"submission_{timestamp}.csv"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    df_sub = pd.DataFrame({'id': ids, 'prediction': predictions})
    df_sub.to_csv(output_path, index=False, quoting=1)
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Submission saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    inference()
