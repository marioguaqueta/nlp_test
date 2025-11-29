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

MODEL_BASE = "Qwen/Qwen3-0.6B-Base" # Must match training
ADAPTER_PATH = "models/qwen_finetuned"
DATA_PATH = "eval.json"
OUTPUT_DIR = "output"

def extract_json(text):
    """
    Extracts the JSON part from the generated text.
    """
    try:
        # Find the first '{'
        start = text.find('{')
        if start == -1:
            return "{}"
        
        # Find the last '}'
        end = text.rfind('}')
        if end == -1:
            return "{}"
            
        # Extract substring
        json_candidate = text[start:end+1]
        
        # Validate
        try:
            json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            # If simple extraction fails, it might be due to unescaped characters or issues.
            # We could try to fix it, but for now let's return the candidate if it looks like JSON
            # or try to find a smaller valid JSON inside?
            # Let's just return the candidate and hope for the best, or return {}
            # Returning the candidate is better than {} for debugging, but for submission valid JSON is preferred.
            # Let's try to be lenient.
            return json_candidate
    except:
        return "{}"

def inference():
    parser = argparse.ArgumentParser(description="Run inference on test data")
    
    # Model and data paths
    parser.add_argument("--model_base", type=str, default=MODEL_BASE,
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH,
                        help="Path to fine-tuned adapter")
    parser.add_argument("--data_path", type=str, default=DATA_PATH,
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for submission file")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    
    # Output options
    parser.add_argument("--no_timestamp", action="store_true",
                        help="Don't add timestamp to output filename")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Custom suffix for output filename")
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
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
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if device == "mps":
        model.to(device)
        
    # Load Adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load Data
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
                temperature=args.temperature, # Low temp for deterministic output
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        # generated_text includes the prompt usually
        response = generated_text[len(prompt):]
        
        json_pred = extract_json(response)
        
        predictions.append(json_pred)
        ids.append(example['id'])

    # Create output filename with timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.no_timestamp:
        # No timestamp
        if args.output_suffix:
            output_filename = f"submission{args.output_suffix}.csv"
        else:
            output_filename = "submission.csv"
    else:
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_suffix:
            output_filename = f"submission_{timestamp}{args.output_suffix}.csv"
        else:
            output_filename = f"submission_{timestamp}.csv"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Create Submission
    df_sub = pd.DataFrame({'id': ids, 'prediction': predictions})
    # Ensure double quotes are escaped properly by using csv.QUOTE_NONNUMERIC or QUOTE_ALL if needed, 
    # but pandas default is usually fine. 
    # The user example: 1,"{""buyer"": ...}" shows standard CSV escaping (double quote becomes two double quotes).
    # Pandas to_csv default handles this.
    df_sub.to_csv(output_path, index=False, quoting=1) # 1 is csv.QUOTE_ALL, ensuring strings are quoted
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Submission saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Output directory: {args.output_dir}")
    
    # Show sample predictions
    print(f"\nSample predictions (first 3):")
    for i in range(min(3, len(predictions))):
        print(f"\nID {ids[i]}:")
        pred_preview = predictions[i][:100] + "..." if len(predictions[i]) > 100 else predictions[i]
        print(f"  {pred_preview}")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    inference()
