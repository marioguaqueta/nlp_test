import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import load_data, format_instruction
import pandas as pd
import json
from tqdm import tqdm
import argparse
import os
from datetime import datetime

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

def inference_batch(model, tokenizer, batch_examples, device, args):
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
        # Use generation parameters from args
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
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
    parser = argparse.ArgumentParser(description="Run optimized inference for Kaggle submission")
    
    # Model and data paths
    parser.add_argument("--model_base", type=str, default="Qwen/Qwen3-0.6B-Base", 
                        help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default="models/qwen_finetuned",
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--data_path", type=str, default="eval.json",
                        help="Path to test data (JSON or CSV)")
    parser.add_argument("--output_path", type=str, default="output/submission.csv",
                        help="Path to save submission CSV")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference (higher = faster but more memory)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for sampling (lower = more deterministic)")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search (1 = greedy)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    
    # Other options
    parser.add_argument("--merge_adapter", action="store_true", default=True,
                        help="Merge adapter weights for faster inference")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix to add to output filename (e.g., '_temp01')")
    
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
    print(f"Output path: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Do sample: {args.do_sample}")
    print(f"Num beams: {args.num_beams}")
    if args.do_sample:
        print(f"Top-p: {args.top_p}")
        print(f"Top-k: {args.top_k}")
    print("="*60)
    print()
    
    # Load model
    print(f"Loading model from {args.adapter_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    if device == "mps":
        model.to(device)
        
    # Load Adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    
    # Merge adapter if requested
    if args.merge_adapter and hasattr(model, 'merge_and_unload'):
        print("Merging adapter weights for faster inference...")
        model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    _, test_dataset = load_data(None, test_path=args.data_path)
    if test_dataset is None:
        print(f"Error: Test dataset not found at {args.data_path}")
        return

    predictions = []
    ids = []

    print(f"\nGenerating predictions for {len(test_dataset)} examples...")
    
    # Create batches
    num_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(test_dataset), args.batch_size), total=num_batches, desc="Inference"):
        batch_examples = test_dataset[i:min(i + args.batch_size, len(test_dataset))]
        
        # Handle single example vs batch
        if not isinstance(batch_examples, list):
            batch_examples = [batch_examples]
        
        # Process batch
        responses = inference_batch(model, tokenizer, batch_examples, device, args)
        
        # Extract JSON from responses
        for j, response in enumerate(responses):
            json_pred = extract_json(response)
            predictions.append(json_pred)
            ids.append(batch_examples[j]['id'])

    # Create output path with suffix if provided
    output_path = args.output_path
    if args.output_suffix:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}{args.output_suffix}{ext}"
    
    # Create Submission
    df_sub = pd.DataFrame({'id': ids, 'prediction': predictions})
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_sub.to_csv(output_path, index=False, quoting=1)
    
    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Submission saved to: {output_path}")
    print(f"Total examples: {len(test_dataset)}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    
    # Show sample predictions
    print(f"\nSample predictions (first 3):")
    for i in range(min(3, len(predictions))):
        print(f"\nID {ids[i]}:")
        print(f"  {predictions[i][:100]}..." if len(predictions[i]) > 100 else f"  {predictions[i]}")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    inference()
