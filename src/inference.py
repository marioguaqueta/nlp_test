import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_loader import load_data, format_instruction
import pandas as pd
import json
import re
from tqdm import tqdm

MODEL_BASE = "Qwen/Qwen3-0.6B-Base" # Must match training
ADAPTER_PATH = "models/qwen_finetuned"
DATA_PATH = "eval.json"
OUTPUT_PATH = "output/submission.csv"

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
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading model from {ADAPTER_PATH}...")
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "mps":
        model.to(device)
        
    # Load Adapter
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    
    # Load Data
    _, test_dataset = load_data(None, test_path=DATA_PATH)
    if test_dataset is None:
        print("Test dataset not found.")
        return

    predictions = []
    ids = []

    print("Generating predictions...")
    for example in tqdm(test_dataset):
        prompt = format_instruction(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1, # Low temp for deterministic output
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        # generated_text includes the prompt usually
        response = generated_text[len(prompt):]
        
        json_pred = extract_json(response)
        
        predictions.append(json_pred)
        ids.append(example['id'])

    # Create Submission
    # Create Submission
    df_sub = pd.DataFrame({'id': ids, 'prediction': predictions})
    # Ensure double quotes are escaped properly by using csv.QUOTE_NONNUMERIC or QUOTE_ALL if needed, 
    # but pandas default is usually fine. 
    # The user example: 1,"{""buyer"": ...}" shows standard CSV escaping (double quote becomes two double quotes).
    # Pandas to_csv default handles this.
    df_sub.to_csv(OUTPUT_PATH, index=False, quoting=1) # 1 is csv.QUOTE_ALL, ensuring strings are quoted
    print(f"Submission saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    inference()
