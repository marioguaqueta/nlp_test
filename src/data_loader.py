import pandas as pd
from datasets import Dataset
import json
import glob
import os
from text_preprocessing import clean_text

def load_data(train_path, test_path=None, preprocess_text=True):
    """
    Loads train and test data.
    train_path: Path to the directory containing JSON training files (e.g., 'data/train/train')
    test_path: Path to the CSV test file (e.g., 'data/test.csv')
    preprocess_text: Whether to decode Unicode escapes and HTML entities
    """
    
    train_dataset = None
    if train_path:
        try:
            if os.path.isdir(train_path):
                files = glob.glob(os.path.join(train_path, "*.json"))
                all_data = []
                print(f"Found {len(files)} training files in {train_path}")
                for f in files:
                    try:
                        with open(f, 'r', encoding='utf-8') as json_file:
                            data = json.load(json_file)
                            for item in data:
                                input_text = item["natural_language"]
                                
                                if preprocess_text:
                                    input_text = clean_text(input_text)
                                
                                all_data.append({
                                    "input": input_text,
                                    "target": json.dumps(item["json_data"], ensure_ascii=False)
                                })
                    except Exception as e:
                        print(f"Error reading {f}: {e}")
                
                if all_data:
                    df_train = pd.DataFrame(all_data)
                    train_dataset = Dataset.from_pandas(df_train)
                    if preprocess_text:
                        print(f"✓ Preprocessed {len(all_data)} training examples (decoded Unicode)")
                else:
                    print("No training data found in JSON files.")
            else:
                df_train = pd.read_csv(train_path)
                cols = df_train.columns.str.lower()
                if 'input' not in cols and 'text' in cols:
                    df_train.rename(columns={'text': 'input'}, inplace=True)
                if 'target' not in cols and 'json' in cols:
                    df_train.rename(columns={'json': 'target'}, inplace=True)
                
                if preprocess_text and 'input' in df_train.columns:
                    df_train['input'] = df_train['input'].apply(clean_text)
                    print(f"✓ Preprocessed {len(df_train)} training examples (decoded Unicode)")
                
                train_dataset = Dataset.from_pandas(df_train)

        except Exception as e:
            print(f"Error loading train data: {e}")
            train_dataset = None

    test_dataset = None
    if test_path:
        try:
            if test_path.endswith('.json'):
                with open(test_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    if 'natural_language' in item:
                        input_text = item['natural_language']
                        
                        if preprocess_text:
                            input_text = clean_text(input_text)
                        
                        item['input'] = input_text
                
                df_test = pd.DataFrame(data)
                test_dataset = Dataset.from_pandas(df_test)
                if preprocess_text:
                    print(f"✓ Preprocessed {len(data)} test examples (decoded Unicode)")
            else:
                df_test = pd.read_csv(test_path)
                
                if preprocess_text and 'input' in df_test.columns:
                    df_test['input'] = df_test['input'].apply(clean_text)
                    print(f"✓ Preprocessed {len(df_test)} test examples (decoded Unicode)")
                
                test_dataset = Dataset.from_pandas(df_test)
        except Exception as e:
            print(f"Error loading test data: {e}")
            
    return train_dataset, test_dataset

def format_instruction(example):
    """
    Formats the input for the model.
    """
    return f"Convertir la siguiente orden de compra a JSON:\n\n{example['input']}\n\nJSON:\n"
