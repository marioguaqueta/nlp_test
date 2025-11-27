import pandas as pd
from datasets import Dataset
import json
import glob
import os

def load_data(train_path, test_path=None):
    """
    Loads train and test data.
    train_path: Path to the directory containing JSON training files (e.g., 'data/train/train')
    test_path: Path to the CSV test file (e.g., 'data/test.csv')
    """
    
    # Load Train
    train_dataset = None
    if train_path:
        try:
            # Check if it's a directory or file
            if os.path.isdir(train_path):
                files = glob.glob(os.path.join(train_path, "*.json"))
                all_data = []
                print(f"Found {len(files)} training files in {train_path}")
                for f in files:
                    try:
                        with open(f, 'r', encoding='utf-8') as json_file:
                            data = json.load(json_file)
                            # data is a list of objects
                            for item in data:
                                all_data.append({
                                    "input": item["natural_language"],
                                    "target": json.dumps(item["json_data"], ensure_ascii=False)
                                })
                    except Exception as e:
                        print(f"Error reading {f}: {e}")
                
                if all_data:
                    df_train = pd.DataFrame(all_data)
                    train_dataset = Dataset.from_pandas(df_train)
                else:
                    print("No training data found in JSON files.")
            else:
                # Fallback to CSV if a file path is provided (legacy support)
                df_train = pd.read_csv(train_path)
                cols = df_train.columns.str.lower()
                if 'input' not in cols and 'text' in cols:
                    df_train.rename(columns={'text': 'input'}, inplace=True)
                if 'target' not in cols and 'json' in cols:
                    df_train.rename(columns={'json': 'target'}, inplace=True)
                train_dataset = Dataset.from_pandas(df_train)

        except Exception as e:
            print(f"Error loading train data: {e}")
            train_dataset = None

    # Load Test
    test_dataset = None
    if test_path:
        try:
            if test_path.endswith('.json'):
                with open(test_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # data is a list of dicts with 'id' and 'natural_language'
                # We need to rename 'natural_language' to 'input' to match the pipeline
                for item in data:
                    if 'natural_language' in item:
                        item['input'] = item['natural_language']
                
                df_test = pd.DataFrame(data)
                test_dataset = Dataset.from_pandas(df_test)
            else:
                df_test = pd.read_csv(test_path)
                test_dataset = Dataset.from_pandas(df_test)
        except Exception as e:
            print(f"Error loading test data: {e}")
            
    return train_dataset, test_dataset

def format_instruction(example):
    """
    Formats the input for the model.
    """
    return f"Convertir la siguiente orden de compra a JSON:\n\n{example['input']}\n\nJSON:\n"
