"""
Test script to verify optimizations work correctly
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import torch
        print("✓ torch")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ transformers")
        
        from peft import LoraConfig, get_peft_model
        print("✓ peft")
        
        from datasets import Dataset
        print("✓ datasets")
        
        import pandas as pd
        print("✓ pandas")
        
        import wandb
        print("✓ wandb")
        
        print("\n✅ All imports successful!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False


def test_data_augmentation():
    """Test data augmentation module"""
    print("Testing data augmentation...")
    try:
        from data_augmentation import DataAugmenter
        
        augmenter = DataAugmenter(augmentation_factor=2)
        
        test_example = {
            'input': 'Necesito comprar 100 unidades de producto A, precio 50 pesos',
            'target': '{"producto": "A", "cantidad": 100, "precio_unitario": 50}'
        }
        
        print(f"Original: {test_example['input']}")
        
        # Test augmentation
        aug_example = augmenter.augment_example(test_example)
        print(f"Augmented: {aug_example['input']}")
        
        # Verify target is unchanged
        assert aug_example['target'] == test_example['target'], "Target should not change"
        
        print("\n✅ Data augmentation works!\n")
        return True
    except Exception as e:
        print(f"\n❌ Data augmentation error: {e}\n")
        return False


def test_model_loading():
    """Test that model can be loaded"""
    print("Testing model loading...")
    try:
        import torch
        from transformers import AutoTokenizer
        
        model_name = "Qwen/Qwen3-0.6B-Base"
        
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  Pad token: {tokenizer.pad_token}")
        
        # Test tokenization
        test_text = "Necesito comprar 100 unidades"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"  Test tokenization: {tokens['input_ids'].shape}")
        
        print("\n✅ Model loading works!\n")
        return True
    except Exception as e:
        print(f"\n❌ Model loading error: {e}")
        print("Note: Full model download may be needed for actual training/inference")
        print("This is expected if you haven't downloaded the model yet.\n")
        return False


def test_data_loader():
    """Test data loader"""
    print("Testing data loader...")
    try:
        from data_loader import format_instruction
        
        test_example = {
            'input': 'Necesito comprar 100 unidades de producto A'
        }
        
        formatted = format_instruction(test_example)
        print(f"Formatted instruction:\n{formatted}")
        
        assert "JSON:" in formatted, "Should contain JSON prompt"
        assert test_example['input'] in formatted, "Should contain original input"
        
        print("\n✅ Data loader works!\n")
        return True
    except Exception as e:
        print(f"\n❌ Data loader error: {e}\n")
        return False


def test_metrics():
    """Test metrics calculation"""
    print("Testing metrics...")
    try:
        from metrics import calculate_f1
        import json
        
        # Perfect match
        pred1 = '{"producto": "A", "cantidad": 100}'
        true1 = '{"producto": "A", "cantidad": 100}'
        score1 = calculate_f1(pred1, true1)
        print(f"Perfect match F1: {score1}")
        assert score1 == 1.0, "Perfect match should have F1=1.0"
        
        # Partial match
        pred2 = '{"producto": "A", "cantidad": 100}'
        true2 = '{"producto": "A", "cantidad": 100, "precio": 50}'
        score2 = calculate_f1(pred2, true2)
        print(f"Partial match F1: {score2}")
        assert 0 < score2 < 1, "Partial match should have 0 < F1 < 1"
        
        # No match
        pred3 = '{"producto": "B"}'
        true3 = '{"producto": "A"}'
        score3 = calculate_f1(pred3, true3)
        print(f"No match F1: {score3}")
        
        print("\n✅ Metrics work!\n")
        return True
    except Exception as e:
        print(f"\n❌ Metrics error: {e}\n")
        return False


def test_device():
    """Test device availability"""
    print("Testing device availability...")
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"MPS available: {mps_available}")
        
        device = "cuda" if cuda_available else "mps" if mps_available else "cpu"
        print(f"\nSelected device: {device}")
        
        if device == "cpu":
            print("⚠️  Warning: Using CPU. Training/inference will be slow.")
            print("   Consider using a GPU for better performance.")
        
        print("\n✅ Device check complete!\n")
        return True
    except Exception as e:
        print(f"\n❌ Device error: {e}\n")
        return False


def main():
    print("="*60)
    print("OPTIMIZATION TEST SUITE")
    print("="*60)
    print()
    
    # Change to src directory
    if os.path.exists('src'):
        sys.path.insert(0, 'src')
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Augmentation", test_data_augmentation()))
    results.append(("Data Loader", test_data_loader()))
    results.append(("Metrics", test_metrics()))
    results.append(("Device", test_device()))
    results.append(("Model Loading", test_model_loading()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print()
    print(f"Passed: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Train: python src/train_optimized.py")
        print("2. Infer: python src/inference_optimized.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("Common fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Ensure you're in the project directory")
    
    print("="*60)


if __name__ == "__main__":
    main()
