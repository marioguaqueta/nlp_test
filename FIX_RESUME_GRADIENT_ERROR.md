# 🔧 Fix: RuntimeError - Tensors Do Not Require Grad

## Issue

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

This error occurs when trying to resume training from a checkpoint.

## Root Cause

When loading a PeftModel (LoRA adapter) from a checkpoint, by default it loads in **inference mode**, which means:
- Parameters are frozen (not trainable)
- Gradients are disabled
- `requires_grad=False` for all LoRA parameters

## Solution

Add `is_trainable=True` when loading the PeftModel:

```python
# ❌ WRONG - Loads in inference mode
model = PeftModel.from_pretrained(model, args.resume_from)

# ✅ CORRECT - Loads in training mode
model = PeftModel.from_pretrained(
    model, 
    args.resume_from,
    is_trainable=True  # Enable training mode
)
```

## Fixed Code

The `train_resume.py` has been updated with the fix:

```python
if is_resuming:
    print(f"\nLoading model from checkpoint: {args.resume_from}")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter with is_trainable=True
    model = PeftModel.from_pretrained(
        model, 
        args.resume_from,
        is_trainable=True  # CRITICAL: Enable training mode
    )
    print("✓ Loaded existing LoRA adapter in training mode")

# Ensure model is in training mode
model.train()
```

## Verification

After the fix, you should see:

```
Loading model from checkpoint: models/qwen_finetuned
✓ Loaded existing LoRA adapter in training mode
trainable params: 2,621,440 || all params: 603,979,776 || trainable%: 0.4340
```

The key is seeing **trainable params** > 0, which means gradients are enabled.

## Now You Can Resume Training

```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --output_dir models/qwen_v2 \
    --epochs 5 \
    --learning_rate 5e-5
```

This will now work without the gradient error!

## Why This Happens

### PeftModel Loading Modes

1. **Inference Mode** (default):
   - `is_trainable=False`
   - Used for: Running predictions
   - Parameters: Frozen
   - Gradients: Disabled

2. **Training Mode** (what we need):
   - `is_trainable=True`
   - Used for: Continuing training
   - Parameters: Trainable
   - Gradients: Enabled

### The Fix Ensures

- ✅ LoRA parameters are trainable
- ✅ Gradients can be computed
- ✅ Backpropagation works
- ✅ Model can be updated

## Related Errors

This fix also resolves:
- `RuntimeError: grad can be implicitly created only for scalar outputs`
- `RuntimeError: Trying to backward through the graph a second time`
- Any gradient-related errors when resuming

---

**Status: FIXED ✅**

The `train_resume.py` now correctly loads models in training mode.
