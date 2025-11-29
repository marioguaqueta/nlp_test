# 🔄 Resume Training Guide - Fine-tune an Existing Model

## Overview

Use `train_resume.py` to:
1. **Continue training** an existing model for more epochs
2. **Fine-tune** with different hyperparameters
3. **Improve** a model that's already partially trained

---

## Quick Start

### Continue Training from Existing Model
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5
```

This will:
- Load your existing trained model
- Train for 5 MORE epochs
- Use lower learning rate (5e-5) for fine-tuning

### Start Fresh (Same as train.py)
```bash
python3 src/train_resume.py \
    --epochs 5 \
    --lora_r 16
```

---

## Use Cases

### 1. Model Needs More Training
**Scenario**: Your model trained for 5 epochs, F1 is 0.75, still improving

**Solution**: Train 5 more epochs
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5
```

**Why lower LR**: Model is already partially trained, needs gentle updates

### 2. Improve Existing Model
**Scenario**: You have a model with F1=0.80, want to push to 0.85

**Solution**: Fine-tune with very low LR
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.02
```

**Why very low LR**: Avoid destroying existing knowledge

### 3. Different Hyperparameters
**Scenario**: Want to try different batch size or scheduler

**Solution**: Resume with new settings
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 5e-5 \
    --batch_size 2 \
    --lr_scheduler_type linear
```

---

## Parameters

### Resume-Specific

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--resume_from` | Path to existing model | `models/qwen_finetuned` |
| `--learning_rate` | Lower for fine-tuning | `5e-5` (vs `2e-4` for new) |
| `--epochs` | ADDITIONAL epochs | `5` |
| `--warmup_ratio` | Lower for resume | `0.05` (vs `0.1` for new) |

### All Other Parameters
Same as `train.py`:
- `--batch_size`
- `--gradient_accumulation_steps`
- `--weight_decay`
- `--lr_scheduler_type`
- `--max_seq_length`
- `--output_dir`

---

## Recommended Learning Rates

### For Resuming Training

| Scenario | Original LR | Resume LR | Reason |
|----------|-------------|-----------|--------|
| Continue training | 2e-4 | **5e-5** | 1/4 of original |
| Fine-tune good model | 2e-4 | **1e-5** | 1/20 of original |
| Polish final model | 2e-4 | **5e-6** | 1/40 of original |

### Rule of Thumb
- **Continue training**: Use 1/4 to 1/2 of original LR
- **Fine-tuning**: Use 1/10 to 1/20 of original LR
- **Polishing**: Use 1/20 to 1/50 of original LR

---

## Example Workflows

### Workflow 1: Incremental Training

**Step 1**: Train initial model (5 epochs)
```bash
python3 src/train.py --epochs 5
# F1: 0.75
```

**Step 2**: Continue for 5 more epochs
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5
# F1: 0.82
```

**Step 3**: Fine-tune for 3 more epochs
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5
# F1: 0.85
```

### Workflow 2: Experiment with Hyperparameters

**Step 1**: Train baseline
```bash
python3 src/train.py --epochs 5 --lora_r 16
# F1: 0.75
```

**Step 2**: Try higher weight decay
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 5e-5 \
    --weight_decay 0.05
# F1: 0.78 (better generalization)
```

**Step 3**: Try different scheduler
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear
# F1: 0.80
```

### Workflow 3: Save Multiple Checkpoints

**Train with different output dirs**:
```bash
# Version 1: 5 epochs
python3 src/train.py --epochs 5 --output_dir models/v1

# Version 2: Continue from v1
python3 src/train_resume.py \
    --resume_from models/v1 \
    --epochs 5 \
    --learning_rate 5e-5 \
    --output_dir models/v2

# Version 3: Continue from v2
python3 src/train_resume.py \
    --resume_from models/v2 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --output_dir models/v3
```

---

## Monitoring Progress

### WandB Tracking

The script automatically logs to WandB with run names:
- New training: `qwen-new-r16-lr0.0002`
- Resumed: `qwen-resume-r16-lr5e-05`

### Console Output

**When resuming**:
```
============================================================
RESUMING TRAINING FROM CHECKPOINT
Checkpoint: models/qwen_finetuned
============================================================
Device: cuda
Epochs: 5 additional
Batch size: 4
Gradient accumulation: 4
Effective batch size: 16
Learning rate: 5e-05
LR scheduler: cosine
============================================================

Loading model from checkpoint: models/qwen_finetuned
✓ Loaded existing LoRA adapter
trainable params: 2,621,440 || all params: 603,979,776 || trainable%: 0.4340

Training on 900 examples
Validating on 100 examples

CONTINUING TRAINING...
Epoch 1/5: 100%|████████████████| 225/225
Epoch 1: Average F1 Score = 0.8123
...
```

---

## Tips for Best Results

### 1. Use Lower Learning Rate
```bash
# ❌ Too high - might destroy existing knowledge
--learning_rate 2e-4

# ✅ Good for resuming
--learning_rate 5e-5

# ✅ Good for fine-tuning
--learning_rate 1e-5
```

### 2. Monitor F1 Scores
- If F1 decreases → LR too high
- If F1 plateaus → Need more epochs or higher LR
- If F1 increases → Good! Continue

### 3. Save Intermediate Checkpoints
```bash
# Save each version separately
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --output_dir models/qwen_v2 \
    --epochs 5
```

### 4. Experiment Safely
Always keep your original model:
```bash
# Copy original first
cp -r models/qwen_finetuned models/qwen_finetuned_backup

# Then experiment
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5
```

---

## Troubleshooting

### Model Not Found
```
Error: models/qwen_finetuned not found
```

**Solution**: Check path exists
```bash
ls -la models/qwen_finetuned
# Should show adapter_config.json, adapter_model.bin, etc.
```

### F1 Score Decreases
```
Epoch 1: F1 = 0.75 (was 0.80)
```

**Solution**: Learning rate too high
```bash
# Reduce LR by 10x
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --learning_rate 1e-5  # Instead of 1e-4
```

### Training Too Slow
```bash
# Increase batch size if memory allows
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --batch_size 8
```

### Out of Memory
```bash
# Reduce batch size
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --batch_size 2
```

---

## Comparison: New vs Resume

| Aspect | New Training | Resume Training |
|--------|--------------|-----------------|
| Learning Rate | 2e-4 | 5e-5 to 1e-5 |
| Warmup | 10% | 5% |
| Epochs | 5-10 | 3-5 |
| Purpose | Learn from scratch | Refine existing |
| Risk | None | Can degrade if LR too high |
| Speed | Full training | Faster (fewer epochs) |

---

## Best Practices

### ✅ Do's
- Use lower learning rate when resuming
- Monitor F1 scores closely
- Save checkpoints with different names
- Keep backup of original model
- Experiment with small epoch counts first

### ❌ Don'ts
- Don't use same LR as initial training
- Don't train too many epochs at once
- Don't overwrite original model immediately
- Don't ignore decreasing F1 scores
- Don't skip validation monitoring

---

## Quick Reference

### Continue Training (5 more epochs)
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5
```

### Fine-tune (gentle polish)
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5
```

### Experiment (new hyperparameters)
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --output_dir models/qwen_experiment
```

---

## Summary

**Resume training is perfect for**:
- ✅ Continuing training that was interrupted
- ✅ Adding more epochs to improve F1
- ✅ Fine-tuning with different hyperparameters
- ✅ Polishing a good model to make it great

**Key differences from new training**:
- 🔻 Lower learning rate (5e-5 vs 2e-4)
- 🔻 Less warmup (5% vs 10%)
- 🔻 Fewer epochs (3-5 vs 5-10)
- ✅ Faster iteration
- ✅ Builds on existing knowledge

---

**Use `train_resume.py` to iteratively improve your model!** 🔄
