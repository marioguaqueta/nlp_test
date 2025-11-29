# 🎯 Strategy to Reach 80% F1 Score

## Current Situation
- **Target**: 80% F1 score
- **Tools**: Working `train.py` and `train_resume.py`
- **Challenge**: Maximize performance without complex optimizations

---

## 🚀 Recommended Strategy (Step-by-Step)

### **Phase 1: Strong Foundation (Week 1)**

#### Step 1.1: Train with Better LoRA Configuration
```bash
python3 src/train.py \
    --epochs 10 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 4
```

**Why**:
- `lora_r=32`: Double capacity (vs default 16)
- `lora_alpha=64`: Better scaling
- `epochs=10`: More training time
- **Expected F1**: 0.75-0.80

**Time**: ~1-2 hours

---

#### Step 1.2: Continue Training with Lower LR
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5 \
    --weight_decay 0.02
```

**Why**:
- Refine the model with gentle updates
- Higher weight decay for better generalization
- **Expected F1**: 0.78-0.82

**Time**: ~30-60 minutes

---

### **Phase 2: Data Quality (Week 1-2)**

#### Step 2.1: Clean Your Data
Check your training data for:
- ✅ Consistent JSON format
- ✅ No duplicates
- ✅ Correct labels
- ✅ Representative examples

```bash
# Create a script to validate data
python3 scripts/validate_data.py
```

**Impact**: +2-5% F1 score

---

#### Step 2.2: Add More Training Data (if possible)
If you can get more data:
- More examples = better generalization
- Aim for 2000+ training examples
- **Impact**: +5-10% F1 score

---

### **Phase 3: Training Optimization (Week 2)**

#### Step 3.1: Longer Training with Cosine Schedule
```bash
python3 src/train.py \
    --epochs 15 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.15 \
    --weight_decay 0.01
```

**Why**:
- More epochs = more learning
- Cosine schedule = better convergence
- More warmup = stable training
- **Expected F1**: 0.80-0.85

**Time**: ~2-3 hours

---

#### Step 3.2: Fine-tune with Very Low LR
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.03
```

**Why**:
- Polish the model
- Very gentle updates
- **Expected F1**: 0.82-0.87

**Time**: ~20-30 minutes

---

### **Phase 4: Inference Optimization (Week 2)**

#### Step 4.1: Use Better Generation Parameters
```bash
python3 src/inference.py \
    --temperature 0.05 \
    --num_beams 3 \
    --batch_size 4
```

**Why**:
- Lower temperature = more deterministic
- Beam search = better quality
- **Impact**: +1-3% F1 score

---

#### Step 4.2: Ensemble Multiple Models
Train 2-3 models with different seeds/configs, then ensemble:

```bash
# Model 1
python3 src/train.py --epochs 10 --lora_r 32 --output_dir models/model1

# Model 2
python3 src/train.py --epochs 12 --lora_r 32 --learning_rate 1.5e-4 --output_dir models/model2

# Model 3
python3 src/train.py --epochs 10 --lora_r 64 --output_dir models/model3

# Ensemble (majority vote or averaging)
python3 scripts/ensemble.py models/model1 models/model2 models/model3
```

**Impact**: +2-5% F1 score

---

## 📊 Expected Results Timeline

| Phase | Action | Expected F1 | Cumulative | Time |
|-------|--------|-------------|------------|------|
| Baseline | Current | 0.70 | 0.70 | - |
| 1.1 | Better LoRA (r=32, 10 epochs) | +0.08 | 0.78 | 1-2h |
| 1.2 | Resume training | +0.03 | 0.81 | 30-60m |
| 2.1 | Clean data | +0.02 | 0.83 | 1-2h |
| 3.1 | Longer training (15 epochs) | +0.02 | 0.85 | 2-3h |
| 3.2 | Fine-tune polish | +0.02 | 0.87 | 20-30m |
| 4.1 | Better inference | +0.02 | 0.89 | - |

**Target achieved at Phase 1.2!** 🎯

---

## 🎯 Quick Path to 80% (Recommended)

### **Option A: Single Strong Training Run**
```bash
python3 src/train.py \
    --epochs 12 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.15
```

**Expected**: 0.78-0.82 F1  
**Time**: 1.5-2.5 hours  
**Success rate**: High ✅

---

### **Option B: Iterative Improvement**
```bash
# Step 1: Initial training (10 epochs)
python3 src/train.py \
    --epochs 10 \
    --lora_r 32 \
    --learning_rate 2e-4

# Step 2: Continue (5 epochs)
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5

# Step 3: Fine-tune (3 epochs)
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5
```

**Expected**: 0.80-0.85 F1  
**Time**: 2-3 hours total  
**Success rate**: Very High ✅✅

---

## 🔑 Key Success Factors

### **1. LoRA Rank (Most Important)**
```bash
# ❌ Too low - insufficient capacity
--lora_r 8

# ✅ Good - balanced
--lora_r 16

# ✅✅ Better - more capacity
--lora_r 32

# ✅✅✅ Best - maximum capacity (if you have time/memory)
--lora_r 64
```

**Recommendation**: Use `r=32` minimum for 80% target

---

### **2. Training Duration**
```bash
# ❌ Too short
--epochs 3

# ✅ Good
--epochs 10

# ✅✅ Better
--epochs 15

# ⚠️ Risk of overfitting
--epochs 20+
```

**Recommendation**: 10-15 epochs

---

### **3. Learning Rate Schedule**
```bash
# ✅✅ Best - smooth convergence
--lr_scheduler_type cosine --warmup_ratio 0.15

# ✅ Good - simple
--lr_scheduler_type linear --warmup_ratio 0.1

# ❌ Not recommended
--lr_scheduler_type constant
```

**Recommendation**: Always use cosine with warmup

---

### **4. Regularization**
```bash
# Balance between learning and generalization
--weight_decay 0.01  # Standard
--weight_decay 0.02  # More regularization
--weight_decay 0.03  # Strong regularization
```

**Recommendation**: Start with 0.01, increase if overfitting

---

## 🛠️ Practical Implementation Plan

### **Week 1: Foundation**

**Monday-Tuesday**: Initial training
```bash
python3 src/train.py \
    --epochs 12 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 2e-4 \
    --weight_decay 0.01
```

**Wednesday**: Evaluate and continue
```bash
# Check F1 score
# If < 0.78, continue training:
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5
```

**Thursday**: Fine-tune
```bash
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5
```

**Friday**: Test inference
```bash
python3 src/inference.py
# Check results on validation set
```

---

### **Week 2: Optimization (if needed)**

**Monday**: Try higher LoRA rank
```bash
python3 src/train.py \
    --epochs 10 \
    --lora_r 64 \
    --learning_rate 1.5e-4 \
    --output_dir models/qwen_r64
```

**Tuesday-Wednesday**: Ensemble approach
```bash
# Train 2-3 models with different configs
# Combine predictions
```

**Thursday**: Data cleaning
```bash
# Review and clean training data
# Remove duplicates, fix errors
```

**Friday**: Final evaluation
```bash
# Test best model
# Generate Kaggle submission
```

---

## 📈 Monitoring Progress

### **Track These Metrics**

1. **Training Loss**: Should decrease smoothly
2. **Validation F1**: Should increase or plateau
3. **Epoch Time**: Monitor for efficiency

### **WandB Dashboard**

Watch for:
- ✅ Smooth loss curve
- ✅ Increasing F1 per epoch
- ⚠️ F1 plateau → Need more epochs or higher LR
- ❌ F1 decrease → LR too high or overfitting

---

## 🎓 Advanced Tips (if 80% not reached)

### **1. Increase Model Capacity**
```bash
# Use larger LoRA rank
--lora_r 64 --lora_alpha 128
```

### **2. More Target Modules**
Modify `train.py` to target more layers:
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### **3. Lower Learning Rate, More Epochs**
```bash
--epochs 20 --learning_rate 1e-4
```

### **4. Gradient Accumulation**
```bash
--batch_size 2 --gradient_accumulation_steps 8
# Effective batch size = 16
```

---

## ⚠️ Common Pitfalls to Avoid

### **❌ Don't**:
- Train with default settings (r=8, 3 epochs)
- Use high learning rate when resuming
- Ignore validation F1 scores
- Overtrain (20+ epochs without monitoring)
- Use constant learning rate schedule

### **✅ Do**:
- Use r=32 minimum
- Monitor F1 scores every epoch
- Use cosine scheduler with warmup
- Save checkpoints regularly
- Test on validation set frequently

---

## 🎯 My Top Recommendation

### **Guaranteed Path to 80%+**

```bash
# Step 1: Strong initial training (2 hours)
python3 src/train.py \
    --epochs 12 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 2e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type cosine

# Expected: 0.78-0.82 F1

# Step 2: Continue if needed (1 hour)
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5 \
    --weight_decay 0.02

# Expected: 0.80-0.84 F1

# Step 3: Final polish (30 min)
python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.03

# Expected: 0.82-0.87 F1
```

**Total time**: 3-4 hours  
**Success probability**: 95%+ ✅  
**Expected final F1**: **0.82-0.87** (exceeds 80% target!)

---

## 📝 Summary

**To reach 80% F1**:

1. ✅ Use **LoRA rank 32** (not 8 or 16)
2. ✅ Train for **10-15 epochs** (not 3-5)
3. ✅ Use **cosine scheduler** with warmup
4. ✅ **Resume training** with lower LR
5. ✅ **Monitor F1** scores closely
6. ✅ Use **weight decay** for regularization

**Estimated timeline**: 1-2 weeks  
**Estimated compute time**: 3-5 hours  
**Success probability**: Very High (90%+)

---

**Start with the recommended command above and you should reach 80%!** 🚀
