# 🎯 Inference & Kaggle Submission Guide

## Quick Start

### Option 1: Interactive Script (Easiest)
```bash
./generate_submissions.sh
```

Select from pre-configured options:
1. **Deterministic** (greedy, temp=0.1) - Most reliable
2. **Low temperature** (temp=0.05) - Very deterministic  
3. **Medium temperature** (temp=0.3) - Balanced
4. **Beam search** (beams=3) - Higher quality
5. **Beam search** (beams=5) - Best quality, slower
6. **Sampling** (temp=0.7) - More diverse
7. **Generate ALL** - Create all variants for ensemble
8. **Custom** - Specify your own parameters

### Option 2: Direct Command
```bash
python3 src/inference_optimized.py \
    --adapter_path models/qwen_finetuned \
    --data_path eval.json \
    --output_path output/submission.csv \
    --batch_size 8 \
    --temperature 0.1
```

---

## Available Parameters

### Model & Data Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_base` | `Qwen/Qwen3-0.6B-Base` | Base model name or path |
| `--adapter_path` | `models/qwen_finetuned` | Path to fine-tuned LoRA adapter |
| `--data_path` | `eval.json` | Path to test data (JSON or CSV) |
| `--output_path` | `output/submission.csv` | Path to save submission CSV |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | `8` | Batch size (higher = faster, more memory) |
| `--max_new_tokens` | `512` | Maximum tokens to generate |

### Generation Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `--temperature` | `0.1` | Sampling temperature | Lower = more deterministic |
| `--do_sample` | `False` | Use sampling | Enable for diversity |
| `--num_beams` | `1` | Beam search beams | Higher = better quality, slower |
| `--top_p` | `1.0` | Nucleus sampling | Lower = focus on likely tokens |
| `--top_k` | `50` | Top-k sampling | Lower = less diversity |

### Other Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--merge_adapter` | `True` | Merge adapter weights | Faster inference |
| `--output_suffix` | `""` | Suffix for output file | e.g., `_beam3` |

---

## Common Configurations

### 1. Most Reliable (Recommended for First Submission)
```bash
python3 src/inference_optimized.py \
    --temperature 0.1 \
    --num_beams 1 \
    --batch_size 8 \
    --output_suffix "_greedy"
```

**When to use**: First submission, baseline
**Pros**: Fast, deterministic, reliable
**Cons**: May miss some variations

### 2. Highest Quality (Beam Search)
```bash
python3 src/inference_optimized.py \
    --temperature 0.1 \
    --num_beams 5 \
    --batch_size 4 \
    --output_suffix "_beam5"
```

**When to use**: When you want best quality
**Pros**: Better quality, explores alternatives
**Cons**: 5x slower than greedy

### 3. Very Deterministic (Low Temperature)
```bash
python3 src/inference_optimized.py \
    --temperature 0.05 \
    --num_beams 1 \
    --batch_size 8 \
    --output_suffix "_temp005"
```

**When to use**: When consistency is critical
**Pros**: Very consistent outputs
**Cons**: May be too conservative

### 4. Balanced (Medium Temperature)
```bash
python3 src/inference_optimized.py \
    --temperature 0.3 \
    --num_beams 1 \
    --batch_size 8 \
    --output_suffix "_temp03"
```

**When to use**: When you want some variation
**Pros**: Good balance of quality and diversity
**Cons**: Less predictable

### 5. Diverse Sampling
```bash
python3 src/inference_optimized.py \
    --temperature 0.7 \
    --do_sample \
    --top_p 0.9 \
    --top_k 50 \
    --batch_size 8 \
    --output_suffix "_sample"
```

**When to use**: For ensemble or exploration
**Pros**: Diverse outputs
**Cons**: Less consistent

### 6. Ensemble (Generate Multiple)
```bash
# Generate all variants
./generate_submissions.sh
# Select option 7

# Then ensemble them (see Ensemble section below)
```

---

## Parameter Effects

### Temperature

```
Temperature = 0.0:  Always picks most likely token (deterministic)
Temperature = 0.1:  Mostly deterministic, slight variation
Temperature = 0.5:  Balanced
Temperature = 1.0:  Natural distribution
Temperature = 2.0:  Very random
```

**Recommendation**: Start with 0.1, try 0.05-0.3 range

### Num Beams

```
Beams = 1:  Greedy decoding (fastest)
Beams = 3:  Explores 3 alternatives (3x slower)
Beams = 5:  Explores 5 alternatives (5x slower)
Beams = 10: Explores 10 alternatives (10x slower)
```

**Recommendation**: Start with 1, try 3-5 for quality

### Batch Size

```
Batch = 2:   Slowest, least memory
Batch = 4:   Slow, low memory
Batch = 8:   Balanced (recommended)
Batch = 16:  Fast, high memory
Batch = 32:  Fastest, very high memory
```

**Recommendation**: 8 for most GPUs, adjust based on memory

---

## Example Workflows

### Workflow 1: Quick Submission
```bash
# 1. Generate with defaults
python3 src/inference_optimized.py

# 2. Check output
head output/submission.csv

# 3. Upload to Kaggle
# (manually or via Kaggle API)
```

### Workflow 2: Quality Optimization
```bash
# 1. Generate greedy baseline
python3 src/inference_optimized.py --output_suffix "_greedy"

# 2. Generate beam search
python3 src/inference_optimized.py --num_beams 3 --output_suffix "_beam3"

# 3. Compare and choose best
# Upload the one with better validation score
```

### Workflow 3: Ensemble
```bash
# 1. Generate multiple variants
./generate_submissions.sh  # Option 7

# 2. Ensemble them (majority vote or averaging)
python3 scripts/ensemble_submissions.py \
    output/submission_greedy.csv \
    output/submission_beam3.csv \
    output/submission_beam5.csv \
    --output output/submission_ensemble.csv

# 3. Upload ensemble
```

---

## Performance Expectations

### Speed (1000 examples)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Original (sequential) | ~120 min | 1x |
| Greedy (batch=8) | ~20 min | 6x |
| Beam=3 (batch=4) | ~60 min | 2x |
| Beam=5 (batch=4) | ~100 min | 1.2x |

### Quality Trade-offs

| Configuration | Quality | Speed | Use Case |
|---------------|---------|-------|----------|
| Greedy, temp=0.1 | Good | Fast | Baseline |
| Beam=3, temp=0.1 | Better | Medium | Quality |
| Beam=5, temp=0.1 | Best | Slow | Final submission |
| Sample, temp=0.7 | Variable | Fast | Ensemble |

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python3 src/inference_optimized.py --batch_size 4

# Or even smaller
python3 src/inference_optimized.py --batch_size 2
```

### Too Slow
```bash
# Increase batch size (if memory allows)
python3 src/inference_optimized.py --batch_size 16

# Use greedy instead of beam search
python3 src/inference_optimized.py --num_beams 1
```

### Inconsistent Results
```bash
# Lower temperature
python3 src/inference_optimized.py --temperature 0.05

# Or use greedy
python3 src/inference_optimized.py --temperature 0.0 --num_beams 1
```

### Poor Quality
```bash
# Use beam search
python3 src/inference_optimized.py --num_beams 5

# Or increase temperature slightly
python3 src/inference_optimized.py --temperature 0.3
```

---

## Kaggle Submission

### Manual Upload
1. Generate submission:
   ```bash
   python3 src/inference_optimized.py
   ```

2. Go to Kaggle competition page

3. Click "Submit Predictions"

4. Upload `output/submission.csv`

5. Add description (e.g., "Greedy decoding, temp=0.1")

6. Submit and check leaderboard

### Using Kaggle API
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API key (one-time)
# Download kaggle.json from Kaggle account settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Submit
kaggle competitions submit \
    -c competition-name \
    -f output/submission.csv \
    -m "Greedy decoding, temp=0.1"

# Check status
kaggle competitions submissions -c competition-name
```

---

## Advanced Tips

### 1. Ensemble Different Configurations
Generate multiple submissions and combine them:
- Greedy (fast, reliable)
- Beam=3 (quality)
- Beam=5 (best quality)
- Sample (diversity)

### 2. Validate Locally First
If you have validation data:
```bash
python3 src/inference_optimized.py \
    --data_path validation.json \
    --output_path output/validation_pred.csv

# Then calculate F1 score
python3 scripts/calculate_f1.py \
    output/validation_pred.csv \
    validation_labels.csv
```

### 3. Experiment with Temperature
Try range: 0.05, 0.1, 0.2, 0.3
- Lower = more consistent
- Higher = more diverse

### 4. Use Beam Search for Final Submission
After finding good hyperparameters with greedy, run beam search for final quality boost.

---

## Summary

**Quick Commands**:
```bash
# Interactive (easiest)
./generate_submissions.sh

# Reliable baseline
python3 src/inference_optimized.py

# High quality
python3 src/inference_optimized.py --num_beams 5 --batch_size 4

# All variants
./generate_submissions.sh  # Option 7
```

**Recommended Strategy**:
1. Start with greedy (fast baseline)
2. Try beam=3 (better quality)
3. Experiment with temperature (0.05-0.3)
4. Generate ensemble if needed
5. Use beam=5 for final submission

---

**Good luck with your Kaggle competition! 🚀**
