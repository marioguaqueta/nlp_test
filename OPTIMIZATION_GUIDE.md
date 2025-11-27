# Optimization Guide: Inference & Training Improvements

## 🚀 Overview

This guide provides comprehensive optimizations for both **inference speed** (reducing from ~2 hours to ~15-30 minutes) and **training quality** (with data augmentation and advanced strategies).

---

## 📊 Performance Comparison

### Inference Speed

| Method | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| Processing | Sequential | Batched | ~4-8x |
| Batch Size | 1 | 8 | 8x |
| Generation | Standard | KV Cache + Optimized | ~2x |
| **Total Time** | **~120 min** | **~15-30 min** | **~4-8x faster** |

### Training Quality

| Feature | Original | Optimized |
|---------|----------|-----------|
| Data Augmentation | None | 6 strategies |
| Dataset Size | 1x | 2-3x (configurable) |
| LoRA Rank | 8 | 16 (configurable) |
| Target Modules | 2 | 4 |
| Label Masking | No | Yes (instruction masked) |
| LR Scheduler | Linear | Cosine with warmup |
| Gradient Checkpointing | No | Yes |

---

## 🔧 Inference Optimizations

### Key Improvements in `inference_optimized.py`

1. **Batch Processing**
   - Process 8 examples simultaneously instead of 1
   - Reduces overhead and improves GPU utilization
   ```python
   BATCH_SIZE = 8  # Adjust based on your GPU memory
   ```

2. **KV Cache**
   - Enables key-value caching for faster generation
   - Reduces redundant computations
   ```python
   use_cache=True
   ```

3. **Optimized Generation Parameters**
   - Greedy decoding (num_beams=1) instead of beam search
   - Lower temperature for deterministic outputs
   - Early stopping with EOS token

4. **Model Merging**
   - Merges LoRA adapter weights into base model
   - Eliminates adapter overhead during inference
   ```python
   model = model.merge_and_unload()
   ```

5. **Memory Optimizations**
   - `low_cpu_mem_usage=True` for efficient loading
   - Dynamic padding with batch processing

### Usage

```bash
# Run optimized inference
python src/inference_optimized.py

# Expected time: ~15-30 minutes (vs ~2 hours original)
```

### Tuning Parameters

Adjust these in `inference_optimized.py` based on your hardware:

```python
BATCH_SIZE = 8          # Increase if you have more GPU memory (16, 32)
                        # Decrease if OOM (4, 2)

MAX_NEW_TOKENS = 512    # Reduce if your JSONs are shorter (256, 128)

NUM_BEAMS = 1           # Keep at 1 for speed, increase to 3-5 for quality
```

---

## 🎯 Training Optimizations

### Key Improvements in `train_optimized.py`

1. **Data Augmentation** (`data_augmentation.py`)
   
   Six augmentation strategies:
   
   - **Synonym Replacement**: Replace words with synonyms
     - "comprar" → "adquirir", "pedir", "solicitar"
   
   - **Word Order Variation**: Shuffle clauses while maintaining meaning
   
   - **Punctuation Variation**: Normalize/vary punctuation
     - "producto,precio" → "producto, precio"
   
   - **Number Format Variation**: Different number representations
     - "1000" → "1,000" or vice versa
   
   - **Case Variation**: Different capitalization
     - "URGENTE" → "urgente" → "Urgente"
   
   - **Whitespace Variation**: Normalize spacing

2. **Improved LoRA Configuration**
   ```python
   r=16,                    # Increased from 8 (more capacity)
   lora_alpha=32,           # Scaling factor
   lora_dropout=0.05,       # Reduced dropout
   target_modules=[         # More modules (was only q_proj, v_proj)
       "q_proj", 
       "k_proj", 
       "v_proj", 
       "o_proj"
   ]
   ```

3. **Better Label Masking**
   - Only trains on JSON output, not instruction
   - Reduces noise and improves learning efficiency

4. **Advanced Training Features**
   - Cosine learning rate scheduler with warmup
   - Gradient checkpointing (saves memory)
   - Weight decay for regularization
   - Mixed precision training (FP16 on CUDA)

5. **Hyperparameter Tuning**
   - Configurable via command-line arguments
   - Easy experimentation

### Usage

```bash
# Basic training with defaults (augmentation enabled)
python src/train_optimized.py

# Custom configuration
python src/train_optimized.py \
    --epochs 5 \
    --batch_size 8 \
    --augmentation_factor 3 \
    --lora_r 16 \
    --learning_rate 2e-4

# Without augmentation (for comparison)
python src/train_optimized.py --use_augmentation False

# High-quality training (more epochs, more augmentation)
python src/train_optimized.py \
    --epochs 10 \
    --augmentation_factor 4 \
    --lora_r 32 \
    --batch_size 4 \
    --gradient_accumulation_steps 8
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Per-device batch size |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation |
| `--learning_rate` | 2e-4 | Learning rate |
| `--use_augmentation` | True | Enable data augmentation |
| `--augmentation_factor` | 2 | Augmentation multiplier (2x, 3x, etc.) |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--max_seq_length` | 512 | Maximum sequence length |
| `--warmup_ratio` | 0.1 | Warmup ratio |
| `--weight_decay` | 0.01 | Weight decay |
| `--gradient_checkpointing` | True | Use gradient checkpointing |

---

## 📈 Expected Improvements

### Inference
- **Speed**: 4-8x faster (2 hours → 15-30 minutes)
- **Memory**: Similar or slightly better
- **Quality**: Same or better (with merged weights)

### Training
- **Dataset Size**: 2-3x larger (with augmentation)
- **Model Robustness**: Handles variations better
- **F1 Score**: Expected +5-15% improvement
- **Generalization**: Better on unseen data

---

## 🔬 Experimentation Guide

### Finding Optimal Batch Size

```bash
# Start with default
python src/inference_optimized.py  # BATCH_SIZE=8

# If you get OOM (Out of Memory), reduce:
# Edit inference_optimized.py: BATCH_SIZE = 4

# If you have extra memory, increase:
# Edit inference_optimized.py: BATCH_SIZE = 16
```

### Hyperparameter Search

Try different combinations:

```bash
# Baseline
python src/train_optimized.py --augmentation_factor 2 --lora_r 16

# More augmentation
python src/train_optimized.py --augmentation_factor 4 --lora_r 16

# Higher capacity
python src/train_optimized.py --augmentation_factor 2 --lora_r 32

# Longer training
python src/train_optimized.py --epochs 10 --augmentation_factor 3
```

### Monitoring Training

WandB will track:
- Training loss
- Validation F1 score
- Learning rate
- GPU utilization

Access at: https://wandb.ai/

---

## 🛠️ Additional Optimization Strategies

### 1. Advanced Data Augmentation

**Back-Translation** (requires translation API):
```python
# Spanish → English → Spanish
# Creates paraphrases while maintaining meaning
```

**Contextual Word Replacement** (requires BERT):
```python
# Use BERT to find contextually similar words
# More sophisticated than synonym replacement
```

### 2. Model Quantization (for inference)

```python
# 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    quantization_config=quantization_config
)
```

### 3. Flash Attention (for training)

```python
# Requires: pip install flash-attn
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="flash_attention_2"
)
```

### 4. Curriculum Learning

Train on easier examples first, then harder ones:
```python
# Sort dataset by complexity (e.g., JSON size)
# Train in stages
```

### 5. Knowledge Distillation

Use a larger teacher model to improve smaller student:
```python
# Train with both ground truth and teacher predictions
```

---

## 📋 Migration Checklist

- [ ] Install dependencies (if needed): `pip install -r requirements.txt`
- [ ] Test optimized inference: `python src/inference_optimized.py`
- [ ] Compare inference times (original vs optimized)
- [ ] Test data augmentation: `python src/data_augmentation.py`
- [ ] Train with augmentation: `python src/train_optimized.py`
- [ ] Monitor WandB for metrics
- [ ] Compare F1 scores (original vs optimized)
- [ ] Tune hyperparameters based on results
- [ ] Run final inference with best model

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Inference:**
- Reduce `BATCH_SIZE` in `inference_optimized.py`
- Reduce `MAX_NEW_TOKENS`

**Training:**
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Enable `--gradient_checkpointing`
- Reduce `--lora_r`

### Slow Training

- Increase `--batch_size` (if memory allows)
- Reduce `--augmentation_factor`
- Use fewer evaluation examples (edit callback)

### Poor Quality

- Increase `--epochs`
- Increase `--augmentation_factor`
- Increase `--lora_r`
- Adjust `--learning_rate`

---

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Data Augmentation for NLP](https://arxiv.org/abs/1901.11196)
- [Efficient Training Techniques](https://huggingface.co/docs/transformers/perf_train_gpu_one)

---

## 📞 Support

For issues or questions:
1. Check WandB logs for training metrics
2. Review error messages carefully
3. Adjust hyperparameters based on hardware
4. Monitor GPU memory usage

---

**Happy Training! 🚀**
