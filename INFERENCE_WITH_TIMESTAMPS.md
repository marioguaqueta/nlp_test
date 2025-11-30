# 📝 Updated Inference Guide - With Timestamps

## ✨ **New Features**

1. **Automatic Timestamps** - Each submission gets a unique timestamp
2. **Command-Line Arguments** - Customize all parameters
3. **Better Output Organization** - All submissions in `output/` directory

---

## 🚀 **Quick Usage**

### **Basic (Default with Timestamp)**
```bash
python3 src/inference.py
```

Output: `output/submission_20241129_131530.csv`

---

### **Without Timestamp (Overwrite)**
```bash
python3 src/inference.py --no_timestamp
```

Output: `output/submission.csv` (always same name)

---

### **With Custom Suffix**
```bash
python3 src/inference.py --output_suffix "_beam3"
```

Output: `output/submission_20241129_131530_beam3.csv`

---

### **Custom Model**
```bash
python3 src/inference.py \
    --adapter_path models/qwen_v2 \
    --output_suffix "_v2"
```

Output: `output/submission_20241129_131530_v2.csv`

---

## 📋 **All Parameters**

### **Model & Data**
```bash
--model_base "Qwen/Qwen3-0.6B-Base"  # Base model
--adapter_path models/qwen_finetuned  # Your trained model
--data_path eval.json                 # Test data
--output_dir output                   # Output directory
```

### **Generation**
```bash
--temperature 0.1        # Lower = more deterministic
--max_new_tokens 512     # Max tokens to generate
--do_sample              # Enable sampling
--num_beams 3            # Beam search (1 = greedy)
```

### **Output**
```bash
--no_timestamp           # Don't add timestamp
--output_suffix "_v2"    # Add custom suffix
```

---

## 💡 **Common Use Cases**

### **1. Multiple Models Comparison**
```bash
# Model 1
python3 src/inference.py \
    --adapter_path models/qwen_v1 \
    --output_suffix "_v1"

# Model 2
python3 src/inference.py \
    --adapter_path models/qwen_v2 \
    --output_suffix "_v2"

# Model 3
python3 src/inference.py \
    --adapter_path models/qwen_best \
    --output_suffix "_best"
```

Output:
```
output/submission_20241129_131530_v1.csv
output/submission_20241129_131545_v2.csv
output/submission_20241129_131600_best.csv
```

---

### **2. Different Generation Settings**
```bash
# Greedy (default)
python3 src/inference.py --output_suffix "_greedy"

# Beam search
python3 src/inference.py \
    --num_beams 3 \
    --output_suffix "_beam3"

# Lower temperature
python3 src/inference.py \
    --temperature 0.05 \
    --output_suffix "_temp005"

# Sampling
python3 src/inference.py \
    --do_sample \
    --temperature 0.7 \
    --output_suffix "_sample"
```

---

### **3. Production Run (No Timestamp)**
```bash
# For automated scripts that expect same filename
python3 src/inference.py \
    --no_timestamp \
    --adapter_path models/qwen_production
```

Output: `output/submission.csv` (always)

---

## 📁 **Output Directory Structure**

After running multiple inferences:

```
output/
├── submission_20241129_131530.csv           # Default run
├── submission_20241129_131545_v2.csv        # Model v2
├── submission_20241129_131600_beam3.csv     # Beam search
├── submission_20241129_131615_temp005.csv   # Low temp
└── submission.csv                            # No timestamp run
```

---

## 🎯 **Recommended Workflow**

### **Step 1: Test Different Models**
```bash
python3 src/inference.py \
    --adapter_path models/qwen_high_quality \
    --output_suffix "_hq"

python3 src/inference.py \
    --adapter_path models/qwen_v2 \
    --output_suffix "_v2"
```

### **Step 2: Test Different Settings**
```bash
# Try beam search on best model
python3 src/inference.py \
    --adapter_path models/qwen_v2 \
    --num_beams 3 \
    --output_suffix "_v2_beam3"
```

### **Step 3: Compare Results**
```bash
# Check which submission has best validation score
# Upload best one to Kaggle
```

### **Step 4: Final Submission**
```bash
# Generate final submission with best config
python3 src/inference.py \
    --adapter_path models/qwen_best \
    --num_beams 3 \
    --temperature 0.05 \
    --output_suffix "_final"
```

---

## 📊 **Example Output**

```
============================================================
INFERENCE CONFIGURATION
============================================================
Device: cuda
Model base: Qwen/Qwen3-0.6B-Base
Adapter path: models/qwen_finetuned
Data path: eval.json
Temperature: 0.1
Max new tokens: 512
Num beams: 1
============================================================

Loading model from models/qwen_finetuned...

Loading test data from eval.json...

Generating predictions for 1000 examples...
Inference: 100%|████████████████████| 1000/1000 [15:23<00:00]

============================================================
INFERENCE COMPLETE
============================================================
Submission saved to: output/submission_20241129_131530.csv
Total predictions: 1000
Output directory: output

Sample predictions (first 3):

ID 1:
  {"buyer": "ACME Corp", "total": 1500.00, ...}

ID 2:
  {"buyer": "Tech Inc", "total": 2300.50, ...}

ID 3:
  {"buyer": "Global LLC", "total": 890.25, ...}

============================================================
```

---

## 🔍 **Tips**

### **Organize by Date**
Timestamps help you track when each submission was generated:
- `submission_20241129_131530.csv` = Nov 29, 1:15:30 PM

### **Use Descriptive Suffixes**
```bash
--output_suffix "_model_v2_beam3_temp005"
```

### **Keep All Submissions**
Don't delete old submissions - they're timestamped so no conflicts!

### **Compare Multiple Runs**
```bash
ls -lh output/submission_*.csv
# See all your submissions with timestamps
```

---

## 📝 **Summary**

**Default behavior**:
- ✅ Automatic timestamp
- ✅ Saves to `output/` directory
- ✅ Unique filename every run
- ✅ No overwriting

**Customization**:
- ✅ Add suffix for identification
- ✅ Choose different models
- ✅ Adjust generation parameters
- ✅ Disable timestamp if needed

**Example**:
```bash
python3 src/inference.py \
    --adapter_path models/qwen_v2 \
    --num_beams 3 \
    --output_suffix "_v2_beam3"
```

Output: `output/submission_20241129_131530_v2_beam3.csv`

---

**Now you can easily track and compare all your inference runs!** 🎯
