#!/bin/bash

# Quick Start Script to Reach 80% F1 Score
# This script implements the recommended strategy

echo "=============================================="
echo "  80% F1 Score - Quick Start"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}This script will train your model to reach 80%+ F1 score${NC}"
echo ""
echo "Estimated time: 3-4 hours total"
echo "  - Step 1: 2 hours (initial training)"
echo "  - Step 2: 1 hour (continue training)"
echo "  - Step 3: 30 min (fine-tune)"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Step 1: Strong Initial Training
echo ""
echo "=============================================="
echo -e "${GREEN}STEP 1: Initial Training (12 epochs)${NC}"
echo "=============================================="
echo "Expected F1: 0.78-0.82"
echo "Time: ~2 hours"
echo ""

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

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Training failed. Check errors above.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Step 1 Complete!${NC}"
echo ""

# Ask if user wants to continue
read -p "Check your F1 score. Continue to Step 2? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Stopping here. You can resume later with:"
    echo "  ./reach_80_percent.sh --step 2"
    exit 0
fi

# Step 2: Continue Training
echo ""
echo "=============================================="
echo -e "${GREEN}STEP 2: Continue Training (5 epochs)${NC}"
echo "=============================================="
echo "Expected F1: 0.80-0.84"
echo "Time: ~1 hour"
echo ""

python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 5 \
    --learning_rate 5e-5 \
    --weight_decay 0.02

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Training failed. Check errors above.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Step 2 Complete!${NC}"
echo ""

# Ask if user wants to continue
read -p "Check your F1 score. Continue to Step 3? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Stopping here. You can resume later with:"
    echo "  ./reach_80_percent.sh --step 3"
    exit 0
fi

# Step 3: Fine-tune
echo ""
echo "=============================================="
echo -e "${GREEN}STEP 3: Fine-tune (3 epochs)${NC}"
echo "=============================================="
echo "Expected F1: 0.82-0.87"
echo "Time: ~30 minutes"
echo ""

python3 src/train_resume.py \
    --resume_from models/qwen_finetuned \
    --epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.03

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Training failed. Check errors above.${NC}"
    exit 1
fi

echo ""
echo "=============================================="
echo -e "${GREEN}✓✓✓ ALL STEPS COMPLETE! ✓✓✓${NC}"
echo "=============================================="
echo ""
echo "Your model should now have 80%+ F1 score!"
echo ""
echo "Next steps:"
echo "  1. Check final F1 score in WandB"
echo "  2. Run inference: python3 src/inference.py"
echo "  3. Generate submission for Kaggle"
echo ""
echo "Model saved at: models/qwen_finetuned"
echo ""
echo "=============================================="
