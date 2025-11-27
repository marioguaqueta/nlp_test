#!/bin/bash

# Comparison script for original vs optimized implementations
# This helps you measure the actual performance improvements

echo "=========================================="
echo "Performance Comparison Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to measure time
measure_time() {
    local script=$1
    local name=$2
    
    echo -e "${BLUE}Running: $name${NC}"
    echo "Script: $script"
    echo "Start time: $(date)"
    
    start_time=$(date +%s)
    python "$script"
    end_time=$(date +%s)
    
    elapsed=$((end_time - start_time))
    minutes=$((elapsed / 60))
    seconds=$((elapsed % 60))
    
    echo "End time: $(date)"
    echo -e "${GREEN}Time taken: ${minutes}m ${seconds}s${NC}"
    echo ""
    
    return $elapsed
}

# Menu
echo "What would you like to compare?"
echo "1. Inference speed (original vs optimized)"
echo "2. Training (original vs optimized)"
echo "3. Data augmentation test"
echo "4. Full pipeline comparison"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "INFERENCE COMPARISON"
        echo "=========================================="
        echo ""
        
        # Original inference
        if [ -f "src/inference.py" ]; then
            measure_time "src/inference.py" "Original Inference"
            original_time=$?
        else
            echo "Original inference.py not found"
            original_time=0
        fi
        
        # Optimized inference
        if [ -f "src/inference_optimized.py" ]; then
            measure_time "src/inference_optimized.py" "Optimized Inference"
            optimized_time=$?
        else
            echo "Optimized inference_optimized.py not found"
            optimized_time=0
        fi
        
        # Calculate speedup
        if [ $original_time -gt 0 ] && [ $optimized_time -gt 0 ]; then
            speedup=$(echo "scale=2; $original_time / $optimized_time" | bc)
            echo "=========================================="
            echo -e "${GREEN}Speedup: ${speedup}x faster${NC}"
            echo "=========================================="
        fi
        ;;
        
    2)
        echo ""
        echo "=========================================="
        echo "TRAINING COMPARISON"
        echo "=========================================="
        echo ""
        
        echo "Note: This will run full training twice."
        read -p "Continue? (y/n): " confirm
        
        if [ "$confirm" = "y" ]; then
            # Original training
            if [ -f "src/train.py" ]; then
                measure_time "src/train.py" "Original Training"
            else
                echo "Original train.py not found"
            fi
            
            # Optimized training
            if [ -f "src/train_optimized.py" ]; then
                measure_time "src/train_optimized.py" "Optimized Training"
            else
                echo "Optimized train_optimized.py not found"
            fi
        fi
        ;;
        
    3)
        echo ""
        echo "=========================================="
        echo "DATA AUGMENTATION TEST"
        echo "=========================================="
        echo ""
        
        if [ -f "src/data_augmentation.py" ]; then
            python src/data_augmentation.py
        else
            echo "data_augmentation.py not found"
        fi
        ;;
        
    4)
        echo ""
        echo "=========================================="
        echo "FULL PIPELINE COMPARISON"
        echo "=========================================="
        echo ""
        
        echo "This will:"
        echo "1. Train with original script"
        echo "2. Run inference with original script"
        echo "3. Train with optimized script"
        echo "4. Run inference with optimized script"
        echo ""
        read -p "Continue? This will take a long time. (y/n): " confirm
        
        if [ "$confirm" = "y" ]; then
            # Original pipeline
            echo -e "${BLUE}=== ORIGINAL PIPELINE ===${NC}"
            measure_time "src/train.py" "Original Training"
            measure_time "src/inference.py" "Original Inference"
            
            # Optimized pipeline
            echo -e "${BLUE}=== OPTIMIZED PIPELINE ===${NC}"
            measure_time "src/train_optimized.py" "Optimized Training"
            measure_time "src/inference_optimized.py" "Optimized Inference"
        fi
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
