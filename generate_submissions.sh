#!/bin/bash

# Kaggle Submission Generation Script
# This script helps you generate different submissions with various parameters

echo "=============================================="
echo "  Kaggle Submission Generator"
echo "=============================================="
echo ""

# Default paths
MODEL_BASE="Qwen/Qwen3-0.6B-Base"
ADAPTER_PATH="models/qwen_finetuned"
DATA_PATH="eval.json"
OUTPUT_DIR="output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Function to run inference with specific parameters
run_inference() {
    local suffix=$1
    local temp=$2
    local beams=$3
    local do_sample=$4
    local description=$5
    
    echo ""
    echo "Generating submission: $description"
    echo "  Temperature: $temp"
    echo "  Num beams: $beams"
    echo "  Do sample: $do_sample"
    echo "  Output: ${OUTPUT_DIR}/submission${suffix}.csv"
    echo ""
    
    python3 src/inference_optimized.py \
        --model_base "$MODEL_BASE" \
        --adapter_path "$ADAPTER_PATH" \
        --data_path "$DATA_PATH" \
        --output_path "${OUTPUT_DIR}/submission.csv" \
        --output_suffix "$suffix" \
        --temperature $temp \
        --num_beams $beams \
        $do_sample \
        --batch_size 8 \
        --max_new_tokens 512
    
    echo "✓ Done!"
    echo ""
}

# Menu
echo "Select submission type:"
echo ""
echo "1. Deterministic (greedy, temp=0.1) - Most reliable"
echo "2. Low temperature (temp=0.05) - Very deterministic"
echo "3. Medium temperature (temp=0.3) - Balanced"
echo "4. Beam search (beams=3) - Higher quality"
echo "5. Beam search (beams=5) - Best quality, slower"
echo "6. Sampling (temp=0.7) - More diverse"
echo "7. Generate ALL variants (for ensemble)"
echo "8. Custom parameters"
echo ""
read -p "Enter choice (1-8): " choice

case $choice in
    1)
        run_inference "_greedy" 0.1 1 "" "Deterministic (greedy)"
        ;;
    2)
        run_inference "_temp005" 0.05 1 "" "Low temperature"
        ;;
    3)
        run_inference "_temp03" 0.3 1 "" "Medium temperature"
        ;;
    4)
        run_inference "_beam3" 0.1 3 "" "Beam search (3 beams)"
        ;;
    5)
        run_inference "_beam5" 0.1 5 "" "Beam search (5 beams)"
        ;;
    6)
        run_inference "_sample" 0.7 1 "--do_sample" "Sampling"
        ;;
    7)
        echo "Generating all variants..."
        run_inference "_greedy" 0.1 1 "" "Deterministic (greedy)"
        run_inference "_temp005" 0.05 1 "" "Low temperature"
        run_inference "_temp03" 0.3 1 "" "Medium temperature"
        run_inference "_beam3" 0.1 3 "" "Beam search (3 beams)"
        run_inference "_beam5" 0.1 5 "" "Beam search (5 beams)"
        run_inference "_sample" 0.7 1 "--do_sample" "Sampling"
        
        echo "=============================================="
        echo "All submissions generated!"
        echo "Files created in $OUTPUT_DIR/:"
        ls -lh $OUTPUT_DIR/submission*.csv
        echo "=============================================="
        ;;
    8)
        echo ""
        read -p "Temperature (0.0-1.0): " temp
        read -p "Num beams (1-10): " beams
        read -p "Do sampling? (y/n): " sample
        read -p "Output suffix (e.g., _custom): " suffix
        
        do_sample_flag=""
        if [ "$sample" = "y" ]; then
            do_sample_flag="--do_sample"
        fi
        
        run_inference "$suffix" $temp $beams "$do_sample_flag" "Custom configuration"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Submission generation complete!"
echo ""
echo "Next steps:"
echo "1. Review the generated CSV file(s) in $OUTPUT_DIR/"
echo "2. Upload to Kaggle competition"
echo "3. Check leaderboard score"
echo "=============================================="
