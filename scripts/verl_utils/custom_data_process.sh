#!/bin/bash

# Script to preprocess custom datasets
# Usage: ./preprocess_custom_dataset.sh [OPTIONS]

set -e  # Exit on any error

# Default values
PYTHON_SCRIPT="/home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/verl_utils/custom_dataset.py"
LOCAL_DATASET_PATH="/home/jianzhnie/llmtuner/hfhub/verl/raw_data/deepscaler_shuf_1K.jsonl"
LOCAL_SAVE_DIR="/home/jianzhnie/llmtuner/hfhub/verl/data/"
DATASET_NAME="deepscaler_1k"
INPUT_KEY="problem"
LABEL_KEY="answer"
TEST_SPLIT_RATIO="0.1"

# Run the preprocessing script
echo "Running dataset preprocessing..."
echo "Dataset path: $LOCAL_DATASET_PATH"
echo "Save directory: $LOCAL_SAVE_DIR"
echo "Dataset name: $DATASET_NAME"


python3 "$PYTHON_SCRIPT" \
    --local_dataset_path "$LOCAL_DATASET_PATH" \
    --local_save_dir "$LOCAL_SAVE_DIR" \
    --dataset_name "$DATASET_NAME" \
    --input_key "$INPUT_KEY" \
    --label_key "$LABEL_KEY" \
    --test_split_ratio "$TEST_SPLIT_RATIO"

echo "Preprocessing completed successfully!"