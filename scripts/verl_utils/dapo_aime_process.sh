#!/bin/bash

# Script to preprocess custom datasets
# Usage: ./preprocess_custom_dataset.sh [OPTIONS]

# Default values
PYTHON_SCRIPT="/home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/verl_utils/dapo_data_process.py"
LOCAL_DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet
DATASET_NAME="dapo-math-17k"
# LOCAL_DATASET_PATH=/home/jianzhnie/llmtuner/hfhub/datasets/HuggingFaceH4/aime_2024
# DATASET_NAME="aime24"
# LOCAL_DATASET_PATH="/home/jianzhnie/llmtuner/hfhub/datasets/yentinglin/aime_2025"
# DATASET_NAME="aime25"
LOCAL_SAVE_DIR="/home/jianzhnie/llmtuner/hfhub/verl/data/"
INPUT_KEY="problem"
LABEL_KEY="answer"
TEST_SPLIT_RATIO="-1"

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