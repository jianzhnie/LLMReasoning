#!/bin/bash

PROJECT_DIR=/home/jianzhnie/llmtuner/llm/LLMReasoning

source $PROJECT_DIR/set_env.sh

python $PROJECT_DIR/llmreasoning/data_process/apply_chat_template.py \
    --model-dir /home/jianzhnie/llmtuner/hfhub/models