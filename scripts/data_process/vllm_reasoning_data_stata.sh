#!/bin/bash

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/vllm_reasonling_data_stata.py \
    --input_data_dir /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets \
    --input_file_pattern aime_combined.jsonl \
    --output_data_dir /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets  \
    --output_filename data_summary.json \
    --model_name_or_path "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-7B" \
    --num_proc 64
