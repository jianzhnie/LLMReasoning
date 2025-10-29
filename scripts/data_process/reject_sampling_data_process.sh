#!/bin/bash

set -ex

input_data_path=/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets/grouped_by_prompt.json
oputput_data_dir=/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets

mkdir -p $oputput_data_dir

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/reject_sampling_data_process_aime.py \
    --input_path $input_data_path \
    --output_path $oputput_data_dir/aime_reject_sampling_chat_template.jsonl \
    --cache_dir /home/jianzhnie/llmtuner/hfhub/cache_dir \
    --subset_output_path $oputput_data_dir/aime_reject_sampling_chat_template_subset.jsonl \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
    --apply_chat_template_method formatted \
    --min_cot_len 32768 \
    --max_cot_len 65536 \
    --num_proc 1 \
    --save_subset \
    --subset_size 32
