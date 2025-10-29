#!/bin/bash

set -ex

input_data_path=/home/jianzhnie/llmtuner/hfhub/pengcheng/am-thinking/math_gt_32K_rl_pcl_qa_no_prompt_8159_with_cot.json
oputput_data_dir=/home/jianzhnie/llmtuner/hfhub/pengcheng/am-thinking/math_gt_32K_offline_pg_pcl_chat_template


mkdir -p $oputput_data_dir

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/offline_pg_data_process.py  \
    --input_path $input_data_path \
    --output_path $oputput_data_dir/am-thinking_dpo_dataset.jsonl \
    --subset_output_path $oputput_data_dir/am-thinking_dpo_dataset_subset.jsonl \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
    --apply_chat_template_method formatted \
    --max_cot_len 65536 \
    --min_cot_len 1024 \
    --num_proc 32 \
    --save_subset \
    --subset_size 16
