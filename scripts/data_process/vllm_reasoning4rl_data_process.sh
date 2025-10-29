#!/bin/bash

set -ex

input_data_path=/home/jianzhnie/llmtuner/hfhub/datasets/pengcheng/am-thinking/math_gt_32K_rl_pcl_qa_no_prompt_8159_with_cot.json
oputput_data_dir=/home/jianzhnie/llmtuner/hfhub/datasets/pengcheng/am-thinking/math_gt_32K_rl_pcl_chat_template

mkdir -p $oputput_data_dir

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/vllm_reasoning4rl_data_process.py \
    --input_path $input_data_path \
    --output_path $oputput_data_dir/am-thinking_gt_32k_rl_data.jsonl \
    --cache_dir /home/jianzhnie/llmtuner/hfhub/cache_dir \
    --subset_output_path $oputput_data_dir/am-thinking_gt_32k_rl_data_subset.jsonl \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
    --max_cot_len 65536 \
    --num_proc 32 \
    --save_subset \
    --subset_size 32
