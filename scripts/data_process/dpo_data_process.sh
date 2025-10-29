#!/bin/bash
set -ex

input_data_path=/home/jianzhnie/llmtuner/llm/QwQ/eval/output/am-thinking/eval_score/merged/all_data_merged.json
oputput_data_dir=/home/jianzhnie/llmtuner/hfhub/mindspeed/datasets/SimPO_V2
mkdir -p $oputput_data_dir

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/dpo_data_process.py  \
    --input_path $input_data_path \
    --output_path $oputput_data_dir/am-thinking_dpo_dataset.jsonl \
    --subset_output_path $oputput_data_dir/am-thinking_dpo_dataset_subset.jsonl \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
    --apply_chat_template_method none \
    --max_cot_len 32768 \
    --num_proc 32 \
    --save_subset \
    --subset_size 128



