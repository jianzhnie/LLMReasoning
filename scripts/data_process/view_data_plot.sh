#!/bin/bash

data_path=/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets/data_summary.json
output_dir=/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets/plot

python  /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/view_data_plot.py \
    --stats-file $data_path \
    --output-dir $output_dir