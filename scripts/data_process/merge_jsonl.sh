#!/bin/bash

set -ex

data_dir=/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/merge_jsonl.py \
        --patterns \
        $data_dir/OpenReasoning-Nemotron-32B/eval_score/aime*.jsonl \
        $data_dir/pcl-reasoner-rl/eval_score/aime*.jsonl \
        $data_dir/pcl-reasoner-v1/eval_score/aime24*.jsonl \
        --output /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets/aime_combined.jsonl
