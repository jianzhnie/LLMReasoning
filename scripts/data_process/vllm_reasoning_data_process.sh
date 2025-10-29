#!/bin/bash

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/vllm_reasoning_data_process.py \
    --input_data_dir "/home/jianzhnie/llmtuner/llm/LLMEval/output/QwQ-32B/eval_score_old" \
    --input_file_pattern "aime*.jsonl" \
    --output_data_dir "/home/jianzhnie/llmtuner/llm/LLMEval/output/QwQ-32B/eval_score" \
    --output_filename "rl_reasoning_data_summary.json" \
    --model_name_or_path "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-7B" \
    --num_proc 64
