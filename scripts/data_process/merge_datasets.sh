python3 /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/merge_datsets.py \
  --input_folder "/home/jianzhnie/llmtuner/llm/QwQ/eval/output/am-thinking/eval_score/merged" \
  --file_pattern "*_grouped_by_prompt.json" \
  --output_path "/home/jianzhnie/llmtuner/llm/QwQ/eval/output/am-thinking/eval_score/merged/all_data_merged.json"


python3 /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/merge_datsets.py \
  --input_folder "/home/jianzhnie/llmtuner/llm/QwQ/eval/output/am-thinking/eval_score/" \
  --file_pattern "infer_qwen25_32B_math_top_30K_rl_verify_part*.jsonl" \
  --output_path "/home/jianzhnie/llmtuner/llm/LLMEval/output/am-thinking-0528/infer_qwen25_32B_math_top_30K_rl_merged.json"


python3 /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/merge_datsets.py \
  --input_folder /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/pcl-reasoner-v1/eval_score \
  --file_pattern aime*.jsonl \
  --output_path /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/pcl-reasoner-v1/eval_score/aime_infer_merged.json


python3 /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/merge_datsets.py \
  --input_folder /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/pcl-reasoner-v1/eval_score \
  --file_pattern aime*.jsonl \
  --output_path /home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/pcl-reasoner-v1/eval_score/aime_infer_merged.json
