python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/rl_data_process_faster.py \
    --data_path /home/jianzhnie/llmtuner/hfhub/pengcheng/omni-math/PCL-Reasoner_omin-math_SFT_final_yao.jsonl \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
    --output_path /home/jianzhnie/llmtuner/hfhub/pengcheng/omni-math/PCL-Reasoner_omin-math_apply_chat_template.jsonl \
    --apply_chat_template_method formatted \
    --system_prompt_type amthinking \
    --input_key problem \
    --response_key gen \
    --label_key answer \
    --use_qwen_math_cot
