

python /home/jianzhnie/llmtuner/llm/LLMReasoning/llmreasoning/data_process/formate_sft_rl_datasets.py \
    --data_path /home/jianzhnie/llmtuner/hfhub/datasets/dicta-il/gpt-oss-120b \
    --model_name_or_path /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B \
    --output_path /home/jianzhnie/llmtuner/hfhub/pengcheng/dicta-il/MathCOT-oss-vs-DeepSeek/MathCOT-oss-vs-DeepSeek_apply_chat_template.jsonl \
    --apply_chat_template_method formatted \
    --system_prompt_type default \
    --use_qwen_math_cot \
    --num_proc 128 
