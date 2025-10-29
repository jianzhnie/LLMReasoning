#!/bin/bash

PROJECT_DIR=/home/jianzhnie/llmtuner/llm/LLMReasoning

source $PROJECT_DIR/set_env.sh


# 定义 Python 脚本的路径
PYTHON_SCRIPT="$PROJECT_DIR/llmreasoning/data_process/aggregate_reasoning_data.py"

# 定义输入文件所在的目录
INPUT_DIR="/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets"

OUTPUT_DIR="/home/jianzhnie/llmtuner/hfhub/pengcheng/aime_reasoning/merged_datasets"

model_name_or_path=/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-32B


mkdir -p $OUTPUT_DIR
# 定义文件名前缀，用于匹配
FILE_PREFIX="aime_combined"

# 遍历所有符合模式的文件
# 例如: infer_qwen25_32B_math_top_30K_rl_verify_part_*.jsonl
for INPUT_FILE in "${INPUT_DIR}/${FILE_PREFIX}"*.jsonl; do
    # 检查文件是否存在，以防通配符没有匹配到任何文件
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Warning: No files found matching pattern ${INPUT_DIR}/${FILE_PREFIX}*.jsonl. Skipping."
        continue
    fi

    # 从文件名中提取基础部分来生成输出文件名
    # 例如: 从 "infer_qwen25_32B_math_top_30K_rl_verify_part_000_bz8.jsonl"
    # 提取 "infer_qwen25_32B_math_top_30K_rl_verify_part_000_bz8"
    BASE_NAME=$(basename "${INPUT_FILE}" .jsonl)

    # 构建输出文件名，加上 "_grouped_by_prompt" 后缀
    OUTPUT_FILE="${OUTPUT_DIR}/grouped_by_prompt.json"

    echo "---"
    echo "Processing: $INPUT_FILE"
    echo "Outputting to: $OUTPUT_FILE"

    # 调用你的 Python 脚本
    # 注意: 确保你的 Python 脚本是可执行的，并且 `python` 命令在 PATH 中
    python3 "$PYTHON_SCRIPT" --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model_name_or_path "$model_name_or_path" --num_proc 32

done
