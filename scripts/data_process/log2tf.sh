#!/bin/bash

# HOME DIR
HOME_DIR=/home/fdd/workspace/projects/Huawei_PCL_MLLMs/

# SCRIPTS DIR
SCRIPTS_DIR=/home/fdd/workspace/projects/LLMPractice

# 设置 PYTHONPATH 以便能找到 llmpractice 模块
source $HOME_DIR/set_env.sh

# 日志文件路径 (请根据实际情况修改)
LOG_PATH="$HOME_DIR/work_dir/exp1/logs/2026-01-23_08-16-33/rank-15_host-10.42.24.145.log"

# TensorBoard 日志输出目录
SAVE_DIR="tf_logs"

# 1. 解析 SFT (Supervised Fine-Tuning) 日志
# python3 llmpractice/data_process/log2tf.py \
#     --log-path "$LOG_PATH" \
#     --save-log-dir "$SAVE_DIR/sft" \
#     --stage sft \
#     --tag-prefix "train/"

# 2. 解析 DPO (Direct Preference Optimization) 日志
# python3 llmpractice/data_process/log2tf.py \
#     --log-path "$LOG_PATH" \
#     --save-log-dir "$SAVE_DIR/dpo" \
#     --stage dpo \
#     --tag-prefix "train/"

# 3. 解析 Pretrain (预训练) 日志
# 示例：解析名为 pretrain.log 的文件
echo "Processing Pretrain Log..."
python3 $SCRIPTS_DIR/llmpractice/data_process/log2tf.py \
    --log-path "$LOG_PATH" \
    --save-log-dir "$SAVE_DIR/pretrain" \
    --stage pretrain \
    --tag-prefix "pretrain/"

echo "Done! Run 'tensorboard --logdir $SAVE_DIR' to view the results."
