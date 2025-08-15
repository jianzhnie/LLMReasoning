#!/bin/bash
#==============================================================#
#   Filename    : run_single_node.sh
#   Description : 启动单节点多卡（NPU/GPU）分布式训练任务
#   Usage       : bash run_single_node.sh <NNODES> <NODE_RANK> <DEVICES_PER_NODE> <MASTER_ADDR> <MASTER_PORT>
#==============================================================#

set -euo pipefail

#----------------------------------------
# 参数校验与定义
#----------------------------------------

if [ $# -ne 5 ]; then
    echo "❌ 错误：脚本需要 5 个参数，但传入了 $# 个"
    echo "Usage: $0 <NNODES> <NODE_RANK> <DEVICES_PER_NODE> <MASTER_ADDR> <MASTER_PORT>"
    exit 1
fi

readonly NNODES=$1
readonly NODE_RANK=$2
readonly DEVICES_PER_NODE=$3 # 与启动器保持一致
readonly MASTER_ADDR=$4
readonly MASTER_PORT=$5

# 校验数值参数
for var in NNODES NODE_RANK DEVICES_PER_NODE MASTER_PORT; do
    if ! [[ ${!var} =~ ^[0-9]+$ ]]; then
        echo "❌ 错误：$var 必须是正整数，当前值: ${!var}"
        exit 1
    fi
done

# 放宽 MASTER_ADDR 校验（支持 IP 和主机名）
if [[ -z "$MASTER_ADDR" ]]; then
    echo "❌ 错误：MASTER_ADDR 不能为空"
    exit 1
fi

#----------------------------------------
# 环境配置
#----------------------------------------
readonly TRAIN_SCRIPT="distributed_allreduce_demo.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "❌ 错误：未找到训练脚本 '$TRAIN_SCRIPT' (当前目录: $(pwd))"
    exit 1
fi

#----------------------------------------
# 输出运行配置（便于调试）
#----------------------------------------
echo "========================================"
echo "🚀 节点 RANK $NODE_RANK 开始执行训练任务"
echo "   总节点数: $NNODES"
echo "   本节点设备数: $DEVICES_PER_NODE"
echo "   主节点: $MASTER_ADDR:$MASTER_PORT"
echo "   主机: $(hostname)"
echo "   Python 训练脚本: $TRAIN_SCRIPT"
echo "========================================"

#----------------------------------------
# 启动 torchrun 分布式任务
#----------------------------------------
echo "▶️ 正在启动 torchrun..."

# 所有输出（stdout/stderr）都将直接打印，并由上层 launch_distributed.sh 捕获到日志文件
# 无需在此处进行任何日志重定向
torchrun \
    --nproc-per-node="$DEVICES_PER_NODE" \
    --nnodes="$NNODES" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --max-restarts=0 \
    "$TRAIN_SCRIPT"

#----------------------------------------
# 检查执行结果
#----------------------------------------
TORCHRUN_EXIT_CODE=$?

if [ $TORCHRUN_EXIT_CODE -eq 0 ]; then
    echo "✅ 节点 RANK $NODE_RANK 训练任务成功完成"
else
    echo "❌ 节点 RANK $NODE_RANK 训练失败，退出码: $TORCHRUN_EXIT_CODE"
fi

# 以 torchrun 的退出码退出，这样上层脚本可以捕获到成功或失败状态
exit $TORCHRUN_EXIT_CODE
