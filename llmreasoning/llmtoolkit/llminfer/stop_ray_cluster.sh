#!/bin/bash
set -eo pipefail  # 启用严格错误处理

# 配置信息
PROJECT_DIR="/root/llmtuner/llm/LLMReasoning/llminfer"

NODES=("10.16.201.108" "10.16.201.198")

MASTER_ADDR=${NODES[0]}
WORKERS=("${NODES[@]:1}")  # 除第一个节点外的所有节点作为worker

# 超时设置(秒)
SSH_TIMEOUT=10
RAY_STOP_TIMEOUT=30

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 状态跟踪
declare -A stop_results
TOTAL_NODES=${#NODES[@]}
SUCCESS_COUNT=0

# 打印集群信息
echo -e "${YELLOW}=== Ray Cluster Shutdown Information ==="
echo -e "Master Node: ${MASTER_ADDR}"
echo -e "Worker Nodes: ${WORKERS[*]}"
echo -e "Total Nodes: ${TOTAL_NODES}"
echo -e "===============================${NC}\n"

# 改进的停止Ray节点函数
stop_ray_node() {
    local node_ip=$1
    local node_type=$2
    local prefix="[${node_type}] ${node_ip}"

    echo -e "${YELLOW}${prefix} Stopping Ray...${NC}"

    # 使用timeout防止命令挂起
    if timeout ${RAY_STOP_TIMEOUT} ssh -o ConnectTimeout=${SSH_TIMEOUT} ${node_ip} \
        "cd ${PROJECT_DIR} && source set_env.sh && ray stop"; then
        echo -e "${GREEN}${prefix} ✔ Successfully stopped${NC}"
        stop_results[$node_ip]="success"
        return 0
    else
        echo -e "${RED}${prefix} ❌ Failed to stop${NC}"
        stop_results[$node_ip]="failed"
        return 1
    fi
}

# 并行停止worker节点
worker_pids=()
for worker in "${WORKERS[@]}"; do
    stop_ray_node "$worker" "WORKER" &
    worker_pids+=($!)
done

# 停止master节点(放在最后)
stop_ray_node "$MASTER_ADDR" "MASTER"

# 等待所有worker节点停止完成
for pid in "${worker_pids[@]}"; do
    if wait $pid; then
        ((SUCCESS_COUNT++))
    fi
done

# 等待master节点停止完成
if wait $!; then
    ((SUCCESS_COUNT++))
fi

# 生成汇总报告
echo -e "\n${YELLOW}=== Shutdown Summary ==="
for node in "${NODES[@]}"; do
    if [ "${stop_results[$node]}" == "success" ]; then
        echo -e "${GREEN}✔ ${node} stopped successfully${NC}"
    else
        echo -e "${RED}❌ ${node} failed to stop${NC}"
    fi
done

# 最终结果
if [ $SUCCESS_COUNT -eq $TOTAL_NODES ]; then
    echo -e "\n${GREEN}✅ All ${TOTAL_NODES} Ray nodes stopped successfully!${NC}"
    exit 0
else
    echo -e "\n${RED}⛔ ${SUCCESS_COUNT}/${TOTAL_NODES} nodes stopped successfully${NC}"
    exit 1
fi
