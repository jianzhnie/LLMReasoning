#!/bin/bash

# =================================================================
# Ray Cluster Launcher (Optimized)
# -----------------------------------------------------------------
# 目的: 启动多节点 Ray 集群，支持配置 NPU 资源。
# 依赖: 所有节点上都安装了 Ray 并配置了无密码 SSH 登录。
# =================================================================

set -euo pipefail  # 严格模式：遇到错误退出，未定义变量报错，管道错误退出

# --- 1. 默认配置与常量 ---
DEFAULT_PROJECT_DIR="/home/jianzhnie/llmtuner/llm/verl"
DEFAULT_MASTER_PORT="6379"         # Ray head node 默认端口
DEFAULT_DASHBOARD_PORT="8266"      # Ray 仪表盘默认端口
DEFAULT_NPUS_PER_NODE=8            # 每个节点的 NPU 数量
DEFAULT_WAIT_TIME=10               # 等待头节点初始化的时间 (秒)

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- 2. 帮助信息与日志函数 ---
usage() {
    echo "Usage: $0 <node_list_file> [options]"
    echo ""
    echo -e "${BLUE}Starts a multi-node Ray cluster, assigning NPU resources.${NC}"
    echo "Options:"
    echo "  --project-dir DIR     Project directory containing 'set_env.sh' (default: $DEFAULT_PROJECT_DIR)"
    echo "  --port PORT           Ray master port (default: $DEFAULT_MASTER_PORT)"
    echo "  --dashboard-port PORT Dashboard port (default: $DEFAULT_DASHBOARD_PORT)"
    echo "  --npus-per-node NUM   Number of NPUs per node (default: $DEFAULT_NPUS_PER_NODE)"
    echo "  --wait-time SEC       Wait time for head node initialization (default: $DEFAULT_WAIT_TIME)"
    echo "  --help                Show this help message"
}

# 日志函数
log_message() {
    local level=$1
    local color=$2
    local msg=$3
    echo -e "${color}[$level]${NC} $msg"
}

log_info() { log_message "INFO" "$GREEN" "$1"; }
log_warn() { log_message "WARN" "$YELLOW" "$1" >&2; }
log_error() { log_message "ERROR" "$RED" "$1" >&2; }
log_fatal() { log_message "FATAL" "$RED" "$1" >&2; exit 1; }

# --- 3. 参数解析与环境设置 ---
PROJECT_DIR="$DEFAULT_PROJECT_DIR"
MASTER_PORT="$DEFAULT_MASTER_PORT"
DASHBOARD_PORT="$DEFAULT_DASHBOARD_PORT"
NPUS_PER_NODE=$DEFAULT_NPUS_PER_NODE
WAIT_TIME=$DEFAULT_WAIT_TIME
NODE_LIST_FILE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        --port) MASTER_PORT="$2"; shift 2 ;;
        --dashboard-port) DASHBOARD_PORT="$2"; shift 2 ;;
        --npus-per-node) NPUS_PER_NODE="$2"; shift 2 ;;
        --wait-time) WAIT_TIME="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        -*) log_fatal "Unknown option '$1'. Use --help for usage." ;;
        *) NODE_LIST_FILE="$1"; shift ;;
    esac
done

# --- 4. 核心函数：远程执行命令 ---

# 在远程节点上执行命令，并确保环境配置 (cd $PROJECT_DIR && source set_env.sh)
remote_exec() {
    local node=$1
    local cmd=$2
    # 使用 'bash -c' 来确保命令组合在一个 shell 中执行
    ssh "$node" "bash -c 'cd \"$PROJECT_DIR\" && source set_env.sh && $cmd'"
}

# 停止节点上的 Ray 进程 (使用 -f 强制停止)
stop_ray_node() {
    local node=$1
    log_info "Stopping existing Ray on $node..."
    # '|| true' 确保即使 Ray 未运行也不会导致脚本退出 (在 set -e 模式下很重要)
    remote_exec "$node" "ray stop -f >/dev/null 2>&1 || true"
}

# 启动 Ray 节点函数
start_ray_node() {
    local node=$1
    local is_head=$2
    local cmd

    # 停止旧进程
    stop_ray_node "$node"

    # 正确转义JSON资源参数
    local resources_json="{\\\"NPU\\\": $NPUS_PER_NODE}"

    if $is_head; then
        log_info "[HEAD] Starting Ray head on $node (Master: $MASTER_ADDR:$MASTER_PORT)..."
        # --dashboard-host=0.0.0.0 允许外部访问仪表盘
        cmd="ray start --head --port $MASTER_PORT --node-ip-address $MASTER_ADDR --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT --resources=\"$resources_json\""
    else
        log_info "[WORKER] Starting Ray worker on $node (Connecting to: $MASTER_ADDR:$MASTER_PORT)..."
        cmd="ray start --address $MASTER_ADDR:$MASTER_PORT --resources=\"$resources_json\""
    fi

    if ! remote_exec "$node" "$cmd"; then
        log_error "Failed to start Ray on node $node"
        return 1
    fi
    return 0
}

# --- 5. 预检与节点列表处理 ---

# 检查必需参数和文件
if [[ -z "$NODE_LIST_FILE" ]]; then
    log_fatal "Node list file is required."
fi

# 检查节点文件是否存在
if [ ! -f "$NODE_LIST_FILE" ]; then
    log_fatal "Node list file '$NODE_LIST_FILE' does not exist!"
fi

# 从文件读取节点列表到数组 (使用 < "$VAR" 语法，并忽略空行和注释行)
mapfile -t NODE_HOSTS < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

# 检查节点列表是否为空
if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    log_fatal "Node list '$NODE_LIST_FILE' is empty or contains no valid hosts."
fi

# 定义集群角色
MASTER_ADDR=${NODE_HOSTS[0]}
NUM_NODES=${#NODE_HOSTS[@]}
WORKERS=("${NODE_HOSTS[@]:1}")  # 除第一个节点外的所有节点作为 worker

# 打印集群信息
log_info "============================================="
log_info "Ray Cluster Setup Configuration"
log_info "============================================="
log_info "Total nodes: $NUM_NODES"
log_info "NPUs per node: $NPUS_PER_NODE"
log_info "Master IP: ${BLUE}$MASTER_ADDR${NC}"
log_info "Master port: $MASTER_PORT"
log_info "Dashboard port: $DASHBOARD_PORT"
log_info "Project directory: $PROJECT_DIR"
log_info "Worker nodes (${#WORKERS[@]}): ${WORKERS[*]:-None}"
log_info "============================================="

# 验证所有节点的 SSH 连接和项目目录
log_info "Verifying SSH connections and project directories on all nodes..."
errors=0
for node in "${NODE_HOSTS[@]}"; do
    # 验证 SSH 连接
    if ! ssh -q "$node" "exit" >/dev/null 2>&1; then
        log_error "SSH connection failed to $node. Ensure SSH keys are set up correctly."
        ((errors++))
        continue
    fi
    # 验证项目目录
    if ! ssh "$node" "[ -d \"$PROJECT_DIR\" ]" >/dev/null 2>&1; then
        log_error "Project directory $PROJECT_DIR not found on node $node. Please check the path."
        ((errors++))
        continue
    fi
    # 验证 set_env.sh 文件
    if ! ssh "$node" "[ -f \"$PROJECT_DIR/set_env.sh\" ]" >/dev/null 2>&1; then
        log_error "set_env.sh not found in $PROJECT_DIR on node $node."
        ((errors++))
        continue
    fi
done

if [ $errors -gt 0 ]; then
    log_fatal "Pre-checks failed with $errors errors. Please fix them before continuing."
fi
log_info "All pre-checks passed."

# --- 6. 启动流程 ---
# 启动头节点
if ! start_ray_node "$MASTER_ADDR" true; then
    log_fatal "Failed to start Ray head node on $MASTER_ADDR"
fi

# 等待头节点完全启动
log_info "Waiting ${WAIT_TIME}s for head node to initialize..."
sleep $WAIT_TIME

# 并行启动工作节点
if [ ${#WORKERS[@]} -gt 0 ]; then
    log_info "Starting ${#WORKERS[@]} worker nodes in parallel..."
    pids=()
    node_names=()

    for worker in "${WORKERS[@]}"; do
        # 在后台启动工作节点
        start_ray_node "$worker" false &
        pids+=($!)
        node_names+=("$worker")
    done

    # 等待所有工作节点启动完成
    success_count=0
    failed_nodes=()

    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        node=${node_names[$i]}
        if wait $pid; then
            ((success_count++))
            log_info "Worker node $node connected successfully."
        else
            log_error "Worker node $node failed to connect."
            failed_nodes+=("$node")
        fi
    done

    failed_count=${#failed_nodes[@]}
else
    log_info "No worker nodes defined. Starting single-node cluster."
    success_count=1 # Head node is running
    failed_count=0
fi

# --- 7. 最终报告 ---

echo ""
echo -e "${BLUE}=============================================${NC}"
if [ $failed_count -eq 0 ]; then
    log_message "SUCCESS" "$GREEN" "Ray cluster setup completed successfully!"
else
    log_warn "Ray cluster setup completed with $failed_count worker node(s) failed!"
    log_info "Failed nodes: ${failed_nodes[*]}"
fi

log_info "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
log_info "Total nodes: $NUM_NODES ($success_count workers running)"
echo -e "${BLUE}=============================================${NC}"

# --- 8. 可选：显示 Ray 状态 ---
log_info "Displaying Ray status..."
remote_exec "$MASTER_ADDR" "ray status"
