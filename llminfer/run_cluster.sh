#!/bin/bash
set -euo pipefail

# ==================================================================================
# 📦 全局常量定义
# ==================================================================================

# 项目目录（所有节点必须存在）
readonly PROJECT_DIR="/root/llmtuner/llm/LLMReasoning/llminfer"

# 节点列表（可根据需要修改）
readonly NODES=("10.16.201.108" "10.16.201.198")

# Ray 配置参数
readonly MASTER_PORT="29500"
readonly DASHBOARD_PORT="8266"
readonly NPUS_PER_NODE=8

# SSH connection options
readonly SSH_OPTS="-o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR"

# 派生变量
readonly NUM_NODES=${#NODES[@]}
readonly MASTER_ADDR=${NODES[0]}
readonly WORKERS=("${NODES[@]:1}")  # All except the first node

# ==============================
# Functions
# ==============================

# Usage: print_usage
print_usage() {
    echo "============================================="
    echo ""
    echo "Usage: $0"
    echo ""
    echo "Starts a Ray cluster across the defined nodes."
    echo ""
    echo "Requirements:"
    echo " - SSH access to all nodes"
    echo " - Set \$PROJECT_DIR exists on all nodes"
    echo " - set_env.sh must be present in project directory"
    echo "============================================="
    echo ""
}

# Usage: check_project_dir <node>
# 检查远程节点是否包含指定的项目目录
check_project_dir() {
    local node="$1"

    # Check project directory exists
    if ! ssh "$node" "[ -d \"$PROJECT_DIR\" ]"; then
        echo "[ERROR] Project directory '$PROJECT_DIR' does not exist on node '$node'" >&2
        return 1
    fi

    # Check set_env.sh exists
    if ! ssh "$node" "[ -f \"$PROJECT_DIR/set_env.sh\" ]"; then
        echo "[ERROR] set_env.sh missing in project directory on node '$node'" >&2
        return 1
    fi

    echo "[INFO] Project directory and set_env.sh verified on '$node'"
}

# 在指定节点上启动 Ray 进程（head 或 worker）
# Arguments:
#   $1 - Node IP/hostname
#   $2 - true for head node, false for worker
start_ray_node() {
    local node="$1"
    local is_head="$2"

    local cmd=""

    if [[ "$is_head" == true ]]; then
        echo "[HEAD] Starting Ray head node on '$node'..."
        cmd="ray start --head \
            --port=$MASTER_PORT \
            --node-ip-address=$node \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=$DASHBOARD_PORT \
            --num-gpus=$NPUS_PER_NODE"
    else
        echo "[WORKER] Starting Ray worker node on '$node'..."
        cmd="ray start --address=$MASTER_ADDR:$MASTER_PORT \
            --node-ip-address=$node \
            --num-gpus=$NPUS_PER_NODE"
    fi

    # 执行远程命令：cd -> source env -> stop old ray -> start new ray
    # Execute remote commands:
    # 1. Change to project directory
    # 2. Source environment setup
    # 3. Export required environment variables
    # 4. Stop any existing Ray processes
    # 5. Start new Ray node
    # Execute remote commands in a single SSH session
    local remote_cmd="
        set -e
        cd '$PROJECT_DIR'
        source set_env.sh
        export VLLM_HOST_IP='$node' HCCL_IF_IP='$node'
        ray stop >/dev/null 2>&1 || true
        $cmd
    "

    if ssh $SSH_OPTS "$node" "$remote_cmd"; then
        echo "[SUCCESS] Ray started on '$node'"
        return 0
    else
        echo "[ERROR] Failed to start Ray on '$node'" >&2
        return 1
    fi
}

# Usage: verify_cluster_setup
# 校验集群部署的前提条件
verify_cluster_setup() {
    echo "[INFO] Verifying cluster setup prerequisites..."

    # Validate master address
    if [[ -z "$MASTER_ADDR" ]]; then
        echo "[FATAL] MASTER_ADDR is empty!" >&2
        exit 1
    fi

    # Validate node list
    if [[ "$NUM_NODES" -lt 1 ]]; then
        echo "[FATAL] No nodes defined!" >&2
        exit 1
    fi

    # Check all nodes have required directories/files
    for node in "${NODES[@]}"; do
        if ! check_project_dir "$node"; then
            echo "[FATAL] Missing project directory on node '$node'. Aborting." >&2
            exit 1
        fi
    done
}

# Usage: print_cluster_info
# 打印当前配置信息
print_cluster_info() {
    echo "============================================="
    echo "Ray Cluster Setup Configuration"
    echo "============================================="
    echo "Number of nodes: $NUM_NODES"
    echo "NPUs per node: $NPUS_PER_NODE"
    echo "Master IP: $MASTER_ADDR"
    echo "Master port: $MASTER_PORT"
    echo "Dashboard port: $DASHBOARD_PORT"
    echo "Worker nodes: ${WORKERS[*]}"
    echo "VLLM_HOST_IP: $VLLM_HOST_IP"
    echo "HCCL_IF_IP: $HCCL_IF_IP"
    echo "============================================="
    echo ""
}

# ==============================
# 🚀 Main Execution
# ==============================

main() {
    print_usage
    verify_cluster_setup
    print_cluster_info

    echo "[INFO] Starting Ray head node..."
    if ! start_ray_node "$MASTER_ADDR" true; then
        echo "[FATAL] Failed to start head node." >&2
        exit 1
    fi

    echo "[INFO] Waiting for head node initialization..."
    sleep 3

    # Start worker nodes in parallel
    if [[ "${#WORKERS[@]}" -gt 0 ]]; then
        echo "[INFO] Starting ${#WORKERS[@]} worker node(s) in parallel..."
        declare -a pids
        local worker

        for worker in "${WORKERS[@]}"; do
            start_ray_node "$worker" false &
                pids+=($!)
        done

        # Wait for background processes
        local pid
        for pid in "${pids[@]}"; do
            if wait "$pid"; then
                echo "[INFO] Worker started successfully (PID: $pid)"
            else
                echo "[WARNING] Worker startup failed (PID: $pid)" >&2
            fi
        done
    else
        echo "[INFO] No worker nodes configured"
    fi


    echo ""
    echo "============================================="
    echo "✅ Ray cluster setup complete!"
    echo "Dashboard URL: http://$MASTER_ADDR:$DASHBOARD_PORT"
    echo "============================================="
}

# Run main function
main "$@"
