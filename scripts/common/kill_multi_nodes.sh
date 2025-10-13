#!/bin/bash

# ==============================================================================
# 多节点进程清理脚本
#
# 该脚本通过 SSH 并发连接到多个节点，根据关键字终止指定的 Python 相关进程。
# 脚本会首先尝试温和地终止进程（SIGTERM），超时后若进程仍存活，则强制终止（SIGKILL）。
# 脚本已优化，会安全地排除 VS Code 相关的后台进程，并增加了用户确认步骤。
# ==============================================================================

# --- 全局配置 ---
# 设置最大并发数，控制同时处理的节点数量，避免 SSH 连接风暴
MAX_JOBS=16

# 节点列表文件路径
NODE_LIST_FILE="/home/jianzhnie/llmtuner/tools/nodes/node_list_all.txt"

# 定义要 kill 的关键词（支持正则）
KEYWORDS=("llmtuner" "llm_workspace" "mindspeed" "ray" "vllm" "python")

# 终止进程的超时时间（秒），用于 SIGTERM。
# 在此时间后，如果进程未退出，将执行 SIGKILL 强制终止
KILL_TIMEOUT=3

# SSH 超时时间（秒），防止 SSH 卡死
SSH_TIMEOUT=5

# --- 辅助函数 ---

# 日志时间戳函数，用于打印带时间戳的日志信息
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---
# 函数：终止指定节点上的进程
# 参数: $1 - 节点名称
# ---
kill_processes_on_node() {
    local node=$1
    log "🔎 Processing node: $node"

    # 构建正则表达式，用于在进程列表中匹配
    local pattern=$(IFS='|'; echo "${KEYWORDS[*]}")

    # 构建远程执行命令
    local remote_cmd="
        # 查找所有匹配关键字的进程 ID，并排除 VS Code 相关的进程
        # 'grep -v' 用于排除指定的关键词
        pids=\$(ps aux | grep -E '$pattern' | grep -v 'grep -E' | grep -v 'vscode-server' | grep -v 'extension' | grep -v 'agent' | awk '{print \$2}')

        if [ -n \"\$pids\" ]; then
            echo \"Found PIDs: \$pids matching '$pattern'.\"

            # 1. 尝试温和终止 (SIGTERM)
            echo 'Attempting to gracefully terminate processes (SIGTERM)...'
            kill -15 \$pids 2>/dev/null

            # 等待一段时间，检查进程是否已退出
            sleep $KILL_TIMEOUT

            # 2. 检查进程是否仍然存活 - 修复语法错误
            remaining_pids=\$(ps -p \"\${pids// /,}\" -o pid= 2>/dev/null)

            if [ -n \"\$remaining_pids\" ]; then
                echo \"Processes still alive: \$remaining_pids. Forcing kill (SIGKILL)...\"
                kill -9 \$remaining_pids 2>/dev/null
                echo 'Successfully killed remaining processes.'
            else
                echo 'All processes terminated gracefully.'
            fi
        else
            echo 'No matching processes found.'
        fi
    "

    # 使用 SSH 执行远程命令，带有超时控制
    # 使用 `timeout` 外部命令来确保整个 SSH 会话不会永久挂起
    if timeout $SSH_TIMEOUT ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes "$node" "$remote_cmd"; then
        log "✅ Successfully processed node: $node"
    else
        local exit_code=$?
        log "❌ Failed to process node: $node (exit code: $exit_code)"
    fi
}

# ---
# 主逻辑开始
# ---
log "�� Starting multi-node process cleanup..."
log "Target keywords: ${KEYWORDS[*]}"
log "Max concurrent jobs: $MAX_JOBS"


# ------------------------------------------------------------------------------
# 添加用户确认步骤
# ------------------------------------------------------------------------------
echo "================================================================"
echo "⚠️  WARNING: This script will kill processes on multiple nodes."
echo "   It targets processes with keywords: ${KEYWORDS[*]}"
echo "   This action is irreversible and may interrupt running jobs."
echo "----------------------------------------------------------------"
read -p "Type 'yes' to continue, or anything else to abort: " user_confirm

if [[ "$user_confirm" != "yes" ]]; then
    log "Aborting process cleanup. No changes have been made."
    exit 0
fi
echo "================================================================"
log "Confirmation received. Proceeding with cleanup..."


# 检查节点列表文件
if [[ ! -f "$NODE_LIST_FILE" ]]; then
    log "ERROR: Node list file not found: $NODE_LIST_FILE"
    exit 1
fi


# 读取节点列表
mapfile -t NODES < "$NODE_LIST_FILE"

if [[ ${#NODES[@]} -eq 0 ]]; then
    log "ERROR: No nodes found in $NODE_LIST_FILE"
    exit 1
fi


# 遍历所有节点，并发执行
for NODE in "${NODES[@]}"; do
    # 确保节点非空
    [[ -z "$NODE" ]] && continue

    # 启动后台任务
    kill_processes_on_node "$NODE" &

    # 控制并发数量
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 0.5
    done
done

# 等待所有后台任务完成
wait
log "�� All specified processes have been cleaned up on all nodes."