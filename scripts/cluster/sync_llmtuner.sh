#!/bin/bash

# 脚本用于将本地的 /home/jianzhengnie/llmtuner 目录同步到所有目标节点

# Note: 根据经验教训内存，对于批量操作的脚本，我们移除了 set -e 以避免单个节点失败中断整体流程
# 但保留了 -u 和 -o pipefail 来检测未定义变量和管道错误
set -uo pipefail

# 默认IP列表文件路径
DEFAULT_IP_LIST_FILE="ip.list.current"
SOURCE_DIR="/home/jianzhnie/llmtuner"
DEST_DIR="/home/jianzhnie/llmtuner"

# 显示帮助信息
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Synchronizes the local llmtuner directory to all nodes listed in the IP file.

OPTIONS:
    -i, --ip-list PATH    Path to IP list file (default: $DEFAULT_IP_LIST_FILE)
    -s, --source PATH     Source directory to sync (default: $SOURCE_DIR)
    -d, --destination PATH Destination directory (default: $DEST_DIR)
    -h, --help            Show this help message

EXAMPLES:
    $0                            # Sync using default settings
    $0 -i /path/to/custom_ips.txt # Sync using custom IP list
    $0 -s /custom/source -d /custom/dest # Sync custom source to destination

EOF
}

# 解析命令行参数
IP_LIST_FILE="$DEFAULT_IP_LIST_FILE"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--ip-list)
            IP_LIST_FILE="$2"
            shift 2
            ;;
        -s|--source)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -d|--destination)
            DEST_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
    esac
done

# 检查源目录是否存在
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# 检查IP列表文件是否存在
if [[ ! -f "$IP_LIST_FILE" ]]; then
    echo "Error: IP list file '$IP_LIST_FILE' does not exist."
    exit 1
fi

echo "Starting synchronization of $SOURCE_DIR to all nodes..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo "IP List: $IP_LIST_FILE"
echo ""

# 读取IP列表并统计总数
total_ips=()
while IFS= read -r ip || [[ -n "$ip" ]]; do
    # 跳过空行和注释行
    if [[ -n "$ip" && ! "$ip" =~ ^[[:space:]]*# ]]; then
        total_ips+=("$ip")
    fi
done < "$IP_LIST_FILE"

total_count=${#total_ips[@]}
echo "Found $total_count IP addresses to process."

# 统计变量
success_count=0
failed_ips=()

# 遍历IP数组进行同步
for ip in "${total_ips[@]}"; do
    echo "Syncing to $ip..."

    # 使用 rsync 命令同步目录
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$ip" "mkdir -p $(dirname "$DEST_DIR")" 2>/dev/null && \
       rsync -avz --delete --timeout=30 -e "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" "$SOURCE_DIR/" "$ip:$DEST_DIR/"; then
        echo "✓ Successfully synced to $ip"
        ((success_count++))
    else
        echo "✗ Failed to sync to $ip"
        failed_ips+=("$ip")
    fi
done

echo ""
echo "Synchronization completed!"
echo "Total: $total_count nodes, Success: $success_count nodes, Failed: ${#failed_ips[@]} nodes"

if [ ${#failed_ips[@]} -gt 0 ]; then
    echo "Failed nodes:"
    printf '%s\n' "${failed_ips[@]}"
fi

if [[ $success_count -eq $total_count ]]; then
    echo "All nodes synchronized successfully!"
    exit 0
else
    echo "Some nodes failed to synchronize."
    exit 1
fi
