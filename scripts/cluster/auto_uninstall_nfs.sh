#!/bin/bash

# 设置脚本选项
set -uo pipefail  # 不使用-e因为我们要处理单个节点的错误而不中断整个流程

# 默认配置
DEFAULT_SHARE_PATH="/home/jianzhnie/llmtuner"
DEFAULT_NODE_LIST_FILE="ip.list.txt"
DEFAULT_MOUNT_POINT="/home/jianzhnie/llmtuner"

# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    -c, --node-list FILE        节点IP列表文件，每行一个IP，第一个为服务器，其余为客户端 (默认: $DEFAULT_NODE_LIST_FILE)
    -p, --share-path PATH       NFS共享路径 (默认: $DEFAULT_SHARE_PATH)
    -m, --mount-point PATH      客户端挂载点 (默认: $DEFAULT_MOUNT_POINT)
    -h, --help                  显示此帮助信息

注意:
    - 如果未指定节点IP列表文件，则会从默认位置读取
    - 节点IP列表文件格式：每行一个IP地址，第一个IP将作为NFS服务器，其余作为客户端
EOF
}

# 初始化变量
SHARE_PATH="$DEFAULT_SHARE_PATH"
NODE_LIST_FILE="$DEFAULT_NODE_LIST_FILE"
MOUNT_POINT="$DEFAULT_MOUNT_POINT"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--node-list)
            NODE_LIST_FILE="$2"
            shift 2
            ;;
        -p|--share-path)
            SHARE_PATH="$2"
            shift 2
            ;;
        -m|--mount-point)
            MOUNT_POINT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ ! "$MOUNT_POINT" =~ ^/ ]]; then
    echo "错误: 挂载点路径必须是绝对路径: $MOUNT_POINT"
    exit 1
fi

if [[ ! "$SHARE_PATH" =~ ^/ ]]; then
    echo "错误: 共享路径必须是绝对路径: $SHARE_PATH"
    exit 1
fi

# 加载节点IP列表
if [[ -f "$NODE_LIST_FILE" ]]; then
    echo "从文件 $NODE_LIST_FILE 加载节点IP列表..."
    mapfile -t nodes < <(grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' "$NODE_LIST_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
else
    echo "错误: 节点IP列表文件 $NODE_LIST_FILE 不存在"
    exit 1
fi

# 验证节点IP数量
if [[ ${#nodes[@]} -eq 0 ]]; then
    echo "错误: 没有找到有效的节点IP地址"
    exit 1
fi

# 第一个节点作为服务器，其余作为客户端
SERVER_IP="${nodes[0]}"
clients=("${nodes[@]:1}")

echo "找到 ${#nodes[@]} 个节点IP地址"
echo "NFS服务器: $SERVER_IP"
echo "NFS客户端数量: ${#clients[@]}"

# 如果没有客户端节点，仍然可以卸载服务器
if [[ ${#clients[@]} -eq 0 ]]; then
    echo "提示: 没有客户端节点，仅卸载服务器上的NFS配置"
fi

echo "共享路径: $SHARE_PATH"
echo "挂载点: $MOUNT_POINT"

# 提示用户确认
read -p "是否继续卸载NFS配置? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# 获取服务器IP的网段
SERVER_NETWORK=$(echo "$SERVER_IP" | sed 's/\.[0-9]*$/.0\/24/')

echo "正在卸载NFS服务器节点: $SERVER_IP"

# 检查SSH连接
if ! ssh -o ConnectTimeout=10 -q "$SERVER_IP" "exit 0" 2>/dev/null; then
    echo "错误: 无法连接到NFS服务器 $SERVER_IP"
    exit 1
fi

# 卸载NFS服务器配置
echo "  - 检查NFS服务是否正在运行"
NFS_RUNNING=$(ssh "$SERVER_IP" "sudo systemctl is-active nfs-server.service 2>/dev/null || echo 'inactive'")

if [[ "$NFS_RUNNING" == "active" ]]; then
    echo "  - 停止NFS服务"
    ssh "$SERVER_IP" "sudo systemctl stop nfs-server.service" || { echo "错误: 无法停止NFS服务"; exit 1; }
else
    echo "  - NFS服务未运行，跳过停止步骤"
fi

echo "  - 检查/etc/exports中是否存在相关配置"
CONFIG_EXISTS=$(ssh "$SERVER_IP" "grep -F '$SHARE_PATH $SERVER_NETWORK' /etc/exports || true")

if [[ -n "$CONFIG_EXISTS" ]]; then
    echo "  - 从/etc/exports中移除NFS导出配置"
    ssh "$SERVER_IP" "sudo sed -i '\@$SHARE_PATH $SERVER_NETWORK@d' /etc/exports" || { echo "错误: 无法从/etc/exports中删除配置"; exit 1; }

    echo "  - 更新export配置"
    ssh "$SERVER_IP" "sudo exportfs -ra" || { echo "错误: 无法更新exportfs"; exit 1; }
    echo "  - NFS服务器配置已移除"
else
    echo "  - NFS导出配置不存在，跳过删除步骤"
fi

echo "  - 删除共享目录 (如果为空)"
ssh "$SERVER_IP" "if [ -d '$SHARE_PATH' ] && [ -z \"\$(ls -A '$SHARE_PATH' 2>/dev/null)\" ]; then sudo rmdir '$SHARE_PATH'; echo '  - 共享目录已删除（如果是空的）'; else echo '  - 共享目录非空或不存在，保留'; fi"

echo "NFS服务器卸载完成"

# 统计变量
SUCCESS_COUNT=0
FAIL_COUNT=0

# 卸载客户端配置
echo "开始卸载 ${#clients[@]} 个NFS客户端配置..."

for i in "${!clients[@]}"; do
    ip="${clients[$i]}"

    echo "正在卸载客户端 ($((i+1))/${#clients[@]}): $ip"

    # 验证IP格式
    if [[ ! "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        echo "  - 跳过无效IP: $ip"
        ((FAIL_COUNT++))
        continue
    fi

    # 检查SSH连接
    if ! ssh -o ConnectTimeout=10 -q "$ip" "exit 0" 2>/dev/null; then
        echo "  - 错误: 无法连接到客户端 $ip"
        ((FAIL_COUNT++))
        continue
    fi

    # 检查是否已挂载
    IS_MOUNTED=$(ssh "$ip" "mount | grep '$MOUNT_POINT' || true")
    if [[ -n "$IS_MOUNTED" ]]; then
        echo "  - 卸载NFS挂载点: $MOUNT_POINT"
        ssh "$ip" "sudo umount '$MOUNT_POINT'" || { echo "  - 错误: 无法卸载挂载点 $MOUNT_POINT"; ((FAIL_COUNT++)); continue; }
    else
        echo "  - 挂载点未挂载或已经卸载"
    fi

    # 从fstab中删除对应条目
    FSTAB_ENTRY="$SERVER_IP:$SHARE_PATH $MOUNT_POINT nfs defaults,_netdev 0 0"
    FSTAB_EXISTS=$(ssh "$ip" "grep -F '$FSTAB_ENTRY' /etc/fstab || true")

    if [[ -n "$FSTAB_EXISTS" ]]; then
        echo "  - 从/etc/fstab中移除条目"
        ssh "$ip" "sudo sed -i '\@$FSTAB_ENTRY@d' /etc/fstab" || { echo "  - 无法从fstab删除条目"; ((FAIL_COUNT++)); continue; }
        echo "  - fstab条目已移除"
    else
        echo "  - fstab条目不存在，跳过删除"
    fi

    # 删除挂载点目录（如果为空）
    echo "  - 删除挂载点目录（如果为空）"
    ssh "$ip" "if [ -d '$MOUNT_POINT' ] && [ -z \"\$(ls -A '$MOUNT_POINT' 2>/dev/null)\" ]; then sudo rmdir '$MOUNT_POINT'; echo '  - 挂载点目录已删除（如果是空的）'; else echo '  - 挂载点非空或不存在，保留'; fi"

    echo "  - 客户端 $ip 卸载完成"
    ((SUCCESS_COUNT++))
done

# 输出最终统计
echo
echo "=== 卸载完成 ==="
echo "成功卸载: $SUCCESS_COUNT 个客户端"
echo "卸载失败: $FAIL_COUNT 个客户端"
echo "总计: ${#clients[@]} 个客户端"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo
    echo "警告: 一些客户端卸载失败，请手动检查这些节点"
    exit 1
fi

echo "NFS卸载全部完成！"
