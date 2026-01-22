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

# 加载IP列表
if [[ -f "$NODE_LIST_FILE" ]]; then
    echo "从文件 $NODE_LIST_FILE 加载节点IP列表..."
    # 读取所有IP地址到数组中
    mapfile -t all_ips < <(grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' "$NODE_LIST_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # 第一个IP作为服务器IP
    if [[ ${#all_ips[@]} -gt 0 ]]; then
        SERVER_IP="${all_ips[0]}"
        # 剩余IP作为客户端列表
        clients=("${all_ips[@]:1}")  # 从第二个元素开始作为客户端列表
    else
        echo "错误: 文件 $NODE_LIST_FILE 中没有找到有效的IP地址"
        exit 1
    fi
else
    echo "错误: IP列表文件 $NODE_LIST_FILE 不存在"
    exit 1
fi

# 验证服务器IP参数
if [[ ! "$SERVER_IP" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    echo "错误: 无效的服务器IP地址: $SERVER_IP"
    exit 1
fi

# 验证客户端IP数量
if [[ ${#clients[@]} -eq 0 ]]; then
    echo "警告: 没有找到有效的客户端IP地址，仅对服务器进行卸载操作"
fi

echo "服务器IP: $SERVER_IP"
echo "找到 ${#clients[@]} 个客户端IP地址"
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
echo "  - 停止NFS服务"
ssh "$SERVER_IP" "sudo systemctl stop nfs-server.service" || { echo "警告: 无法停止NFS服务，可能服务未运行"; }

echo "  - 从/etc/exports中移除NFS导出配置"
ssh "$SERVER_IP" "sudo sed -i '/$(printf '%s' "$SHARE_PATH $SERVER_NETWORK" | sed 's/[[\.*^$()+?{|]/\\&/g')/d' /etc/exports" || { echo "错误: 无法从/etc/exports中删除配置"; exit 1; }

echo "  - 更新export配置"
ssh "$SERVER_IP" "sudo exportfs -ra" || { echo "错误: 无法更新exportfs"; exit 1; }

echo "  - 删除共享目录 (如果为空)"
ssh "$SERVER_IP" "if [ -d '$SHARE_PATH' ] && [ -z \"\$(ls -A '$SHARE_PATH')\" ]; then sudo rmdir '$SHARE_PATH'; echo '  - 共享目录已删除（如果是空的）'; else echo '  - 共享目录非空或不存在，保留'; fi"

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

    # 尝试卸载挂载点
    echo "  - 卸载NFS挂载点: $MOUNT_POINT"
    ssh "$ip" "sudo umount '$MOUNT_POINT'" 2>/dev/null || { echo "  - 挂载点可能未挂载或已经卸载"; }

    # 从fstab中删除对应条目
    FSTAB_ENTRY_PATTERN="$SERVER_IP:$SHARE_PATH[[:space:]]\+$MOUNT_POINT[[:space:]]\+nfs[[:space:]]\+defaults,_netdev[[:space:]]\+0[[:space:]]\+0"
    FSTAB_ENTRY_ESCAPED="$SERVER_IP:$SHARE_PATH $MOUNT_POINT nfs defaults,_netdev 0 0"

    echo "  - 从/etc/fstab中移除条目"
    ssh "$ip" "sudo sed -i '\@$(printf '%s' "$FSTAB_ENTRY_ESCAPED" | sed 's/[[\.*^$()+?{|]/\\&/g')@d' /etc/fstab" || { echo "  - 无法从fstab删除条目，可能不存在"; }

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
