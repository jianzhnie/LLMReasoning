#!/bin/bash

# 设置脚本选项
set -uo pipefail  # 不使用-e因为我们要处理单个节点的错误而不中断整个流程

# 默认配置
DEFAULT_SERVER_IP="10.42.24.194"
DEFAULT_SHARE_PATH="/home/jianzhnie/llmtuner"
DEFAULT_CLIENT_LIST_FILE="ip.list.txt"
DEFAULT_MOUNT_POINT="/home/jianzhnie/llmtuner"

# 显示帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    -s, --server-ip IP          NFS服务器IP地址 (默认: $DEFAULT_SERVER_IP)
    -c, --client-list FILE      客户端IP列表文件，每行一个IP (默认: $DEFAULT_CLIENT_LIST_FILE)
    -p, --share-path PATH       NFS共享路径 (默认: $DEFAULT_SHARE_PATH)
    -m, --mount-point PATH      客户端挂载点 (默认: $DEFAULT_MOUNT_POINT)
    -h, --help                  显示此帮助信息

注意:
    - 如果未指定客户端IP列表文件，则会从默认位置读取
    - 客户端IP列表文件格式：每行一个IP地址
EOF
}

# 初始化变量
SERVER_IP="$DEFAULT_SERVER_IP"
SHARE_PATH="$DEFAULT_SHARE_PATH"
CLIENT_LIST_FILE="$DEFAULT_CLIENT_LIST_FILE"
MOUNT_POINT="$DEFAULT_MOUNT_POINT"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--server-ip)
            SERVER_IP="$2"
            shift 2
            ;;
        -c|--client-list)
            CLIENT_LIST_FILE="$2"
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

# 验证输入参数
if [[ ! "$SERVER_IP" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
    echo "错误: 无效的服务器IP地址: $SERVER_IP"
    exit 1
fi

if [[ ! "$MOUNT_POINT" =~ ^/ ]]; then
    echo "错误: 挂载点路径必须是绝对路径: $MOUNT_POINT"
    exit 1
fi

if [[ ! "$SHARE_PATH" =~ ^/ ]]; then
    echo "错误: 共享路径必须是绝对路径: $SHARE_PATH"
    exit 1
fi

# 加载客户端IP列表
if [[ -f "$CLIENT_LIST_FILE" ]]; then
    echo "从文件 $CLIENT_LIST_FILE 加载客户端IP列表..."
    mapfile -t clients < <(grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' "$CLIENT_LIST_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
else
    echo "警告: 客户端IP列表文件 $CLIENT_LIST_FILE 不存在，使用默认IP列表"
    clients=(
        10.42.24.195
        10.42.24.196
        10.42.24.197
        10.42.24.198
        10.42.24.199
        10.42.24.200
        10.42.24.201
        10.42.24.202
        10.42.24.203
        10.42.24.204
        10.42.24.205
        10.42.24.206
        10.42.24.207
        10.42.24.208
        10.42.24.209
    )
fi

# 验证客户端IP数量
if [[ ${#clients[@]} -eq 0 ]]; then
    echo "错误: 没有找到有效的客户端IP地址"
    exit 1
fi

echo "找到 ${#clients[@]} 个客户端IP地址"
echo "NFS服务器: $SERVER_IP"
echo "共享路径: $SHARE_PATH"
echo "挂载点: $MOUNT_POINT"

# 提示用户确认
read -p "是否继续? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# 获取服务器IP的网段
SERVER_NETWORK=$(echo "$SERVER_IP" | sed 's/\.[0-9]*$/.0\/24/')

echo "正在配置NFS服务器节点: $SERVER_IP"
# 检查SSH连接
if ! ssh -o ConnectTimeout=10 -q "$SERVER_IP" "exit 0" 2>/dev/null; then
    echo "错误: 无法连接到NFS服务器 $SERVER_IP"
    exit 1
fi

echo "检查NFS服务是否已安装..."
# 建议添加：
ssh "$SERVER_IP" "command -v nfsd || (echo 'NFS服务未安装' && exit 1)"


# 配置NFS服务器
echo "  - 创建共享目录: $SHARE_PATH"
ssh "$SERVER_IP" "sudo mkdir -p '$SHARE_PATH'" || { echo "错误: 无法在服务器上创建目录"; exit 1; }

echo "  - 设置目录权限"
ssh "$SERVER_IP" "sudo chmod 777 '$SHARE_PATH'" || { echo "错误: 无法设置目录权限"; exit 1; }

echo "  - 检查/etc/exports中是否已有相关配置"
CONFIG_EXISTS=$(ssh "$SERVER_IP" "grep -F '$SHARE_PATH $SERVER_NETWORK' /etc/exports || true")

if [[ -z "$CONFIG_EXISTS" ]]; then
    echo "  - 添加NFS导出配置到/etc/exports"
    ssh "$SERVER_IP" "echo '$SHARE_PATH $SERVER_NETWORK(rw,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports" || { echo "错误: 无法写入/etc/exports"; exit 1; }
else
    echo "  - NFS导出配置已存在"
fi

echo "  - 更新export配置"
ssh "$SERVER_IP" "sudo exportfs -ra" || { echo "错误: 无法更新exportfs"; exit 1; }

echo "  - 重启NFS服务"
ssh "$SERVER_IP" "sudo systemctl restart nfs-server.service" || { echo "错误: 无法重启NFS服务"; exit 1; }

echo "NFS服务器配置完成"

# 统计变量
SUCCESS_COUNT=0
FAIL_COUNT=0

# 配置客户端
echo "开始配置 ${#clients[@]} 个NFS客户端..."

for i in "${!clients[@]}"; do
    ip="${clients[$i]}"

    echo "正在配置客户端 ($((i+1))/${#clients[@]}): $ip"

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

    # 创建挂载点
    echo "  - 创建挂载点: $MOUNT_POINT"
    ssh "$ip" "sudo mkdir -p '$MOUNT_POINT'" || { echo "  - 错误: 无法在客户端 $ip 上创建挂载点"; ((FAIL_COUNT++)); continue; }

    # 检查fstab中是否已存在配置
    FSTAB_ENTRY="$SERVER_IP:$SHARE_PATH $MOUNT_POINT nfs defaults,_netdev 0 0"
    FSTAB_EXISTS=$(ssh "$ip" "grep -F '$FSTAB_ENTRY' /etc/fstab || true")

    if [[ -z "$FSTAB_EXISTS" ]]; then
        echo "  - 添加fstab条目以实现开机自动挂载"
        ssh "$ip" "echo '$FSTAB_ENTRY' | sudo tee -a /etc/fstab" || { echo "  - 错误: 无法写入/etc/fstab"; ((FAIL_COUNT++)); continue; }
    else
        echo "  - fstab条目已存在"
    fi

    # 尝试立即挂载
    echo "  - 执行挂载操作"
    if ssh "$ip" "sudo mount -a"; then
        echo "  - 客户端 $ip 配置成功"
        ((SUCCESS_COUNT++))
    else
        echo "  - 警告: 客户端 $ip 挂载失败，但fstab配置已完成"
        ((FAIL_COUNT++))
    fi
done

# 输出最终统计
echo
echo "=== 配置完成 ==="
echo "成功配置: $SUCCESS_COUNT 个客户端"
echo "配置失败: $FAIL_COUNT 个客户端"
echo "总计: ${#clients[@]} 个客户端"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo
    echo "警告: 一些客户端配置失败，请手动检查这些节点"
    exit 1
fi

echo "NFS配置全部完成！"
