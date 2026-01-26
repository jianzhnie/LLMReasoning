#!/bin/bash
#
# Auto Uninstall NFS Server & Clients Script
#
# 功能：
# 1. 停止并禁用 NFS 服务
# 2. 清理 NFS 服务端配置 (exports, 防火墙)
# 3. 清理 NFS 客户端配置 (卸载挂载点, fstab)
# 4. 支持多节点批量操作
#
# 对应安装脚本: scripts/cluster/auto_build_nfs.sh

set -u

# ================= 配置与默认值 =================
DEFAULT_SHARE_PATH="/home/jianzhnie/llmtuner"
DEFAULT_NODE_LIST_FILE="ip.list.txt"
DEFAULT_MOUNT_POINT="/home/jianzhnie/llmtuner"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================= 辅助函数 =================

log_info() {
    echo -e "${GREEN}[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1${NC}"
}

show_help() {
    cat << EOF
用法: $0 [选项]

选项:
    -c, --node-list FILE        节点IP列表文件 (默认: $DEFAULT_NODE_LIST_FILE)
                                格式: 第一行为服务端IP，其余为客户端IP
    -p, --share-path PATH       NFS服务端共享路径 (默认: $DEFAULT_SHARE_PATH)
    -m, --mount-point PATH      客户端挂载点路径 (默认: $DEFAULT_MOUNT_POINT)
    -n, --network CIDR          (可选) 指定需要从防火墙规则中移除的网段
                                如不指定，自动根据服务端IP计算 /24 网段
    -h, --help                  显示此帮助信息

示例:
    $0 -c ips.txt -p /data/nfs -m /data/nfs
EOF
}

# 远程执行命令封装
remote_exec() {
    local ip="$1"
    local cmd="$2"
    local desc="${3:-Execute command}"

    log_info "[$ip] $desc..."
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$ip" "$cmd" 2>&1 | sed "s/^/[$ip] /"; then
        return 0
    else
        log_error "[$ip] 执行失败: $desc"
        return 1
    fi
}

# ================= 主逻辑 =================

# 初始化变量
SHARE_PATH="$DEFAULT_SHARE_PATH"
NODE_LIST_FILE="$DEFAULT_NODE_LIST_FILE"
MOUNT_POINT="$DEFAULT_MOUNT_POINT"
ALLOW_NETWORK=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--node-list) NODE_LIST_FILE="$2"; shift 2 ;;
        -p|--share-path) SHARE_PATH="$2"; shift 2 ;;
        -m|--mount-point) MOUNT_POINT="$2"; shift 2 ;;
        -n|--network) ALLOW_NETWORK="$2"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "未知选项: $1"; show_help; exit 1 ;;
    esac
done

# 校验参数
if [[ ! "$SHARE_PATH" =~ ^/ ]] || [[ ! "$MOUNT_POINT" =~ ^/ ]]; then
    log_error "路径必须是绝对路径"
    exit 1
fi

if [[ ! -f "$NODE_LIST_FILE" ]]; then
    log_error "节点列表文件不存在: $NODE_LIST_FILE"
    exit 1
fi

# 读取节点列表
mapfile -t nodes < <(grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' "$NODE_LIST_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^#')

if [[ ${#nodes[@]} -eq 0 ]]; then
    log_error "未找到有效的节点IP"
    exit 1
fi

SERVER_IP="${nodes[0]}"
CLIENT_IPS=("${nodes[@]:1}")

# 如果未指定网段，自动计算 /24 (用于清理防火墙规则)
if [[ -z "$ALLOW_NETWORK" ]]; then
    ALLOW_NETWORK=$(echo "$SERVER_IP" | sed 's/\.[0-9]*$/.0\/24/')
    log_info "自动识别需清理的网段: $ALLOW_NETWORK"
fi

echo "================ 卸载预览 ================"
echo "NFS 服务端: $SERVER_IP"
echo "共享路径:   $SHARE_PATH"
echo "清理网段:   $ALLOW_NETWORK"
echo "NFS 客户端: ${#CLIENT_IPS[@]} 个 (${CLIENT_IPS[*]})"
echo "挂载点:     $MOUNT_POINT"
echo "=========================================="
echo "注意: 此操作将停止 NFS 服务并卸载挂载点，请确保没有进程正在使用这些文件。"
echo

read -p "确认开始卸载? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# ================= 服务端卸载函数 =================

undeploy_server() {
    local ip=$1
    local share_path=$2
    local network=$3

    local uninstall_script="
        set -e
        # 1. 识别服务名称
        SERVICE_NAME=\"\"
        if systemctl list-unit-files | grep -q nfs-kernel-server; then
            SERVICE_NAME=nfs-kernel-server
        elif systemctl list-unit-files | grep -q nfs-server; then
            SERVICE_NAME=nfs-server
        fi

        # 2. 停止并禁用服务
        if [[ -n \"\$SERVICE_NAME\" ]]; then
            echo \"Stopping \$SERVICE_NAME...\"
            systemctl stop \$SERVICE_NAME || true
            systemctl disable \$SERVICE_NAME || true
        else
            echo \"NFS service not found, skipping stop/disable.\"
        fi

        # 3. 清理 Exports
        if [ -f /etc/exports ]; then
            if grep -qF \"$share_path\" /etc/exports; then
                # 备份
                cp /etc/exports /etc/exports.bak.uninstall.\$(date +%s)
                # 删除配置行
                sed -i \"\|$share_path|d\" /etc/exports
                echo \"Removed export config for $share_path\"

                # 刷新配置 (即使服务停止了，最好也清理一下状态)
                exportfs -ra || true
            else
                echo \"Export config not found for $share_path, skipping.\"
            fi
        fi

        # 4. 清理防火墙
        if command -v ufw >/dev/null && systemctl is-active --quiet ufw; then
            echo \"Cleaning UFW rules...\"
            # 尝试删除规则，忽略错误（如果规则不存在）
            ufw delete allow from $network to any port nfs || true
            ufw delete allow from $network to any port 2049 || true
            ufw delete allow from $network to any port 111 || true
        elif command -v firewall-cmd >/dev/null && systemctl is-active --quiet firewalld; then
            echo \"Cleaning Firewalld rules...\"
            firewall-cmd --permanent --remove-service=nfs || true
            firewall-cmd --permanent --remove-service=rpc-bind || true
            firewall-cmd --permanent --remove-service=mountd || true
            firewall-cmd --reload
        fi

        # 5. 清理共享目录 (可选，这里只在目录为空时删除，避免误删数据)
        if [ -d \"$share_path\" ]; then
            if [ -z \"\$(ls -A \"$share_path\")\" ]; then
                rmdir \"$share_path\"
                echo \"Removed empty share directory: $share_path\"
            else
                echo \"Share directory is not empty, keeping it: $share_path\"
            fi
        fi
    "

    remote_exec "$ip" "$uninstall_script" "卸载 NFS 服务端"
}

# ================= 客户端卸载函数 =================

undeploy_client() {
    local ip=$1
    local mount_point=$2

    local uninstall_script="
        set -e

        # 1. 卸载挂载点
        if mountpoint -q \"$mount_point\"; then
            echo \"Unmounting $mount_point...\"
            # 尝试正常卸载，失败则尝试强制卸载 (lazy unmount)
            umount \"$mount_point\" || umount -l \"$mount_point\"
            echo \"Unmounted successfully\"
        else
            echo \"Not mounted, skipping unmount.\"
        fi

        # 2. 清理 fstab
        if grep -qF \"$mount_point\" /etc/fstab; then
            # 备份
            cp /etc/fstab /etc/fstab.bak.uninstall.\$(date +%s)
            # 删除配置行
            sed -i \"\|$mount_point|d\" /etc/fstab
            echo \"Removed fstab entry for $mount_point\"

            # 重新加载 daemon 以防万一 (对于 systemd)
            systemctl daemon-reload || true
        else
            echo \"No fstab entry found for $mount_point, skipping.\"
        fi

        # 3. 删除挂载点目录 (仅当为空时)
        if [ -d \"$mount_point\" ]; then
            if [ -z \"\$(ls -A \"$mount_point\")\" ]; then
                rmdir \"$mount_point\"
                echo \"Removed empty mount point: $mount_point\"
            else
                echo \"Mount point directory is not empty (or busy), keeping it: $mount_point\"
            fi
        fi
    "

    remote_exec "$ip" "$uninstall_script" "卸载 NFS 客户端"
}

# ================= 执行流程 =================

SUCCESS_COUNT=0
FAIL_COUNT=0

# 1. 先卸载客户端 (防止服务端停了客户端卡死)
log_info ">>> 开始卸载 NFS 客户端 (共 ${#CLIENT_IPS[@]} 台)"
for client_ip in "${CLIENT_IPS[@]}"; do
    if [[ "$client_ip" == "$SERVER_IP" ]]; then
        log_info "跳过服务端本机 ($client_ip) 作为客户端的清理 (将在服务端环节处理或请手动检查)"
        continue
    fi

    if undeploy_client "$client_ip" "$MOUNT_POINT"; then
        log_info "客户端 $client_ip 卸载成功"
        ((SUCCESS_COUNT++))
    else
        log_error "客户端 $client_ip 卸载失败"
        ((FAIL_COUNT++))
    fi
done

# 2. 再卸载服务端
log_info ">>> 开始卸载 NFS 服务端: $SERVER_IP"
if undeploy_server "$SERVER_IP" "$SHARE_PATH" "$ALLOW_NETWORK"; then
    log_info "服务端卸载成功"
else
    log_error "服务端卸载失败"
    # 服务端失败不应该影响脚本的退出状态码，除非它真的很重要。
    # 这里我们记录错误，但允许脚本继续完成（实际上是最后一步了）。
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo
echo "================ 卸载完成 ================"
echo "客户端成功: $SUCCESS_COUNT"
echo "客户端失败/服务端失败: $FAIL_COUNT"
echo "=========================================="

if [[ $FAIL_COUNT -gt 0 ]]; then
    log_warn "部分节点卸载失败，请检查上方日志。"
    exit 1
fi
