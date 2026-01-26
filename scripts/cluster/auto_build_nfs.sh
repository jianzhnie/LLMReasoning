#!/bin/bash
#
# Auto Build NFS Server & Clients Script
#
# 功能：
# 1. 自动识别 OS (Ubuntu/Debian/CentOS/RHEL) 并安装 NFS 依赖
# 2. 配置 NFS 服务端 (exports, 权限, 防火墙)
# 3. 配置 NFS 客户端 (fstab, 自动挂载)
# 4. 支持多节点批量部署
#
# 参考文档: docs/docs/build_nfs_server.md

set -u

# ================= 配置与默认值 =================
DEFAULT_SHARE_PATH="/home/jianzhnie/llmtuner"
DEFAULT_NODE_LIST_FILE="ip.list.txt"
DEFAULT_MOUNT_POINT="/home/jianzhnie/llmtuner"
# 默认挂载参数:
# rw: 读写
# noatime: 不更新访问时间(提升性能)
# rsize=1048576,wsize=1048576: 增大读写块大小(提升吞吐, 适合大文件)
# namlen=255: 文件名长度
# hard: 硬挂载, 网络断开会挂起直到恢复(保护数据一致性)
# proto=tcp: 使用TCP协议
# timeo=600: 超时时间
# retrans=2: 重试次数
DEFAULT_MOUNT_OPTS="defaults,noatime,rsize=1048576,wsize=1048576,namlen=255,hard,proto=tcp,timeo=600,retrans=2,_netdev"
# 默认导出参数
DEFAULT_EXPORT_OPTS="rw,sync,no_subtree_check,no_root_squash"

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
    -n, --network CIDR          允许访问的客户端网段 (例如: 10.42.24.0/24)
                                如不指定，自动根据服务端IP计算 /24 网段
    -o, --mount-opts OPTS       挂载参数 (默认: $DEFAULT_MOUNT_OPTS)
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
MOUNT_OPTS="$DEFAULT_MOUNT_OPTS"
ALLOW_NETWORK=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--node-list) NODE_LIST_FILE="$2"; shift 2 ;;
        -p|--share-path) SHARE_PATH="$2"; shift 2 ;;
        -m|--mount-point) MOUNT_POINT="$2"; shift 2 ;;
        -n|--network) ALLOW_NETWORK="$2"; shift 2 ;;
        -o|--mount-opts) MOUNT_OPTS="$2"; shift 2 ;;
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

# 如果未指定网段，自动计算 /24
if [[ -z "$ALLOW_NETWORK" ]]; then
    ALLOW_NETWORK=$(echo "$SERVER_IP" | sed 's/\.[0-9]*$/.0\/24/')
    log_info "自动识别允许网段: $ALLOW_NETWORK"
fi

echo "================ 配置概览 ================"
echo "NFS 服务端: $SERVER_IP"
echo "共享路径:   $SHARE_PATH"
echo "允许网段:   $ALLOW_NETWORK"
echo "NFS 客户端: ${#CLIENT_IPS[@]} 个 (${CLIENT_IPS[*]})"
echo "挂载点:     $MOUNT_POINT"
echo "挂载参数:   $MOUNT_OPTS"
echo "=========================================="

read -p "确认开始部署? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# ================= 服务端部署函数 =================

deploy_server() {
    local ip=$1
    local share_path=$2
    local network=$3

    # 1. 识别 OS 并安装软件
    # 生成安装脚本
    local install_script="
        set -e
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=\$ID
        else
            echo '无法识别操作系统'
            exit 1
        fi

        echo \"Detected OS: \$OS\"

        if [[ \"\$OS\" == \"ubuntu\" ]] || [[ \"\$OS\" == \"debian\" ]]; then
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-kernel-server
            systemctl enable nfs-kernel-server
            SERVICE_NAME=nfs-kernel-server
        elif [[ \"\$OS\" == \"centos\" ]] || [[ \"\$OS\" == \"rhel\" ]] || [[ \"\$OS\" == \"fedora\" ]]; then
            yum install -y nfs-utils
            systemctl enable nfs-server
            SERVICE_NAME=nfs-server
        else
            echo \"不支持的操作系统: \$OS\"
            exit 1
        fi

        # 2. 创建目录
        mkdir -p \"$share_path\"
        chown nobody:nogroup \"$share_path\" || chown nobody:nobody \"$share_path\"
        chmod 777 \"$share_path\"

        # 3. 配置 Exports
        # 备份
        cp /etc/exports /etc/exports.bak.\$(date +%s)
        # 检查是否存在，不存在则追加
        if ! grep -qF \"$share_path\" /etc/exports; then
            echo \"$share_path $network($DEFAULT_EXPORT_OPTS)\" >> /etc/exports
            echo \"Added export config\"
        else
            # 如果存在，尝试替换 (这里简单处理，建议手动检查如果配置复杂)
            sed -i \"\|$share_path|d\" /etc/exports
            echo \"$share_path $network($DEFAULT_EXPORT_OPTS)\" >> /etc/exports
            echo \"Updated export config\"
        fi

        exportfs -ra
        systemctl restart \$SERVICE_NAME

        # 4. 防火墙配置
        if command -v ufw >/dev/null && systemctl is-active --quiet ufw; then
            ufw allow from $network to any port nfs
            ufw allow from $network to any port 2049
            ufw allow from $network to any port 111
            echo \"Configured UFW\"
        elif command -v firewall-cmd >/dev/null && systemctl is-active --quiet firewalld; then
            firewall-cmd --permanent --add-service=nfs
            firewall-cmd --permanent --add-service=rpc-bind
            firewall-cmd --permanent --add-service=mountd
            firewall-cmd --reload
            echo \"Configured Firewalld\"
        else
            echo \"Warning: No active firewall detected or managed manually. Ensure ports 2049/tcp, 111/tcp/udp are open.\"
        fi
    "

    remote_exec "$ip" "$install_script" "部署 NFS 服务端"
}

# ================= 客户端部署函数 =================

deploy_client() {
    local ip=$1
    local server_ip=$2
    local share_path=$3
    local mount_point=$4
    local opts=$5

    local install_script="
        set -e
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=\$ID
        fi

        # 1. 安装客户端
        if [[ \"\$OS\" == \"ubuntu\" ]] || [[ \"\$OS\" == \"debian\" ]]; then
            # 避免 apt update 过于频繁，可以根据需要开启
            # apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-common
        elif [[ \"\$OS\" == \"centos\" ]] || [[ \"\$OS\" == \"rhel\" ]]; then
            yum install -y nfs-utils
        fi

        # 2. 创建挂载点
        mkdir -p \"$mount_point\"

        # 3. 挂载
        # 检查是否已经挂载
        if mountpoint -q \"$mount_point\"; then
            echo \"Already mounted\"
        else
            mount -t nfs -o \"$opts\" \"$server_ip:$share_path\" \"$mount_point\"
            echo \"Mounted successfully\"
        fi

        # 4. 持久化 (fstab)
        # 备份
        cp /etc/fstab /etc/fstab.bak.\$(date +%s)
        # 移除旧配置 (防止重复)
        sed -i \"\|$mount_point|d\" /etc/fstab
        # 添加新配置
        echo \"$server_ip:$share_path $mount_point nfs $opts 0 0\" >> /etc/fstab
        echo \"Updated /etc/fstab\"
    "

    remote_exec "$ip" "$install_script" "部署 NFS 客户端"
}

# ================= 执行流程 =================

log_info ">>> 开始部署 NFS 服务端: $SERVER_IP"
if deploy_server "$SERVER_IP" "$SHARE_PATH" "$ALLOW_NETWORK"; then
    log_info "服务端部署成功"
else
    log_error "服务端部署失败，终止流程"
    exit 1
fi

SUCCESS_COUNT=0
FAIL_COUNT=0

log_info ">>> 开始部署 NFS 客户端 (共 ${#CLIENT_IPS[@]} 台)"
for client_ip in "${CLIENT_IPS[@]}"; do
    # 跳过服务端IP (如果它也在客户端列表中)
    if [[ "$client_ip" == "$SERVER_IP" ]]; then
        # 即使是本机，也可能需要挂载（看需求），这里假设如果是同一台机器，做本地挂载
        log_info "跳过服务端本机 ($client_ip) 作为客户端的远程部署 (建议本机使用 bind mount 或直接访问)"
        continue
    fi

    if deploy_client "$client_ip" "$SERVER_IP" "$SHARE_PATH" "$MOUNT_POINT" "$MOUNT_OPTS"; then
        log_info "客户端 $client_ip 部署成功"
        ((SUCCESS_COUNT++))
    else
        log_error "客户端 $client_ip 部署失败"
        ((FAIL_COUNT++))
    fi
done

echo
echo "================ 部署完成 ================"
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo "=========================================="

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
