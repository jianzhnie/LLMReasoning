#!/bin/bash

# =================================================================
# 脚本名称: auto_ssh_config.sh
# 核心功能: 实现所有节点(及其相互之间)的完全免密登录
# 特色点:
#   1. 自动跳过指纹确提示 (StrictHostKeyChecking=no)
#   2. 自动修复 openEuler/CentOS 家目录权限
#   3. 实现 Mesh 型全互联 (任意两台皆免密)
# =================================================================

show_help() {
    echo "
Usage: $0 [OPTIONS]

Automate SSH key distribution for multiple nodes to enable passwordless login.

OPTIONS:
    -f, --file PATH     Path to IP list file (default: ./ip.list.txt)
    -u, --user NAME     Default username for hosts without @ specified (default: jianzhnie)
    -p, --password PASS Password for SSH connection (default: pcl@0312)
    -h, --help          Show this help message

IP LIST FORMAT:
    Each line should contain either:
    - IP address (will use default username)
    - user@host format

    Example:
        192.168.1.10
        admin@192.168.1.11
        # This is a comment

NOTES:
    - Comments must start with #
    - Empty lines are ignored
    - Ensure sshpass is installed on the system
"
}

# --- 配置区 ---
filename="./ip.list.txt"  # 修改默认文件名为更通用的名称
default_user="jianzhnie"
hostpassword='pcl@0312'  # 建议使用单引号包裹，防止特殊字符被转义

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            filename="$2"
            shift 2
            ;;
        -u|--user)
            default_user="$2"
            shift 2
            ;;
        -p|--password)
            hostpassword="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# SSH 参数：静默模式、自动接受指纹、不再读取/写入 known_hosts、超时5秒
SSH_OPTS="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5"

# 1. 检查并安装依赖
if ! command -v sshpass &> /dev/null; then
    echo "📦 正在安装依赖 sshpass..."
    if command -v yum &> /dev/null; then
        sudo yum install -y sshpass
    elif command -v apt-get &> /dev/null; then
        sudo apt-get install -y sshpass
    else
        echo "❌ 无法检测到包管理器，请手动安装 sshpass"
        exit 1
    fi
fi

# 2. 检查必需的命令
for cmd in ssh ssh-keygen sshpass; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ 错误: 未找到命令 $cmd"
        exit 1
    fi
done

# 3. 生成本地密钥 (若无)
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "🔑 正在生成本地 SSH 密钥..."
    ssh-keygen -t rsa -b 4096 -q -f ~/.ssh/id_rsa -N "" || {
        echo "❌ 生成本地 SSH 密钥失败"
        exit 1
    }
fi

# 4. 解析 IP 列表文件
if [ ! -f "$filename" ]; then
    echo "❌ 错误: 找不到文件 $filename"
    echo "💡 提示: 使用 -h 查看帮助信息"
    exit 1
fi

nodes=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" == *"@"* ]]; then
        nodes+=("$line")
    else
        nodes+=("$default_user@$line")
    fi
done < "$filename"

# 输出找到的节点数量
if [ ${#nodes[@]} -eq 0 ]; then
    echo "❌ 错误: 在 $filename 中未找到有效的节点"
    exit 1
fi

echo "🔍 发现 ${#nodes[@]} 个节点:"
for node in "${nodes[@]}"; do
    echo "   - $node"
done

# 5. 创建临时空间收集公钥
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
all_keys_file="$temp_dir/combined_authorized_keys"

# 首先加入本地公钥
cat ~/.ssh/id_rsa.pub > "$all_keys_file"

echo "------------------------------------------------"
echo "Step 1: 正在生成并收集各节点的公钥 (已跳过指纹确认)..."

success_count=0
failed_nodes=()

for i in "${!nodes[@]}"; do
    node="${nodes[$i]}"
    echo "[$((i+1))/${#nodes[@]}] -> 正在处理: $node"

    # 远程执行：修复权限 -> 创建.ssh -> 生成密钥 -> 传回公钥内容
    pub_content=$(sshpass -p "$hostpassword" ssh $SSH_OPTS "$node" "
        chmod 755 ~
        mkdir -p ~/.ssh && chmod 700 ~/.ssh
        [ ! -f ~/.ssh/id_rsa ] && ssh-keygen -t rsa -b 4096 -q -f ~/.ssh/id_rsa -N '' > /dev/null
        cat ~/.ssh/id_rsa.pub
    " 2>/dev/null || echo "FAILED")

    if [ "$pub_content" != "FAILED" ]; then
        echo "$pub_content" >> "$all_keys_file"
        ((success_count++))
        echo "    ✅ 成功处理: $node"
    else
        echo "    ⚠️  连接失败: $node (请检查网络或密码)"
        failed_nodes+=("$node")
    fi
done

# 汇总去重
sort -u "$all_keys_file" -o "$all_keys_file"

echo "------------------------------------------------"
echo "Step 2: 正在全网分发互信授权文件..."

# 更新本地 authorized_keys，先备份再更新
if [ -f ~/.ssh/authorized_keys ]; then
    cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak
    sort -u ~/.ssh/authorized_keys "$all_keys_file" -o ~/.ssh/authorized_keys
else
    cp "$all_keys_file" ~/.ssh/authorized_keys
fi
chmod 600 ~/.ssh/authorized_keys

# 分发到所有远程节点
deploy_success_count=0
deploy_failed_nodes=()

for i in "${!nodes[@]}"; do
    node="${nodes[$i]}"
    echo "[$((i+1))/${#nodes[@]}] -> 部署全量公钥至: $node"

    # 1. 传输汇总后的文件
    if sshpass -p "$hostpassword" scp $SSH_OPTS "$all_keys_file" "$node:.ssh/authorized_keys.tmp" 2>/dev/null; then
        # 2. 远程执行：备份原文件，替换，设置权限
        if sshpass -p "$hostpassword" ssh $SSH_OPTS "$node" "
            mkdir -p ~/.ssh
            chmod 700 ~/.ssh
            [ -f ~/.ssh/authorized_keys ] && mv ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak 2>/dev/null || true
            mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys
            chmod 600 ~/.ssh/authorized_keys
            chown \$(id -un):\$(id -gn) ~/.ssh ~/.ssh/authorized_keys 2>/dev/null || true
        " 2>/dev/null; then
            echo "    ✅ 成功部署至: $node"
            ((deploy_success_count++))
        else
            echo "    ❌ 部署失败: $node"
            deploy_failed_nodes+=("$node")
        fi
    else
        echo "    ❌ 文件传输失败: $node"
        deploy_failed_nodes+=("$node")
    fi
done

echo "------------------------------------------------"
echo "------------------------------------------------"
echo "✅ 全部完成! 收集公钥成功 ${success_count}/${#nodes[@]} 个节点"
echo "✅ 部署授权成功 ${deploy_success_count}/${#nodes[@]} 个节点"

if [ ${#failed_nodes[@]} -gt 0 ]; then
    echo "⚠️  以下节点公钥收集失败:"
    for failed_node in "${failed_nodes[@]}"; do
        echo "   - $failed_node"
    done
fi

if [ ${#deploy_failed_nodes[@]} -gt 0 ]; then
    echo "⚠️  以下节点授权部署失败:"
    for failed_node in "${deploy_failed_nodes[@]}"; do
        echo "   - $failed_node"
    done
fi

echo "💡 提示: 现在可以从任何节点 SSH 到任何其他节点而无需密码"
