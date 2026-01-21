#!/bin/bash

# =================================================================
# 脚本名称: auto_del_user.sh
# 核心功能: 在多个节点上自动删除用户
# 特色点:
#   1. 支持命令行参数配置
#   2. 自动跳过指纹确认
#   3. 错误隔离，单个节点失败不影响其他节点
#   4. 显示处理进度和统计结果
# =================================================================

show_help() {
    echo "
Usage: $0 [OPTIONS]

Automate user deletion across multiple nodes.

OPTIONS:
    -f, --file PATH     Path to IP list file (default: ./ip.list.txt)
    -u, --user NAME     Username to delete on nodes (default: jianzhnie)
    -h, --help          Show this help message

IP LIST FORMAT:
    Each line should contain an IP address
    Comments must start with #

    Example:
        192.168.1.10
        192.168.1.11
        # This is a comment
"
}

# --- 配置区 ---
filename="./ip.list.txt"
default_username="jianzhnie"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            filename="$2"
            shift 2
            ;;
        -u|--user)
            default_username="$2"
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
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5"

# 检查必需的命令
for cmd in ssh sshpass; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ 错误: 未找到命令 $cmd"
        exit 1
    fi
done

# 检查IP列表文件是否存在
if [ ! -f "$filename" ]; then
    echo "❌ 错误: 找不到文件 $filename"
    echo "💡 提示: 使用 -h 查看帮助信息"
    exit 1
fi

# 解析IP列表文件
nodes=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    nodes+=("${line}")
done < "$filename"

# 检查是否有有效的节点
if [ ${#nodes[@]} -eq 0 ]; then
    echo "❌ 错误: 在 $filename 中未找到有效的节点"
    exit 1
fi

echo "🔍 发现 ${#nodes[@]} 个节点:"
for node in "${nodes[@]}"; do
    echo "   - $node"
done

# 初始化统计变量
success_count=0
failed_nodes=()

# 遍历每个节点
for i in "${!nodes[@]}"; do
    ip="${nodes[$i]}"
    echo "[$((i+1))/${#nodes[@]}] >>> 正在处理节点: $ip"

    # 构建删除用户的命令
    remote_cmd="
        if id '$default_username' &>/dev/null; then
            # 删除用户及其主目录
            userdel -r '$default_username'
            if [ \$? -eq 0 ]; then
                echo '✅ 用户 $default_username 在 $ip 删除成功'
            else
                echo '❌ 用户 $default_username 在 $ip 删除失败'
            fi
        else
            echo '⚠️ 用户 $default_username 不存在，跳过删除'
        fi
    "

    # 执行远程命令
    if ssh $SSH_OPTS "root@$ip" "$remote_cmd"; then
        ((success_count++))
    else
        echo "❌ 在节点 $ip 上执行失败"
        failed_nodes+=("$ip")
    fi
done

echo "------------------------------------------------"
echo "✅ 全部完成! 成功处理 ${success_count}/${#nodes[@]} 个节点"

if [ ${#failed_nodes[@]} -gt 0 ]; then
    echo "⚠️  以下节点处理失败:"
    for failed_node in "${failed_nodes[@]}"; do
        echo "   - $failed_node"
    done
fi
