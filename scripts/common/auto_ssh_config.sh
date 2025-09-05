#!/bin/bash

#----------------------------------------
# Shell 脚本设置
#----------------------------------------
# -e: 任何命令执行失败时立即退出脚本
# -u: 使用未定义的变量时报错并退出
# -o pipefail: 管道中的任何命令失败时，整个管道命令失败并返回非零状态
set -euo pipefail

#----------------------------------------
# 帮助信息和参数解析
#----------------------------------------
usage() {
    echo "Usage: $0 [NODE_LIST_FILE]"
    echo
    echo "用于在多个 Linux 服务器上配置 SSH 免密登录。"
    echo
    echo "Arguments:"
    echo "  NODE_LIST_FILE    包含节点 IP 或主机名的文件路径 (默认为: ./node_list_all.txt)"
    exit 1
}

# 检查参数数量
if [ "$#" -gt 1 ]; then
    echo "❌ 错误: 参数过多。"
    usage
fi

# 设置节点列表文件路径，如果未提供则使用默认值
NODE_LIST_FILE="${1:-"./node_list_all.txt"}"

#----------------------------------------
# 用户配置
#----------------------------------------
# 远程服务器的用户名和密码
# **警告: 将密码明文存储在脚本中存在安全风险。建议在使用后立即删除脚本。**
readonly USERNAME=""
readonly PASSWORD=""

#----------------------------------------
# 节点列表处理
#----------------------------------------
# 检查节点文件是否存在
if [ ! -f "$NODE_LIST_FILE" ]; then
    echo "❌ 错误: 节点列表文件 '$NODE_LIST_FILE' 不存在！"
    usage
fi

# 从文件读取节点列表到数组，同时忽略空行和注释行
mapfile -t NODE_HOSTS < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

# 检查节点列表是否为空
if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    echo "❌ 错误: 节点列表 '$NODE_LIST_FILE' 为空。"
    exit 1
fi

#----------------------------------------
# Expect 函数库
#----------------------------------------
# `expect_ssh` 函数: 通过 expect 自动登录并执行命令
# 该函数只返回远程命令的标准输出
expect_ssh() {
    local host="$1"
    local username="$2"
    local password="$3"
    local command="$4"

    host=$(echo "$host" | tr -d '\r')
    # expect 脚本
    local expect_script=$(cat <<-EOF
        set timeout 30
        set output ""
        spawn -noecho ssh $username@$host "$command"
        expect {
            "*(yes/no)?" { send "yes\r"; exp_continue }
            "*assword:*" { send "$password\r"; exp_continue }
            timeout { puts "❌ 错误: 连接 $host 超时"; exit 1 }
            eof {
                set output \$expect_out(buffer)
            }
        }
        # puts \$output
        exit
EOF
)

    # 通过管道将 expect 脚本传递给 expect 命令，并捕获其输出
    echo "$expect_script" | expect -
}

# `expect_scp` 函数: 通过 expect 自动进行文件传输
expect_scp() {
    local host="$1"
    local username="$2"
    local password="$3"
    local source="$4"
    local destination="$5"

    expect -c "
        set timeout 30
        spawn scp -r \"$source\" \"$username@$host:$destination\"
        expect {
            \"(yes/no)?\" { send \"yes\r\"; exp_continue }
            \"*assword:\" { send \"$password\r\" }
            \"100%\" { puts \"\n✅ 传输成功\" }
            timeout { puts \"\n❌ 错误: 传输到 $host 超时\"; exit 1 }
            eof
        }
        expect eof
    "
}

#----------------------------------------
# 主要逻辑
#----------------------------------------
echo "✨ 脚本开始执行 SSH 免密登录配置"
echo "----------------------------------------"

# 1. 在每个节点上生成 SSH 密钥对
echo "➡️ 步骤 1: 在每个远程节点上生成 SSH 密钥对..."
for host in "${NODE_HOSTS[@]}"; do
    echo "   - 节点: $host"
    expect_ssh "$host" "$USERNAME" "$PASSWORD" "
        mkdir -p /home/$USERNAME/.ssh
        chmod 700 /home/$USERNAME/.ssh
        rm -f /home/$USERNAME/.ssh/id_rsa*
        ssh-keygen -t rsa -f /home/$USERNAME/.ssh/id_rsa -P ''
    "
done
echo "✅ SSH 密钥生成完成。"
echo

# 2. 收集所有节点的公钥到本地
echo "➡️ 步骤 2: 收集所有节点的公钥到本地..."
temp_authorized_keys=$(mktemp)
echo "   - 临时文件: $temp_authorized_keys"

for host in "${NODE_HOSTS[@]}"; do
    echo "   - 正在从 $host 获取公钥..."
    expect_ssh "$host" "$USERNAME" "$PASSWORD" "cat /home/$USERNAME/.ssh/id_rsa.pub" >> "$temp_authorized_keys"
done
echo "✅ 所有公钥已收集到本地。"
echo

# 3. 将整合后的公钥分发回所有节点
echo "➡️ 步骤 3: 将整合后的公钥分发到所有节点..."
for host in "${NODE_HOSTS[@]}"; do
    echo "   - 正在将公钥分发至 $host..."
    expect_scp "$host" "$USERNAME" "$PASSWORD" "$temp_authorized_keys" "/home/$USERNAME/.ssh/authorized_keys"

    # 确保 authorized_keys 文件的权限正确
    expect_ssh "$host" "$USERNAME" "$PASSWORD" "chmod 600 /home/$USERNAME/.ssh/authorized_keys"
done
echo "✅ 公钥分发完成。"
echo

# 4. 清理临时文件
echo "➡️ 步骤 4: 清理临时文件..."
rm -f "$temp_authorized_keys"
echo "✅ 清理完成。"
echo

echo "🎉 所有节点间的 SSH 免密登录配置成功！"
