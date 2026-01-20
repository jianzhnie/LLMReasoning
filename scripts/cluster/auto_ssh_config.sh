#!/bin/bash

# =================================================================
# 脚本名称: mesh_ssh.sh
# 核心功能: 实现所有节点(及其相互之间)的完全免密登录
# 特色点:
#   1. 自动跳过指纹确提示 (StrictHostKeyChecking=no)
#   2. 自动修复 openEuler/CentOS 家目录权限
#   3. 实现 Mesh 型全互联 (任意两台皆免密)
# =================================================================

# --- 配置区 ---
filename="./ip.list.current"
default_user="jianzhnie"
hostpassword='pcl@0312'  # 建议使用单引号包裹，防止特殊字符被转义

# SSH 参数：静默模式、自动接受指纹、不再读取/写入 known_hosts、超时5秒
SSH_OPTS="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5"

# 移除 set -e 使脚本在单个节点失败时仍能继续执行
# set -e

# 1. 检查并安装依赖
if ! command -v sshpass &> /dev/null; then
    echo "📦 正在安装依赖 sshpass..."
    sudo yum install -y sshpass || sudo apt-get install -y sshpass
fi

# 2. 生成本地密钥 (若无)
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "🔑 正在生成本地 SSH 密钥..."
    ssh-keygen -t rsa -b 4096 -q -f ~/.ssh/id_rsa -N ""
fi

# 3. 解析 IP 列表文件
if [ ! -f "$filename" ]; then
    echo "❌ 错误: 找不到文件 $filename"
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
echo "🔍 发现 ${#nodes[@]} 个节点:"

# 4. 创建临时空间收集公钥
temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
all_keys_file="$temp_dir/combined_authorized_keys"

# 首先加入本地公钥
cat ~/.ssh/id_rsa.pub > "$all_keys_file"

echo "------------------------------------------------"
echo "Step 1: 正在生成并收集各节点的公钥 (已跳过指纹确认)..."

success_count=0
for node in "${nodes[@]}"; do
    echo " -> 正在处理: $node"

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
    fi
done

# 汇总去重
sort -u "$all_keys_file" -o "$all_keys_file"

echo "------------------------------------------------"
echo "Step 2: 正在全网分发互信授权文件..."

# 更新本地
cat "$all_keys_file" >> ~/.ssh/authorized_keys
sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 分发到所有远程节点
for node in "${nodes[@]}"; do
    echo " -> 部署全量公钥至: $node"

    # 1. 传输汇总后的文件
    if sshpass -p "$hostpassword" scp $SSH_OPTS "$all_keys_file" "$node:~/.ssh/authorized_keys" 2>/dev/null; then
        # 2. 强制修正远程权限及 SELinux
        sshpass -p "$hostpassword" ssh $SSH_OPTS "$node" "
            chmod 600 ~/.ssh/authorized_keys
            [ -x /sbin/restorecon ] && /sbin/restorecon -Rv ~/.ssh >/dev/null 2>&1 || true
        " 2>/dev/null || echo "    ⚠️  权限设置失败: $node"
    else
        echo "    ⚠️  文件传输失败: $node"
    fi
done

echo "------------------------------------------------"
echo "✅ 任务完成！成功处理 $success_count 个节点，总共 ${#nodes[@]} 个节点。"
echo "所有可达节点已建立两两免密互联。你可以直接输入 'ssh IP' 测试，不再有交互提示。"
