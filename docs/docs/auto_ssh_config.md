# 自动化 SSH 免密登录配置脚本

## 功能介绍

这个 Bash 脚本专为需要管理多个 Linux 服务器的用户设计，旨在简化 \*\*SSH 免密登录（passwordless SSH）\*\*的配置过程。通过该脚本，您可以快速地在集群内的所有节点之间建立起相互信任的 SSH 连接，从而无需在每次登录或执行远程命令时手动输入密码。

该脚本的核心工作流程如下：

1.  **生成密钥对**：在每个远程服务器上，脚本会使用 `ssh-keygen` 自动生成一对新的 RSA 密钥（公钥和私钥）。
2.  **收集公钥**：脚本会遍历所有节点，将每个节点的公钥文件 (`id_rsa.pub`) 收集到本地的一个临时文件中。
3.  **整合与分发**：将所有收集到的公钥内容整合成一个完整的 `authorized_keys` 文件。随后，脚本将这个文件分发回集群中的每一个节点，并存放在用户的 `.ssh` 目录下。
4.  **设置权限**：确保 `.ssh` 目录和 `authorized_keys` 文件的权限正确，以符合 SSH 协议的安全要求。

脚本特别使用了 **`expect`** 工具，实现了自动化输入密码，避免了手动交互。这使得整个过程完全自动化，极大地提高了效率，尤其是在配置大规模集群时。

> **安全警告：** 脚本中直接明文存储了远程服务器的密码。这存在潜在的安全风险。在非生产环境中使用时，请务必注意安全，并在使用完毕后立即删除脚本，或将密码替换为更安全的凭证管理方式。

-----

## 使用教程

## 1\. 前置条件

在运行脚本前，请确保您的系统已满足以下条件：

  * **Linux/macOS 环境**：脚本是为类 Unix 系统（如 Linux、macOS）设计的。
  * **安装 `expect` 工具**：此脚本依赖 `expect` 来实现自动化交互。您可以使用包管理器进行安装：
      * 在基于 Debian/Ubuntu 的系统上：`sudo apt-get install expect`
      * 在基于 RedHat/CentOS 的系统上：`sudo yum install expect`
  * **准备节点列表文件**：创建一个文本文件，每行包含一个远程服务器的 IP 地址或主机名。默认文件名为 `node_list_all.txt`。

## 2\. 准备工作

  * **克隆或下载脚本**：将脚本文件保存到您的本地机器上，例如命名为 `auto_ssh_config.sh`。

  * **设置执行权限**：使用 `chmod` 命令为脚本添加可执行权限：

    ```bash
    chmod +x auto_ssh_config.sh
    ```

  * **创建节点列表文件**：在脚本所在的目录下，创建一个名为 `node_list_all.txt` 的文件，并填入您的服务器列表，例如：

    ```
    # node_list_all.txt
    192.168.1.101
    192.168.1.102
    ```

    > 脚本会忽略文件中的空行和以 `#` 开头的注释行。

  * **修改用户名和密码**：打开脚本文件，找到 `USERNAME` 和 `PASSWORD` 变量，将其修改为您的远程服务器用户名和密码。

    ```bash
    readonly USERNAME="your_username"
    readonly PASSWORD="your_password"
    ```

## 3\. 运行脚本

完成上述准备后，您可以在终端中直接运行脚本。

  * **使用默认节点列表文件**：
    ```bash
    ./auto_ssh_config.sh
    ```
  * **指定自定义节点列表文件**：如果您使用的文件名不是 `node_list_all.txt`，可以在运行脚本时作为参数传入。
    ```bash
    ./auto_ssh_config.sh /path/to/your_node_list.txt
    ```

脚本运行后，您将看到实时的输出日志，详细显示了每个步骤的执行情况。当看到 "🎉 所有节点间的 SSH 免密登录配置成功！" 的消息时，表示整个配置过程已顺利完成。



## Auto SSH Config
```bash
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
```
