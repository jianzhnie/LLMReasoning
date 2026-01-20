# 自动化 SSH 免密登录配置脚本

## 功能介绍

这个 Bash 脚本专为需要管理多个 Linux 服务器的用户设计，旨在简化 \*\*SSH 免密登录（passwordless SSH）\*\*的配置过程。通过该脚本，您可以快速地在集群内的所有节点之间建立起相互信任的 SSH 连接，从而无需在每次登录或执行远程命令时手动输入密码。

该脚本的核心工作流程如下：

1.  **生成密钥对**：在每个远程服务器上，脚本会使用 `ssh-keygen` 自动生成一对新的 RSA 密钥（公钥和私钥）。
2.  **收集公钥**：脚本会遍历所有节点，将每个节点的公钥文件 (`id_rsa.pub`) 收集到本地的一个临时文件中。
3.  **整合与分发**：将所有收集到的公钥内容整合成一个完整的 `authorized_keys` 文件。随后，脚本将这个文件分发回集群中的每一个节点，并存放在用户的 `.ssh` 目录下。
4.  **设置权限**：确保 `.ssh` 目录和 `authorized_keys` 文件的权限正确，以符合 SSH 协议的安全要求。

> **安全警告：** 脚本中直接明文存储了远程服务器的密码。这存在潜在的安全风险。在非生产环境中使用时，请务必注意安全，并在使用完毕后立即删除脚本，或将密码替换为更安全的凭证管理方式。

## 使用教程

## 1\. 前置条件

在运行脚本前，请确保您的系统已满足以下条件：

  * **Linux/macOS 环境**：脚本是为类 Unix 系统（如 Linux、macOS）设计的。
  * **安装 `sshpass` 工具**：此脚本依赖 `sshpass` 来实现自动化交互。您可以使用包管理器进行安装：
      * 在基于 Debian/Ubuntu 的系统上：`sudo apt-get install sshpass`
      * 在基于 RedHat/CentOS 的系统上：`sudo yum install sshpass`
  * **准备节点列表文件**：创建一个文本文件，每行包含一个远程服务器的 IP 地址或主机名。默认文件名为 `node_list.txt`。

## 2\. 准备工作

  * **克隆或下载脚本**：将脚本文件保存到您的本地机器上，例如命名为 `auto_ssh_config.sh`。

  * **设置执行权限**：使用 `chmod` 命令为脚本添加可执行权限：

    ```bash
    chmod +x auto_ssh_config.sh
    ```

  * **创建节点列表文件**：在脚本所在的目录下，创建一个名为 `node_list.txt` 的文件，并填入您的服务器列表，例如：

    ```
    # node_list.txt
    192.168.1.101
    192.168.1.102
    ```

    > 脚本会忽略文件中的空行和以 `#` 开头的注释行。

## 3\. 运行脚本

完成上述准备后，您可以在终端中直接运行脚本。

### 基本运行
```bash
./auto_ssh_config.sh
```

### 使用自定义参数
```bash
# 指定IP列表文件、用户名和密码
./auto_ssh_config.sh -f /path/to/your/ip_list.txt -u myusername -p mypassword
```

或者使用长参数：
```bash
./auto_ssh_config.sh --file /path/to/your/ip_list.txt --user myusername --password mypassword
```

### 查看帮助信息
```bash
./auto_ssh_config.sh -h
# 或
./auto_ssh_config.sh --help
```

### 3.1 参数说明

- `-f, --file PATH`: 指定IP列表文件路径（默认为 `./ip.list.current`）
- `-u, --user NAME`: 指定默认用户名（默认为 `jianzhnie`）
- `-p, --password PASS`: 指定SSH密码（默认为 `pcl`）
- `-h, --help`: 显示帮助信息

### 3.2 脚本执行流程

1. **依赖检查**: 检查是否安装了 `sshpass`, `ssh`, `ssh-keygen` 等必要工具
2. **本地密钥生成**: 如果本地没有SSH密钥对，则生成一个新的
3. **IP列表解析**: 读取IP列表文件，解析出所有要处理的节点
4. **公钥收集**: 连接到每个节点，生成SSH密钥（如果不存在），并将公钥收集到本地临时文件
5. **公钥分发**: 将所有收集到的公钥分发到每个节点的 `~/.ssh/authorized_keys` 文件

### 3.3 注意事项

- 确保所有节点都可以通过SSH访问
- 确保已安装 `sshpass` 工具（如果没有会尝试自动安装）
- 确保所有节点使用相同的密码
- 脚本会自动处理节点间的双向免密登录（网状全连接）

### 3.4 运行后验证

脚本执行完成后，您可以从任一节点SSH到其他任何节点而无需输入密码：

```bash
ssh user@node1
# 从node1连接到node2应无需密码
ssh user@node2
```
