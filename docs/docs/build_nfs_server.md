
# NFS 搭建

## 背景

有一个16个节点的集群，需要将数据存储在某个节点上，并希望在多个节点之间共享文件。例如，主节点 IP 是 10.42.24.130，其他节点的 IP 分别是 10.42.24.131, 10.42.24.132, ..., 10.42.24.145。


## 概述

搭建 NFS（Network File System，网络文件系统）可以让你在多台 Linux 服务器之间共享目录，就像访问本地硬盘一样方便。

以下是搭建 NFS 系统的标准步骤，分为 **服务端（Server）** 和 **客户端（Client）** 两个部分。


## 第一步：服务端配置 (Server)

服务端是存放物理文件并提供共享服务的机器。通常选择其中一个性能较好、磁盘空间充足的节点作为服务端（Server），其余节点作为客户端（Client）。

### 1. 安装软件

首先更新包列表并安装 NFS 内核服务器：

```bash
sudo apt update
sudo apt install nfs-kernel-server  # Ubuntu/Debian
# 或者
sudo yum install nfs-utils         # CentOS/RHEL
```

### 2. 创建共享目录

创建一个你想共享给其他机器访问的文件夹，并设置权限：

```bash
sudo mkdir -p /mnt/nfs_share
sudo chown nobody:nogroup /mnt/nfs_share  # 确保权限通用
sudo chmod 777 /mnt/nfs_share
```

### 3. 配置导出路径

由于节点较多，我们直接授权 10.42.24.0/24 这个网段的所有机器访问，避免手动输入其他 15 个 IP：

编辑 `/etc/exports` 文件，告知 NFS 哪些目录可以共享给谁：

```bash
sudo nano /etc/exports
```

在文件末尾添加一行（将 `client_ip` 替换为客户端的 IP，或者用 `*` 表示允许所有 IP）：

```text
/mnt/nfs_share  10.42.24.0/24(rw,sync,no_subtree_check,no_root_squash)
```
> 注意：no_root_squash 允许客户端以 root 权限操作文件，这在集群环境中很常用，但请确保环境安全。

* **rw**: 读写权限。
* **sync**: 数据同步写入内存和磁盘，更安全。
* **no_subtree_check**: 禁用子树检查，提高性能。

### 4. 启动服务

```bash
sudo exportfs -a
sudo systemctl restart nfs-server.service
```

## 第二步：客户端配置 (Client)

客户端是需要挂载并使用共享目录的机器。

### 1. 安装客户端工具

```bash
sudo apt update
sudo apt install nfs-common  # Ubuntu/Debian
# 或者
sudo yum install nfs-utils    # CentOS/RHEL
```

### 2. 创建挂载点

在客户端本地创建一个目录，用于映射服务端的共享文件夹：

```bash
sudo mkdir -p /mnt/nfs_clientside
```

### 3. 挂载共享目录

使用 `mount` 命令进行连接（将 `server_ip` 替换为服务端的实际 IP）：

```bash
sudo mount 10.42.24.130:/mnt/nfs_share /mnt/nfs_clientside
```

## 第三步：验证与自动挂载

### 1. 验证

在客户端运行 `df -h`，你应该能看到挂载成功的 NFS 分区。或者在客户端的 `/mnt/nfs_clientside` 创建一个文件，看看服务端是否同步出现。

### 2. 设置开机自动挂载

如果不配置这一步，重启后挂载会失效。编辑 `/etc/fstab`：

```bash
sudo nano /etc/fstab
```

添加以下行：

```text
echo "10.42.24.130:/mnt/nfs_share  /mnt/nfs_clientside  nfs  defaults  0  0" | sudo tee -a /etc/fstab
```

## 3. 快速检查列表

| 节点类型    | IP 地址                | 角色说明                           |
| ----------- | ---------------------- | ---------------------------------- |
| **Server**  | `10.42.24.130`         | 存放数据，运行 `nfs-kernel-server` |
| **Clients** | `10.42.24.131` ~ `145` | 挂载数据，运行 `nfs-common`        |


## 💡 给您的专业建议

1. **防火墙策略**：如果你开启了防火墙，请在 **130 节点**上放行来自 `10.42.24.0/24` 的流量。
2. **性能优化**：如果你在进行高并发读写（如深度学习训练），可以在客户端挂载参数中添加 `rsize=32768,wsize=32768` 以提升吞吐量。
3. **单点故障**：NFS 服务端（130）一旦宕机，所有客户端的挂载点都会变得不可用（“Stale file handle”）。如果这是生产环境，后期可以考虑使用 **Keepalived + NFS** 做高可用。

您是否有批量管理工具（如 Ansible），需要我为您提供一份 **Ansible Playbook** 来一键部署这 16 台机器吗？


## 常见问题排查（避坑指南）

* **防火墙问题**：如果连接失败，请确保服务端的防火墙允许 NFS 流量。
* Ubuntu (UFW): `sudo ufw allow from [client_ip] to any port nfs`
* CentOS (firewalld): `sudo firewall-cmd --permanent --add-service=nfs`

* **权限拒绝**：如果无法写入，请检查服务端的目录权限以及 `/etc/exports` 中的 `rw` 配置。



# 统一挂载路径

要实现在所有节点（包括服务端和客户端）以**完全相同的路径**访问数据，最标准的做法是采用**统一挂载点策略**。

假设我们选定的统一路径是 `/data/nfs`。

## 1. 统一挂载路径方案

为了让代码、脚本或程序在任何一台机器上运行都能通过 `/data/nfs/my_file.txt` 找到文件，我们需要统一服务端和客户端的目录结构。

### 在服务端 (10.42.24.130)

服务端不仅是提供者，它自己也可以通过本地路径访问。

1. **创建物理目录**：
```bash
sudo mkdir -p /data/nfs
sudo chown nobody:nogroup /data/nfs
sudo chmod 777 /data/nfs

```

2. **导出配置** (`/etc/exports`)：
```text
/data/nfs  10.42.24.0/24(rw,sync,no_subtree_check,no_root_squash)
```



### 在所有客户端 (10.42.24.131 - 145)

客户端需要将远程路径挂载到**相同的本地路径**：

1. **创建相同的挂载点**：
```bash
sudo mkdir -p /data/nfs
```

2. **执行挂载**：
```bash
sudo mount 10.42.24.130:/data/nfs /data/nfs
```

## 2. 自动化配置（推荐）

由于节点较多，手动操作极易出错。建议使用循环脚本在所有客户端执行。

### 客户端批量挂载脚本

在你的管理机（或 130 节点）上运行以下循环（假设已配置 SSH 免密登录）：

```bash
# 定义所有客户端 IP
clients=(10.42.24.131 10.42.24.132 10.42.24.133 10.42.24.134 10.42.24.135 10.42.24.136 10.42.24.137 10.42.24.138 10.42.24.139 10.42.24.140 10.42.24.141 10.42.24.142 10.42.24.143 10.42.24.144 10.42.24.145)

for ip in "${clients[@]}"; do
    echo "正在配置节点: $ip"
    # 创建目录
    ssh $ip "sudo mkdir -p /data/nfs"
    # 写入 fstab 实现开机自启（先备份，再追加）
    ssh $ip "echo '10.42.24.130:/data/nfs /data/nfs nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab"
    # 立即执行挂载
    ssh $ip "sudo mount -a"
done
```


## 3. 如何验证“路径一致性”？

配置完成后，你可以进行如下测试：

1. **在 130 节点**：
`touch /data/nfs/test_file.txt`
2. **在 145 节点（或其他任意节点）**：
`ls /data/nfs/test_file.txt`

如果能看到该文件，说明路径已完全统一。


## ⚠️ 进阶建议：符号链接 (Symbolic Link)

如果你发现原始数据已经在服务端的某个奇怪路径下（比如 `/var/lib/docker/data/...`），但你想在所有节点通过 `/projects/data` 访问，你可以使用**软链接**：

1. 在所有节点上执行：
```bash
sudo ln -s /data/nfs /projects/data
```

2. 这样，无论实际挂载点在哪，你的应用程序只需要访问 `/projects/data` 即可。
