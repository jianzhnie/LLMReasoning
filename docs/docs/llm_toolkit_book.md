# 大语言模型（LLM）训练与运维技术指南

本文档旨在为基于昇腾（Ascend）NPU平台的大语言模型训练与部署提供完整的技术指导，涵盖环境配置、依赖安装、代码管理、数据获取、分布式通信测试及常用运维操作等关键环节。

## CANN 安装

```bash
bash Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run --install
bash Atlas-A3-cann-kernels_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install
source /usr/local/Ascend/nnal/atb/set_env.sh
```



## 一、CANN 环境配置

请根据实际安装路径加载昇腾AI软件栈（CANN）相关环境变量：

```bash
install_path=/usr/local/Ascend
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh
```

> **说明**：`set_env.sh` 脚本用于设置编译器、库路径和运行时依赖，确保后续操作能正确调用 NPU 驱动与算子库。



## 二、NPU 设备状态检查

使用 `npu-smi` 工具查看 NPU 设备运行状态：

```bash
npu-smi info
```

该命令将输出设备健康状态、内存使用情况、温度及驱动版本等信息，建议在启动训练前执行以确认硬件可用性。



## 三、PyTorch 与 Torch-NPU 安装

安装指定版本的 `torch` 与 `torch-npu`，以确保与 CANN 版本兼容：

```bash
pip install numpy==1.26.0
pip install torch==2.5.1 && pip install torch-npu==2.5.1rc1
```

> **注意**：`torch-npu` 是华为针对 Ascend 芯片优化的 PyTorch 后端扩展，必须与 `torch` 版本严格匹配。



## 四、PIP 镜像源配置

为加速 Python 包下载，可使用国内镜像源：

### 清华源

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn numpy==1.26.0
```

### 阿里源
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com numpy==1.26.0
```

> **建议**：将常用镜像源写入 `pip.conf` 以避免重复指定。

### 全局配置

#### **设置阿里源(推荐)**

```text
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

pip config set install.trusted-host mirrors.aliyun.com
```

#### **设置清华源**

```text
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```

#### 删除其他配置

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config unset global.extra-index-url
```

第一个命令将默认索引源设置为 `https://mirrors.aliyun.com/pypi/simple/`。

第二个命令将删除所有额外的索引源，包括 `https://download.pytorch.org/whl/cpu/` 和 `https://mirrors.huaweicloud.com/ascend/repos/pypi`。



## 五、数据存储挂载

挂载分布式持久化存储（DPC）至本地目录：

```bash
mount -t dpc /llmdir llmdir
```

请确保挂载路径具备读写权限，并在训练任务中正确引用该路径以实现数据共享。



## 六、文件权限配置

修改 Miniconda 安装目录权限，确保当前用户可正常访问：

```bash
chown -R HwHiAiUser:users miniconda3/
chmod -R 755 miniconda3/
```

> **说明**：`HwHiAiUser` 为昇腾平台默认用户，若使用其他账户请相应调整。



## 七、Weights & Biases（Wandb）登录

登录 Wandb 以启用实验跟踪与可视化功能：

```bash
wandb login
```

执行后将提示输入 API Key，登录成功后可在项目中通过 `wandb.init()` 记录训练指标。



## 八、Conda 环境克隆（实现“重命名”）

Conda 不支持直接重命名环境，但可通过**克隆 + 删除**方式间接实现。

### 场景
将环境 `old_env_name` 重命名为 `new_env_name`。

### 操作步骤

#### 1. 克隆环境

```bash
conda create --name new_env_name --clone old_env_name
```

或显式指定路径：

```bash
conda create --prefix /root/llmdir/miniconda3/envs/rlhf --clone /root/llm_workspace/miniconda3/envs/openRLHF
```

#### 2. 验证新环境

```bash
conda info --envs
```

确认 `new_env_name` 存在且包列表一致。

#### 3. 删除原环境（确认无误后）

```bash
conda remove --name old_env_name --all
```

> **注意事项**：
> - 克隆过程会复制所有包与依赖，耗时较长。
> - 删除前务必验证新环境功能完整性。



## 九、代码与资源获取

### 使用 GitHub 加速代理

为提升 GitHub 资源下载速度，推荐使用代理服务 [https://gh-proxy.com/](https://gh-proxy.com/)。

**示例**：
- 原始链接：`https://github.com/volcengine/verl.git`
- 代理链接：`https://gh-proxy.com/https://github.com/volcengine/verl.git`

### 克隆代码仓库（通过代理）

```bash
git clone https://gh-proxy.com/https://github.com/huggingface/transformers.git
git clone https://gh-proxy.com/https://github.com/volcengine/verl.git
git clone https://gh-proxy.com/https://github.com/wangshuai09/vllm.git
git clone https://gh-proxy.com/https://gitee.com/ascend/MindSpeed.git
git clone https://gh-proxy.com/https://github.com/NVIDIA/Megatron-LM.git
git clone https://gh-proxy.com/https://github.com/huggingface/trl.git
git clone https://gh-proxy.com/https://github.com/hkust-nlp/simpleRL-reason.git
git clone https://gh-proxy.com/https://github.com/as12138/verl.git verl-npu
```



## 十、Git 仓库合并外部 Pull Request（PR）

### 1. 权限要求

需具备目标仓库的**写入权限**或**PR 合并权限**。对于开源项目，通常需维护者审核通过后方可合并。

### 2. Web 界面操作（以 GitHub 为例）

#### （1）进入 PR 页面
在仓库的 **Pull Requests** 标签页中定位目标 PR。

#### （2）检查状态
- CI/CD 流水线应全部通过。
- 分支间无合并冲突。

#### （3）选择合并策略
点击 `Merge pull request` 下拉菜单，可选：
- **Create a merge commit**：生成合并提交，保留完整历史（推荐）。
- **Squash and merge**：压缩为单个提交，简化历史。
- **Rebase and merge**：变基合并，保持线性历史（慎用于主分支）。

#### （4）确认合并
点击 `Confirm merge` 完成操作，可选删除源分支。

### 3. 命令行合并（适用于本地验证）

#### （1）拉取主仓库与 PR 分支
```bash
git clone <主仓库URL>
cd <仓库目录>
git remote add <贡献者名> <PR来源仓库URL>
git fetch <贡献者名> <PR分支名>
```

#### （2）切换并合并
```bash
git checkout main
git merge <PR分支名>
```

#### （3）解决冲突（如有）
手动编辑冲突文件后提交：
```bash
git add .
git commit -m "Merge PR #<编号>: <描述>"
```

#### （4）推送至主仓库
```bash
git push origin main
```

### 4. 合并后验证

- 查看 `Commits` 或 `Network Graph` 确认合并结果。
- 执行单元测试与集成测试，确保系统稳定性。

### 注意事项
1. **分支保护规则**：主分支可能要求 CI 通过或指定审核人。
2. **代码审查**：建议通过 `Review` 功能进行同行评审。
3. **跨仓库 PR**：来自 Fork 的 PR 需先拉取至本地审查。

### 其他平台差异
- **GitLab**：称为 Merge Request (MR)，操作类似。
- **Bitbucket**：流程基本一致，界面略有不同。



## 十一、本地仓库同步官方上游代码

当用户 Fork 并修改了官方仓库（如 OpenRLHF），仍需定期同步上游更新。

### 步骤 0：配置远程仓库

查看当前远程：
```bash
git remote -v
```

添加上游仓库：
```bash
git remote add online https://github.com/OpenRLHF/OpenRLHF.git
```

验证配置：
```bash
git remote -v
```
应显示 `origin`（个人 Fork）与 `online`（官方源）。



### 步骤 1：获取最新远程信息

```bash
git fetch origin
git fetch online
```

查看所有分支：
```bash
git branch -a
```
确保 `remotes/origin/main` 与 `remotes/online/main` 均存在。



### 步骤 2：创建本地合并分支（推荐）

若当前处于 `HEAD detached` 状态，创建临时分支：

```bash
git checkout -b temp-merge-branch origin/main
```



### 步骤 3：合并上游变更

执行合并操作：
```bash
git merge online/main
```

- 若存在冲突，需手动解决后执行：
  ```bash
  git add .
  git commit -m "Merge upstream/main"
  ```

- 若无冲突，Git 将自动生成合并提交。

检查提交历史：
```bash
git log --oneline --graph --all
```



### 步骤 4：推送至个人仓库

将合并结果推送到 `origin/main`：

```bash
git push origin temp-merge-branch:main
```

> **权限说明**：若无推送权限，需通过 PR 提交合并变更。



### 完整命令流程

```bash
# 获取最新代码
git fetch origin
git fetch online

# 创建本地分支并合并上游
git checkout -b temp-merge-branch origin/main
git merge online/main

# 推送合并结果
git push origin temp-merge-branch:main
```



## 十二、模型与数据集下载（使用国内镜像）

参考镜像站：[https://hf-mirror.com](https://hf-mirror.com)

### 1. 安装依赖

```bash
pip install -U huggingface_hub
```

### 2. 配置环境变量

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0
```

> **注意**：若无法访问镜像站，请检查网络连接或更换可用源。

### 3. 下载模型权重

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir /root/llmdir/hfhub/models/Qwen/Qwen2.5-0.5B
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/llmdir/hfhub/models/Qwen/Qwen2.5-0.5B-Instruct
```

### 4. 下载数据集

```bash
huggingface-cli download --repo-type dataset openai/gsm8k --local-dir /root/llmdir/hfhub/datasets/openai/gsm8k
huggingface-cli download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17k --local-dir /root/llmdir/hfhub/datasets/BytedTsinghua-SIA/DAPO-Math-17k
```

### 5. 自动化脚本示例（`download_script.sh`）

```bash
#!/bin/bash

# 设置国内镜像
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0

# 模型下载
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir /root/llmdir/hfhub/models/Qwen/Qwen2.5-0.5B
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/llmdir/hfhub/models/Qwen/Qwen2.5-0.5B-Instruct

# 数据集下载
huggingface-cli download --repo-type dataset openai/gsm8k --local-dir /root/llmdir/hfhub/datasets/openai/gsm8k
huggingface-cli download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17k --local-dir /root/llmdir/hfhub/datasets/BytedTsinghua-SIA/DAPO-Math-17k
```



## 十三、分布式通信测试（All-Reduce）

### 文件：`allreduce_demo.py`

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
import os


def main():
    # 获取分布式环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # 设置 NPU 设备
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{torch.npu.current_device()}")

    # 初始化 HCCL 通信后端
    dist.init_process_group(backend="hccl")

    # 构造测试张量
    tensor = torch.ones(2, 2, dtype=torch.float16, device=device) * (rank + 1)

    print(f'Rank {rank} 初始张量:\n{tensor}')
    print(f'数据类型: {tensor.dtype}, 设备: {tensor.device}')

    # 执行 All-Reduce（求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f'Rank {rank} All-Reduce 结果:\n{tensor}')

    # 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### 启动脚本：`run.sh`

```bash
#!/bin/bash

# 指定可见 NPU 设备
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 使用 torchrun 启动 8 卡分布式任务
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    allreduce_demo.py
```

执行：
```bash
bash run.sh
```

预期输出：每个 Rank 的输出张量值为 $ \sum_{i=0}^{7}(i+1) = 36 $，验证 HCCL 通信正常。



## 十四、常用集群运维命令

### 1. 后台运行任务

使用 `nohup` 在后台执行脚本并记录日志：

```bash
nohup sh download_hf_weights.sh > output2.log 2>&1 &
```

### 2. 查看后台进程

```bash
ps aux | grep hugg
```

### 3. 终止 Python 进程

```bash
ps aux | grep python | awk '{print $2}' | xargs kill -9
```

> **警告**：`kill -9` 强制终止，可能导致数据丢失，请谨慎使用。

### 4. 清理 Torch 扩展缓存

```bash
rm -rf /root/.cache/torch_extensions/py310_cpu
```



## 十五、Wandb 日志同步

### 1. 在线模式（自动同步）

使用 `wandb.init()` 且网络正常时，日志将自动上传至云端。

### 2. 离线模式日志上传

若以离线模式运行（`WANDB_MODE=offline`），日志保存在本地，需手动同步。

#### （1）日志存储路径
- 默认路径：`~/.wandb/`
- 项目路径：`./wandb/run-<RUN_ID>/`

#### （2）同步命令

同步当前目录下所有日志：
```bash
wandb sync wandb/
```

同步特定运行：
```bash
wandb sync wandb/run-<RUN_ID>/
```

同步所有未上传记录：
```bash
wandb sync --sync-all
```

#### （3）强制同步与清理

```bash
wandb sync --include-offline
wandb sync --clean    # 清理无效缓存
wandb sync
```

#### （4）指定项目上传

```bash
WANDB_PROJECT="your_project_name" wandb sync wandb/
```

## 十六、Wandb 同步 tensorbaord 日志

如果你有 本地 TensorBoard 日志（如 `.tfevents` 文件）并希望同步到 Weights & Biases (W&B)，可以按照以下步骤操作：

### 1：使用 `wandb sync` 命令

W&B 提供了 wandb sync 命令，可以直接同步本地 TensorBoard 文件：

```bash
wandb sync --sync-tensorboard <tensorboard_log_dir>
```

- `<tensorboard_log_dir>` 是你的 TensorBoard 日志目录，例如 `logs/`。这会自动解析 TensorBoard 的 `.tfevents` 文件，并将数据上传到 W&B。

### 2：使用 Python 代码直接同步

如果你在 Python 代码中运行 TensorBoard，可以手动集成 W&B：

#### 步骤 1：初始化 W&B

在你的训练脚本中：

```python
import wandb
wandb.init(project="your_project_name")
```

#### 步骤 2：让 W&B 监测 TensorBoard 日志

你可以让 W&B 监听 TensorBoard 目录：

```python
wandb.tensorboard.patch(root_logdir="logs/")
```

或者直接：

```python
wandb.init(sync_tensorboard=True)
```

#### 示例

```python
import wandb
import tensorflow as tf

# 初始化 wandb
wandb.init(project="my_project", sync_tensorboard=True)

# 定义 TensorBoard 目录
tensorboard_logdir = "logs/"

# 创建 TensorBoard 记录器
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)

# 训练模型
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])
```

这样，TensorBoard 生成的 `.tfevents` 日志会 自动同步 到 W&B。

###  常见问题排查

| 问题                     | 可能原因                   | 解决方案                            |
| ------------------------ | -------------------------- | ----------------------------------- |
| `wandb sync` 无响应      | 网络不通、日志损坏、已同步 | 检查网络，使用 `--clean` 清理缓存   |
| 无法访问 `hf-mirror.com` | 镜像站失效或网络限制       | 更换镜像源或配置代理                |
| NPU 设备未识别           | 驱动未安装或环境未加载     | 检查 `npu-smi` 输出，确认 CANN 配置 |

## 十七. Git 生成 SSH Key

### 1. 检查现有 SSH Key

首先，检查你的电脑上是否已经有 SSH Key。

打开终端（Terminal）或命令行工具，输入以下命令：

```bash
ls -al ~/.ssh
```

查看是否有以下文件：

- `id_rsa` 和 `id_rsa.pub`（RSA）
- `id_ed25519` 和 `id_ed25519.pub`（Ed25519）

如果文件已存在，你可以选择使用现有的 Key 或生成新的。

### 2. 生成新的 SSH Key

运行以下命令生成一个新的 SSH Key（建议使用更安全的 Ed25519 算法）：

```bash
ssh-keygen -t ed25519 -C "jianzhnie@126.com"
```

> 如果你使用的是较老版本的系统不支持 Ed25519，可以使用 RSA：
>
> ```bash
> ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
> ```

系统会提示你：

- **保存位置**：直接按回车使用默认路径（如 `~/.ssh/id_ed25519`）。
- **设置密码（passphrase）**：可选。建议设置以增强安全性。每次使用 Key 时会要求输入密码，也可以配合 `ssh-agent` 自动管理。

### 3. 启动 SSH Agent 并添加 Key

确保 `ssh-agent` 正在运行：

```bash
eval "$(ssh-agent -s)"
```

将你的 SSH Key 添加到 `ssh-agent`：

```bash
ssh-add ~/.ssh/id_ed25519
```

（如果使用的是 RSA，则为 `id_rsa`）

### 4. 将 SSH Key 添加到 GitHub

1. 复制公钥内容到剪贴板：

   **macOS:**

   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub
   ```

   **Linux (需要 xclip):**

   ```bash
   xclip -selection clipboard < ~/.ssh/id_ed25519.pub
   ```

   **Windows (Git Bash):**

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   然后手动复制输出内容。

2. 登录 GitHub，点击右上角头像 → **Settings** → **SSH and GPG keys** → **New SSH key**。

3. 输入标题（如 "My Laptop"），粘贴公钥内容，点击 **Add SSH key**。

### 5. 测试连接

在终端运行：

```bash
ssh -T git@github.com
```

如果看到类似以下提示：

```
Hi username! You've successfully authenticated...
```

说明配置成功！

---

✅ 现在你就可以使用 SSH 方式克隆、推送仓库了，例如：

```bash
git clone git@github.com:username/repo.git
```
