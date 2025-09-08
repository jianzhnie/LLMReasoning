## 第一部分：功能介绍

###  优化 .bashrc 脚本：专为 AI/ML 开发者打造

这份 `.bashrc` 脚本是一个高度优化的 Bash 配置文件，专为在集群环境上进行人工智能（AI）算法开发和模型训练的用户设计。它旨在简化日常操作、提升工作效率，并通过统一的配置确保开发环境的一致性。

**核心功能**

- **路径管理**：脚本通过一个自定义函数，使你能够轻松地将常用工具或脚本目录添加到 `$PATH` 环境变量中，避免手动配置的繁琐。
- **Python 环境集成**：它无缝集成了 **Conda** 环境管理。脚本不仅提供了方便的别名，如 `ca`（`conda activate`）和 `cenv`（`conda env list`），还能在命令行提示符中显示当前激活的 Conda 环境，让你随时掌握状态。
- **别名与快捷方式**：为了减少重复输入，脚本定义了一系列常用命令的别名：
  - **通用命令**：`ll`、`la` 等命令提供更友好的 `ls` 输出。`rm -i`、`cp -i` 等别名则在执行高风险操作前提供安全确认。
  - **AI/ML 专用**：`gpu` 别名可以让你通过 `watch` 命令实时监控 GPU 的使用情况。
- **统一的命令行提示符（PS1）**：这是脚本的一大亮点。它创建了一个智能的命令行提示符，可以同时显示当前 **Conda 环境**、**用户名**、**主机名**、**当前路径**，甚至还能集成 **Git 分支**信息，让你一目了然地掌握所有关键信息。
- **模块管理与环境加载**：脚本包含了加载集群系统（如 Lmod）环境模块的逻辑，确保你的环境变量配置与集群的预设相符。


### .bashrc  脚本内容

你可以直接将以下内容保存为 `bashrc.sh` 文件，并根据你的集群环境进行必要的路径调整。 文件中的注释部分详细解释了每个配置项的作用，方便你根据需要进行定制。 部分内容需要根据你的集群环境进行调整，例如 Conda 的安装路径和 ascend toolkit 的路径。

```bash
# ====================================================================
# .bashrc 文件，专为在集群上进行人工智能（AI）开发和模型训练优化
# ====================================================================

# --------------------------------------------------------------------
# 1. 基础设置与路径管理
# --------------------------------------------------------------------

# 确保在非交互式shell中不加载此文件
[ -z "$PS1" ] && return

# 忽略重复或以空格开头的命令历史
export HISTCONTROL=ignoreboth

# 调整命令历史记录大小
export HISTSIZE=10000
export HISTFILESIZE=20000

# 函数：在PATH中添加新路径，并检查是否已存在
add_to_path() {
    if [[ ":$PATH:" != *":$1:"* ]]; then
        export PATH="$1${PATH:+:$PATH}"
    fi
}

# 示例：将自定义的脚本目录添加到PATH
add_to_path "$HOME/bin"

# --------------------------------------------------------------------
# 2. Python 环境管理
# --------------------------------------------------------------------

# Conda/Miniconda 环境变量配置
# 这部分通常在 Conda 安装时自动添加。
# 如果你发现问题，请重新运行 'conda init bash'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/path/to/your/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/path/to/your/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/path/to/your/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/path/to/your/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Conda 别名
alias ca='conda activate'
alias cenv='conda env list'

# --------------------------------------------------------------------
# 3. 常用别名 (Aliases)
# --------------------------------------------------------------------

# ls 命令别名
alias ls='ls --color=auto'
alias ll='ls -lh'
alias la='ls -lha'
alias lrt='ls -lht'     # 按修改时间倒序排列

# 通用别名
alias rm='rm -i'        # 删除前确认
alias cp='cp -i'        # 复制前确认
alias mv='mv -i'        # 移动前确认
alias grep='grep --color=auto'

# AI/ML 相关别名
alias llm='cd $HOME/llmtuner/llm'
alias hub='cd $HOME/llmtuner/hfhub'

# 监控 GPU 使用情况 (通常使用 nvidia-smi)
alias npu='watch -n 1 nvidia-smi'

# --------------------------------------------------------------------
# 4. 模块管理 (Modules)
# --------------------------------------------------------------------
if [ -f /etc/profile.d/modules.sh ]; then
   . /etc/profile.d/modules.sh
fi

# --------------------------------------------------------------------
# 5. 自定义命令行提示符 (PS1)
# --------------------------------------------------------------------

# 基础提示符：用户名@主机名:当前路径
PS1_BASE='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]'

# 检查 Conda 环境并添加到提示符
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    PS1_CONDA="\[\033[01;33m\]($CONDA_DEFAULT_ENV)\[\033[00m\] "
else
    PS1_CONDA=""
fi

# 检查 Git 分支并添加到提示符
if [ -f /usr/lib/git-core/git-prompt.sh ]; then
   . /usr/lib/git-core/git-prompt.sh
   GIT_PS1_SHOWDIRTYSTATE=true
   PS1_GIT='$(__git_ps1 " (%s)")'
else
   PS1_GIT=""
fi

# 组合最终的 PS1 提示符
export PS1="$PS1_CONDA$PS1_BASE$PS1_GIT\$ "

# --------------------------------------------------------------------
# 6. 其他常用工具路径
# --------------------------------------------------------------------

# CUDA Toolkit 和 cuDNN 的路径设置
# 请根据你的集群系统安装路径进行修改
# export PATH="/usr/local/cuda/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"


# Ascend Toolkit 环境设置
# 请确保下面的路径是正确的
export install_path=/usr/local/Ascend
if [ -d "$install_path" ]; then
    echo "Loading Ascend Toolkit Env..."
    source $install_path/ascend-toolkit/set_env.sh
    source $install_path/nnal/atb/set_env.sh
fi
```


## 第二部分：多节点同步

`sync_bashrc_from_file.sh` 是一个自动化脚本，它解决了在多台集群节点上维护 `.bashrc` 文件一致性的痛点。你只需维护一个本地的 `.bashrc` 文件和一个节点列表文件，脚本就能自动将配置同步到所有指定节点。

**使用前准备**

1. 创建节点列表文件：在本地创建一个文本文件（例如 nodes.txt），将所有目标节点的地址（格式为 用户名@节点名）逐行写入。

   nodes.txt 示例：

   ```bash
   user@compute-node01
   user@compute-node02
   # 你可以添加注释
   # user@compute-node03
   ```

2. **配置 SSH 无密码登录**：强烈建议在本地机器和所有集群节点之间配置 SSH 公钥认证。这将允许脚本在无需手动输入密码的情况下，自动完成文件传输，使同步过程完全自动化。

**使用教程**

1. **将脚本保存到本地**：将 `sync_bashrc_from_file.sh` 脚本文件保存到你的本地计算机上。

2. **赋予执行权限**：打开终端，使用 `chmod` 命令为脚本添加执行权限：

   ```bash
   chmod +x sync_bashrc_from_file.sh
   ```

3. **执行同步**：运行脚本时，将节点列表文件的路径作为唯一参数传入。

   ```bash
   ./sync_bashrc_from_file.sh /path/to/your/nodes.txt
   ```

**工作流程**

- 脚本首先会检查你传入的节点列表文件和本地 `.bashrc` 文件是否存在。
- 然后，它会逐行读取节点列表文件。
- 对于列表中的每一个节点，脚本会使用 `scp`（Secure Copy）命令，安全地将你的本地 `.bashrc` 文件复制到远程节点的家目录中。
- 如果同步成功，脚本会输出一条成功的消息。如果失败，它也会给出清晰的错误提示。

**重要提示**

- 同步脚本只负责将文件复制到远程节点。要使新的 `.bashrc` 配置生效，你**必须登录到每个节点**，并手动运行 `source ~/.bashrc` 命令。
- 请务必确保你的本地 `.bashrc` 文件是最新的，因为每次运行脚本都会用本地版本覆盖远程节点上的文件。



### `sync_bashrc_from_file.sh`  内容

通过此脚本，你可以将本地的 `.bashrc` 文件同步到多个远程节点。 

```bash
#!/bin/bash

# ====================================================================
# 脚本名称: sync_bashrc_from_file.sh
# 功能: 从文件中读取节点列表，自动同步本地的 .bashrc 文件到这些节点
# 使用方法: ./sync_bashrc_from_file.sh /path/to/your/node_list.txt
# 
# 前提: 
# 1. 确保你已配置了 SSH 无密码登录，或者你将能够手动输入密码。
# 2. 节点列表文件中的每一行应包含一个节点地址，例如: user@node1
# ====================================================================

# --------------------------------------------------------------------
# 1. 变量和参数检查
# --------------------------------------------------------------------
# 检查命令行参数，确保节点文件路径已提供
if [ -z "$1" ]; then
    echo "使用方法: $0 <节点文件路径>"
    echo "示例: $0 nodes.txt"
    exit 1
fi

NODE_LIST_FILE="$1"

# 检查节点列表文件是否存在
if [ ! -f "$NODE_LIST_FILE" ]; then
    echo "错误: 节点列表文件 '$NODE_LIST_FILE' 未找到。"
    exit 1
fi


# 从文件读取节点列表到数组 (使用 < "$VAR" 语法，并忽略空行和注释行)
mapfile -t NODE_HOSTS < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

# 检查节点列表是否为空
if [ ${#NODE_HOSTS[@]} -eq 0 ]; then
    echo "❌ 错误: 节点列表 '$NODE_LIST_FILE' 为空。"
    exit 1
fi

# 要同步的本地文件
LOCAL_FILE="$HOME/bashrc.sh"
# 远程文件路径
REMOTE_FILE="$HOME/.bashrc"

# 检查本地 .bashrc 文件是否存在
if [ ! -f "$LOCAL_FILE" ]; then
    echo "错误: 本地文件 '$LOCAL_FILE' 未找到。"
    exit 1
fi

# --------------------------------------------------------------------
# 2. 同步过程
# --------------------------------------------------------------------

echo "开始从文件 '$NODE_LIST_FILE' 同步 '$LOCAL_FILE' 到以下节点..."
echo "-------------------------------------"

# 逐行读取文件并同步
for node in "${NODE_HOSTS[@]}"; do
    
    echo "正在同步到节点: $node ..."
    
    # 使用 scp 命令复制文件
    # -p 参数用于保留文件的修改时间和权限
    # -q 参数用于静默模式，减少输出
    scp -p "$LOCAL_FILE" "$node:$REMOTE_FILE"
    
    # 检查 scp 命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "✅ 同步成功!"
    else
        echo "❌ 同步失败。请检查网络连接、节点名或 SSH 权限。"
    fi
    echo "-------------------------------------"
done

echo "所有节点同步任务完成。"
echo "提示: 在每个节点上，你可能需要运行 'source ~/.bashrc' 来使更改生效。"
```