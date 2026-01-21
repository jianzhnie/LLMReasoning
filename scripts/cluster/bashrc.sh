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
__conda_setup="$('/home/jianzhnie/llmtuner/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jianzhnie/llmtuner/software/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jianzhnie/llmtuner/software/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jianzhnie/llmtuner/software/miniconda3/bin:$PATH"
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
