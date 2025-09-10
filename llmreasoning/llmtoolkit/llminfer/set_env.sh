#!/bin/bash

# hf-mirror
export HF_ENDPOINT=https://hf-mirror.com

# cann 相关环境
install_path=/home/jianzhnie/llmtuner/Ascend
# install_path=/usr/local/Ascend
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$install_path/driver/lib64/driver/:$LD_LIBRARY_PATH


# ASCEND 相关
export ASCEND_LAUNCH_BLOCKING=0

# 以支持torch2.5以上版本
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_DISABLE=1

# HCCL相关
export HCCL_BUFFSIZE=512
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export HCCL_CONNECT_TIMEOUT=3600

# 通信
export GLOO_SOCKET_IFNAME=enp66s0f5
export TP_SOCKET_IFNAME=enp66s0f5
export HCCL_SOCKET_IFNAME=enp66s0f5

# 线程相关
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

# DEVICES
export RAY_memory_usage_threshold=0.98
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# VLLM
export VLLM_USE_V1=0
export TASK_QUEUE_ENABLE=1


# ray 相关
# 激活conda环境
# source /home/jianzhnie/llmtuner/miniconda3/bin/activate vllm
source /root/llm_workspace/miniconda3/bin/activate mindspeed_rl_0620
