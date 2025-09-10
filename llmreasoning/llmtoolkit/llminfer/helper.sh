## 节点1
ssh 10.16.201.108

cd /home/jianzhnie/llmtuner/llm/LLMReasoning/llminfer

source set_env.sh
source ray_cluster_env_1.sh

echo $HCCL_WHITELIST_DEVICE
echo $HCCL_IF_IP
echo $HCCL_SOCKET_IFNAME

## 节点2

ssh 10.16.201.198
cd /home/jianzhnie/llmtuner/llm/LLMReasoning/llminfer
source set_env.sh
source ray_cluster_env_2.sh


echo $HCCL_WHITELIST_DEVICE
echo $HCCL_IF_IP
echo $HCCL_SOCKET_IFNAME

## start ray cluster

bash run_cluster.sh
nohup bash vllm_model_server.sh > vllm_model_server.log 2>&1 &

#  Test
ssh 10.16.201.108

cd /home/jianzhnie/llmtuner/llm/QwQ/eval
source /home/jianzhnie/llmtuner/llm/LLMReasoning/llminfer/set_env.sh
nohup bash  scripts/math_eval_qwq.sh > model_eval_output.log 2>&1 &
