model_path="/home/jianzhnie/llmtuner/hfhub/models/Qwen/QwQ-32B"
model_name="Qwen/QwQ-32B"

# model_path="/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen2.5-7B"
# model_name="Qwen/Qwen2.5-7B"

python -m vllm.entrypoints.openai.api_server \
       --model $model_path \
       --trust-remote-code \
       --served-model-name $model_name \
       --enforce-eager \
       --distributed_executor_backend "ray" \
       --tensor-parallel-size 4 \
       --pipeline-parallel-size 4 \
       --disable-frontend-multiprocessing \
       --port 8090
