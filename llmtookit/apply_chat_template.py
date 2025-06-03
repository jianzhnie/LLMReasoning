from typing import Dict, List, Optional

from transformers import AutoTokenizer

# 模型名称与路径映射
MODEL_PATHS: Dict[str, str] = {
    'Qwen2.5-7B': 'Qwen/Qwen2.5-7B',
    'QwQ-32B': 'Qwen/QwQ-32B',
    'Qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'AM-Thinking-v1': 'a-m-team/AM-Thinking-v1',
    'DeepSeek-R1-Distill-Qwen-32B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'Skywork-OR1-32B': 'Skywork/Skywork-OR1-32B',
    'DeepSeek-R1': 'deepseek-ai/DeepSeek-R1',
}

# 各种系统 prompt 类型
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'qwen_math_cot':
    'Please reason step by step, and put your final answer within \\boxed{}.',
    'deepseek_r1':
    ('A conversation between User and Assistant. The User asks a question, and the Assistant solves it. '
     'The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. '
     'The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer>.'
     ),
    'default':
    'You are a helpful assistant.',
    'none':
    None,
}

qwen_math_cot = (
    'Please reason step by step, and put your final answer within \\boxed{}.')


# 示例对话生成函数
def get_sample_chat(system_prompt: Optional[str]) -> List[dict]:
    chat = []
    if system_prompt is not None:
        chat.append({'role': 'system', 'content': system_prompt})
    chat += [{
        'role':
        'user',
        'content':
        ('Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.'
         'How many clips did Natalia sell altogether in April and May?' +
         '\n' + qwen_math_cot)
    }, {
        'role':
        'assistant',
        'content':
        ('Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n\\boxed{72}'
         )
    }]
    return chat


# 应用模板并打印结果
def try_chat_template(model_name: str, model_path: str, prompt_name: str,
                      system_prompt: Optional[str]):
    print(f"\n{'='*60}\nModel: {model_name}\nPrompt Type: {prompt_name}\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        chat = get_sample_chat(system_prompt)
        result = tokenizer.apply_chat_template(chat, tokenize=False)
        print(result)
    except Exception as e:
        print(
            f"[ERROR] Failed for {model_name} with prompt '{prompt_name}':\n{e}"
        )


if __name__ == '__main__':
    for model_name, model_path in MODEL_PATHS.items():
        for prompt_name, system_prompt in SYSTEM_PROMPT_FACTORY.items():
            try_chat_template(model_name, model_path, prompt_name,
                              system_prompt)
