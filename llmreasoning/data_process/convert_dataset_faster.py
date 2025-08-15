import os
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# === System prompts used for formatting conversations ===
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'qwen_math_cot':
    'Please reason step by step, and put your final answer within \\boxed{}.',
    'deepseek_r1':
    ('A conversation between User and Assistant. The User asks a question, and the Assistant solves it.'
     'The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. '
     'The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, '
     'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
     ),
    'no':
    None,
}

# === Optional template for formatting input ===
INPUT_TEMPLATE: Dict[str, str] = {
    'prompt':
    ('{instruction}. \n\n Please reason step by step, and put your final answer within \\boxed{{}}.'
     )
}


def load_custom_dataset(data_path):
    """
    根据路径自动判断数据格式：
    - 如果是 .json 或 .jsonl，使用 json 加载器
    - 否则认为是 Hugging Face Hub 上的数据集名称，使用默认加载方式
    """
    _, ext = os.path.splitext(data_path)

    if ext in ['.json', '.jsonl']:
        print(f'🔍 检测到本地文件格式：{ext}，使用 json 加载方式')
        dataset = load_dataset('json', data_files=data_path)['train']
    else:
        print('🌐 检测到数据集名称，尝试从 Hugging Face Hub 加载')
        dataset = load_dataset(data_path, split='train')

    print(f'✅ 数据加载完成，共 {len(dataset)} 条样本')
    return dataset


def maybe_apply_chat_template(
        example: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase) -> Dict[str, str]:
    """Placeholder: Replace with actual chat template function if needed."""
    # Example: apply tokenizer's chat template (if available)
    return {
        'prompt': tokenizer.apply_chat_template(example['prompt'],
                                                tokenize=False)
    }


def preprocess_data(
    data: Dict[str, Any],
    input_key: str = 'input',
    label_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    cot_prompt: Optional[str] = None,
    input_template: Optional[Dict[str, str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, str]:
    input_prompt: str = data.get(input_key, '')
    prompt_text: str = ''

    if tokenizer:
        if system_prompt:
            prompt = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': input_prompt
                },
            ]
        elif cot_prompt:
            prompt = [{'role': 'user', 'content': input_prompt + cot_prompt}]
        else:
            prompt = [{'role': 'user', 'content': input_prompt}]
        prompt_text = maybe_apply_chat_template({'prompt': prompt},
                                                tokenizer)['prompt']
    elif input_template:
        prompt_text = input_template['prompt'].format(instruction=input_prompt)
    else:
        prompt_text = input_prompt

    label_text = data.get(label_key, '') if label_key else ''

    return {
        'problem': prompt_text,
        'answer': label_text,
    }


def process_and_save_dataset(
    data_path: str,
    model_name_or_path: str,
    output_path: str,
    input_key: str = 'Problem',
    label_key: str = 'Answer',
    system_prompt: Optional[str] = None,
    cot_prompt: Optional[str] = None,
    input_template: Optional[Dict[str, str]] = None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset: Dataset = load_custom_dataset(data_path)

    def hf_wrapper(example):
        return preprocess_data(
            example,
            input_key=input_key,
            label_key=label_key,
            system_prompt=system_prompt,
            cot_prompt=cot_prompt,
            input_template=input_template,
            tokenizer=tokenizer,
        )

    # Use multiprocessing to speed up processing
    processed_dataset = dataset.map(hf_wrapper,
                                    num_proc=16)  # Adjust num_proc as needed

    # Save as JSONL file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed_dataset.to_json(output_path, orient='records', lines=True)
    print(
        f'Saved {len(processed_dataset)} processed examples to {output_path}')


if __name__ == '__main__':
    # === Configuration ===
    data_path = '/root/llmtuner/hfhub/datasets/EleutherAI/hendrycks_math/test.jsonl'
    model_name_or_path = '/root/llmtuner/hfhub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    output_path = '/root/llmtuner/hfhub/datasets/EleutherAI/hendrycks_math/hendrycks_math_qwen_math_cot.jsonl'

    # === Run Processing ===
    process_and_save_dataset(
        data_path=data_path,
        model_name_or_path=model_name_or_path,
        output_path=output_path,
        input_key='problem',
        label_key='solution',
        system_prompt=SYSTEM_PROMPT_FACTORY['qwen_math_cot'],
        # cot_prompt=SYSTEM_PROMPT_FACTORY["qwen_math_cot"],
        input_template=None,  # or use INPUT_TEMPLATE if needed
    )
