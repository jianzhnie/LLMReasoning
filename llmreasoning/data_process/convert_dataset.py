import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl.data_utils import maybe_apply_chat_template

# === System prompts used for formatting conversations ===
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'qwen_math_cot':
    'Please reason step by step, and put your final answer within \\boxed{}.',
    'deepseek_r1':
    ('A conversation between User and Assistant. The User asks a question, and the Assistant solves it. '
     'The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. '
     'The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer>.'
     ),
    'no':
    None,
}

# === Optional input formatting templates ===
INPUT_TEMPLATE: Dict[str, str] = {
    'prompt':
    ('{instruction}. \n\n Please reason step by step, and put your final answer within \\boxed{{}}.'
     )
}


def load_custom_dataset(data_path: str) -> Dataset:
    """
    Load dataset from either a local JSON/JSONL file or from the Hugging Face Hub.

    Args:
        data_path: Path to a JSON/JSONL file or Hugging Face dataset name.

    Returns:
        Loaded Hugging Face Dataset object.
    """
    _, ext = os.path.splitext(data_path)

    try:
        if ext in ['.json', '.jsonl']:
            print(f'🔍 Detected local file format: {ext}, using JSON loader.')
            dataset = load_dataset('json', data_files=data_path)['train']
        else:
            print('🌐 Detected dataset name, loading from Hugging Face Hub.')
            dataset = load_dataset(data_path, split='train')
    except Exception as e:
        raise RuntimeError(f'Failed to load dataset from {data_path}: {e}')

    print(f'✅ Loaded dataset with {len(dataset)} samples.')
    return dataset


def preprocess_data(
    data: Dict[str, Any],
    input_key: str = 'input',
    label_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    cot_prompt: Optional[str] = None,
    input_template: Optional[Dict[str, str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Dict[str, str]:
    """
    Preprocess a single data entry into a prompt-answer pair.

    Args:
        data: Dictionary representing one example from the dataset.
        input_key: Key used to retrieve the input prompt text.
        label_key: Key used to retrieve the label or answer.
        system_prompt: Optional system prompt to format the conversation.
        cot_prompt: Optional chain-of-thought prompt for reasoning.
        input_template: Optional formatting template for the input prompt.
        tokenizer: Tokenizer used with maybe_apply_chat_template.

    Returns:
        A dictionary with keys 'problem' and 'answer'.
    """
    input_prompt: str = data.get(input_key, '')

    if system_prompt and tokenizer:
        prompt = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': input_prompt
        }]
        prompt_text = maybe_apply_chat_template({'prompt': prompt},
                                                tokenizer)['prompt']
    elif cot_prompt and tokenizer:
        prompt = [{
            'role': 'user',
            'content': input_prompt + '\n' + cot_prompt
        }]
        prompt_text = maybe_apply_chat_template({'prompt': prompt},
                                                tokenizer)['prompt']
    elif input_template:
        prompt_text = input_template['prompt'].format(instruction=input_prompt)
    else:
        prompt_text = input_prompt

    label_text: str = data.get(label_key, '') if label_key else ''

    return {'problem': prompt_text, 'answer': label_text}


def process_and_save_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    input_key: str = 'Problem',
    label_key: str = 'Answer',
    system_prompt: Optional[str] = None,
    cot_prompt: Optional[str] = None,
    input_template: Optional[Dict[str, str]] = None,
) -> None:
    """
    Process a dataset and save the results to a JSONL file.

    Args:
        dataset: Dataset to process.
        tokenizer: Tokenizer used for formatting.
        output_path: Path to output .jsonl file.
        input_key: Key to access input text in the dataset.
        label_key: Key to access labels/answers in the dataset.
        system_prompt: Optional system prompt.
        cot_prompt: Optional chain-of-thought prompt.
        input_template: Optional template for input formatting.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w', encoding='utf-8') as f:
        for i, example in enumerate(dataset, start=0):
            processed = preprocess_data(
                example,
                input_key=input_key,
                label_key=label_key,
                system_prompt=system_prompt,
                cot_prompt=cot_prompt,
                input_template=input_template,
                tokenizer=tokenizer,
            )
            # Write each example as a single line JSON
            json_line = json.dumps(processed, ensure_ascii=False)
            f.write(json_line + '\n')
            if i % 1000 == 0:
                print(f'✅ Processed {i} examples...')

    print(f'🎉 Finished processing {len(dataset)} examples.')
    print(f'💾 Saved to {output_file}')


SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'qwen_math_cot':
    'Please reason step by step, and put your final answer within \\boxed{}.',
    'deepseek_r1':
    ('A conversation between User and Assistant. The User asks a question, and the Assistant solves it. '
     'The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. '
     'The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer>.'
     ),
    'no':
    None,
}


def chat_template_example(tokenizer: PreTrainedTokenizerBase) -> None:
    """
    Example usage of `apply_chat_template` for debugging purposes.
    """
    system_prompt = SYSTEM_PROMPT_FACTORY['']
    chat = [{
        'role': 'system',
        'content': system_prompt
    }, {
        'role':
        'user',
        'content':
        ('Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.'
         'How many clips did Natalia sell altogether in April and May?')
    }, {
        'role':
        'assistant',
        'content':
        'Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n\\boxed{72}'
    }]
    results = tokenizer.apply_chat_template(chat, tokenize=False)
    return results


if __name__ == '__main__':
    # === Config ===
    # data_path = "/root/llmtuner/hfhub/datasets/deepmath_8k.json"
    # output_path = "/root/llmtuner/hfhub/datasets/deepmath/skywork_qwen_math_cot.jsonl"

    data_path = '/root/llmtuner/hfhub/datasets/Skywork/clean_data/clean0516_skywork_1-8.json'
    # model_name_or_path = "/root/llmtuner/hfhub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # model_name_or_path = "/root/llmtuner/hfhub/models/Skywork/Skywork-OR1-32B"
    # model_name_or_path = "/root/llmtuner/hfhub/models/deepseek-ai/DeepSeek-R1"
    model_name_or_path = '/root/llm_workspace/models/a-m-team/AM-Thinking-v1'
    output_path = '/root/llmtuner/hfhub/datasets/Skywork/clean_data/clean0516_skywork_1-8_qwen_math_cot.jsonl'
    input_key = 'prompt'
    label_key = 'answer'

    # === Load resources ===
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # dataset = load_custom_dataset(data_path)
    # # === Process and save ===
    # process_and_save_dataset(
    #     dataset,
    #     tokenizer,
    #     output_path=output_path,
    #     input_key=input_key,
    #     label_key=label_key,
    #     cot_prompt=SYSTEM_PROMPT_FACTORY["qwen_math_cot"],
    #     input_template=None,  # Or use INPUT_TEMPLATE
    # )
    # === Optional debug example ===
    results = chat_template_example(tokenizer)
    print(results)
