import json
import os
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from trl.data_utils import maybe_apply_chat_template


def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)


def load_json_data(path: str) -> Union[List[Dict], Dict]:
    """
    Reads a dataset file in JSON or JSONL format.

    Args:
        path (str): Path to the data file.

    Returns:
        Union[List[Dict], Dict]: Parsed data. A list of dictionaries for JSONL,
        or a dictionary (or list) for JSON.

    Raises:
        NotImplementedError: If file extension is not .json or .jsonl.
        ValueError: If the file content cannot be parsed.
    """
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f'Failed to parse JSON file: {e}')
    elif path.endswith('.jsonl'):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f'Failed to parse line in JSONL file: {e}')
        return data
    else:
        raise NotImplementedError(
            'Only .json and .jsonl formats are supported.')


def lower_keys(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lowercase all dictionary keys.

    Args:
        example (Dict[str, Any]): Original dictionary.

    Returns:
        Dict[str, Any]: Dictionary with all keys in lowercase.
    """
    return {key.lower(): value for key, value in example.items()}


def load_data(data_name: str, split: str, data_dir: str = './data') -> Dataset:
    """
    Load and return a HuggingFace Dataset. Supports local JSONL and HF datasets.

    Args:
        data_name (str): Name of the dataset (e.g., 'math', 'gsm8k', 'gsm-hard').
        split (str): Dataset split to load ('train', 'test', or 'validation').
        data_dir (str, optional): Local directory to load/save datasets. Defaults to './data'.
        use_cache (bool, optional): If True, load from cached JSONL if available. Defaults to True.

    Returns:
        Dataset: A HuggingFace-compatible Dataset object.
    """
    os.makedirs(os.path.join(data_dir, data_name), exist_ok=True)
    data_file = os.path.join(data_dir, data_name, f'{split}.jsonl')

    if os.path.exists(data_file):
        # Load from local JSONL and convert to HuggingFace dataset
        examples = list(load_json_data(data_file))
        if 'idx' not in examples[0]:
            for i, example in enumerate(examples):
                example['idx'] = i
        return Dataset.from_list(sorted(examples, key=lambda x: x['idx']))

    # Load from HuggingFace datasets
    if data_name == 'math':
        dataset = load_dataset(
            'competition_math',
            name='main',
            split=split,
            cache_dir=os.path.join(data_dir, 'temp'),
        )
    elif data_name == 'gsm8k':
        dataset = load_dataset(data_name, split=split)
    elif data_name == 'gsm-hard':
        dataset = load_dataset('reasoning-machines/gsm-hard', split='train')
    else:
        raise NotImplementedError(f"Dataset '{data_name}' is not supported.")

    # Normalize key names
    examples = [lower_keys(ex) for ex in dataset]

    # Ensure each sample has an 'idx'
    for i, ex in enumerate(examples):
        ex.setdefault('idx', i)

    # Save processed data to disk
    Dataset.from_list(examples).to_json(data_file)

    return Dataset.from_list(sorted(examples, key=lambda x: x['idx']))


def preprocess_data(
    data: Dict[str, Any],
    input_key: str = 'input',
    label_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    input_template: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    apply_chat_template: bool = False,
) -> Dict[str, str]:
    """
    Preprocesses a single data point for training, optionally applying a chat template.

    Args:
        data (Dict[str, Any]): A dictionary containing input and optionally label.
        input_key (str): Key to retrieve input text from `data`.
        label_key (Optional[str]): Key to retrieve label text from `data`, if any.
        system_prompt (Optional[str]): Optional system prompt for chat models.
        input_template (Optional[str]): Template to format the input text.
        tokenizer (Optional[Any]): Tokenizer used for formatting with chat templates.
        apply_chat_template (bool): Whether to apply a chat format (e.g., OpenAI style).

    Returns:
        Dict[str, str]: A dictionary with preprocessed `prompt` and `label`.
    """
    if apply_chat_template:
        prompt = []
        if system_prompt is not None:
            prompt.append({'role': 'system', 'content': system_prompt})
        prompt.append({'role': 'user', 'content': data[input_key]})
        example = {'prompt': prompt}

        # Assuming maybe_apply_chat_template is a helper that returns a dict with "prompt" text
        prompt_text = maybe_apply_chat_template(example, tokenizer)['prompt']
    else:
        prompt_text = data[input_key]
        if input_template:
            # Format the input text using the provided template
            prompt_text = input_template.format(input=prompt_text)

    # Use empty string if label_key is not provided
    label_text = '' if label_key is None else str(data.get(label_key, ''))

    return {'prompt': prompt_text, 'label': label_text}


class PromptDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for training reinforcement learning models (e.g., PPO).

    Args:
        dataset (List[Dict[str, Any]]): Raw dataset, where each item is a dictionary.
        tokenizer (Any): Tokenizer to apply if using chat templates.
        input_key (str): Key used to retrieve the input text from each example.
        label_key (str): Key used to retrieve the label text from each example.
        systerm_template (Optional[str]): Optional system prompt for chat templates.
        input_template (Optional[str]): Template string for formatting inputs.
        apply_chat_template (bool): Whether to use chat-style formatting for the inputs.
    """

    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        tokenizer: Any,
        input_key: str,
        label_key: str,
        systerm_template: Optional[str] = None,
        input_template: Optional[str] = None,
        apply_chat_template: bool = True,
    ) -> None:
        super().__init__()

        self.processed_inputs: List[Dict[str, str]] = []
        for data in tqdm(dataset, desc='Preprocessing data'):
            processed_input = preprocess_data(
                data=data,
                input_key=input_key,
                label_key=label_key,
                system_prompt=systerm_template,
                input_template=input_template,
                tokenizer=tokenizer,
                apply_chat_template=apply_chat_template,
            )
            self.processed_inputs.append(processed_input)

    def __len__(self) -> int:
        """Returns the number of processed examples."""
        return len(self.processed_inputs)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Retrieves the processed example at the given index."""
        return self.processed_inputs[idx]
