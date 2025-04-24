import json
import os
import random
from typing import Any, Dict, List, Union

import numpy as np
from datasets import Dataset, load_dataset


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


def load_data(data_name: str,
              split: str,
              data_dir: str = './data',
              use_cache: bool = True) -> Dataset:
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

    if use_cache and os.path.exists(data_file):
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
