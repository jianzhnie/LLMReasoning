import os
import json
from typing import Union, Iterable, Any, List, Dict
from pathlib import Path
from datasets import load_dataset, Dataset


def load_jsonl(file: Union[str, Path]) -> Iterable[Dict[str, Any]]:
    """
    Load a JSONL file line-by-line into dictionaries.

    Args:
        file (Union[str, Path]): Path to the .jsonl file.

    Yields:
        Dict[str, Any]: Parsed JSON object from each line.
    """
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print("Error in loading:", line)
                raise


def lower_keys(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lowercase all dictionary keys.

    Args:
        example (Dict[str, Any]): Original dictionary.

    Returns:
        Dict[str, Any]: Dictionary with all keys in lowercase.
    """
    return {key.lower(): value for key, value in example.items()}


def load_data(
    data_name: str, split: str, data_dir: str = "./data"
) -> List[Dict[str, Any]]:
    """
    Load dataset from local file or download and process from HuggingFace.

    Args:
        data_name (str): Name of the dataset.
        split (str): Data split, e.g., 'train', 'test', or 'validation'.
        data_dir (str, optional): Directory to load/save datasets. Defaults to './data'.

    Returns:
        List[Dict[str, Any]]: Loaded and processed examples.
    """
    data_file = os.path.join(data_dir, data_name, f"{split}.jsonl")
    examples: List[Dict[str, Any]] = []

    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        # Load datasets from HuggingFace or local JSONs
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "gsm-hard":
            dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        else:
            raise NotImplementedError(f"Dataset '{data_name}' is not supported.")

        # Normalize keys and save locally
        examples = [lower_keys(example) for example in list(dataset)]
        os.makedirs(os.path.join(data_dir, data_name), exist_ok=True)
        Dataset.from_list(examples).to_json(data_file)

    # Add 'idx' if not present
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # Sort by index
    examples = sorted(examples, key=lambda x: x["idx"])

    return examples
