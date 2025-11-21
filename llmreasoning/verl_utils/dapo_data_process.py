"""
Data Processing Script for Custom Datasets.

This script preprocesses custom datasets into parquet format for training LLMs.
It supports various input formats and applies transformations to standardize
the data structure for downstream tasks.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Final, Union

import datasets

# System prompts used for different model types
OPENR1_SYSTEM_PROMPT: Final[str] = (
    'You are a helpful AI Assistant that provides well-reasoned and detailed responses. '
    'You first think about the reasoning process as an internal monologue and then '
    'provide the user with the answer.'
    'The reasoning process and answer are enclosed within <think>\\n...\\n</think> and '
    '<answer>\\n...\\n</answer> tags, respectively, i.e., '
    '<think> reasoning process here </think> <answer> answer here </answer>.')

QWEN_MATH_COT_PROMPT: Final[str] = (
    'Please reason step by step, and put your final answer within \\boxed{}.')


def extract_question(content: str) -> str:
    """
    Extract clean question from content using multiple strategies for robustness.

    This function tries multiple approaches in order:
    1. First attempts precise prefix/suffix removal for structured data
    2. Falls back to robust splitting approach for less structured data
    3. Returns original content if both methods fail

    Args:
        content: Raw content string containing the question

    Returns:
        Cleaned question content
    """
    if not isinstance(content, str):
        return str(content) if content is not None else ''

    # Try precise extraction first (for structured data with known prefixes/suffixes)
    prefix = (
        'Solve the following math problem step by step. '
        'The last line of your response should be of the form Answer: $Answer (without quotes) '
        'where $Answer is the answer to the problem.\n\n')
    suffix = '\n\nRemember to put your answer on its own line after "Answer:".'

    # Check if content has the expected structure
    if content.startswith(prefix) and content.endswith(suffix):
        try:
            # Remove precise prefixes and suffixes
            clean_content = content[len(prefix):-len(suffix)].strip()
            if clean_content:  # Only return if we got non-empty result
                return clean_content
        except (IndexError, ValueError) as e:
            # Continue to fallback method if precise extraction fails
            pass

    # Fallback to robust splitting approach
    try:
        parts = content.split('\n\n')
        # For well-structured content with [Prompt]\n\n[Question]\n\n[Suffix] format
        if len(parts) >= 3:
            # Reassemble middle parts to prevent Question being split by internal \n\n
            clean_content = '\n\n'.join(parts[1:-1]).strip()
            if clean_content:  # Only return if we got non-empty result
                return clean_content
    except (AttributeError, TypeError) as e:
        # Handle cases where content is not a string
        pass

    # Ultimate fallback - return original content stripped
    return content.strip()


def dapo_process_fn(example: Dict[str, Any],
                    data_source: str) -> Dict[str, Any]:
    """
    Process DAPO dataset examples.

    Args:
        example: A single dataset example
        idx: Index of the example

    Returns:
        Processed example with standardized structure
    """
    # Extract the content from the prompt
    content = example['prompt'][0]['content']
    question = extract_question(content)

    # Add the instruction following text
    question = question + '\n' + QWEN_MATH_COT_PROMPT

    return {
        'data_source':
        data_source,
        'prompt': [{
            'role': 'system',
            'content': OPENR1_SYSTEM_PROMPT
        }, {
            'role': 'user',
            'content': question
        }],
        'ability':
        example['ability'],
        'reward_model':
        example['reward_model'],
        'extra_info':
        example['extra_info']
    }


def process_fn(example: Dict[str, Any], idx: int, input_key: str,
               label_key: str, data_source: str) -> Dict[str, Any]:
    """
    Process general dataset examples.

    Args:
        example: A single dataset example
        idx: Index of the example
        input_key: Key for input text in the example
        label_key: Key for label/answer in the example
        data_source: Name of the data source

    Returns:
        Processed example with standardized structure
    """
    # Use get() method to avoid KeyError
    question = example.get(input_key, '')
    answer = example.get(label_key, '')

    question = question + '\n' + QWEN_MATH_COT_PROMPT

    # Return only the new structure, removing original keys
    return {
        'data_source': data_source,
        'prompt': [{
            'role': 'user',
            'content': question
        }],
        'ability': 'math',
        'reward_model': {
            'style': 'rule',
            'ground_truth': answer
        },
        'extra_info': {
            'split': 'null',
            'index': idx
        },
    }


def load_dataset_from_path(dataset_path: str) -> datasets.Dataset:
    """
    Load dataset from a given path based on file extension.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Loaded dataset

    Raises:
        RuntimeError: If dataset loading fails
    """
    try:
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            return datasets.load_dataset('json',
                                         data_files=dataset_path,
                                         split='train')
        elif dataset_path.endswith('.parquet'):
            return datasets.load_dataset('parquet',
                                         data_files=dataset_path,
                                         split='train')
        else:
            return datasets.load_dataset(dataset_path, split='train')
    except Exception as e:
        raise RuntimeError(f'Failed to load dataset: {e}')


def save_example(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save a single example as JSON for reference.

    Args:
        data: Data to save
        filepath: Path where to save the file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main() -> None:
    """Main function to preprocess custom datasets to parquet format."""
    parser = argparse.ArgumentParser(
        description='Preprocess custom datasets to parquet format')
    parser.add_argument('--local_dataset_path',
                        required=True,
                        help='The local path to the raw dataset (required)')
    parser.add_argument(
        '--local_save_dir',
        default='~/data/math',
        help='The save directory for the preprocessed dataset.')
    parser.add_argument('--dataset_name',
                        default='deepscaler',
                        help='Name of the dataset')
    parser.add_argument('--input_key',
                        default='question',
                        help='Key for input text')
    parser.add_argument('--label_key',
                        default='answer',
                        help='Key for label/answer')
    parser.add_argument('--test_split_ratio',
                        type=float,
                        default=0.1,
                        help='Ratio of test split (default: 0.1)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.local_dataset_path):
        raise FileNotFoundError(
            f'Dataset path does not exist: {args.local_dataset_path}')

    data_source = 'custom_' + args.dataset_name

    print(
        f'Loading the {data_source} dataset from {args.local_dataset_path}...',
        flush=True)

    # Load dataset
    raw_dataset = load_dataset_from_path(args.local_dataset_path)
    print(f'Dataset loaded with {len(raw_dataset)} samples', flush=True)

    if len(raw_dataset) > 0:
        print(raw_dataset[0])

    print('Processing dataset...', flush=True)

    # Process the dataset and remove original columns
    if args.dataset_name == 'dapo-math-17k':
        processed_dataset = raw_dataset.map(
            function=dapo_process_fn,
            with_indices=True,
            remove_columns=raw_dataset.column_names,
            num_proc=64)
    else:
        # Create a wrapper function to pass additional arguments
        def _process_fn_wrapper(example: Dict[str, Any],
                                idx: int) -> Dict[str, Any]:
            return process_fn(example, idx, args.input_key, args.label_key,
                              data_source)

        processed_dataset = raw_dataset.map(
            function=_process_fn_wrapper,
            with_indices=True,
            remove_columns=raw_dataset.column_names)

    # Split into train/test if needed
    if args.test_split_ratio > 0:
        split_dataset = processed_dataset.train_test_split(
            test_size=args.test_split_ratio)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']
    else:
        train_dataset = processed_dataset
        test_dataset = processed_dataset

    local_dir = os.path.join(args.local_save_dir, 'dapo')
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    print(f'Saving datasets to {local_dir}...', flush=True)

    # Save train dataset
    train_dataset.to_parquet(
        os.path.join(local_dir, f'{args.dataset_name}_train.parquet'))

    # Save test dataset
    test_dataset.to_parquet(
        os.path.join(local_dir, f'{args.dataset_name}_test.parquet'))

    # Save one example as JSON for reference
    if len(train_dataset) > 0:
        example = train_dataset[0]
        save_example(
            example,
            os.path.join(local_dir, f'{args.dataset_name}_train_example.json'))

    if len(test_dataset) > 0:
        example = test_dataset[0]
        save_example(
            example,
            os.path.join(local_dir, f'{args.dataset_name}_test_example.json'))

    print('Dataset preprocessing completed successfully!', flush=True)


if __name__ == '__main__':
    main()
