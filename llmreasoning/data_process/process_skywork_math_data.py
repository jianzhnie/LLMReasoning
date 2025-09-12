import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Final, List, TypedDict, Union

from datasets import Dataset, load_dataset

# --------------------
# 📄 Configuration & Setup
# --------------------

# Configure logging for better visibility and control.
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --------------------
# 📚 Type Definitions
# --------------------


class Message(TypedDict):
    """Represents a single message in a conversation."""
    role: str
    content: str


class RewardModel(TypedDict, total=False):
    """
    Represents the reward model information for an example.

    'ground_truth' is optional because it might not exist initially.
    """
    ground_truth: Union[str, List[str]]
    style: str


class ModelDifficulty(TypedDict, total=False):
    """
    Represents the difficulty scores for different models.

    The keys are defined with `total=False` to handle potential
    missing keys gracefully during filtering.
    """
    DeepSeek_R1_Distill_Qwen_1_5B: int
    DeepSeek_R1_Distill_Qwen_7B: int
    DeepSeek_R1_Distill_Qwen_32B: int


class ExtraInfo(TypedDict):
    """Represents extra information for an example."""
    index: int
    model_difficulty: ModelDifficulty


class RawExample(TypedDict):
    """Represents the raw structure of an example from the dataset."""
    prompt: List[Message]
    reward_model: RewardModel
    extra_info: ExtraInfo


class MetaData(TypedDict):
    data_source: str
    model_difficulty: ModelDifficulty


class ProcessedExample(TypedDict):
    """Represents the processed structure of an example."""
    prompt: str
    ground_truth: str
    metadata: MetaData


# --------------------
# 🛠️ Utility Functions
# --------------------


def process_ground_truth(item: dict) -> dict:
    """
    Safely parses the 'ground_truth' string into a list using JSON.

    This function is designed to be used with `dataset.map()`.

    Args:
        item: A dictionary representing a single dataset example.

    Returns:
        The updated dictionary with 'ground_truth' as a list if parsing
        was successful, otherwise the original item.
    """
    # Use nested .get() for safe access to avoid KeyError.
    if isinstance(item.get('reward_model'), dict) and isinstance(
            item['reward_model'].get('ground_truth'), str):
        try:
            # Safely attempt to parse the JSON string.
            parsed_ground_truth = json.loads(
                item['reward_model']['ground_truth'])
            item['reward_model']['ground_truth'] = parsed_ground_truth
        except (json.JSONDecodeError, KeyError) as e:
            # Log the error for debugging but don't halt the process.
            logger.warning(f'Failed to parse ground_truth for item: {e}')
            pass  # Do nothing if parsing fails, the original string remains.
    return item


def contains_url(text: str) -> bool:
    """
    Checks if the given text contains a URL.

    Args:
        text: The input string to check.

    Returns:
        True if a URL is found, otherwise False.
    """
    url_pattern: Final[re.Pattern[str]] = re.compile(
        r'https?://[^\s]+|www\.[^\s]+')
    return bool(url_pattern.search(text))


def is_valid_integer_string(s: str) -> bool:
    """
    Checks if a string represents a valid integer using a regular expression.

    Args:
        s: The string to check.

    Returns:
        True if the string is a valid integer, False otherwise.
    """
    if not isinstance(s, str):
        return False
    # Use re.fullmatch to ensure the pattern matches the entire string.
    return bool(re.fullmatch(r'[+-]?\d+', s.strip()))


def contains_chinese(text: str) -> bool:
    """
    Checks if the given text contains any Chinese characters.

    Args:
        text: The input string to check.

    Returns:
        True if Chinese characters are found, otherwise False.
    """
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# --------------------
# ⚙️ Dataset Processing
# --------------------


def convert_example(example: RawExample) -> ProcessedExample:
    """
    Converts a raw dataset example into a simplified, processed format.

    Args:
        example: The raw example from the dataset.

    Returns:
        A dictionary with 'prompt', 'ground_truth', and 'model_difficulty'.
    """
    # Safely access nested dictionary values.
    prompt_content: str = example['prompt'][0]['content']
    # 'ground_truth' should be a list after `process_ground_truth`.
    ground_truth: str = example['reward_model']['ground_truth'][0]
    model_difficulty: ModelDifficulty = example['extra_info'][
        'model_difficulty']
    meta_data: MetaData = MetaData(data_source=example['data_source'],
                                   model_difficulty=model_difficulty)
    return ProcessedExample(question=prompt_content,
                            ground_truth=ground_truth,
                            meta_data=meta_data)


def filter_fn(example: ProcessedExample) -> bool:
    """
    Filters examples based on specific criteria.

    Criteria:
    1. The prompt does not contain URLs or Chinese characters.
    2. The ground truth is a valid integer string.
    3. All model difficulty scores are within the inclusive range [4, 15].

    Args:
        example: A dictionary representing a processed sample from the dataset.

    Returns:
        True if the example should be kept; False otherwise.
    """
    # 1. Check prompt content.
    prompt = example.get('prompt', '')
    if contains_url(prompt) or contains_chinese(prompt):
        return False

    # 2. Check ground truth format.
    ground_truth = example.get('ground_truth', '')
    if is_valid_integer_string(ground_truth):
        return True

    # 3. Check model difficulty scores.
    difficulty_dict = example.get('model_difficulty', {})
    qwen32_difficulty = difficulty_dict.get('DeepSeek_R1_Distill_Qwen_32B')
    qwen7_difficulty = difficulty_dict.get('DeepSeek_R1_Distill_Qwen_7B')
    qwen1p5_difficulty = difficulty_dict.get('DeepSeek_R1_Distill_Qwen_1_5B')

    if qwen32_difficulty is None or qwen7_difficulty is None or qwen1p5_difficulty is None:
        return False

    if qwen32_difficulty < 10 or qwen7_difficulty < 10 or qwen1p5_difficulty < 10:
        return True

    # A single, correct boolean expression for all conditions.
    return False


# --------------------
# 🚀 Main Execution Logic
# --------------------


def main() -> None:
    """
    Main function to parse arguments, load the dataset, process, and save it.
    """
    parser = argparse.ArgumentParser(
        description='Process and filter a Hugging Face dataset.')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='HuggingFace dataset identifier (e.g., "AI-MATH-PROMPT-DPO").')
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=os.getenv('HF_CACHE_DIR',
                          '/home/jianzhnie/llmtuner/hfhub/cache'),
        help='HuggingFace cache directory. Defaults to HF_CACHE_DIR env var.')
    parser.add_argument(
        '--num_proc',
        type=int,
        default=32,
        help='Number of processes for parallel data processing.')
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help=
        'Output path for the processed dataset (e.g., "data/filtered_dataset.jsonl").'
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f'Loading dataset from "{args.data_dir}"...')
    try:
        # Assume 'train' split as a common practice.
        dataset = load_dataset(args.data_dir,
                               cache_dir=args.cache_dir,
                               split='math')
        logger.info(f'Loaded dataset with {len(dataset)} examples.')
    except Exception as e:
        logger.exception(f'Failed to load dataset: {e}')
        sys.exit(1)

    logger.info('Processing ground truth...')
    # Use a lambda for the map function to directly call `process_ground_truth`.
    dataset = dataset.map(process_ground_truth, num_proc=args.num_proc)

    logger.info('Converting examples...')
    processed_dataset = dataset.map(
        convert_example,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
        batched=False,
        desc='Building RL Reasonling data',
    )

    logger.info('Filtering processed dataset...')
    # The filter function now has corrected logic.
    filtered_dataset = processed_dataset.filter(filter_fn,
                                                num_proc=args.num_proc)

    logger.info(
        f'Initial dataset size: {len(processed_dataset)}, Filtered size: {len(filtered_dataset)}.'
    )
    logger.info(f'Saving filtered dataset to "{output_path}"...')
    try:
        # Use to_json with lines=True to save in JSONL format.
        filtered_dataset.to_json(str(output_path), lines=True)
        logger.info('Dataset saved successfully.')
    except Exception as e:
        logger.exception(f'Failed to save final dataset: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
