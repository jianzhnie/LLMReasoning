import argparse
import logging
import random
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

# 配置日志记录器
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 常量定义 ---
SFT_FIELDS: List[str] = ['input_ids', 'labels', 'attention_mask']
DPO_FIELDS: List[Tuple[str, str]] = [
    ('chosen', 'input_ids'),
    ('chosen', 'labels'),
    ('rejected', 'input_ids'),
    ('rejected', 'labels'),
]

# --- 类型别名 ---
IndexedDatasetDict = Dict[str, IndexedDataset]
DPODatasetDict = Dict[Tuple[str, str], IndexedDataset]

# --- 数据加载函数 ---


def load_sft_datasets(prefix: str) -> IndexedDatasetDict:
    """
    Load SFT (Supervised Fine-Tuning) datasets from pre-processed files.

    The expected file format is:
    - prefix_input_ids_document.*
    - prefix_labels_document.*
    - prefix_attention_mask_document.*

    These files store input_ids, labels, and attention_mask respectively.
    The labels for non-predicted tokens are typically padded with -100.

    Args:
        prefix (str): The common prefix for all dataset files.

    Returns:
        Dict[str, IndexedDataset]: A dictionary where keys are field names and values
                                   are the loaded IndexedDataset objects.

    Raises:
        RuntimeError: If any of the required files cannot be loaded.
    """
    datasets = {}
    for field in SFT_FIELDS:
        path = f'{prefix}_{field}_document'
        try:
            dataset = IndexedDataset(path_prefix=path)
            datasets[field] = dataset
            logger.info(
                f"✅ Successfully loaded '{field}' dataset from: {path}")
        except Exception as e:
            raise RuntimeError(
                f'❌ Failed to load dataset from: {path}. Error: {e}')

    return datasets


def load_dpo_datasets(prefix: str) -> DPODatasetDict:
    """
    Load DPO (Direct Preference Optimization) datasets.

    The expected file format is:
    - prefix_chosen_input_ids_document.*
    - prefix_chosen_labels_document.*
    - prefix_rejected_input_ids_document.*
    - prefix_rejected_labels_document.*

    Args:
        prefix (str): The common prefix for all dataset files.

    Returns:
        Dict[Tuple[str, str], IndexedDataset]: A dictionary where keys are tuples of
                                               (type, field) and values are the loaded datasets.

    Raises:
        RuntimeError: If any of the required files cannot be loaded.
    """
    datasets = {}
    for pair_type, field in DPO_FIELDS:
        path = f'{prefix}_{pair_type}_{field}_document'
        try:
            dataset = IndexedDataset(path_prefix=path)
            datasets[(pair_type, field)] = dataset
            logger.info(
                f"✅ Successfully loaded '{pair_type}/{field}' dataset from: {path}"
            )
        except Exception as e:
            raise RuntimeError(
                f'❌ Failed to load dataset from: {path}. Error: {e}')

    return datasets


# --- 解码和打印函数 ---


def _decode_and_print_dpo(
    tokenizer: PreTrainedTokenizer,
    chosen_input_ids: List[int],
    chosen_labels: List[int],
    rejected_input_ids: List[int],
    rejected_labels: List[int],
) -> None:
    """
    Helper function to decode and print DPO chosen and rejected samples.
    """
    print('=' * 100)
    print('🌟 Chosen Sample 🌟')
    print('-' * 20)
    print('Input IDs Decoded:')
    print(tokenizer.decode(chosen_input_ids, skip_special_tokens=True))

    # Replace -100 in labels for decoding purposes
    temp_chosen_labels = copy(chosen_labels)
    # The `copy()` is essential to avoid modifying the original data.
    temp_chosen_labels[temp_chosen_labels == -100] = tokenizer.pad_token_id
    print('\nLabels Decoded:')
    print(tokenizer.decode(temp_chosen_labels, skip_special_tokens=True))

    print('=' * 100)
    print('🚫 Rejected Sample 🚫')
    print('-' * 20)
    print('Input IDs Decoded:')
    print(tokenizer.decode(rejected_input_ids, skip_special_tokens=True))

    temp_rejected_labels = copy(rejected_labels)
    temp_rejected_labels[temp_rejected_labels == -100] = tokenizer.pad_token_id
    print('\nLabels Decoded:')
    print(tokenizer.decode(temp_rejected_labels, skip_special_tokens=True))
    print('=' * 100)


def _decode_and_print_sft(
    tokenizer: PreTrainedTokenizer,
    input_ids: List[int],
    labels: List[int],
    attention_mask: List[int],
) -> None:
    """
    Helper function to decode and print SFT samples.
    """
    print('=' * 100)
    print('📚 SFT Sample 📚')
    print('-' * 20)
    print('Input IDs Decoded:')
    print(tokenizer.decode(input_ids, skip_special_tokens=True))

    # Replace -100 in labels for decoding purposes
    temp_labels = copy(labels)
    temp_labels[temp_labels == -100] = tokenizer.pad_token_id
    print('\nLabels Decoded:')
    print(tokenizer.decode(temp_labels, skip_special_tokens=True))

    # The attention mask is a list of 0s and 1s, which cannot be decoded into meaningful text.
    # It's more helpful to show the raw values.
    print('\nAttention Mask (Raw):')
    print(attention_mask)
    print('=' * 100)


# --- 主逻辑函数 ---


def main(
    dataset_prefix: str,
    dataset_type: str,
    tokenizer_path: str,
    index: Optional[int] = None,
) -> None:
    """
    Main function to load, analyze, and print samples from an SFT or DPO dataset.

    Args:
        dataset_prefix (str): The base path prefix for the dataset files.
        dataset_type (str): The type of dataset, either 'sft' or 'dpo'.
        tokenizer_path (str): The path to the tokenizer model.
        index (Optional[int]): The index of the sample to view. If None, a random
                               sample will be chosen.
    """
    if dataset_type not in ['sft', 'dpo']:
        raise ValueError("❌ Invalid dataset_type. Must be 'sft' or 'dpo'.")

    # 1. 自动添加 '_packed' 后缀并加载 Tokenizer
    prefix = f'{dataset_prefix}_packed'
    try:
        logger.info(f'🧠 Loading tokenizer from: {tokenizer_path}')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        raise RuntimeError(f'❌ Failed to load tokenizer. Error: {e}')

    # 2. 根据类型加载数据集
    if dataset_type == 'sft':
        datasets = load_sft_datasets(prefix)
    else:  # 'dpo'
        datasets = load_dpo_datasets(prefix)

    # 3. 确定样本总数和要查看的索引
    if dataset_type == 'sft':
        total_samples = len(datasets[SFT_FIELDS[0]])
    else:  # 'dpo'
        total_samples = len(datasets[DPO_FIELDS[0]])

    logger.info(f'📊 Total samples in dataset: {total_samples}')

    if index is not None:
        if not (0 <= index < total_samples):
            raise ValueError(
                f'❌ Index {index} is out of range. Valid range: [0, {total_samples - 1}]'
            )
        sample_index = index
    else:
        sample_index = random.randint(0, total_samples - 1)
        logger.info(
            f'🎲 No index provided. Randomly selected index: {sample_index}')

    # 4. 根据类型获取数据并打印
    logger.info(f'\n🔍 Viewing sample at index: {sample_index}\n')

    if dataset_type == 'sft':
        input_ids = datasets['input_ids'][sample_index]
        labels = datasets['labels'][sample_index]
        attention_mask = datasets['attention_mask'][sample_index]
        _decode_and_print_sft(tokenizer, input_ids, labels, attention_mask)
    else:  # 'dpo'
        chosen_input_ids = datasets[('chosen', 'input_ids')][sample_index]
        chosen_labels = datasets[('chosen', 'labels')][sample_index]
        rejected_input_ids = datasets[('rejected', 'input_ids')][sample_index]
        rejected_labels = datasets[('rejected', 'labels')][sample_index]
        _decode_and_print_dpo(
            tokenizer,
            chosen_input_ids,
            chosen_labels,
            rejected_input_ids,
            rejected_labels,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Load and inspect samples from pre-processed SFT or DPO datasets.')
    parser.add_argument(
        '--dataset_prefix',
        type=str,
        required=True,
        help="The file path prefix for the dataset, e.g., 'data/my_dataset'.",
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['sft', 'dpo'],
        required=True,
        help="The type of dataset to load: 'sft' or 'dpo'.",
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        required=True,
        help='The path to the tokenizer model directory.',
    )
    parser.add_argument(
        '--index',
        type=int,
        default=None,
        help=
        'The specific sample index to view. If not provided, a random index is chosen.',
    )
    args = parser.parse_args()
    main(args.dataset_prefix, args.dataset_type, args.tokenizer_path,
         args.index)
