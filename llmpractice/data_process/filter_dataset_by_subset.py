import argparse
import json
from typing import Any, Dict, Iterator, Set

from datasets import IterableDataset, load_dataset
from tqdm import tqdm


def load_subset_prompts(subset_path: str) -> Set[str]:
    """
    使用 load_dataset 流式读取 JSONL 格式的子集文件，提取所有 prompt。

    Args:
        subset_path (str): 子集文件路径（JSONL 格式）。

    Returns:
        Set[str]: 所有 prompt 的集合。
    """
    print('🔄 Loading prompts from subset JSONL...')
    dataset = load_dataset('json',
                           data_files=subset_path,
                           split='train',
                           streaming=True)

    prompts = set()
    for example in tqdm(dataset, desc='Reading subset prompts'):
        prompt = example.get('prompt')
        if prompt:
            prompts.add(prompt)

    print(f'✅ Loaded {len(prompts)} unique prompts from subset JSONL.')
    return prompts


def filter_dataset_by_prompts(
        dataset: IterableDataset,
        allowed_prompts: Set[str]) -> Iterator[Dict[str, Any]]:
    """
    过滤出 prompt 在 allowed_prompts 中的数据项。

    Args:
        dataset (IterableDataset): 原始 JSONL 数据集（流式加载）。
        allowed_prompts (Set[str]): 允许保留的 prompt 集合。

    Yields:
        Iterator[Dict[str, Any]]: 符合条件的数据项。
    """
    for example in tqdm(dataset, desc='Filtering dataset'):
        if example.get('prompt') in allowed_prompts:
            yield example


def save_filtered_data(filtered_data: Iterator[Dict[str, Any]],
                       output_path: str) -> None:
    """
    将过滤后的数据保存为 JSONL 文件。

    Args:
        filtered_data (Iterator[Dict[str, Any]]): 过滤后的数据流。
        output_path (str): 输出文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'✅ Filtered data saved to: {output_path}')


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description='Filter JSONL dataset by prompts from a subset JSONL file.'
    )

    parser.add_argument(
        '--subset_path',
        type=str,
        required=True,
        help='Path to the subset JSONL file containing prompts.')
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to the input JSONL dataset.')
    parser.add_argument('--output_path',
                        type=str,
                        default='filtered_dataset.jsonl',
                        help='Path to save the filtered JSONL dataset.')

    return parser.parse_args()


def main():
    args = parse_args()

    print('🔄 Loading subset prompts from JSONL...')
    allowed_prompts = load_subset_prompts(args.subset_path)

    print('🔄 Loading and filtering main dataset...')
    raw_dataset = load_dataset('json',
                               data_files=args.input_path,
                               split='train',
                               streaming=True)
    filtered_data = filter_dataset_by_prompts(raw_dataset, allowed_prompts)

    print('💾 Saving filtered dataset...')
    save_filtered_data(filtered_data, args.output_path)

    print('🎉 Filtering complete.')


if __name__ == '__main__':
    main()
