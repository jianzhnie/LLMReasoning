import argparse
import glob
import json
import os
from typing import Any, Dict

from datasets import Dataset, load_dataset


def merge_datasets_from_folder(input_folder: str, file_pattern: str,
                               output_path: str) -> None:
    """
    Merges multiple JSON datasets from a folder into a single JSON file.

    Args:
        input_folder: The directory containing the JSON files.
        file_pattern: A glob pattern to match the files (e.g., "*.json").
        output_path: The path for the final merged JSON file.
    """
    # 构建搜索路径
    search_path = os.path.join(input_folder, file_pattern)
    print(f'🔍 Searching for files with pattern: {search_path}')

    # 使用 glob 查找所有匹配的文件
    # load_dataset 接受一个文件列表作为输入
    input_files = sorted(glob.glob(search_path))

    if not input_files:
        print('❌ No files found. Please check the input folder and pattern.')
        return

    print(f'✅ Found {len(input_files)} files to merge.')

    try:
        # 使用 datasets.load_dataset 来加载并合并所有文件
        # 'json' 加载器会自动处理文件列表
        merged_dataset = load_dataset('json',
                                      data_files=input_files,
                                      split='train')

        print('📊 Dataset loaded and merged successfully!')
        print(f'Total number of examples: {len(merged_dataset)}')

        # 将合并后的数据集保存为 JSON 文件
        merged_dataset.to_json(output_path, lines=True)
        print(f'💾 Merged dataset saved to: {output_path}')

    except Exception as e:
        print(f'❌ An error occurred during dataset merging: {e}')


def main():
    """
    Main function to handle command-line arguments and run the merging process.
    """
    parser = argparse.ArgumentParser(
        description=
        'Merge multiple JSON datasets from a folder using the datasets library.'
    )
    parser.add_argument(
        '--input_folder',
        '-i',
        type=str,
        required=True,
        help='Path to the folder containing the JSON files.',
    )
    parser.add_argument(
        '--file_pattern',
        '-p',
        type=str,
        default='*.json',
        help=
        "Glob pattern to match files (e.g., '*.json' or '*_grouped_by_prompt.json').",
    )
    parser.add_argument(
        '--output_path',
        '-o',
        type=str,
        required=True,
        help='Path for the final merged JSON file.',
    )

    args = parser.parse_args()

    # 调用合并函数
    merge_datasets_from_folder(input_folder=args.input_folder,
                               file_pattern=args.file_pattern,
                               output_path=args.output_path)


if __name__ == '__main__':
    main()
