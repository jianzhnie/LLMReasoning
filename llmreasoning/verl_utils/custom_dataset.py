#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess custom datasets to parquet format
"""

import argparse
import json
import os
from pathlib import Path

import datasets


def main():
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

    # Expand user home directory
    local_save_dir = os.path.expanduser(args.local_save_dir)

    data_source = 'custom' + '_' + args.dataset_name
    print(
        f'Loading the {data_source} dataset from {args.local_dataset_path}...',
        flush=True)

    try:
        if args.local_dataset_path.endswith(
                '.json') or args.local_dataset_path.endswith('.jsonl'):
            raw_dataset = datasets.load_dataset(
                'json', data_files=args.local_dataset_path, split='train')
        elif args.local_dataset_path.endswith('.parquet'):
            raw_dataset = datasets.load_dataset(
                'parquet', data_files=args.local_dataset_path, split='train')
        else:
            raw_dataset = datasets.load_dataset(args.local_dataset_path,
                                                split='train')
    except Exception as e:
        raise RuntimeError(f'Failed to load dataset: {e}')

    print(f'Dataset loaded with {len(raw_dataset)} samples', flush=True)

    raw_dataset = raw_dataset.train_test_split(test_size=args.test_split_ratio)

    train_dataset = raw_dataset['train']
    test_dataset = raw_dataset['test']

    # Made instruction more configurable
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):

        def process_fn(example, idx):
            # Use get() method to avoid KeyError
            question = example.get(args.input_key, '')
            answer = example.get(args.label_key, '')

            question = question + ' ' + instruction_following

            data = {
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
                    'split': split,
                    'index': idx
                },
            }
            return data

        return process_fn

    print('Processing training dataset...', flush=True)
    train_dataset = train_dataset.map(function=make_map_fn('train'),
                                      with_indices=True)

    print('Processing test dataset...', flush=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'),
                                    with_indices=True)

    local_dir = os.path.join(local_save_dir, args.dataset_name)

    # Fixed: Use exist_ok=True to prevent crashes if directory exists
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    print(f'Saving datasets to {local_dir}...', flush=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Save one example as JSON for reference
    if len(train_dataset) > 0:
        example = train_dataset[0]
        with open(
                os.path.join(local_dir,
                             f'{args.dataset_name}_train_example.json'),
                'w') as f:
            json.dump(example, f, indent=2)
    if len(test_dataset) > 0:
        example = test_dataset[0]
        with open(
                os.path.join(local_dir,
                             f'{args.dataset_name}_test_example.json'),
                'w') as f:
            json.dump(example, f, indent=2)

    print('Dataset preprocessing completed successfully!', flush=True)


if __name__ == '__main__':
    main()
