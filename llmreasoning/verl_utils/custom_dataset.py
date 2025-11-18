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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os
import datasets
  

def main()
    parser = argparse.ArgumentParser()    
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--dataset_name", default="deep_scaler", help=""
    )
    parser.add_argument("--input_key", default="question", help="")
    parser.add_argument("--label_key", default="answer", help="")

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "custom" + "_" + args.dataset_name
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    raw_dataset = datasets.load_dataset('json',
                               data_files=args.local_dataset_path,
                               split='train')

    raw_dataset = raw_dataset.train_test_split(test_size=0.1)

    train_dataset = raw_dataset["train"]
    test_dataset = raw_dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop(args.input_key)

            question = question + " " + instruction_following

            answer = example.pop(args.label_key)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)


    local_save_dir = args.local_save_dir
    local_dir = os.path.join(local_save_dir, args.dataset_name)
    os.makedirs(local_dir, exist_ok=False)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)


if __name__ == "__main__":
    main()