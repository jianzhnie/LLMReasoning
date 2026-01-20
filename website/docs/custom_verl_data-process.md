# 自定义数据集预处理器文档

## 1. 功能介绍

这个脚本 (`custom_dataset.py`) 用于将自定义的 JSON 数据集预处理成适用于机器学习工作流的标准 parquet 格式。它将原始 JSON 数据转换为结构化的训练和测试数据集，并添加可用于数学推理任务的元数据。

主要功能包括：
- 从本地文件加载 JSON 数据集
- 将数据拆分为训练集和测试集
- 添加标准化的元数据和提示词用于模型训练
- 将数据转换为高效的 parquet 格式
- 生成示例 JSON 文件供参考

## 2. 使用教程

### 基本用法

```bash
python custom_dataset.py --local_dataset_path /path/to/dataset.json
```

### 完整参数说明

| 参数                   | 默认值        | 描述                       |
| ---------------------- | ------------- | -------------------------- |
| `--local_dataset_path` | (必需)        | 原始 JSON 数据集的本地路径 |
| `--local_save_dir`     | `~/data/math` | 预处理后数据集的保存目录   |
| `--dataset_name`       | `deepscaler`  | 数据集名称                 |
| `--input_key`          | `question`    | 输入文本的键名             |
| `--label_key`          | `answer`      | 标签/答案的键名            |
| `--test_split_ratio`   | `0.1`         | 测试集分割比例             |

### 示例命令

```bash
# 基本用法
python custom_dataset.py --local_dataset_path ./my_dataset.json

# 自定义参数
python custom_dataset.py \
  --local_dataset_path ./math_problems.json \
  --local_save_dir ~/processed_datasets \
  --dataset_name math_dataset_v1 \
  --input_key problem \
  --label_key solution \
  --test_split_ratio 0.2
```

## 3. 输入数据和输出数据格式及样例

### 输入数据格式

输入必须是 JSON 格式的文件，其中包含一个对象数组。每个对象应至少包含指定的 `input_key` 和 `label_key` 字段。

**默认格式示例:**
```json
[
  {
    "question": "What is 2+2?",
    "answer": "4"
  },
  {
    "question": "Solve for x: x + 5 = 10",
    "answer": "x = 5"
  }
]
```

### 输出数据格式

脚本会生成以下文件:

1. `train.parquet` - 训练数据集
2. `test.parquet` - 测试数据集
3. `train_example.json` - 训练样本示例
4. `test_example.json` - 测试样本示例

**输出数据结构示例:**
```json
{
  "data_source": "custom_deepscaler",
  "prompt": [
    {
      "role": "user",
      "content": "What is 2+2? Let's think step by step and output the final answer within \\boxed{}."
    }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "4"
  },
  "extra_info": {
    "split": "train",
    "index": 0
  }
}
```
