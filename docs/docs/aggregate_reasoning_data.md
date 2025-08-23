# `aggregate_reasoning_data.py` 脚本介绍文档

本文档旨在详细介绍 `aggregate_reasoning_data.py` 脚本的功能、设计思路和使用方法。

## 1. 功能介绍

`aggregate_reasoning_data.py` 是一个数据处理脚本，其核心功能是将一个包含多条思维链（Chain-of-Thought, CoT）推理记录的 `JSONL` 文件，聚合成一个结构化的 `JSON` 文件。

在大型语言模型的推理任务中，同一个问题（prompt）可能会生成多个不同的推理路径和答案。此脚本的主要目的就是将这些针对同一问题的多条记录进行分组和整理，形成一个以问题为核心的、包含多个思维链变体的结构化数据集。

**核心功能点：**

*   **流式数据加载**：为高效处理可能非常大的输入文件，脚本采用流式（streaming）方式加载数据，有效降低了内存消耗。
*   **数据预处理**：脚本会从原始数据中提取关键字段，如 `prompt`（问题）、`gen`（生成的文本）、`accuracy`（准确率）和 `extracted_answer`（提取的答案），并将其转换为统一的内部格式。
*   **按问题分组**：脚本的核心逻辑。它会遍历所有数据记录，并将具有相同 `prompt` 的记录聚合在一起。
*   **结构化输出**：将分组后的数据整理成一个清晰的 `JSON` 结构。每个问题对应一个顶层对象，该对象包含问题本身、标准答案以及一个名为 `cots` 的字典，其中存放了所有与该问题相关的思维链推理过程。
*   **元数据生成**：在最终输出中，脚本会自动为每个条目生成一个唯一的 `id`，并计算每个问题的 `difficulty`（难度，此处定义为与该问题关联的思维链数量）以及每条思维链的 `cot_token_len`（Token 长度）。

最终生成的 `JSON` 文件可以被用于后续的分析、模型训练或评测，特别适用于需要对不同推理路径进行比较和研究的场景。

## 2. 使用教程

### 2.1. 环境准备

运行此脚本需要一个标准的 Python 环境，并安装 `datasets` 和 `tqdm` 库。您可以通过以下命令安装所需依赖：

```bash
pip install datasets tqdm
```

### 2.2. 命令行参数

脚本通过命令行接收两个必需的参数：

*   `--input` / `-i`：**（必需）** 指定输入的 `JSONL` 文件的路径。此文件应包含原始的推理数据。
*   `--output` / `-o`：**（必需）** 指定处理后输出的 `JSON` 文件的保存路径。

### 2.3. 输入文件格式

输入的 `JSONL` 文件（每行一个 JSON 对象）应至少包含以下或类似的字段：

*   `prompt`: (字符串) 提给模型的问题。
*   `gen`: (字符串或列表) 模型生成的推理过程和答案。
*   `accuracy`: (浮点数) 该条推理的准确度评分。
*   `extracted_answer`: (字符串) 从 `gen` 中提取出的最终答案。

**示例 `input.jsonl`：**
```json
{"prompt": "1+1=?", "gen": ["1+1=2"], "accuracy": 1.0, "extracted_answer": "2"}
{"prompt": "1+1=?", "gen": ["one plus one equals two"], "accuracy": 1.0, "extracted_answer": "2"}
{"prompt": "2+2=?", "gen": ["2+2=4"], "accuracy": 1.0, "extracted_answer": "4"}
```

### 2.4. 输出文件格式

脚本执行成功后，会生成一个 `JSON` 文件，其结构如下：

```json
[
  {
    "id": "1",
    "question": "1+1=?",
    "answer": "2",
    "difficulty": 2,
    "cots": {
      "cot_1": {
        "cot": "1+1=2",
        "cot_token_len": 3,
        "is_correct": true
      },
      "cot_2": {
        "cot": "one plus one equals two",
        "cot_token_len": 5,
        "is_correct": true
      }
    }
  },
  {
    "id": "2",
    "question": "2+2=?",
    "answer": "4",
    "difficulty": 1,
    "cots": {
      "cot_1": {
        "cot": "2+2=4",
        "cot_token_len": 3,
        "is_correct": true
      }
    }
  }
]
```

### 2.5. 使用示例

假设您的输入文件位于 `/path/to/input_data.jsonl`，并且您希望将处理结果保存到 `/path/to/aggregated_output.json`，您可以执行以下命令：

```bash
python /Users/jianzhengnie/work_dir/chatgpt/LLMReasoning/llmreasoning/data_process/aggregate_reasoning_data.py \
  --input /path/to/input_data.jsonl \
  --output /path/to/aggregated_output.json
```
