# VLLM 数学采样推理数据后处理


## 📜 功能介绍

该脚本是一个功能强大的数据处理工具，专为分析大型语言模型（LLM）的评估结果而设计。它利用 **Hugging Face `datasets` 库**，能够高效地并行处理多个 JSONL 文件，并汇总关键指标。该脚本的核心功能如下：

  * **数据聚合**：它能够读取指定目录下的所有 JSONL 文件，并将每个文件中关于提示（prompt）的评估数据进行聚合。
  * **准确性分析**：对于每个独特的提示，它计算并汇总其平均准确性（`average_accuracy`）、总计数（`count`）和总准确性计数（`total_accuracy_count`）。
  * **生成文本长度分析**：它使用指定的 Hugging Face tokenizer 计算每个模型生成文本的词元（token）长度，并提供平均长度（`avg_gen_length`）和最大长度（`max_gen_length`）。
  * **并行处理**：通过 `datasets` 库的 `map` 和 `filter` 方法，脚本可以利用多核 CPU 进行并行处理，从而显著提升处理速度，特别是在处理大规模数据集时。
  * **统一输出**：最终，所有分析结果被合并到一个单一的 JSON 文件中，为后续的数据分析和可视化提供了便利。



## 📂 输入/输出数据样例

为了更好地理解脚本的功能，下面提供了输入 JSONL 文件和脚本生成的最终输出 JSON 文件的结构示例。

### 📥 输入数据样例（`example_input.jsonl`）

输入数据是 JSONL (JSON Lines) 格式，这意味着每一行都是一个独立的、有效的 JSON 对象。每个 JSON 对象通常代表一个评估示例。

**文件内容示例：**

```json
{"prompt": "中国的首都是哪里？", "gen": "北京", "answer": "北京", "accuracy": 1.0}
{"prompt": "中国的首都是哪里？", "gen": "上海", "answer": "北京", "accuracy": 0.0}
{"prompt": "美国的国旗有多少颗星星？", "gen": "50", "answer": "50", "accuracy": 1.0}
{"prompt": "美国的国旗有多少颗星星？", "gen": "50颗", "answer": "50", "accuracy": 1.0}
{"prompt": "美国的国旗有多少颗星星？", "gen": ["50颗"], "answer": "50", "accuracy": 1.0}
{"prompt": "日本的首都是哪里？", "gen": "东京", "answer": "东京", "accuracy": 1.0}
```

  * **`prompt`**: 模型的输入提示。这是用于聚合数据的唯一标识符。
  * **`gen`**: 模型生成的文本。
  * **`answer`**: 期望的正确答案。
  * **`accuracy`**: 准确性评分，通常是一个浮点数（例如 `1.0` 表示正确，`0.0` 表示不正确）。

### 📤 输出数据样例（`combined_accuracy_summary.json`）

脚本将多个输入 JSONL 文件中的数据进行聚合和统计，然后将结果保存为一个单一的 JSON 文件。每个 JSON 对象代表一个 **聚合后的提示** 的所有统计信息。

**文件内容示例：**

```json
[
  {
    "prompt": "中国的首都是哪里？",
    "answer": "北京",
    "count": 2,
    "total_accuracy_count": 1.0,
    "average_accuracy": 0.5,
    "max_gen_length": 2,
    "avg_gen_length": 2.5,
    "gen_lengths": [2, 3]
  },
  {
    "prompt": "美国的国旗有多少颗星星？",
    "answer": "50",
    "count": 3,
    "total_accuracy_count": 3.0,
    "average_accuracy": 1.0,
    "max_gen_length": 3,
    "avg_gen_length": 2.3333333333333335,
    "gen_lengths": [1, 2, 2]
  },
  {
    "prompt": "日本的首都是哪里？",
    "answer": "东京",
    "count": 1,
    "total_accuracy_count": 1.0,
    "average_accuracy": 1.0,
    "max_gen_length": 2,
    "avg_gen_length": 2.0,
    "gen_lengths": [2]
  }
]
```

  * **`prompt`**: 聚合后的提示文本。
  * **`answer`**: 对应的标准答案。
  * **`count`**: 该提示在所有输入文件中出现的总次数。
  * **`total_accuracy_count`**: 该提示的总准确性计数（所有 `accuracy` 值的总和）。
  * **`average_accuracy`**: 平均准确性，等于 `total_accuracy_count` 除以 `count`。
  * **`max_gen_length`**: 该提示所有生成文本中的最大词元（token）长度。
  * **`avg_gen_length`**: 该提示所有生成文本的平均词元长度。
  * **`gen_lengths`**: 包含所有生成文本词元长度的列表，用于详细分析。


## 👨‍💻 使用教程

### 📋 先决条件

在运行此脚本之前，请确保你已安装所有必需的 Python 库。你可以使用以下命令安装它们：

```bash
pip install datasets transformers
```

### 🚀 运行命令

该脚本是一个命令行工具。你需要在终端中运行它，并提供相应的参数。

**基本语法**：

```bash
python your_script_name.py \
--input_data_dir <输入目录> \
--model_name_or_path <模型路径> \
--output_data_dir <输出目录> \
--output_filename <输出文件名> \
--input_file_pattern <文件匹配模式> \
--num_proc <并行进程数>
```



### 示例

假设你的 JSONL 评估文件存储在 `./data/evals/` 目录下，文件名为 `inference_part_1.jsonl`, `inference_part_2.jsonl` 等。你想将结果汇总到 `./summary/` 目录下的 `final_report.json` 文件中。你正在使用的模型是 `Qwen/Qwen2.5-7B`。

**运行命令**：

```bash
python batch_processor.py \
--input_data_dir ./data/evals \
--input_file_pattern "inference_*.jsonl" \
--output_data_dir ./summary \
--output_filename final_report.json \
--model_name_or_path "/root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B" \
--num_proc 8
```



### 📊 参数说明

  * `--input_data_dir` (必需): 包含输入 JSONL 文件的目录路径。
  * `--model_name_or_path` (必需): Hugging Face 模型的路径或名称。**此参数仅用于加载 tokenizer**，以正确计算生成文本的词元长度。
  * `--output_data_dir` (可选): 存放最终汇总 JSON 文件的目录。默认为 `./summary`。
  * `--output_filename` (可选): 最终汇总 JSON 文件的文件名。默认为 `combined_accuracy_summary.json`。
  * `--input_file_pattern` (可选): 用于匹配输入 JSONL 文件的 glob 模式。默认为 `*.jsonl`，表示处理目录下所有以 `.jsonl` 结尾的文件。
  * `--num_proc` (可选): 并行处理的进程数。默认为 `64`。你可以根据你的 CPU 核数进行调整，以优化性能。

通过遵循这些步骤，你可以轻松地自动化对大量评估结果的分析和报告生成过程。
