# RL Reasoning 数据集生成脚本

## 概述

这是一个用于预处理数据集的 Python 脚本，主要目标是将原始数据集转换为适合大语言模型（LLM）训练的格式。它支持多种数据源（本地 JSON/JSONL 文件或 Hugging Face Hub 上的数据集），并能根据不同的聊天模板和提示（如 Qwen 的 Chain-of-Thought）对数据进行格式化。该脚本的核心优势在于其灵活性和可配置性，所有参数都可通过命令行传递，无需修改代码，极大地提高了易用性和可复用性。

### 主要功能

- **灵活的数据加载**：支持从本地文件（`.json`, `.jsonl`）或 Hugging Face Hub 加载数据集。
- **多样的聊天模板**：能够应用不同的系统提示（如 DeepSeek 的 `deepseek_r1`）和模型特定的提示（如 Qwen 的数学 Chain-of-Thought `\\boxed{}`）。
- **命令行参数化**：所有配置项（数据集路径、模型路径、输出路径、键名等）均通过 `argparse` 进行管理，方便用户通过命令行进行定制。
- **格式化输出**：将处理后的数据以 JSON Lines (`.jsonl`) 格式保存，每行一个 JSON 对象，非常适合作为模型微调的输入。
- **进度显示**：在处理大量数据时，会显示处理进度，让用户了解任务状态。

## 使用教程

### 环境准备

在运行脚本之前，请确保你已经安装了所有必需的 Python 库。

```shell
pip install datasets transformers
```

### 命令行参数

该脚本通过命令行参数进行配置。你可以使用 `python reasoning_data_process.py --help` 查看所有可用的参数及其说明。

```shell
usage: reasoning_data_process.py [-h] --data_path DATA_PATH --model_name_or_path MODEL_NAME_OR_PATH --output_path OUTPUT_PATH [--input_key INPUT_KEY] [--label_key LABEL_KEY]
                           [--system_prompt_type {deepseek_r1,openr1_prompt,none}] [--use_qwen_math_cot] [--run_example]

Preprocess datasets for model training using a specified chat template.

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the local dataset file (.json, .jsonl) or Hugging Face dataset name.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Hugging Face model name or path, used to load the tokenizer.
  --output_path OUTPUT_PATH
                        Path to save the processed .jsonl output file.
  --input_key INPUT_KEY
                        Key in the dataset for the input text (e.g., "prompt" or "Problem").
  --label_key LABEL_KEY
                        Key in the dataset for the label/answer text (e.g., "answer" or "Answer").
  --system_prompt_type {deepseek_r1,openr1_prompt,none}
                        Type of system prompt to use. Available choices: deepseek_r1, openr1_prompt, none
  --use_qwen_math_cot   Use the Qwen math Chain-of-Thought prompt. Overrides --system_prompt_type.
  --run_example         Run only the chat template example without processing the dataset.
```

### 实际操作示例

#### 示例 1：运行一个模板示例

如果你只想测试聊天模板的输出效果，可以使用 `--run_example` 参数。

```shell
python reasoning_data_process.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --run_example
```

**注意**：`--data_path` 和 `--output_path` 是必需参数，但当你使用 `--run_example` 时，它们的值可以任意设置，因为它们不会被实际使用。为了简化操作，你可以在此命令中也提供它们。

#### 示例 2：使用 DeepSeek 模板处理本地数据集

这个命令会从本地加载一个 JSON 文件，使用 `deepseek_r1` 系统提示来格式化数据，并将结果保存到一个新的 `.jsonl` 文件中。

```shell
python reasoning_data_process.py \
    --data_path /path/to/your/dataset.json \
    --model_name_or_path /path/to/your/model_tokenizer \
    --output_path /path/to/save/processed_data.jsonl \
    --input_key prompt \
    --label_key answer \
    --system_prompt_type deepseek_r1
```

- `--data_path`: 指定你的数据集文件路径。
- `--model_name_or_path`: 指定用于加载 tokenizer 的模型路径或名称，确保该模型支持你使用的聊天模板。
- `--output_path`: 指定处理后的 `.jsonl` 文件保存路径。
- `--input_key` 和 `--label_key`: 如果你的数据集中的输入和答案键名不同于默认值（`prompt` 和 `answer`），请使用这两个参数指定正确的键名。

#### 示例 3：使用 Qwen CoT 模板处理 Hugging Face Hub 数据集

这个命令会从 Hugging Face Hub 加载数据集，并使用 `QWEN_MATH_COT` 提示来格式化每个示例的输入。

```shell
python reasoning_data_process.py \
    --data_path gemini/math_qa_datasets \
    --model_name_or_path Qwen/Qwen1.5-7B-Chat \
    --output_path ./processed_qwen_math_qa.jsonl \
    --input_key Problem \
    --label_key Answer \
    --use_qwen_math_cot
```

- `--use_qwen_math_cot`: 这个布尔标志会自动启用 Qwen 的数学 Chain-of-Thought 提示，并会覆盖 `system_prompt_type` 的设置。

通过以上教程，你可以轻松地根据不同的任务和模型需求，灵活地配置和运行此脚本，为你的大语言模型微调任务准备高质量的数据。
