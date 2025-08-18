## 模型聊天模板测试工具

### 简介

这是一个用于测试和验证 Hugging Face 模型聊天模板的实用脚本。它遍历一个预定义的模型列表和系统提示词类型，将一个示例对话应用到每个模型的聊天模板上，并打印出格式化后的结果。

通过运行此脚本，你可以快速查看不同模型如何处理和格式化对话输入，这对于确保模型在推理阶段能正确理解对话历史至关重要。

### 主要功能

- **多模型支持**：内置一个包含多个 Hugging Face 模型的列表，方便批量测试。
- **多提示词类型**：支持多种系统提示词类型，例如 DeepSeek R1 的特定格式、Qwen 的数学 COT（Chain-of-Thought）提示词，以及通用提示词。
- **健壮的错误处理**：脚本能优雅地处理因模型或分词器加载失败而引起的错误，并提供清晰的错误信息。
- **可移植性**：通过命令行参数和环境变量，轻松适应不同的本地模型存储路径。
- **代码规范**：代码结构清晰、可读性高，并添加了详细的类型提示和文档字符串，遵循了 Python 最佳实践。

### 使用方法

#### 前提条件

1. **安装依赖**：确保你已安装 `transformers` 库。

   ```
   pip install transformers
   ```

2. **下载模型**：脚本默认从本地路径加载模型。你需要先将想要测试的模型下载到本地。

   - **方法一**：使用 `huggingface-cli` 下载。

     ```
     huggingface-cli download Qwen/Qwen2.5-7B --local-dir /path/to/your/models/Qwen/Qwen2.5-7B
     ```

   - **方法二**：直接使用 `git lfs` 克隆仓库。

     ```
     git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B /path/to/your/models/Qwen/Qwen2.5-7B
     ```

#### 运行脚本

脚本支持以下命令行参数：

| 参数                  | 类型   | 默认值                         | 描述                                                         |
| --------------------- | ------ | ------------------------------ | ------------------------------------------------------------ |
| `--model-dir`         | `str`  | `/root/llmtuner/hfhub/models/` | 本地模型存储的基础目录。此值可通过环境变量 `HF_MODEL_DIR` 覆盖。 |
| `--use-qwen-math-cot` | `flag` | `False`                        | 如果添加此参数，则会在用户消息后附加一个 Qwen 数学 COT 提示。 |

1. **基本运行**

   在命令行中执行脚本，它将遍历 `MODEL_PATHS` 中定义的所有模型，并对每种系统提示类型进行测试。

   ```
   python apply_chat_template.py
   ```

2. **指定模型目录**

   如果你的模型存储在其他位置，可以使用 `--model-dir` 参数指定路径。

   ```
   python apply_chat_template.py --model-dir /data/my_models
   ```

3. **使用 Qwen 数学 COT 提示**

   如果你想测试 Qwen 的数学提示词如何与聊天模板结合，可以添加 `--use-qwen-math-cot` 标志。

   ```
   python apply_chat_template.py --use-qwen-math-cot
   ```

4. **通过环境变量指定模型目录**

   你也可以设置 `HF_MODEL_DIR` 环境变量，脚本将优先使用此值。

   ```
   export HF_MODEL_DIR=/data/llm_models
   python apply_chat_template.py
   ```

### 预期输出

脚本会为每个模型和每种提示词类型打印一个分隔块，展示其聊天模板的最终输出结果。如果加载或应用模板失败，会打印一个清晰的错误信息，方便你排查问题。

**示例输出：**

```shell
============================================================
Model: Qwen2.5-7B
Prompt Type: default

<|system|>
You are a helpful assistant.
<|endoftext|>
<|user|>
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.How many clips did Natalia sell altogether in April and May?
<|endoftext|>
<|assistant|>
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
\boxed{72}
<|endoftext|>
```