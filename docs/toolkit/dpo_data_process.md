# DPO 数据集生成脚本介绍

> 脚本地址：[Script](https://gitee.com/jianzhnie/LLMReasoning/blob/main/llmreasoning/data_process/dpo_data_process.py)



这个 Python 脚本是一个强大的数据处理工具，旨在为\*\*直接偏好优化（DPO）**训练任务生成高质量的数据集。它从包含多个思维链（CoT）和问题的数据源中，自动构建 DPO 所需的**`（prompt, chosen, rejected）`\*\*三元组。

**核心功能:**

  * **智能配对:** 自动识别并配对正确的（`chosen`）和错误的（`rejected`）CoT 回答。
  * **长短配对:** 如果数据集中只存在正确回答，脚本会自动选择**最短**的 CoT 作为 `chosen`，**最长**的 CoT 作为 `rejected`，从而创建有价值的偏好对。
  * **模板格式化:** 支持使用 Hugging Face `transformers` 库的 `AutoTokenizer` 聊天模板，将 `prompt` 和 `chosen/rejected` 答案格式化成特定模型的输入格式。
  * **长度过滤:** 使用 tokenizer 精确计算 token 长度，自动过滤掉长度超过指定阈值的 CoT，确保生成的训练数据符合模型的上下文限制。
  * **并行处理:** 利用 `datasets` 库的并行处理能力，高效处理大规模数据集。
  * **灵活配置:** 所有关键参数（如模型路径、最大长度、提示词等）都可以通过命令行参数进行自定义。

-----


## 输入数据格式


为了让脚本能正确处理你的数据，输入文件需要是一个 JSONL（每行一个 JSON 对象）文件。每个 JSON 对象都必须包含一个 `question` 字段，以及一个 `cots` 字段。`cots` 字段可以是一个字典或一个列表，其中包含了多个思维链（CoT）回答及其正确性信息。

下面是两种支持的输入数据格式示例。

-----

### 格式一：`cots` 字段为字典

在这种格式中，`cots` 是一个字典，键（如 `"cot_1"`）用于唯一标识每个 CoT，值是一个包含 `cot` 和 `is_correct` 信息的对象。

```json
{
  "question": "If a car is traveling at 60 mph, how long will it take to travel 180 miles?",
  "cots": {
    "cot_1": {
      "cot": "The distance is 180 miles and the speed is 60 mph. Time = distance / speed = 180 / 60 = 3. It will take 3 hours. <answer>3 hours</answer>",
      "is_correct": true
    },
    "cot_2": {
      "cot": "The distance is 180 miles and the speed is 60 mph. Time = distance / speed = 180 * 60 = 10800. It will take 10800 hours. <answer>10800 hours</answer>",
      "is_correct": false
    },
    "cot_3": {
      "cot": "To find the time, you divide the distance by the speed. 180 miles / 60 mph = 3 hours. <answer>3 hours</answer>",
      "is_correct": true
    }
  }
}
```

-----

### 格式二：`cots` 字段为列表

在这种格式中，`cots` 是一个列表，列表中的每个元素都是一个包含 `cot` 和 `is_correct` 信息的对象。

```json
{
  "question": "What is the capital of Japan?",
  "cots": [
    {
      "cot": "The capital of Japan is a major city known for its blend of modern and traditional culture. That city is Tokyo. <answer>Tokyo</answer>",
      "is_correct": true
    },
    {
      "cot": "The capital of Japan is Kyoto, which used to be the ancient capital. <answer>Kyoto</answer>",
      "is_correct": false
    },
    {
      "cot": "Japan is a country in East Asia. Its capital is the largest metropolitan area in the world, Tokyo. <answer>Tokyo</answer>",
      "is_correct": true
    }
  ]
}
```

-----

### 字段说明

  * `question`: **(必需)** 用户的原始问题，类型为 `string`。
  * `cots`: **(必需)** 包含多个 CoT 回答的字段。可以是 `dict` 或 `list`，其中每个 CoT 条目都必须包含以下子字段：
      * `cot`: **(必需)** 模型的详细思维过程和最终答案，类型为 `string`。
      * `is_correct`: **(必需)** 标识该 CoT 回答是否正确，可以是 `boolean` (`true`/`false`)、`integer` (`1`/`0`) 或 `string` (`"true"`/`"false"`)。

**提示:** 无论你使用哪种格式，脚本都会自动识别并正确解析。请确保你的输入文件是**JSONL**格式，即每个 JSON 对象都位于单独的一行，而不是将整个文件作为一个大的 JSON 数组。



## 输出数据格式

下面是你处理完数据的格式，它将是一个 JSONL 文件，每一行都代表一个完整的 DPO 训练样本。

-----

### 处理完成后的数据格式

脚本的输出是一个 JSONL（每行一个 JSON 对象）文件，其中每个对象都包含了用于 DPO 训练的 **`prompt`**, **`chosen`** 和 **`rejected`** 字段。

`prompt` 字段是已经应用了聊天模板（`chat_template`）的用户提示，而 `chosen` 和 `rejected` 字段则是同样应用了模板的助手回答。

下面是处理完成后的一个数据样本示例：

```json
{
  "prompt": "You are a helpful assistant...\n<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n",
  "chosen": "The capital of Japan is a major city known for its blend of modern and traditional culture. That city is Tokyo. <answer>Tokyo</answer>",
  "rejected": "The capital of Japan is Kyoto, which used to be the ancient capital. <answer>Kyoto</answer>"
}
```

-----

### 字段说明

  * `prompt`: **(必需)** 已经过 `tokenizer.apply_chat_template` 格式化的完整提示词。它通常包含了系统提示词、用户问题，以及模型开始生成回答所需的特殊 Token（例如 `<|im_start|>assistant\n`）。
  * `chosen`: **(必需)** 已经过 `tokenizer.apply_chat_template` 格式化的“**正确**”或“**更优**”的助手回答。在上面的示例中，`chosen` 是一个正确的 CoT 回答。
  * `rejected`: **(必需)** 已经过 `tokenizer.apply_chat_template` 格式化的“**错误**”或“**较差**”的助手回答。在上面的示例中，`rejected` 是一个错误的 CoT 回答。
  * `system`（可选）: 如果你的 `prompt` 模板中包含了系统提示词，这个字段可能也会被保留，但这取决于你的具体实现。在 DPO 训练中，最重要的三个字段是 `prompt`、`chosen` 和 `rejected`。

这种格式直接适配了大多数 DPO 训练框架，你可以将这个输出文件作为 `chosen_datasets` 和 `rejected_datasets` 的输入，进行模型的微调。



## 如何使用

本脚本通过命令行运行，需要指定输入文件、输出文件和模型路径等参数。

 前提条件

你需要安装以下 Python 库：

```bash
pip install datasets transformers
```

 命令行参数

以下是脚本支持的所有命令行参数：

| 参数名                 | 类型   | 默认值                           | 描述                                                    |
| :--------------------- | :----- | :------------------------------- | :------------------------------------------------------ |
| `--input_path`         | `str`  | **无**                           | \*\*必需。\*\*输入 JSONL 文件路径。                     |
| `--output_path`        | `str`  | **无**                           | \*\*必需。\*\*输出 JSONL 文件路径。                     |
| `--model_name_or_path` | `str`  | **无**                           | \*\*必需。\*\*用于加载 tokenizer 的模型名称或本地路径。 |
| `--cache_dir`          | `str`  | `/root/llmtuner/hfhub/cache_dir` | Hugging Face 缓存目录。                                 |
| `--num_proc`           | `int`  | `32`                             | 并行处理进程数。                                        |
| `--max_cot_len`        | `int`  | `32768`                          | CoT 的最大 token 长度。                                 |
| `--system_prompt`      | `str`  | （内置）                         | 自定义系统提示词模板。如果未指定，将使用内置默认值。    |
| `--math_cot_prompt`    | `str`  | （内置）                         | 自定义数学 CoT 提示词。如果未指定，将使用内置默认值。   |
| `--save_subset`        | `bool` | `False`                          | 启用此标志，将额外保存一个较小的子集。                  |
| `--subset_size`        | `int`  | `256`                            | 子集的大小。                                            |
| `--subset_output_path` | `str`  | `dpo_output_subset.jsonl`        | 子集输出文件的路径。                                    |

 示例

假设你有一个名为 `raw_data.jsonl` 的原始数据集，并且你想使用 `Qwen/Qwen2-7B-Instruct` 模型的 tokenizer 来处理。你可以这样运行脚本：

**1. 简单运行**

```bash
python dpo_data_process.py \
  --input_path raw_data.jsonl \
  --output_path dpo_data.jsonl \
  --model_name_or_path Qwen/Qwen2-7B-Instruct \
  --num_proc 8
```

这将会：

  * 使用 `Qwen2-7B-Instruct` 的 tokenizer。
  * 使用默认的系统提示词和数学 CoT 提示词。
  * 生成并保存一个名为 `dpo_data.jsonl` 的 DPO 数据集。

**2. 包含自定义提示词和保存子集**

如果你想自定义提示词，并同时保存一个大小为 `500` 的子集，可以这样做：

```bash
python dpo_data_process.py \
  --input_path raw_data.jsonl \
  --output_path dpo_data_custom_prompt.jsonl \
  --model_name_or_path Qwen/Qwen2-7B-Instruct \
  --system_prompt "你是一个专业的AI助手，请按照我的要求进行回复。" \
  --math_cot_prompt "请详细分析并给出最终答案。" \
  --max_cot_len 16384 \
  --save_subset \
  --subset_size 500 \
  --subset_output_path dpo_data_subset.jsonl
```
这将会：

  * 使用自定义的系统提示词和数学 CoT 提示词。
  * 将 CoT 的最大长度限制为 `16384` tokens。
  * 生成并保存一个名为 `dpo_data_custom_prompt.jsonl` 的 DPO 数据集。
  * 同时保存一个名为 `dpo_data_subset.jsonl` 的大小为 `500` 的子集。
