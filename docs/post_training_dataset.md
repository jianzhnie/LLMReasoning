# Post-Training-Dataset

## nvidia/Nemotron-Post-Training-Dataset-v1

### Data Distribution

|   Category   | Value          |
| :----------: | -------------- |
|     chat     | 746,622        |
|     code     | 1,896,395      |
|     math     | 2,044,407      |
|     stem     | 20,662,167     |
| tool_calling | 310,051        |
|  **Total**   | **25,659,642** |

### Prompts

提示词来源包括公开开放语料库或合成生成。所有回答均通过公开开放模型合成生成。

提示词经过提取后，根据质量与复杂度进行筛选，或通过生成满足质量与复杂度要求。筛选过程包括移除不一致提示词、答案易猜测的提示词，以及语法错误的提示词。

### Responses

回答由多种模型合成生成，部分提示词包含推理开启与关闭双模式回答，以训练模型区分两种模式。

|      Model       | Number of Samples |
| :--------------: | ----------------- |
| DeepSeek-R1-0528 | 24,602,969        |
| Qwen3-235B-A22B  | 1,056,673         |
|    **Total**     | **25,659,642**    |

### Recommended Training Formats

本数据集以原始格式提供数据（例如数学问题、编程挑战）。为在监督微调期间获得最佳性能，我们建议使用指令模板包装输入字段。以下是我们训练中使用的模板示例。

#### For the chat split:

聊天分支专为对话调优设计。输入字段代表用户轮次，通常可直接使用。训练时模型的系统提示可以是：

```text
You are a helpful and friendly AI assistant.
```

需要注意的是，聊天分支中的某些提示来源于外部。对于这些条目，“messages”中的“input”字段为空，用户必须从原始来源 lmsys-chat-1m 下载所需数据。

#### For the code split:


若要指导模型生成带有详细解释的代码，请使用同时要求解释和代码块的格式：

```text
Write a solution for the following programming challenge. Provide a brief explanation of your approach, followed by the complete code.
{problem}
```

需要注意的是，代码分支中的某些提示来源于外部。对于这些条目，“input”字段为空，用户必须从原始来源网站下载所需数据。更多信息请参阅 OpenCodeReasoning-2 的 README 文档。

#### For the math split:

若要引导模型提供分步解答并明确标注最终答案，请使用如下格式：

```text
Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.
{problem}
```

#### For the stem split:

对于一般推理、科学及人文问题，直接给出指令是有效的：

```text
Read the following problem carefully and provide a detailed, step-by-step answer.
{problem}
```

#### For the tool calling split:

工具调用分割覆盖了单轮、多轮和多步骤工具调用场景。元数据中的“工具”和助手消息中的“tool_calls”应按照模型的工具调用模板进行格式化以用于训练。





## a-m-team/AM-DeepSeek-R1-0528-Distilled

### Dataset Summary


该数据集是一个高质量推理语料库， **源自 DeepSeek-R1-0528 的蒸馏 ，这是 DeepSeek-R1 大语言模型的改进版。相较于初始版本，DeepSeek-R1-0528 在推理能力、指令遵循和多轮对话方面展现出显著进步。基于这些改进，我们以 DeepSeek-R1-0528 作为教师模型，收集并精炼了跨多个领域的 **260 万条多样化查询。

DeepSeek-R1-0528 的一个显著特点是其输出长度远超先前版本，尤其在数学领域：对于某些数学问题，输出长度达到早期版本的 **1.5 至 2 倍。这体现了更详尽、明确的逐步推理过程。

该数据集采用统一格式与验证流程，可直接与其他开源蒸馏语料库进行对比。其旨在支持开发具备强大、可验证推理能力的下一代语言模型。

### Dataset Statistics

- Shared query base: **2.6 million** unique prompts
- Responses distilled from **DeepSeek-R1-0528**
- Task Category Breakdown:
  - **general chat**: 1,223K (47.3%)
  - **math**: 674K (26.1%)
  - **code**: 412K (16.0%)
  - **science**: 220K (8.5%)
  - **if**: 54K (2.1%)
- Each sample is verified and filtered for output quality.

### Dataset Structure

#### Data Fields

Each sample is a dictionary with the following fields:

- system: The system prompt used during distillation, typically guiding structured reasoning via <think>and<answer>tags.
  - Note: Some instance's 'system' fields in our dataset are empty. The 'system' field is not used in training. Feel free to use them.

- conversations
  : A list of dialogue turns structured as:
  - `from`: Either `'human'` or `'assistant'`.
  - `value`: Full message content.
  - info
    : Metadata dictionary containing:
    - `source`: Dataset origin (e.g., `OpenHermes-2.5`).
    - `category`: Task domain (e.g., `math`, `code`, `general chat`).
    - `ground_truth`: Ground truth reference (if applicable).
    - `test_case`: Associated test case ID (optional).
    - `instruction_constrain`: Instruction constraint metadata (optional).
    - `think_content`: Assistant’s reasoning trace.
    - `answer_content`: Final answer segment.
    - `verify_score`: Verification confidence score (float ≥ 0.9).
    - `model_name`: Name of the teacher model (`deepseek-r1-0528`).
    - `ppl`: Perplexity of the assistant’s output.
