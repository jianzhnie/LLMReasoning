### 功能介绍

该 Python 脚本是一个强大的工具，旨在帮助您**自动化测试和验证**各种 Hugging Face 语言模型的聊天模板。这个脚本特别有用，因为不同的模型对对话格式有独特的偏好（例如，使用特殊的标记如 `<|user|>` 或 `<s>`）。

该脚本的主要功能包括：

- **自动化测试：** 脚本会遍历预定义的模型列表和系统提示类型。对于每一对组合，它都会应用相应的聊天模板，并打印出格式化的结果。这让您可以轻松查看不同模型的对话输入格式。
- **模板应用：** 它使用 Hugging Face `AutoTokenizer` 的 `apply_chat_template` 方法。这个方法是业界标准，可以根据模型的配置文件自动处理对话历史，生成模型期望的输入字符串。
- **参数化配置：** 脚本采用 `argparse` 模块处理命令行参数，使得它高度灵活。您不需要修改源代码，就可以通过 `--model-dir` 参数指定模型的本地存储路径。

### 使用教程

使用此脚本非常简单。只需几个步骤，您就可以开始验证模型的聊天模板。

#### 1. 前置准备

请确保您的环境中已经安装了必要的 Python 库。如果尚未安装，请在终端中运行以下命令：

```
pip install transformers torch
```

#### 2. 命令行使用

运行脚本时，您可以通过 `--model-dir` 参数指定模型所在的本地目录。脚本会根据 `MODEL_PATHS` 字典中定义的子路径，在您提供的根目录中查找模型。

**示例命令：**

假设您的脚本文件名为 `chat_template_util.py`，并且您的所有 Hugging Face 模型都下载到了 `/root/llmtuner/hfhub/models/` 目录中。

您只需在终端中运行以下命令即可：

```
python chat_template_util.py --model-dir "/root/llmtuner/hfhub/models/"
```

脚本运行后，它会依次加载每个模型，应用不同的系统提示，并打印出最终的格式化字符串。如果某个模型或提示应用失败，它也会友好地打印出错误信息。

#### 3. 自定义配置

您可以根据自己的需求轻松自定义脚本：

- **添加/删除模型：** 只需编辑 `MODEL_PATHS` 字典。您可以添加新的模型名称和其在 Hugging Face Hub 上的相对路径，或者移除您不感兴趣的模型。
- **修改/添加系统提示：** 编辑 `SYSTEM_PROMPT_FACTORY` 字典。您可以修改现有提示的内容，或者添加新的提示类型，以测试不同的模型行为。

例如，要添加一个用于代码生成的系统提示，您可以将字典修改为：

```
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    # ... 其他提示
    'code_assistant': 'You are a professional code assistant. Please write clean and well-commented code.'
}
```

希望这份教程能帮助您更好地理解和使用这个脚本。