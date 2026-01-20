# 轻量级实验追踪库 Trackio (From Hugging Face)

[Trackio](https://github.com/gradio-app/trackio) 是 Hugging Face 推出的一款全新开源、免费的 Python 实验追踪库。它提供本地可视化仪表板，并支持无缝集成 Hugging Face Spaces，用户只需分享一个 URL 即可实现结果共享。Spaces 支持设置为公开或组织内私有，满足不同场景下的协作需求。由于 `trackio` 与 `wandb` 的 API 完全兼容，开发者可以使用已熟悉的语法快速上手。

## 背景

在机器学习模型训练过程中，对指标、参数和超参数进行有效追踪，并在训练后可视化分析，是提升模型开发效率的关键环节。大多数研究人员依赖专门的实验追踪工具来完成这些任务。然而，现有工具往往存在收费、配置复杂或共享不便等问题，难以满足快速迭代和开放协作的需求。

## 设计理念

- **API 兼容性**：与主流追踪工具（如 wandb）保持接口一致，迁移成本极低。
- **本地优先**：默认在本地运行并持久化日志，可选同步至 Hugging Face Spaces。
- **轻量可扩展**：核心代码不足 1000 行 Python，结构清晰，易于理解和二次开发。
- **完全开源免费**：所有功能，包括 Spaces 托管，均无任何费用。
- **基于成熟生态构建**：底层依赖 Hugging Face Datasets 和 Spaces，保障数据管理与可视化能力的稳定性。

## 为什么选择 Trackio？

Hugging Face 科研团队已在多个研究项目中采用 [Trackio](https://github.com/gradio-app/trackio)，相较于其他追踪方案，其具备以下核心优势：

### 1. 简化共享与嵌入
Trackio 支持通过 iframe 将训练曲线或关键指标直接嵌入博客、文档或报告中，便于向团队成员或社区展示实验进展。用户无需注册账户或登录复杂系统即可查看结果，显著提升了信息传递效率。

### 2. 标准化与透明化
追踪如 GPU 能耗等指标对于评估模型训练的能源消耗和环境影响至关重要。Trackio 直接调用 `nvidia-smi` 获取硬件能耗数据，使能耗信息的量化与对比更加便捷，并可轻松集成至 [模型卡片（Model Cards）](https://huggingface.co/docs/hub/model-cards)，增强研究透明度。

### 3. 数据可访问性强
不同于某些将数据锁定在专有 API 后的追踪工具，Trackio 保证了日志数据的高度可访问性。研究人员可随时导出原始数据，用于自定义分析或将指标集成至其他研究流程中，提升了工作流的灵活性。

### 4. 轻量灵活，便于实验创新
Trackio 的轻量设计使其易于扩展和定制。例如，在记录张量（tensor）时，开发者可自主控制张量从 GPU 到 CPU 的转移时机，在不影响训练性能的前提下高效追踪模型中间状态，显著提升大规模实验的吞吐效率。

## 安装与使用

### 安装

使用 `pip` 安装 Trackio：

```bash
pip install trackio
```

或使用 `uv`（推荐用于现代 Python 环境）：

```bash
uv pip install trackio
```

### 使用方法

Trackio 的设计目标是作为主流实验追踪库（如 `wandb`）的“即插即用”替代方案。其 API 与 `wandb.init`、`wandb.log` 和 `wandb.finish` 完全兼容，因此只需在代码中做简单替换即可迁移。

```diff
- import wandb
+ import trackio as wandb
```

#### 示例代码

以下是一个模拟训练过程的示例：

```python
import trackio
import random
import time

runs = 3
epochs = 8

def simulate_multiple_runs():
    for run in range(runs):
        trackio.init(project="fake-training", config={
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 64
        })
        for epoch in range(epochs):
            train_loss = random.uniform(0.2, 1.0)
            train_acc = random.uniform(0.6, 0.95)
            val_loss = train_loss - random.uniform(0.01, 0.1)
            val_acc = train_acc + random.uniform(0.01, 0.05)
            trackio.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            time.sleep(0.2)
        trackio.finish()

simulate_multiple_runs()
```

#### 可视化结果

完成实验记录后，可通过命令行启动本地仪表板查看结果：

```bash
trackio show
```

或在 Python 中调用：

```python
import trackio
trackio.show()
```

支持指定项目名称：

```bash
trackio show --project "my project"
```

Python 调用方式：

```python
trackio.show(project="my project")
```

> **提示**：仪表板支持多项目管理，便于对比不同实验。

#### 通过 Hugging Face Spaces 共享

只需在 `init` 时传入 `space_id`，即可将本地仪表板同步至 Hugging Face Spaces：

```python
trackio.init(project="fake-training", space_id="org_name/space_name")
```

同步后，可通过 URL 直接分享，或使用 iframe 嵌入网页：

```html
<iframe
  src="https://jianzhnie-llmreasoning.hf.space/?project=trackio-demo&amp;metrics=train/loss,train/accuracy&amp;sidebar=hidden"
  width="600"
  height="600"
  frameborder="0"
  style="border-radius: 8px; border: 1px solid #e0e0e0; margin: 20px 0;">
</iframe>
```

<iframe
  src="https://jianzhnie-llmreasoning.hf.space/?project=trackio-demo&amp;metrics=train/loss,train/accuracy&amp;sidebar=hidden"
  width="600"
  height="600"
  frameborder="0"
  style="border-radius: 8px; border: 1px solid #e0e0e0; margin: 20px 0;">
</iframe>
Spaces 支持公开或组织内私有部署，所有功能均免费使用。

#### 数据持久化机制

当仪表板部署在 Spaces 上时，数据默认存储于临时 SQLite 数据库中。为防止因实例重启导致数据丢失，Trackio 每 5 分钟自动将 SQLite 数据导出为 Parquet 格式，并备份至 Hugging Face Dataset。用户可随时访问该数据集进行深度分析或长期归档。

> **提示**：可通过 `dataset_id` 参数自定义备份数据集的名称：
> ```python
> trackio.init(project="my-project", space_id="org/space", dataset_id="org/my-dataset")
> ```

### 与 Hugging Face 生态深度集成

Trackio 原生支持 `transformers` 和 `accelerate` 等 Hugging Face 核心库，几乎无需额外配置即可启用指标追踪。

#### 集成 `transformers.Trainer`

```python
import numpy as np
from datasets import Dataset
from transformers import Trainer, AutoModelForCausalLM, TrainingArguments

# 构造示例数据集
data = np.random.randint(0, 1000, (8192, 64)).tolist()
dataset = Dataset.from_dict({"input_ids": data, "labels": data})

# 使用 Trainer 进行训练
trainer = Trainer(
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B"),
    args=TrainingArguments(run_name="fake-training", report_to="trackio"),
    train_dataset=dataset,
)
trainer.train()
```

#### 集成 `accelerate`

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with="trackio")
accelerator.init_trackers("fake-training")

# 准备模型、数据加载器等

for step, batch in enumerate(dataloader):
    # 训练逻辑
    accelerator.log({"training_loss": loss}, step=step)

accelerator.end_training()
```

无需额外封装或配置，开箱即用。

## Trackio API 参考文档

本文档详细介绍了 Trackio 库的核心类与函数接口。

### `trackio.Run`

```python
class trackio.Run(url: str, project: str, client: gradio_client.Client | None, name: str | None = None, config: dict | None = None, space_id: str | None = None)
```

表示一次实验运行（Run）的实例，用于管理指标记录与会话生命周期。

#### 方法：`finish()`

```python
finish()
```

清理当前运行的相关资源，结束本次实验追踪。建议在训练流程结束时显式调用。

### `trackio.init`

```python
trackio.init(
    project: str,
    name: str | None = None,
    space_id: str | None = None,
    dataset_id: str | None = None,
    config: dict | None = None,
    resume: str = 'never',
    settings: Any = None
) -> trackio.Run
```

初始化一个新的实验项目，并返回一个 `Run` 对象，用于后续的日志记录。

#### 参数说明

- **`project`** (`str`)
  项目名称。若项目已存在，则继续向该项目追加记录；否则创建新项目。

- **`name`** (`str`, 可选)
  当前运行的名称。若未提供，系统将自动生成唯一名称。

- **`space_id`** (`str`, 可选)
  指定 Hugging Face Space 的标识符（如 `"username/space_name"` 或 `"orgname/space_name"`）。若提供，所有日志将同步至该 Space 而非本地目录。若 Space 不存在，则自动创建。

- **`dataset_id`** (`str`, 可选)
  若指定了 `space_id`，可同时指定用于持久化存储的 Hugging Face Dataset 标识符（如 `"username/dataset_name"`）。若未提供，系统将默认创建一个与 Space 同名但后缀为 `_dataset` 的 Dataset。

- **`config`** (`dict`, 可选)
  实验配置参数字典（如超参数），用于兼容 `wandb.init()` 接口。

- **`resume`** (`str`, 可选，默认 `'never'`)
  控制运行恢复策略：
  - `'must'`：必须恢复指定名称的运行，若不存在则报错。
  - `'allow'`：若运行存在则恢复，否则创建新运行。
  - `'never'`：始终创建新运行，不恢复。

- **`settings`** (`Any`, 可选)
  预留参数，当前未使用，仅为兼容 `wandb.init()` 而保留。

#### 返回值

返回一个 [`trackio.Run`](#class-trackiorun) 实例，可用于记录指标和管理运行生命周期。

### `trackio.log`

```python
trackio.log(metrics: dict, step: int | None = None)
```

向当前运行中记录一组指标。

#### 参数说明

- **`metrics`** (`dict`)
  待记录的指标字典，键为指标名，值为数值。

- **`step`** (`int`, 可选)
  当前训练步数。若未提供，系统将自动递增步数计数器。

### `trackio.finish`

```python
trackio.finish()
```

结束当前运行，释放相关资源。等效于对当前 `Run` 实例调用 `finish()` 方法。

### `trackio.show`

```python
trackio.show(project: str | None = None, theme: str | gradio.themes.ThemeClass = 'citrus')
```

启动本地 Trackio 可视化仪表板。

#### 参数说明

- **`project`** (`str`, 可选)
  指定要展示的项目名称。若未提供，则显示所有项目供用户选择。

- **`theme`** (`str` 或 `ThemeClass`, 可选，默认 `'citrus'`)
  仪表板使用的视觉主题。可选内置主题（如 `'soft'`, `'default'`）、Hub 上的主题（如 `"gstaff/xkcd"`）或自定义主题类。

### `trackio.import_csv`

```python
trackio.import_csv(
    csv_path: str | pathlib.Path,
    project: str,
    name: str | None = None,
    space_id: str | None = None,
    dataset_id: str | None = None
)
```

从 CSV 文件导入实验数据至 Trackio 项目。

#### 参数说明

- **`csv_path`** (`str` 或 `Path`)
  CSV 文件路径。

- **`project`** (`str`)
  目标项目名称。该项目必须为新项目，不能与现有项目同名。

- **`name`** (`str`, 可选)
  导入运行的名称。若未提供，系统将生成默认名称。

- **`space_id`** (`str`, 可选)
  若指定，则将数据导入至对应的 Hugging Face Space。

- **`dataset_id`** (`str`, 可选)
  若指定，则同步至对应的 Hugging Face Dataset。若未提供且指定了 `space_id`，则自动创建同名 `_dataset`。

#### 数据格式要求

CSV 文件必须包含：
- `step` 列：表示训练步数。
- （可选）`timestamp` 列：时间戳。
- 其他列视为指标。
- 首行为列名（header）。

> **注意**：当前版本导入后不返回 `Run` 对象，暂不支持继续追加日志（待实现）。

### `trackio.import_tf_events`

```python
trackio.import_tf_events(
    log_dir: str | pathlib.Path,
    project: str,
    name: str | None = None,
    space_id: str | None = None,
    dataset_id: str | None = None
)
```

从 TensorFlow Events 日志目录导入数据。

#### 参数说明

- **`log_dir`** (`str` 或 `Path`)
  包含 TensorBoard Events 文件的目录路径。

- **`project`** (`str`)
  目标项目名称，必须为新项目。

- **`name`** (`str`, 可选)
  运行名称前缀。每个子目录将被导入为一个独立运行，名称基于子目录名生成。

- **`space_id`** (`str`, 可选)
  指定目标 Space，实现云端同步。

- **`dataset_id`** (`str`, 可选)
  指定持久化 Dataset，用于长期存储。

#### 导入规则

- `log_dir` 下的每个子目录被视为一次独立运行。
- 自动解析 Events 文件中的标量指标并导入。

## Reference

- https://huggingface.co/docs/trackio/index
- https://github.com/gradio-app/trackio
