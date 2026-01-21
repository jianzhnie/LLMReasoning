# SimPO 代码解析

> SimPO 代码实现： https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/mindspeed_llm/tasks/posttrain/dpo/simpo_trainer.py

## 整体概述

`SimPOTrainer` 类实现了 SimPO（Simple Preference Optimization，简单偏好优化）算法，这是一种专为大型语言模型（LLMs）设计的离线偏好学习方法。其核心目标是利用成对的人类偏好数据（例如，一对“优选”和“劣选”的回答）来进一步微调模型，使其生成的回答在指令遵循、安全性和帮助性上更符合人类标准。SimPO 的提出标志着大型语言模型训练范式从传统、复杂且计算成本高昂的强化学习（如 RLHF）向更简洁、高效的**端到端监督学习**的演进趋势。与 DPO（Direct Preference Optimization）等方法类似，SimPO 将强化学习中的奖励最大化问题重构为一个监督损失最小化问题，从而避免了强化学习训练过程中常见的复杂性和不稳定性。

## 关键概念与知识点

- **SimPO（Simple Preference Optimization）**：一种新颖的偏好优化方法，通过简化对齐训练过程，仅依赖于单个模型（即 **policy model**）进行训练。该方法无需额外的 **reference model**（参考模型）来计算奖励，从而降低了计算开销和内存占用。其核心思想是通过直接比较 policy model 对“偏好”和“拒绝”响应的对数概率比值来构建损失函数。
- **DPO（Direct Preference Optimization）**：另一种流行的偏好对齐算法。SimPO 是在 DPO 基础上的简化与改进。
- **Megatron-LM**：NVIDIA 开发的大型模型训练框架，提供高效的**多维并行训练**能力，包括张量并行（Tensor Parallelism）、上下文并行（Context Parallelism）和数据并行（Data Parallelism）。这些并行策略共同解决了训练超大模型或处理长序列时单个 GPU 内存不足的根本瓶颈。
- **张量并行（Tensor Parallelism）**：一种分布式训练策略，将单个模型的权重矩阵和计算分布在多个 GPU 上，以适应超大模型的训练需求。代码中的 `mpu`（Megatron Process Utilities）模块用于处理张量并行相关的操作。
- **函数式编程技巧**: `functools.partial` 的使用是一种优雅的编程方式，它将函数和其部分参数打包，延迟计算，使代码模块化程度更高。
- **`torch.nn.functional`**：PyTorch 提供的函数库，包含激活函数、损失函数等无状态函数。代码中使用了 `F.logsigmoid` 和 `torch.relu` 等函数。

## 代码逐段解析

### 1. 导入与类定义

```python
import os
from typing import Dict, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from megatron.training import get_args
from megatron.core import mpu
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.dpo import DPOTrainer
from mindspeed_llm.tasks.posttrain.utils import vocab_parallel_log_softmax

class SimPOTrainer(BaseTrainer):
    """
    A trainer class for Simple Preference Optimization (SimPO).
    ...
    """
    IGNORE_INDEX = -100
```

- **`import`**：导入必要的模块，包括操作系统接口 `os`、类型提示 `typing`、函数工具 `functools`、PyTorch 框架 `torch`，以及 Megatron 训练和核心模块。
- **`class SimPOTrainer(BaseTrainer)`**：定义 `SimPOTrainer` 类，继承自 `BaseTrainer`，表明其为用于 SimPO 任务的基础训练器的特化版本。
- **`IGNORE_INDEX = -100`**：定义常量，用于在处理标签时忽略特定 token，通常用于填充（padding）token。

### 2. 初始化方法

```python
    def __init__(self):
        """
        Initializes the SimPOTrainer instance.
        ...
        """
        super().__init__()
        self.args.actual_micro_batch_size = self.args.micro_batch_size * 2
```

- **`super().__init__()`**：调用父类 `BaseTrainer` 的初始化方法，继承其属性和行为。

在 `__init__` 方法中，`SimPOTrainer` 将实际微批次大小（`actual_micro_batch_size`）设置为设定值（`micro_batch_size`）的两倍。这一设计反映了 SimPO/DPO 这类**成对偏好学习**算法与传统监督微调（SFT）的本质区别。在 SFT 中，一个批次通常只包含单一类型的样本，在偏好学习中，数据通常是**成对**出现的（一个 "chosen" 回答和一个 "rejected" 回答）。如果 `micro_batch_size` 指的是 "偏好对" 的数量，那么实际送入模型进行计算的样本数量是其两倍。这行代码就是为了正确设置实际的批次大小，以确保资源（如内存）被正确计算和分配。

### 3. 损失函数

```python
    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """SimPO Loss function.
        ...
        """
        args = get_args()
        all_policy_logits = output_tensor
        labels = input_tensor
        loss, metrics = self.get_batch_loss_metrics(
            all_policy_logits,
            labels
        )
        ...
        # Reduce loss for logging.
        metrics['lm loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])
        return loss, metrics
```

该方法作为训练步骤的入口，负责调度损失计算和指标收集。

- **输入参数**:
  - `input_tensor (torch.Tensor)`：标签张量。根据代码逻辑，该张量包含用于损失计算的标签序列，其内容是优选（chosen）和劣选（rejected）数据的拼接。
  - `output_tensor (torch.Tensor)`：模型输出的 Logits 张量，是模型前向传播的结果，包含模型对每个词汇在词表上的预测分数。
- **返回值**:
  - `loss (torch.Tensor)`：一个标量损失值，代表经过数据并行（DP）聚合后的平均损失，用于指导反向传播和模型参数更新。
  - `metrics (Dict)`：包含各种训练指标（如 `'lm loss'` 和 `'rewards/accuracies'`）的字典。这些指标同样是经过数据并行聚合后的平均值，用于监控训练过程和评估模型性能。

- **`loss, metrics = self.get_batch_loss_metrics(...)`**：调用 `get_batch_loss_metrics` 方法计算实际的 SimPO 损失及相关指标，这是核心逻辑所在。
- **`average_losses_across_data_parallel_group`**：Megatron 框架的实用函数，用于在数据并行组内对损失进行平均，确保在分布式训练中每个 GPU 上的损失被正确汇总，得到全局平均损失。

在分布式训练中，单个进程的损失出现 `NaN` 会导致整个训练失败。因此，代码在执行 `average_losses_across_data_parallel_group` 前加入了对 `NaN` 的检查，这是一种重要的**健壮性设计**，用于在分布式规约（all-reduce）前尽早捕获并报告异常。该函数是 Megatron-LM 框架中实现**数据并行（Data Parallelism）**的核心工具，通过在所有数据并行组内的 GPU 之间执行 `all_reduce` 操作对损失和指标进行平均，确保每个 GPU 上的梯度基于整个批次（而非本地数据分片）计算，从而实现同步梯度下降。

### 4. 前向传播

```python
    def forward_step(self, data_iterator, model):
        """SimPO Forward training step.
        ...
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, attention_mask, position_ids = self.get_batch(data_iterator)
        self.timers('batch-generator').stop()
        output_tensor = model(tokens, position_ids, attention_mask)
        return output_tensor, partial(self.loss_func, labels)
```



- `get_batch(data_iterator)`: 从数据迭代器中获取一个批次的数据。值得注意的是，`get_batch` 方法直接复用了 `DPOTrainer` 的实现，这表明 SimPO 和 DPO（Direct Preference Optimization）在数据处理上是兼容的。获取的数据包括 `tokens`（模型输入）、`labels`（期望输出）等。
- `output_tensor = model(...)`: 将 `tokens` 输入到模型中，执行标准的前向传播，得到模型的原始输出 `output_tensor`（通常是 logits，即未经过 Softmax 的概率分布）。
- `return output_tensor, partial(self.loss_func, labels)`: 这是 Megatron-LM 框架中一种常见的设计模式。它不直接在 `forward_step` 中计算损失，而是返回模型输出和**一个准备好计算损失的偏函数（partial function）**
- `partial(self.loss_func, labels)` 创建了一个新函数，该函数只需要再接收一个 `output_tensor` 参数就可以执行损失计算。这种分离使得训练流程更加灵活和清晰。

### 5. SimPO 损失计算核心

```python
def simpo_loss(
    self,
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
    """
    Compute the SimPO loss for a batch of policy model log probabilities.
    """
    pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs

    logits = pi_log_ratios - self.args.gamma_beta_ratio

    if self.args.simpo_loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(self.args.simpo_beta * logits) * (1 - self.args.simpo_label_smoothing)
            - F.logsigmoid(-self.args.simpo_beta * logits) * self.args.simpo_label_smoothing
        )
    # ... other loss types ...

    chosen_rewards = (self.args.simpo_beta * policy_chosen_log_probs.detach())
    rejected_rewards = (self.args.simpo_beta * policy_rejected_log_probs.detach())

    return losses, chosen_rewards, rejected_rewards
```

这是 SimPO 算法的**核心实现**。

- `pi_log_ratios`: 计算 "chosen" 回答和 "rejected" 回答的对数概率之差 `log P(chosen) - log P(rejected)`。这个差值反映了当前模型对这两个回答的偏好程度。差值越大，说明模型越倾向于 "chosen" 回答。

- `logits = pi_log_ratios - self.args.gamma_beta_ratio`: 在 DPO-style 的算法中，我们希望 `pi_log_ratios` 尽可能大。这里引入了一个**奖励边际（reward margin）** `gamma/beta`。目标是让 `pi_log_ratios` 不仅大于0，而且要大于这个设定的边际，从而更好地将好的和坏的回答区分开。

- **不同损失类型 (`sigmoid`, `hinge`, `ipo`)**:
  - **`sigmoid`**：这是最核心的损失形式，类似于逻辑回归的损失。它使用 `logsigmoid` 函数来惩罚那些 `logits` 不够大的样本。代码还考虑了**标签平滑（label smoothing）**，这是一种正则化技术，可以防止模型对标签过于自信，提高泛化能力。
  - **`hinge`**：使用 hinge 损失，一种非平滑的损失函数。若 $1 - \beta \cdot \text{logits} > 0$，则产生损失，否则为 0。
  - **`ipo`**：使用 **IPO (Identity Preference Optimization)** 损失，基于均方误差（MSE）的损失形式。

- `chosen_rewards`, `rejected_rewards`: 计算隐式奖励（implicit rewards）。在偏好学习中，奖励值不是显式给定的，而是通过模型的对数概率派生出来的。这里用 `beta * log_probs` 作为奖励的代理。使用 `.detach()` 是因为这些奖励值仅用于监控和分析，我们不希望它们的梯度影响模型的训练。



#### SimPO 算法的损失计算

该函数首先计算优选（chosen）和劣选（rejected）响应的平均对数概率之差。这是 SimPO 算法中的关键对比信号。SimPO 的核心在于**使用自身的平均对数概率作为隐式奖励**，即：

$$
r_{\text{SimPO}}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)
$$
这与 DPO 使用优选/劣选响应与参考模型之间的对数比率作为奖励：

$$
\beta \log \left( \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right)
$$
有本质区别。SimPO 的设计消除了对额外参考模型的依赖，显著节省了计算和内存资源，使其训练更加高效。

### 6. 批次损失与指标计算

```python
def get_batch_loss_metrics(
        self,
        all_policy_logits,
        label
) -> Tuple[torch.Tensor, Dict]:
    """
    Computes the sum log probabilities of the labels under the given logits.
    """
    # ...
    (
        policy_chosen_log_probs,
        policy_rejected_log_probs,
        policy_chosen_log_probs_avg,
    ) = self._compute_log_probs(all_policy_logits, label)

    losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
        policy_chosen_log_probs,
        policy_rejected_log_probs,
    )

    sft_loss = -policy_chosen_log_probs_avg
    if self.args.pref_ftx > 1e-6:
        losses += self.args.pref_ftx * sft_loss

    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    # ...
    metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean()
    # ...

    return losses.mean(), metrics
```

这个函数将各个部分组合起来，计算最终的损失和监控指标。

- `_compute_log_probs(...)`: 调用辅助函数，从模型的原始 `logits` 和 `labels` 中计算出 "chosen" 和 "rejected" 回答的**序列对数概率**。
- `compute_preference_loss(...)`: 调用我们上面分析的 `simpo_loss` 函数，得到核心的偏好损失。
- `sft_loss = -policy_chosen_log_probs_avg`: 计算一个额外的 **SFT (Supervised Fine-Tuning) 损失**。这相当于在 "chosen" 数据上做一个标准的语言模型训练。
- `losses += self.args.pref_ftx * sft_loss`: 将 SFT 损失以一定的权重（`pref_ftx`）加到总损失中。这是一个重要的技巧，可以**防止模型在学习偏好的过程中遗忘其基础的语言能力**，确保生成质量。
- `reward_accuracies`: 计算奖励准确率，即模型认为 "chosen" 回答的奖励高于 "rejected" 回答的比例。这是一个非常直观的监控指标，用来衡量模型是否在朝着正确的方向学习。
- `return losses.mean(), metrics`: 返回批次的平均损失（用于反向传播）和包含各种监控指标的字典（用于日志记录）。

### 7. 批次对数概率计算（支持分布式处理）

```python
def _get_batch_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ...
    if mpu.get_tensor_model_parallel_world_size() > 1:
        # Tensor Parallelism logic
        # ...
        torch.distributed.all_reduce(...)
    else:
        # Single GPU / Data Parallelism logic
        # ...

    if mpu.get_context_parallel_world_size() > 1:
        # Context Parallelism logic
        torch.distributed.all_reduce(...)

    return all_log_probs, valid_length
```

- 这是一个健壮的函数，同时支持**张量并行（Tensor Parallelism）**和非张量并行的情况。
- **`mpu.get_tensor_model_parallel_world_size() > 1`**：检查是否启用了张量并行。
- **张量并行下的处理**：
  - **`labels -= mpu.get_tensor_model_parallel_rank() * tp_vocab_size`**：由于词表（vocab）在张量并行中被切分，需根据当前 GPU 的 `rank` 对标签索引进行偏移，以匹配其拥有的词表分片。
  - **`vocab_parallel_log_softmax(logits)`**：Megatron 框架提供的特殊函数，用于在张量并行模式下对分布在多个 GPU 上的 logits 正确执行 `log_softmax` 计算。
  - **`torch.distributed.all_reduce(...)`**：在张量并行模式下，每个 GPU 仅计算其本地词表分片上的对数概率。为获得完整序列的对数概率和有效长度，需使用 `all_reduce` 操作对所有 GPU 的结果进行求和。
- **非张量并行下的处理**：直接使用 `logits.log_softmax(-1)` 计算对数概率。
- **上下文并行下的处理**：若启用了上下文并行，同样需使用 `all_reduce` 操作来汇总对数概率和有效长度。

该段代码首先判断是否启用了**张量并行（Tensor Parallelism, TP）**。在 TP 模式下，模型权重（尤其是词嵌入和输出层）及词表被切分并分布到多个 GPU 上。为在 `torch.gather` 操作中正确索引词汇，需根据当前 GPU 的 `rank` 对标签 ID 进行偏移调整。

`vocab_parallel_log_softmax` 是 Megatron-LM 专为 TP 设计的 softmax 计算函数。它首先在所有 TP 进程间执行 `all_reduce` 操作同步 logits，再进行 softmax 归一化。这确保了即使词表被分片，每个 GPU 上的 `log_softmax` 值也是基于完整词表计算的。

在 TP 模式下，每个 GPU 仅计算其本地词表分片上的对数概率。为获得完整序列的对数概率，需将所有 TP 进程上的局部结果通过 `all_reduce`（`SUM` 模式）进行聚合。这种**“先局部计算，后全局同步”**的设计是张量并行的核心，它最大化了计算与通信的重叠效率，是大型模型分布式训练的基石。

```python
if mpu.get_context_parallel_world_size() > 1:
    torch.distributed.all_reduce(...)
```

该条件判断是否启用了**上下文并行（Context Parallelism, CP）**。CP 将长序列切分到不同 GPU 上，以缓解长序列带来的显存压力。CP 模式下的 `all_reduce` 同样以 `SUM` 模式聚合在不同 GPU 上分片计算出的 `all_log_probs` 和 `valid_length`，从而确保总损失覆盖整个长序列。TP 和 CP 可同时启用，形成**多维并行**，以应对超大模型和超长序列的双重挑战。

### 8. 超参数与训练策略

`SimPOTrainer` 提供了丰富的超参数以灵活控制训练行为。

| 参数名                  | 作用                                 | 备注                                         |
| ----------------------- | ------------------------------------ | -------------------------------------------- |
| `simpo_beta`            | 控制奖励值缩放，影响损失函数陡峭程度 | SimPO 中通常需比 DPO **大得多**，可能高达 10 |
| `gamma_beta_ratio`      | 设定优选与劣选响应的奖励边际         | 鼓励模型建立更明确的区分度                   |
| `simpo_label_smoothing` | 引入标签不确定性，用于平滑损失       | 默认值为 0，可根据需要调整                   |
| `pref_ftx`              | 控制 SFT 损失的权重                  | 用于在偏好训练中保留模型的通用语言能力       |

代码中的 `self.args.actual_micro_batch_size = self.args.micro_batch_size * 2` 不仅反映了数据格式要求，更体现了一种**训练策略**。它与**临界批次大小（Critical Batch Size, CBS）**的研究相关。这种设计表明训练者在训练效率与模型性能之间进行了权衡，并采用了适合偏好学习任务的数据组织方式。

## 总结与对比

SimPO 算法的核心优势在于对 DPO 的简化与改进。

| 算法名称 | 核心思想                     | 是否需要参考模型 | 损失函数类型                                   | 训练效率 | 优劣势                                         |
| -------- | ---------------------------- | ---------------- | ---------------------------------------------- | -------- | ---------------------------------------------- |
| SimPO    | 平均对数概率作为隐式奖励     | 否               | $- \log \sigma(\beta (r_w - r_l) - \gamma)$    | 高       | 简洁、高效，但可能缺乏 KL 惩罚，对超参敏感     |
| DPO      | 对数概率比作为隐式奖励       | 是               | –                                              | 中       | 稳定、广泛应用，但需额外维护参考模型           |
| ORPO     | SFT 损失与对数赔率比损失结合 | 否               | $L_{\text{SFT}} + \lambda \cdot L_{\text{OR}}$ | 高       | 单步融合、效果优秀，但在特定任务上可能表现不佳 |

DPO 损失函数公式为：

$$
-\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)
$$

**无参考模型（Reference-Free）**：SimPO 摒弃了 DPO 所需的参考模型（通常为 SFT 模型），直接使用当前策略模型（Policy Model）的平均对数概率作为隐式奖励。这一设计显著降低了训练的内存与计算开销，无需在训练过程中加载和维护额外模型。

**长度归一化（Length Normalization）**：SimPO 的隐式奖励采用**平均**对数概率，即 $ r_{\text{SimPO}}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y|x) $，而非总和。这一设计有效缓解了“长度偏差”问题——即更长序列因包含更多 token 而总对数概率更低，导致模型倾向于生成短答案。长度归一化通过将总概率除以序列长度，纠正了这一偏差，使奖励机制与模型解码目标更一致。
