# SimPO 高效优化策略

在偏好学习任务中，将“优选”（chosen）和“劣选”（rejected）样本拼接为单一批次进行一次前向传播，虽然实现简单，但会使微批次大小翻倍，从而显著增加显存消耗。尤其在训练大模型时，这种做法可能成为显存瓶颈。

通过将计算过程拆分为两次独立的前向传播，并累积梯度，可以在不改变算法数学本质的前提下，有效降低峰值显存占用。该策略确保最终的梯度更新与原始方法完全一致，是大规模模型训练中广泛应用的一种显存优化技术，其本质是**梯度累积（Gradient Accumulation）** 的灵活应用。

---

## 1. 整体设计目标

本优化方案的核心目标是在**不改变 SimPO 算法数学逻辑**的前提下，重构模型的前向传播与损失计算流程。具体而言，将原本在一个微批次中同时处理 `chosen` 与 `rejected` 数据的操作，拆解为两个独立、顺序执行的前向传播步骤。

关键设计点如下：

- **梯度累积**：在两次前向传播之间不调用 `optimizer.zero_grad()`，确保两次计算产生的梯度能够累积。
- **单次参数更新**：在完成两次前向与反向传播后，仅执行一次 `optimizer.step()`。
- **显存优化**：将峰值显存占用降低约 50%，从而在有限显存条件下支持更大批次或更长序列的训练。

此方法在显存效率与训练正确性之间实现了良好平衡。

---

## 2. 输入/输出接口说明

### **输入参数及格式**

- **原始方法**：`forward_step` 接收一个拼接后的单一批次数据，其尺寸为 `B × 2`（即 `B` 个 `chosen` 和 `B` 个 `rejected` 样本合并）。
- **优化后方法**：`forward_step` 将处理两个独立的微批次：
  - 一个大小为 `B` 的 `chosen_batch`
  - 一个大小为 `B` 的 `rejected_batch`

这两个批次可由 `get_batch` 函数直接返回，或在 `forward_step` 内部从拼接批次中拆分得到。

### **返回值类型与语义**

- 优化后的 `forward_step` 不再返回单一损失值，而是负责完成两次前向传播、损失计算与梯度累积。
- 为兼容外部训练循环，函数可返回组合后的损失张量（如 `losses.mean()`）或度量指标（metrics），用于日志记录。
- 实际的 `backward()` 操作可在 `forward_step` 内部执行，或由主训练循环调用。

---

## 3. 关键逻辑实现（按执行顺序）

我们重构 `forward_step` 函数，避免将 `chosen` 和 `rejected` 数据拼接后一次性送入模型。

### **步骤1：数据拆分与加载**

```python
# 修改前（原始方法）：
# tokens, labels, attention_mask, position_ids = self.get_batch(data_iterator)
# output_tensor = model(tokens, position_ids, attention_mask)  # 输入为拼接批次

# 修改后（优化方法）：
chosen_batch, rejected_batch = self.get_batch(data_iterator)
```

原代码中通过 `self.args.actual_micro_batch_size = self.args.micro_batch_size * 2` 实现数据拼接。为实现显存优化，需调整 `get_batch` 返回两个独立批次，或在 `forward_step` 中显式拆分输入数据。

### **步骤2：两次独立前向传播**

```python
# 第一次前向：处理 chosen 数据
chosen_output = model(
    chosen_batch['tokens'],
    chosen_batch['position_ids'],
    chosen_batch['attention_mask']
)

# 第二次前向：处理 rejected 数据
rejected_output = model(
    rejected_batch['tokens'],
    rejected_batch['position_ids'],
    rejected_batch['attention_mask']
)
```

这是显存优化的核心。通过分别处理两个批次，GPU 上的激活值（activations）仅需保留一个批次的中间状态，从而将峰值显存需求降低约 50%。

### **步骤3：分别计算对数概率**

```python
chosen_log_probs, _, _ = self._compute_log_probs(
    chosen_output, chosen_batch['labels']
)

_, rejected_log_probs, _ = self._compute_log_probs(
    rejected_output, rejected_batch['labels']
)
```

复用原有的 `_compute_log_probs` 函数，分别传入两次前向输出与对应标签，即可得到 `chosen_log_probs` 和 `rejected_log_probs`。

### **步骤4：计算 SimPO 损失**

```python
losses, chosen_rewards, rejected_rewards = self.simpo_loss(
    chosen_log_probs, rejected_log_probs
)
```

`simpo_loss` 函数的输入正是 `chosen_log_probs` 和 `rejected_log_probs`，与拆分后的输出完美匹配。因此，最终计算出的损失值与原始拼接方法**完全等价**，保证了算法的数学一致性。

### **步骤5：梯度累积与参数更新**

在主训练循环中，封装新的训练步进逻辑：

```python
def train_step(self, data_iterator, model, optimizer):
    optimizer.zero_grad()  # 清零梯度（仅在每轮累积开始时调用一次）

    # Step 1: 前向传播 - chosen
    chosen_batch = next(data_iterator)
    chosen_logits = model(chosen_batch['tokens'], ...)
    chosen_log_probs, _, _ = self._compute_log_probs(chosen_logits, chosen_batch['labels'])

    # Step 2: 前向传播 - rejected
    rejected_batch = next(data_iterator)
    rejected_logits = model(rejected_batch['tokens'], ...)
    _, rejected_log_probs, _ = self._compute_log_probs(rejected_logits, rejected_batch['labels'])

    # Step 3: 计算联合损失
    losses = self.simpo_loss(chosen_log_probs, rejected_log_probs)

    # Step 4: 反向传播（累积梯度）
    losses.mean().backward()

    # Step 5: 更新参数
    optimizer.step()
```

在 Megatron-LM 等框架中，`forward_step` 可返回 `(chosen_log_probs, rejected_log_probs)` 元组，由外部 `train_step` 统一调用 `backward()`，实现解耦与灵活性。

---

## 拆分计算是否保证梯度正确性？

这是一个关键问题，涉及深度学习框架中自动微分与梯度计算的底层机制。

**结论：是的，该拆分方法能保证梯度的完全正确性，最终更新与原始方法等价。**

### **核心原理：梯度累积机制**

PyTorch 的自动微分（Autograd）引擎默认采用**梯度累加**模式。其行为如下：

1. **前向传播与计算图构建**：
   - 第一次前向传播（`chosen`）构建计算图 A，保存中间激活值。
   - 第二次前向传播（`rejected`）构建独立计算图 B，同样保存激活值。
   - 由于未调用 `zero_grad()`，模型参数的 `.grad` 字段保持可累加状态。

2. **损失聚合与反向传播**：
   - `simpo_loss` 函数接收两个独立输出，计算出最终损失 $ L $。
   - 当调用 `L.backward()` 时，Autograd 会同时追溯图 A 和图 B，计算两部分对模型参数的梯度，并将其累加至 `.grad` 属性。

### **数学等价性证明**

设模型参数为 $ \theta $，原始拼接方法的损失为：

$$
L_{\text{original}} = \text{SimPO\_Loss}\left(f([x_c, x_r]; \theta)\right)
$$

拆分方法中，损失为：

$$
L_{\text{split}} = \text{SimPO\_Loss}\left(f(x_c; \theta), f(x_r; \theta)\right)
$$

由于 SimPO 损失函数的结构仅依赖于 `chosen` 和 `rejected` 的对数概率，且两次前向共享参数 $ \theta $，故有：

$$
L_{\text{original}} = L_{\text{split}}
$$

根据链式法则，梯度为：

$$
\nabla_\theta L = \nabla_\theta \text{SimPO\_Loss} = \frac{\partial \text{SimPO\_Loss}}{\partial \log p_c} \cdot \frac{\partial \log p_c}{\partial \theta} + \frac{\partial \text{SimPO\_Loss}}{\partial \log p_r} \cdot \frac{\partial \log p_r}{\partial \theta}
$$

这正是两次独立反向传播结果的**梯度和**，与拼接方法完全一致。

### **显存优化与分布式框架协同**

通过该策略，GPU 在任一时刻仅需保存一个批次的激活值，显著降低显存峰值。在 Megatron-LM 等框架中，此优化与张量并行（TP）、上下文并行（CP）无缝集成。框架在后台处理分布式通信，但梯度累积机制确保了无论数据如何切分，最终的梯度都能正确同步与更新。

---

综上所述，该优化方案在**不牺牲算法正确性**的前提下，通过**梯度累积**实现了显著的显存节省，是大模型偏好训练中一项高效且可靠的工程实践。
