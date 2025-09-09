## 摘要

在可验证奖励强化学习（RLVR）中，通常采用 **Pass@1** 作为奖励函数。然而，这种设置在探索与利用的平衡上存在挑战，容易导致模型策略倾向于保守动作，并陷入局部最优。因此，如何设计一个更有效的奖励指标至关重要。



## 引言：Pass@k 作为奖励的新探索


虽然 **Pass@k** 指标在模型评估中已有广泛应用，但其与大型语言模型（LLMs）探索能力之间的关联却被长期忽视。为了系统地探究这一关联，我们首先将 **Pass@k** 作为奖励函数来训练策略模型（我们称之为“**Pass@k 训练**”），并观察其对模型探索能力的提升效果。

在此基础上，我们进一步推导了 **Pass@k 训练** 优势函数（advantage function）的解析解，这不仅显著提升了训练的效率和有效性，还为我们带来了全新的洞察：**探索（exploration）与利用（exploitation）并非是天然冲突的目标，它们可以相互促进，共同增强**。

此外，我们的分析表明，采用带有解析解的 **Pass@k 训练** 本质上是一种**直接设计优势函数**的方法。受此启发，我们初步探索了在 RLVR 框架下直接设计优势函数的可能性，并取得了令人鼓舞的初步成果，这为未来的研究开辟了一个全新的方向。

## Pass@k 训练详解

### 1. Pass@k 指标的定义



给定一个问题 $x$，我们使用策略模型 $π_θ$ 通过特定的解码策略（例如，基于采样的解码或蒙特卡洛树搜索 MCTS）生成 $k$ 个候选答案 $y^1,…,y^k$。每个候选答案 $y^i$ 会通过一个验证器获得一个奖励 $R_i$。

**Pass@k** 指标被定义为从这 $k$ 个候选答案中获得的**最大奖励的期望值**。其数学表达式如下：

$$
\text{Pass@k} = \mathbb{E}\_{(x,y)\sim D,\\{\hat{y}\_i\\}_{i=1}^K\sim \pi\_\theta(\cdot|x)}\left[\max\left(R_1, \dots, R_K)\right)\right].
$$


简单来说，只要在这 $k$ 个答案中，有至少一个答案是正确的（即奖励为1），那么这次采样就获得了满分奖励。这鼓励模型去生成更多样的、更有可能包含正确答案的候选集，从而增强其探索能力。

### 2. Pass@k 训练与优势函数的解析解


为了进一步提升 **Pass@k 训练** 的效率和效果，我们不再依赖于传统的强化学习中的基线（baseline）来估计优势函数，而是直接推导出了一个解析解。这个解析解可以更精确地量化每个答案对最终 **Pass@k** 奖励的贡献。

我们计算了**正向答案（positive responses）**和**负向答案（negative responses）**的优势值解析解。这里，$\bar{R}^{\text{group}}$ 表示一个批次中所有采样（rollout）的平均奖励，$\sigma^\text{group}$ 是其标准差。$N_{rollout} $ 是总采样数，$N_{neg} $ 是负向答案的数量。

**解析解推导**：

首先，我们定义一个批次中 Pass@k 奖励的平均值和标准差：


$$
\bar{R}^{\text{group}}=1-\frac{\binom{N\_\text{neg}}{k}}{\binom{N\_\text{rollout}}{k}}
$$

$$
\sigma^\text{group}=\sqrt{\bar{R}^\text{group}\times\left(1-\bar{R}^\text{group}\right)}
$$

基于均值$\bar{R}^{\text{group}}$ 和标准差 $\sigma^\text{group}$，我们可以计算正向和负向答案的优势值。

$$
\hat{A}\_{\text{pos}}=\frac{1-\bar{R}^{\text{group}}}{\sigma^{\text{group}}}
$$

$$
\hat{A}\_{\text{neg}}=\left(1-\bar{R}^\text{group}-\frac{\binom{N\_\text{neg}-1}{k-1}}{\binom{N\_\text{rollout}-1}{k-1}}\right)\times\left(\sigma^\text{group}\right)^{-1}
$$


通过这个解析解，我们能更精准地计算每个答案的优势值，并指导模型进行更有效的更新。

## 实现细节

我们实现了 **Pass@k 训练与解析解** 的具体代码。

```python
import torch
from scipy.special import comb
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

class PassKAdvantage:
    def __init__(self, k: int):
        """
        Initializes the Pass@k advantage calculator.
        
        Args:
            k (int): The k parameter for the Pass@k metric.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def __call__(
        self,
        token_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        group_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates advantages for batched responses based on the Pass@k metric.

        Args:
            token_rewards (torch.Tensor): Token-level rewards [batch_size, seq_len].
                                          Rewards are assumed to be 0 or 1.
            response_mask (torch.Tensor): Mask for valid tokens [batch_size, seq_len].
                                          This is crucial for handling variable-length sequences.
            group_indices (List[int]): A list mapping each response in the batch to its group ID.
                                       Responses with the same group ID are considered part of the same Pass@k group.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - advantages (torch.Tensor): The calculated advantage values [batch_size, seq_len].
                - baselines (torch.Tensor): The baseline values, which are identical to the advantages 
                                          in this implementation.
        """
        with torch.no_grad():
            # 1. Aggregate rewards to the response level
            # Sum rewards for each sequence, resulting in a binary reward (0 or 1)
            # as per the problem definition.
            # `response_rewards` will be a tensor of shape [batch_size].
            response_rewards = (token_rewards.sum(dim=-1) > 0).float()
            
            # 2. Prepare data for group-wise calculation
            unique_groups = sorted(list(set(group_indices)))
            
            # `advantage_map` will store the calculated advantage for each group
            advantage_map = {}
            
            # 3. Calculate advantages for each group
            for group_id in unique_groups:
                # Get the indices of responses belonging to the current group
                group_mask = torch.tensor([idx == group_id for idx in group_indices], device=response_rewards.device)
                
                # Extract rewards for the current group
                group_rewards_tensor = response_rewards[group_mask]
                
                # Convert to NumPy for combination calculations
                group_rewards_np = group_rewards_tensor.cpu().numpy()
                
                num_rollout = len(group_rewards_np)
                num_neg = np.sum(group_rewards_np == 0)
                
                # Skip calculation if k is greater than the number of responses or negative responses
                if self.k > num_rollout or self.k > num_neg:
                    # In this case, Pass@k is always 1, so the advantage is undefined or 0.
                    # We can assign a default positive advantage to the correct responses and 0 to the rest.
                    # A_pos = (1 - R_group) / sigma_group. If R_group=1, A_pos is 0.
                    # If R_group=0, A_neg is undefined.
                    # The following assignment is a numerically stable fallback.
                    advantage_map[group_id] = np.where(group_rewards_np == 1, 1.0, 0.0)
                    continue

                # Calculate group-level statistics
                R_group = 1.0 - (comb(num_neg, self.k, exact=True) / comb(num_rollout, self.k, exact=True))
                sigma_group = np.sqrt(R_group * (1.0 - R_group))
                
                # Check for numerical stability
                if sigma_group < 1e-8:
                    advantages_np = np.zeros_like(group_rewards_np, dtype=np.float32)
                else:
                    # Calculate advantages for positive and negative samples
                    A_pos = (1.0 - R_group) / sigma_group
                    A_neg = (R_group - (comb(num_neg - 1, self.k - 1, exact=True) / comb(num_rollout - 1, self.k - 1, exact=True))) / sigma_group
                    
                    # Assign advantages based on reward values
                    advantages_np = np.where(group_rewards_np == 1, A_pos, A_neg)
                
                advantage_map[group_id] = advantages_np
            
            # 4. Create a result tensor and populate it with the calculated advantages
            advantages_tensor = torch.zeros_like(response_rewards)
            for group_id, adv_np in advantage_map.items():
                group_mask = torch.tensor([idx == group_id for idx in group_indices], device=advantages_tensor.device)
                advantages_tensor[group_mask] = torch.from_numpy(adv_np).to(advantages_tensor.device)

        # 5. Expand advantages to the token level and apply the response mask
        advantages = advantages_tensor.unsqueeze(-1) * response_mask
        
        # In this specific context, baseline is identical to the advantages
        baselines = advantages

        return advantages, baselines
```


另外一个实现版本：


```python
import numpy as np
import torch
from scipy.special import comb
from typing import List, Tuple, Dict
from collections import defaultdict

class PassKAdvantage:
    def __init__(self, k: int):
        """Initialize Pass@k advantage calculator
        
        Args:
            k (int): The k parameter for Pass@k metric
        """
        self.k = k

    def calculate_group_statistics(self, rewards: np.ndarray) -> Tuple[float, float]:
        """Calculate group-level statistics based on rewards
        
        Args:
            rewards (np.ndarray): Binary rewards array (0 for negative, 1 for positive)
            
        Returns:
            Tuple[float, float]: Mean reward and standard deviation
        """
        N_total = len(rewards)
        N_neg = len(np.where(rewards == 0)[0])
        N_pos = N_total - N_neg
        
        # Calculate mean reward using combination formula
        R_group = 1 - (comb(N_neg, self.k) / comb(N_total, self.k))
        
        # Calculate standard deviation
        sigma_group = np.sqrt(R_group * (1 - R_group))
        
        return R_group, sigma_group

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """Compute advantage values for each response
        
        Args:
            rewards (np.ndarray): Binary rewards array
            
        Returns:
            np.ndarray: Advantage values for each response
        """
        R_group, sigma_group = self.calculate_group_statistics(rewards)
        N_total = len(rewards)
        N_neg = len(np.where(rewards == 0)[0])
        
        # Calculate advantages for positive and negative samples
        A_pos = (1 - R_group) / (sigma_group + 1e-8)  # Add epsilon for numerical stability
        A_neg = (1 - R_group - comb(N_neg-1, self.k-1)/comb(N_total-1, self.k-1)) / (sigma_group + 1e-8)
        
        # Assign advantages based on rewards
        advantages = np.where(rewards == 1, A_pos, A_neg)
        
        return advantages

    def __call__(self, token_rewards: torch.Tensor, 
                 response_mask: torch.Tensor, 
                 group_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate advantages for batched responses
        
        Args:
            token_rewards (torch.Tensor): Token-level rewards [batch_size, seq_len]
            response_mask (torch.Tensor): Mask for valid tokens [batch_size, seq_len]
            group_indices (List[int]): Group indices for each response
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Advantages and baseline values
        """
        # Get response-level scores
        scores = token_rewards.sum(dim=-1)
        
        # Group responses by index
        grouped_scores = defaultdict(list)
        index_to_batch = defaultdict(list)
        
        with torch.no_grad():
            # Group scores by index
            for i, idx in enumerate(group_indices):
                grouped_scores[idx].append(scores[i].item())
                index_to_batch[idx].append(i)
            
            # Calculate advantages for each group
            for idx in grouped_scores:
                group_rewards = np.array(grouped_scores[idx])
                advantages = self.compute_advantages(group_rewards)
                
                # Assign advantages back to original batch positions
                for i, batch_idx in enumerate(index_to_batch[idx]):
                    scores[batch_idx] = advantages[i]
        
        # Apply response mask and expand dimensions
        advantages = scores.unsqueeze(-1) * response_mask
        
        return advantages, advantages

# Usage example:
"""
advantage_calculator = PassKAdvantage(k=5)
advantages, baseline = advantage_calculator(
    token_rewards=token_rewards,
    response_mask=response_mask,
    group_indices=indices
)
```

