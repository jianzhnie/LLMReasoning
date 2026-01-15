import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

# 模拟原始logits
np.random.seed(42)
logits = np.random.normal(0, 1, 50)  # 50个token的logits
temperatures = [0.5, 1.0, 1.5]

# 创建第一个图形：原始分布视图
plt.figure(figsize=(12, 8))
original_probs = softmax(logits)
plt.plot(original_probs, 'k--', alpha=0.5, label='Original (T=1.0)')

for temp in temperatures:
    scaled_logits = logits / temp
    probs = softmax(scaled_logits)
    plt.plot(probs, 'o-', label=f'T={temp}', alpha=0.7)

plt.title('Effect of Temperature on Probability Distribution', fontsize=14)
plt.xlabel('Token Index', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(len(logits)))
plt.ylim(0, max(original_probs) * 1.2)

# 高亮最高概率token
max_idx = np.argmax(original_probs)
for temp in temperatures:
    scaled_logits = logits / temp
    probs = softmax(scaled_logits)
    plt.scatter(max_idx, probs[max_idx], color='red', zorder=5)

plt.tight_layout()
plt.savefig('temperature_effect_raw.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存

# 创建第二个图形：排序视图
plt.figure(figsize=(12, 8))
for temp in temperatures:
    scaled_logits = logits / temp
    probs = softmax(scaled_logits)
    sorted_probs = np.sort(probs)[::-1]  # 降序排列
    plt.plot(sorted_probs, 'o-', label=f'T={temp}', alpha=0.7)

plt.title('Sorted Probability Distributions at Different Temperatures',
          fontsize=14)
plt.xlabel('Rank (0=highest prob)', fontsize=12)
plt.ylabel('Probability (log scale)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('temperature_effect_sorted.png', dpi=300, bbox_inches='tight')
plt.close()
