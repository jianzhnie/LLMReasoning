import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Optional


class TopKTopPFilter:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        top_k >0: keep only top k tokens with highest probability (top-k filtering).

        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        filter_value: 被过滤的词的 logits 设为该值（默认 -inf）。
    """

    def __init__(
        self,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float("Inf"),
    ) -> None:
        self.top_k = top_k
        self.top_p = top_p
        self.filter_value = filter_value

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        对输入的 logits 应用 top-k 和 top-p 过滤。

        Args:
            logits: (vocab_size,) 的 logits 张量。

        Returns:
            过滤后的 logits 张量。
        """
        logits = logits.clone()  # 避免原地修改
        vocab_size = logits.size(0)

        # Top-K 过滤
        if self.top_k > 0:
            top_k = min(self.top_k, vocab_size)
            threshold = torch.topk(logits, top_k).values[-1]
            logits[logits < threshold] = self.filter_value

        # Top-P 过滤
        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = self.filter_value

        return logits


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    """
    使用 huggingface 模型自定义生成文本。

    Args:
        model: 加载的 transformers 模型。
        tokenizer: 对应的 tokenizer。
        prompt: 初始输入文本。
        max_new_tokens: 要生成的最大 token 数。
        temperature: 控制采样的多样性，越高越随机。
        top_k: top-k 采样限制。
        top_p: top-p（nucleus）采样限制。
        eos_token_id: 生成终止符，如果为 None 则不停。
        device: 使用的设备（如 'cuda' 或 'cpu'）。默认自动检测。

    Returns:
        生成的文本字符串。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    # 创建过滤器实例
    filter = TopKTopPFilter(top_k=top_k, top_p=top_p)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            next_token_logits = outputs.logits[0, -1, :] / temperature

            filtered_logits = filter(next_token_logits)
            probs = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
