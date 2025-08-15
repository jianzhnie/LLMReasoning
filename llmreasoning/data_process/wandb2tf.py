import os
import re
from typing import Dict, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

# Define prefixes of interest for filtering metrics.
INTERESTED_PREFIXES: Tuple[str, ...] = ('response_length/', 'prompt_length/',
                                        'grpo/', 'actor/', 'timing/',
                                        'vllm_throughput', 'grad_norm')


def extract_metrics(line: str) -> Tuple[Optional[int], Dict[str, float]]:
    """
    Extract iteration number and relevant metrics from a log line.

    Args:
        line (str): A single line from the log file.

    Returns:
        Tuple[Optional[int], Dict[str, float]]: A tuple containing the iteration
        number (if found) and a dictionary of metrics that match interested prefixes.
    """
    metrics: Dict[str, float] = {}

    # Extract iteration number
    iter_match = re.search(r'iteration: (\d+)', line)
    if not iter_match:
        return None, {}

    iteration = int(iter_match.group(1))

    # Extract all "key : value" style metrics
    kv_pairs = re.findall(r'([\w/]+)\s*:\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)',
                          line)
    for key, val in kv_pairs:
        if key.startswith(INTERESTED_PREFIXES):
            metrics[key] = float(val)

    return iteration, metrics


def process_log_file(log_path: str, log_dir: str) -> None:
    """
    Read a log file line by line and write matched metrics to TensorBoard logs.

    Args:
        log_path (str): Path to the input log file.
        log_dir (str): Directory where TensorBoard logs will be written.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f'Log file not found: {log_path}')

    os.makedirs(log_dir, exist_ok=True)

    with SummaryWriter(log_dir=log_dir) as writer:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                iteration, metrics = extract_metrics(line)
                if iteration is None:
                    continue
                for key, value in metrics.items():
                    writer.add_scalar(key, value, iteration)

    print(f'✅ TensorBoard logs written to: {log_dir}')
    print('Use the following command to view them: \n')
    print(f'  tensorboard --logdir {log_dir}')


if __name__ == '__main__':
    # Example paths - adjust according to your environment
    wandb_log_path = (
        '/root/llmtuner/llm/MindSpeed-RL-master/work_dir/wandb_log/'
        'wandb/run-20250530_154406-37fteudp/files/output.log')
    tf_log_path = ('/root/llmtuner/llm/MindSpeed-RL-master/work_dir/tf_logs/'
                   'r1_zero_distill-qwen-7b_skywork_1-8_8k/eval_logs')

    try:
        process_log_file(wandb_log_path, tf_log_path)
    except Exception as e:
        print(f'❌ Error: {e}')
