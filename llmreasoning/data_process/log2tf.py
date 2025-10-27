"""
A script to process log files containing training metrics and write them to
TensorBoard for visualization.

This module provides functions to extract key-value metrics from log lines
and an entry point to read a log file and write the extracted data to
a TensorBoard SummaryWriter.
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

# Pre-compile regular expressions for efficiency.
# This pattern matches the main training log line format, now including MFU.
# Pre-compile regular expressions for efficiency.
DPO_TRAINING_LOG_PATTERN = re.compile(
    r'iteration\s+(\d+)/\s*\d+\s*\|.*?'
    r'throughput per GPU \(TFLOP/s/GPU\):\s*([\d\.E+-]+)\s*\|.*?'
    r'learning rate:\s*([\d\.E+-]+)\s*\|.*?'
    r'rewards/accuracies:\s*([\d\.E+-]+)\s*\|.*?'
    r'lm loss:\s*([\d\.E+-]+)\s*\|.*?'
    r'grad norm:\s*([\d\.E+-]+).*?\|\s*number of skipped iterations:.*?')

SFT_TRAINING_LOG_PATTERN = re.compile(
    r'iteration\s+(\d+)/\s*\d+\s*\|.*?'
    r'throughput per GPU \(TFLOP/s/GPU\):\s*([\d\.E+-]+)\s*\|.*?'
    r'learning rate:\s*([\d\.E+-]+)\s*\|.*?'
    r'lm loss:\s*([\d\.E+-]+)\s*\|.*?'
    r'grad norm:\s*([\d\.E+-]+).*?\|\s*number of skipped iterations:.*?')

# Map stages to their respective patterns and field configurations
TRAINING_LOG_CONFIGS = {
    'sft': {
        'pattern': SFT_TRAINING_LOG_PATTERN,
        'fields': ['tflops_per_gpu', 'learning_rate', 'lm_loss', 'grad_norm']
    },
    'dpo': {
        'pattern':
        DPO_TRAINING_LOG_PATTERN,
        'fields': [
            'tflops_per_gpu', 'learning_rate', 'rewards_accuracies', 'lm_loss',
            'grad_norm'
        ]
    }
}


def extract_metrics_from_specific_log(
        line: str,
        stage: str = 'sft') -> Tuple[Optional[int], Dict[str, float]]:
    """
    Extract the iteration number and relevant metrics from a specific log line
    format.

    This function is tailored to the user's specific log format. It uses a
    single, comprehensive regex pattern to capture all key metrics at once.

    Args:
        line (str): A single line from the log file.
        stage (str): The training stage, either 'sft' or 'dpo'. Defaults to 'sft'.

    Returns:
        Tuple[Optional[int], Dict[str, float]]: A tuple containing the
        iteration number and a dictionary of extracted metrics. Returns `(None, {})`
        if the line does not match the expected format.
    """
    config = TRAINING_LOG_CONFIGS.get(stage)
    if not config:
        raise ValueError(
            f'Invalid stage: {stage}. Supported stages: {list(TRAINING_LOG_CONFIGS.keys())}'
        )

    pattern = config['pattern']
    fields = config['fields']

    match = pattern.search(line)

    if not match:
        return None, {}
    metrics: Dict[str, float] = {}

    try:
        iteration = int(match.group(1))
        # Extract metrics based on the field configuration
        for i, field_name in enumerate(fields):
            metrics[field_name] = float(
                match.group(i + 2))  # +2 because group(1) is iteration

    except (ValueError, IndexError) as e:
        print(
            f'Warning: Failed to parse metrics from line: "{line.strip()}". Error: {e}'
        )
        return None, {}

    return iteration, metrics


def register_new_log_format(stage: str, pattern: re.Pattern,
                            fields: List[str]) -> None:
    """
    Register a new log format for parsing.

    Args:
        stage (str): Name of the training stage.
        pattern (re.Pattern): Compiled regex pattern for matching log lines.
        fields (List[str]): List of field names corresponding to capture groups (excluding iteration).
    """
    TRAINING_LOG_CONFIGS[stage] = {'pattern': pattern, 'fields': fields}


def process_log_file(log_path: str, log_dir: str, stage: str = 'sft') -> None:
    """
    Read a log file line by line and write matched metrics to TensorBoard logs.

    Args:
        log_path (str): Path to the input log file.
        log_dir (str): Directory to save TensorBoard logs.
        stage (str): Training stage type ('sft' or 'dpo').
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f'Log file not found at: {log_path}')

    os.makedirs(log_dir, exist_ok=True)
    print(f'Starting to process log file: {log_path}')

    with SummaryWriter(log_dir=log_dir) as writer:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                iteration, metrics = extract_metrics_from_specific_log(
                    line, stage)

                if iteration is None:
                    continue

                for key, value in metrics.items():
                    # Sanitize key to be a valid TensorBoard tag
                    sanitized_key = key.replace(' ', '_').replace('/', '_')
                    tensorboard_tag = f'train/{sanitized_key}'
                    writer.add_scalar(tensorboard_tag, value, iteration)

    print(f'✅ TensorBoard logs successfully written to: {log_dir}')
    print('\nTo view the logs, run this command in your terminal:')
    print(f'  tensorboard --logdir "{log_dir}"')


def main():
    """
    Main function to handle command-line argument parsing and script execution.
    """
    parser = argparse.ArgumentParser(
        description='Process log files and write metrics to TensorBoard.')
    parser.add_argument('--log-path',
                        type=str,
                        required=True,
                        help='The full path to the input log file.')
    parser.add_argument('--save-log-dir',
                        type=str,
                        required=True,
                        help='The directory to save the TensorBoard logs.')
    parser.add_argument('--stage',
                        type=str,
                        default='sft',
                        choices=['sft', 'dpo'],
                        help='Training stage type (default: sft).')

    args = parser.parse_args()

    try:
        process_log_file(args.log_path, args.save_log_dir, args.stage)
    except FileNotFoundError as fnfe:
        print(f'❌ Error: {fnfe}')
        print('Please ensure the log file path is correct and accessible.')
    except Exception as e:
        print(f'❌ An unexpected error occurred: {e}')


if __name__ == '__main__':
    main()
