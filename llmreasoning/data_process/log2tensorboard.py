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
from typing import Dict, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

# Define prefixes of interest for filtering metrics.
# This constant is a tuple for immutability and is used to efficiently check
# if a metric key should be processed.
INTERESTED_PREFIXES: Tuple[str, ...] = (
    'response_length/',
    'prompt_length/',
    'grpo/',
    'actor/',
    'timing/',
    'vllm_throughput',
    'grad_norm',
)

# Pre-compile regular expressions for efficiency, as they will be used in a loop.
# This avoids re-compiling the pattern on every line of the log file.
_ITERATION_PATTERN = re.compile(r'iteration: (\d+)')
# This pattern matches key-value pairs where the key is a word/path
# and the value is a number (integer, float, or scientific notation, possibly negative).
_METRICS_PATTERN = re.compile(
    r'([\w/]+)\s*:\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)')


def extract_metrics(line: str) -> Tuple[Optional[int], Dict[str, float]]:
    """
    Extract the iteration number and relevant metrics from a log line.

    The function first looks for an 'iteration' number. If found, it then
    searches for all 'key: value' pairs in the line and filters them based
    on the `INTERESTED_PREFIXES` constant.

    Args:
        line (str): A single line from the log file.

    Returns:
        Tuple[Optional[int], Dict[str, float]]: A tuple containing the
        iteration number (if found, otherwise None) and a dictionary of
        metrics that match the interested prefixes. Returns `(None, {})` if
        no iteration number is found, as metrics without an iteration cannot
        be plotted.
    """
    metrics: Dict[str, float] = {}

    # Find the iteration number using the pre-compiled regex.
    iter_match = _ITERATION_PATTERN.search(line)
    if not iter_match:
        return None, {}

    iteration = int(iter_match.group(1))

    # Find all "key : value" pairs using the pre-compiled regex.
    # The findall method returns a list of tuples, e.g., [('key1', 'val1'), ...].
    kv_pairs = _METRICS_PATTERN.findall(line)

    for key, val_str in kv_pairs:
        # Check if the key starts with any of the defined prefixes.
        if key.startswith(INTERESTED_PREFIXES):
            try:
                metrics[key] = float(val_str)
            except ValueError:
                # Log a warning or skip if a value can't be converted to a float.
                print(
                    f"Warning: Could not convert '{val_str}' to float for key '{key}'."
                )
                continue

    return iteration, metrics


def process_log_file(log_path: str, log_dir: str) -> None:
    """
    Read a log file line by line and write matched metrics to TensorBoard logs.

    This is the main function that orchestrates the log processing. It handles
    file I/O, directory creation, and writing data to a TensorBoard
    SummaryWriter.

    Args:
        log_path (str): The full path to the input log file.
        log_dir (str): The directory where TensorBoard logs will be written.
                      The directory will be created if it doesn't exist.
    """
    # Use a more explicit check and raise an error if the file doesn't exist.
    if not os.path.exists(log_path):
        raise FileNotFoundError(f'Log file not found at: {log_path}')

    os.makedirs(log_dir, exist_ok=True)
    print(f'Starting to process log file: {log_path}')

    # The `SummaryWriter` is used as a context manager to ensure it's closed
    # properly even if an error occurs.
    with SummaryWriter(log_dir=log_dir) as writer:
        # Open the log file for reading with UTF-8 encoding.
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                iteration, metrics = extract_metrics(line)
                # Skip lines that don't contain a valid iteration number.
                if iteration is None:
                    continue
                # Loop through all extracted metrics and add them to TensorBoard.
                for key, value in metrics.items():
                    # The `add_scalar` method logs a single scalar value.
                    writer.add_scalar(key, value, iteration)

    print(f'✅ TensorBoard logs successfully written to: {log_dir}')
    print('\nTo view the logs, run this command in your terminal:')
    print(f'  tensorboard --logdir "{log_dir}"')


def main():
    """
    Main function to handle command-line argument parsing and script execution.
    """
    # Create the parser and define the arguments
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

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main processing function with the parsed arguments
    try:
        process_log_file(args.log_path, args.save_log_dir)
    except FileNotFoundError as fnfe:
        print(f'❌ Error: {fnfe}')
        print('Please ensure the log file path is correct and accessible.')
    except Exception as e:
        print(f'❌ An unexpected error occurred: {e}')


if __name__ == '__main__':
    main()
