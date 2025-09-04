"""
This script batch processes JSONL evaluation files to analyze accuracy and
other key statistics for a set of prompts. It aggregates results from
multiple files into a single, comprehensive summary JSON file.

The script uses HuggingFace's `datasets` library for efficient data loading
and parallel processing, making it suitable for large datasets.
"""

import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class SummaryStats:
    """
    A dataclass to hold the aggregated statistics for a single prompt.

    Using a dataclass provides a clear, structured, and type-hinted
    way to manage the aggregated data, improving readability and
    reducing the risk of typos.
    """
    count: int = 0
    answer: str = ''
    accuracy_count: float = 0.0
    gen_lengths: List[int] = field(default_factory=list)


def safe_mean(data: List[int]) -> float:
    """Calculates the mean of a list of numbers, handling empty lists.

    Args:
        data: A list of integers.

    Returns:
        The mean as a float, or 0.0 if the list is empty.
    """
    if not data:
        return 0.0
    return sum(data) / len(data)


def preprocess_example(
        example: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase) -> Optional[Dict[str, Any]]:
    """
    Processes a single example from the dataset for aggregation.

    This function extracts and validates key fields from an example
    (a single line from a JSONL file), calculates the tokenized length
    of the generated text, and prepares a dictionary of statistics
    for further aggregation.

    Args:
        example: A dictionary representing a single line from the JSONL file.
        tokenizer: A HuggingFace tokenizer for tokenizing the generated text.

    Returns:
        A dictionary with processed statistics for a single example. Returns
        `None` if the example is invalid or cannot be parsed, allowing the
        `datasets` library's filter to remove it.
    """
    try:
        # Get and validate the prompt, which is the unique identifier for aggregation
        prompt = example.get('prompt', '')
        if not prompt or not isinstance(prompt, str):
            print("Skipping example due to missing or invalid 'prompt' key.")
            return None

        # Safely convert accuracy and ensure 'gen' field is a string
        accuracy = float(example.get('accuracy', 0.0))
        answer = str(example.get('answer', ''))
        gen_field = example.get('gen', '')

        # Handle cases where 'gen' is a list and take the first item
        gen = gen_field[0] if isinstance(
            gen_field, list) and gen_field else str(gen_field)

        gen_length = len(tokenizer.tokenize(gen))

        return {
            'prompt': prompt.strip(),
            'answer': answer,
            'count': 1,
            'accuracy_count': accuracy,
            'gen_length': gen_length,
        }
    except (ValueError, TypeError) as e:
        print(f'Skipping example due to parsing error: {e}')
        return None


def aggregate_results(dataset: Dataset) -> Dict[str, SummaryStats]:
    """
    Aggregates processed example data by prompt.

    This function iterates through a dataset of pre-processed examples and
    consolidates statistics for each unique prompt using the `SummaryStats`
    dataclass.

    Args:
        dataset: A `datasets.Dataset` object containing pre-processed examples.

    Returns:
        A dictionary mapping each unique prompt string to its
        `SummaryStats` object.
    """
    prompt_stats: Dict[str, SummaryStats] = {}
    for example in dataset:
        prompt = example.get('prompt')
        # Skip if the prompt is missing or invalid, as it's the aggregation key
        if not prompt:
            continue

        # Use setdefault with a dataclass to simplify initialization
        stats = prompt_stats.setdefault(
            prompt, SummaryStats(answer=example.get('answer', '')))

        # Accumulate statistics for the current prompt
        stats.accuracy_count += example['accuracy_count']
        stats.count += example['count']
        stats.gen_lengths.append(example['gen_length'])

    return prompt_stats


def analyze_and_get_summary(jsonl_file_path: str,
                            tokenizer: PreTrainedTokenizerBase,
                            num_proc: int = 64) -> List[Dict[str, Any]]:
    """
    Analyzes a single JSONL file and returns a list of summary dictionaries.

    This function orchestrates the loading, processing, and aggregation steps
    for a single file, providing a complete summary of its contents.

    Args:
        jsonl_file_path: The path to the JSONL file to be analyzed.
        tokenizer: The HuggingFace tokenizer to use.
        num_proc: The number of processes for parallel data processing.

    Returns:
        A list of dictionaries, where each dictionary contains the
        aggregated statistics for a single prompt. Returns an empty
        list if an error occurs.
    """
    try:
        print(f'Starting analysis for {jsonl_file_path}...')
        dataset = load_dataset('json',
                               data_files=jsonl_file_path,
                               split='train')

        # Map and filter in a single, efficient chain
        processed_dataset = dataset.map(
            lambda x: preprocess_example(x, tokenizer),
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        ).filter(lambda x: x is not None and bool(x.get('prompt')))

        prompt_stats = aggregate_results(processed_dataset)

        summary = []
        for prompt, stats in prompt_stats.items():
            count = stats.count
            average_accuracy = stats.accuracy_count / count if count > 0 else 0.0
            average_gen_length = safe_mean(stats.gen_lengths)
            max_gen_length = max(stats.gen_lengths) if stats.gen_lengths else 0

            summary.append({
                'prompt': prompt,
                'answer': stats.answer,
                'count': count,
                'accuracy_count': stats.accuracy_count,
                'average_accuracy': average_accuracy,
                'max_gen_length': max_gen_length,
                'avg_gen_length': average_gen_length,
                'gen_lengths': stats.gen_lengths,
            })

        print(f'Finished analysis for {jsonl_file_path}')
        return summary
    except Exception as e:
        print(f'An error occurred while processing {jsonl_file_path}: {e}')
        return []


def process_and_summarize(
    input_dir: str,
    file_pattern: str,
    output_dir: str,
    summary_filename: str,
    model_name_or_path: str,
    num_proc: int,
) -> None:
    """
    Finds and processes multiple JSONL files and merges the results into a single summary file.

    This is the main orchestration function. It finds all files matching a
    pattern, loads the tokenizer once for efficiency, and then processes
    each file sequentially, collecting all results before writing them to
    a final summary file.

    Args:
        input_dir: The directory containing the input JSONL files.
        file_pattern: The glob pattern to match input files (e.g., '*.jsonl').
        output_dir: The directory to save the combined summary JSON file.
        summary_filename: The filename for the final combined summary JSON.
        model_name_or_path: The HuggingFace model path for the tokenizer.
        num_proc: Number of processes to use for parallel data processing.
    """
    full_pattern = str(Path(input_dir) / file_pattern)
    jsonl_files = glob.glob(full_pattern)

    if not jsonl_files:
        print(f'No files found matching the pattern: {full_pattern}')
        return

    print(f'Found {len(jsonl_files)} files to process.')

    # Load the tokenizer once for efficiency, before the loop
    print(f'Loading tokenizer from {model_name_or_path}...')
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name_or_path)

    all_summaries: List[Dict[str, Any]] = []

    for file_path in jsonl_files:
        file_summary = analyze_and_get_summary(file_path, tokenizer, num_proc)
        all_summaries.extend(file_summary)

    # Ensure the output directory exists before writing
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all summaries to a single file with proper encoding
    output_json_path = output_path / summary_filename
    with open(output_json_path, 'w', encoding='utf-8') as out_file:
        json.dump(all_summaries, out_file, indent=2, ensure_ascii=False)

    print(
        f'Batch processing complete. All {len(all_summaries)} prompt summaries saved to {output_json_path}'
    )


def main() -> None:
    """
    Main function to parse command-line arguments and run the batch process.
    """
    parser = argparse.ArgumentParser(
        description=
        'Batch process JSONL evaluation files and generate a combined accuracy summary.'
    )
    parser.add_argument('--input_data_dir',
                        type=str,
                        required=True,
                        help='The directory containing the input JSONL files.')
    parser.add_argument(
        '--input_file_pattern',
        type=str,
        default='*.jsonl',
        help=
        "The glob pattern to match input files (e.g., 'infer_qwen25_*.jsonl')."
    )
    parser.add_argument(
        '--output_data_dir',
        type=str,
        default='./summary',
        help='The directory to save the combined summary JSON file.')
    parser.add_argument(
        '--output_file_name',
        type=str,
        default='combined_accuracy_summary.json',
        help='The filename for the final combined summary JSON.')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help=
        "The HuggingFace model path for the tokenizer (e.g., '/root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B')."
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=64,
        help='Number of processes to use for parallel data processing.')

    args = parser.parse_args()

    # Pass the corrected argument names to the function
    process_and_summarize(input_dir=args.input_data_dir,
                          file_pattern=args.input_file_pattern,
                          output_dir=args.output_data_dir,
                          summary_filename=args.output_file_name,
                          model_name_or_path=args.model_name_or_path,
                          num_proc=args.num_proc)


if __name__ == '__main__':
    main()
