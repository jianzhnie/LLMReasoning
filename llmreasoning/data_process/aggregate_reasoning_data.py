import argparse
import json
import logging
import os  # 导入 os 库用于检查文件路径
from collections import defaultdict
from typing import Any, Dict, Iterator, List

from datasets import IterableDataset, load_dataset
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_streaming(input_path: str) -> IterableDataset:
    """
    Loads a JSONL dataset in streaming mode.

    Args:
        input_path: The file path to the input JSONL file.

    Returns:
        A streaming `IterableDataset` ready for processing.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Error: Input file not found at '{input_path}'")
    return load_dataset('json',
                        data_files=input_path,
                        split='train',
                        streaming=True)


def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocesses a single data example by extracting and cleaning relevant fields.

    The 'gen' field is handled to correctly extract the first element if it's a list.
    The 'is_correct' field is a boolean derived from 'accuracy'.

    Args:
        example: An input dictionary representing a single data example.

    Returns:
        A processed dictionary containing 'prompt', 'cot', 'is_correct', and 'answer'.
    """
    try:
        gen_data = example.get('gen', '')
        cot_text = gen_data[0] if isinstance(gen_data,
                                             list) and gen_data else ''
        return {
            'prompt': example.get('prompt', ''),
            'cot': cot_text,
            'is_correct': float(example.get('accuracy', 0.0)) >= 0.5,
            'answer': example.get('extracted_answer', ''),
        }
    except Exception as e:
        logger.warning(f'Skipping example due to preprocessing error: {e}')
        logger.debug(f'Problematic example: {example}')
        return None


def group_by_prompt(
        dataset: Iterator[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups data examples by their 'prompt' field.

    This function iterates through the preprocessed dataset and aggregates
    all related examples under a single prompt key.

    Args:
        dataset: An iterator over the preprocessed dataset.

    Returns:
        A dictionary where each key is a 'prompt' and the value is a list of
        dictionaries, each containing the 'cot', 'is_correct', and 'answer'
        for that prompt.
    """
    grouped_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    processed_count = 0
    error_count = 0

    for item in tqdm(dataset, desc='Processing Data'):
        try:
            # Skip None items (failed preprocessing)
            if item is None:
                error_count += 1
                continue

            prompt = item['prompt']
            # Skip items with empty prompts
            if not prompt:
                error_count += 1
                logger.warning('Skipping example with empty prompt')
                continue

            grouped_data[prompt].append({
                'cot': item['cot'],
                'is_correct': item['is_correct'],
                'answer': item['answer'],
            })
            processed_count += 1
        except Exception as e:
            error_count += 1
            logger.warning(f'Skipping example due to grouping error: {e}')
            logger.debug(f'Problematic item: {item}')

    logger.info(
        f'Processed {processed_count} examples, skipped {error_count} examples due to errors'
    )
    return grouped_data


def build_final_output(
        grouped_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Builds the final structured output from the grouped data.

    This function organizes the grouped data into the desired output format,
    including an 'id', 'question', 'answer', and a dictionary of 'cots'.
    Each 'cot' entry includes the text, token length, and correctness.

    Args:
        grouped_data: A dictionary where prompts are keys and values are lists
                      of associated data examples.

    Returns:
        A list of dictionaries, each representing a complete, structured output entry.
    """
    final_output: List[Dict[str, Any]] = []
    id_counter = 1
    error_count = 0

    for prompt, cots_list in tqdm(grouped_data.items(),
                                  desc='Building Output Structure'):
        try:
            # Skip items with empty prompts or empty cots_list
            if not prompt or not cots_list:
                error_count += 1
                logger.warning(
                    'Skipping output building for empty prompt or cots_list')
                continue

            answer = cots_list[0].get('answer', '')
            cots = {
                f'cot_{i+1}': {
                    'cot': cot_info['cot'],
                    'cot_token_len': len(cot_info['cot'].split()),
                    'is_correct': cot_info['is_correct'],
                }
                for i, cot_info in enumerate(cots_list)
            }

            final_output.append({
                'id': str(id_counter),
                'question': prompt,
                'answer': answer,
                'difficulty': len(cots_list),
                'cots': cots,
            })
            id_counter += 1
        except Exception as e:
            error_count += 1
            logger.warning(
                f'Skipping prompt due to output building error: {e}')
            logger.debug(
                f'Problematic prompt: {prompt}, cots_list: {cots_list}')

    logger.info(
        f'Built final output for {len(final_output)} entries, skipped {error_count} due to errors'
    )

    return final_output


def save_to_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Saves the processed data to a JSON file with pretty-printing.

    Args:
        data: The final structured data to be saved.
        output_path: The file path where the JSON file will be written.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'✅ Result saved to: {output_path}')
    except IOError as e:
        print(f'❌ Error saving file: {e}')


def main(input_path: str, output_path: str) -> None:
    """
    Main function to orchestrate the entire data processing pipeline.

    The pipeline loads a streaming dataset, preprocesses it, groups it by prompt,
    structures the final output, and saves it to a JSON file.

    Args:
        input_path: The file path to the input JSONL file.
        output_path: The file path where the output JSON file will be saved.
    """
    print('🔄 Loading dataset...')
    raw_dataset = load_data_streaming(input_path)

    print('⚙️ Preprocessing data...')
    mapped_dataset_iterator = raw_dataset.map(preprocess)

    print('🧩 Grouping data by prompt...')
    grouped_data = group_by_prompt(mapped_dataset_iterator)

    print('📦 Building final output format...')
    final_data = build_final_output(grouped_data)

    print('💾 Saving to JSON...')
    save_to_json(final_data, output_path)

    print('🎉 Processing complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process a JSONL dataset and group it by prompt.')
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help='Path to the input JSONL file.',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='Path for the output JSON file.',
    )
    args = parser.parse_args()

    main(args.input, args.output)
