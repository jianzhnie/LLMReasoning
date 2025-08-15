#!/usr/bin/env python3
"""
Generate DPO (Direct Preference Optimization) pairs from reasoning data.

This script processes a dataset containing questions and multiple "Chain of Thought" (CoT)
reasoning paths, along with their correctness labels. It then generates pairs of
(prompt, chosen, rejected) suitable for Direct Preference Optimization (DPO) training.

The core logic involves:
1. Loading a JSON/JSONL dataset with fields "question" and "cots".
2. Differentiating between correct and incorrect CoT responses.
3. Forming DPO pairs:
    - If both correct and incorrect CoTs exist for a question, it pairs each correct CoT
      with each incorrect one.
    - If only correct CoTs are present, it creates pairs by choosing the shortest CoTs
      as "chosen" and the longest as "rejected" to promote conciseness and efficiency.
4. Applying a model's chat template or a custom string format to prompts and responses.
5. Saving the resulting DPO pairs to a new JSONL file.

Author: jianzhnie
"""

import argparse
import logging
from functools import partial
from itertools import chain, product
from typing import (Any, Dict, Final, Iterable, List, MutableMapping, Optional,
                    TypedDict, Union)

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------

# Configure logging for better visibility
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger('dpo_pair_builder')

# Defaults (used only if CLI values are not provided)
DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant. To answer the user's question, you first think "
    'about the reasoning process and then provide the user with the answer. '
    'The reasoning process and answer are enclosed within <think> </think> and '
    '<answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think> <answer> answer here </answer>.')
DEFAULT_MATH_COT_PROMPT: Final[str] = (
    'Please reason step by step, and put your final answer within \\boxed{}.')

PROMPT_FORMAT_TEMPLATE: Final[str] = (
    '<|im_start|>system\n{system_prompt}<|im_end|>\n'
    '<|im_start|>user\n{user_question}\n{additional_prompt}<|im_end|>\n'
    '<|im_start|>assistant\n')

RESPONSE_FORMAT_TEMPLATE: Final[str] = (
    '<|im_start|>assistant\n{assistant_response}<|im_end|>\n')

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


class DpoPair(TypedDict):
    """
    Defines the structure of a DPO training pair.

    Attributes:
        system (Optional[str]): An optional system prompt.
        prompt (str): The formatted user prompt.
        chosen (str): The preferred (correct) response.
        rejected (str): The dis-preferred (incorrect) response.
    """
    system: Optional[str]
    prompt: str
    chosen: str
    rejected: str


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def item_as_bool(value: Any) -> bool:
    """
    Convert various truthy/falsey encodings to bool.

    Accepts: bool, int, str such as "true"/"false", "yes"/"no", "1"/"0".
    Defaults to False for unrecognized strings.

    Args:
        value (Any): The input value to convert.

    Returns:
        bool: True if the value is truthy, False otherwise.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        return s in {'1', 'true', 't', 'yes', 'y'}
    return bool(value)


def get_token_len(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """
    Calculates the number of tokens for a given text using the specified tokenizer.

    Args:
        text (str): The text to tokenize.
        tokenizer (PreTrainedTokenizerBase): The tokenizer instance.

    Returns:
        int: The number of tokens.
    """
    try:
        # Use tokenizer.__call__ for better handling of special tokens and padding
        encoded = tokenizer(text,
                            add_special_tokens=False,
                            return_attention_mask=False)
        return len(encoded.get('input_ids', []))
    except Exception as e:
        logger.warning(
            f'Falling back to naive length computation due to tokenization error: {e}'
        )
        return len(text.split())


def get_iter_cots(
    cots_field: Union[Dict[str, Any], List[Any], Any]
) -> Iterable[MutableMapping[str, Any]]:
    """
    Normalizes the 'cots' field from a dataset item to an iterable of dictionaries.

    Args:
        cots_field (Union[Dict, List, Any]): The raw 'cots' data from a dataset item.
                                             Can be a dictionary, a list of dictionaries, or other types.

    Returns:
        Iterable[MutableMapping[str, Any]]: An iterator over the valid CoT entries.

    """
    if isinstance(cots_field, dict):
        # Handle cases where 'cots' is a dict of CoTs
        return (v for v in cots_field.values() if isinstance(v, dict))
    if isinstance(cots_field, list):
        # Handle cases where 'cots' is a list of CoTs
        return (v for v in cots_field if isinstance(v, dict))
    # Return an empty iterator for any other type
    return iter([])


def apply_model_chat_template(
    item: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> DpoPair:
    """
    Formats the user's prompt and chosen and rejected responses using the model's chat template for the assistant's turn.

    This function constructs a chat history with optional system and user prompts,
    and then applies the tokenizer's chat template to format the conversation.

    Args:
        item (Dict[str, Any]): A dictionary containing 'chosen' and 'rejected' fields with raw text.
        tokenizer (PreTrainedTokenizerBase): The tokenizer with the chat template.
        system_prompt (Optional[str]): The system prompt to be included.
        additional_prompt (Optional[str]): An optional prompt to append to the user's text.
                                           For example, "Please reason step by step...".
        add_generation_prompt (bool): If True, the template will include a prompt for
                                      the assistant's turn, such as '<|im_start|>assistant\n'.
    Returns:
        DpoPair: The dictionary with 'chosen' and 'rejected' fields now formatted
                 for the assistant's turn, with a TypedDict signature.
    """
    # Use .get() with a default to avoid KeyError if fields are missing
    user_prompt = item.get('prompt', '')
    chosen_cot_text = item.get('chosen', '')
    rejected_cot_text = item.get('rejected', '')

    if additional_prompt:
        user_prompt += '\n' + additional_prompt

    prompt_messages: List[Dict[str, str]] = []
    if system_prompt:
        prompt_messages.append({'role': 'system', 'content': system_prompt})

    prompt_messages.append({'role': 'user', 'content': user_prompt})

    chosen_messages: List[Dict[str, str]] = [{
        'role': 'assistant',
        'content': chosen_cot_text
    }]
    rejected_messages: List[Dict[str, str]] = [{
        'role': 'assistant',
        'content': rejected_cot_text
    }]

    # Apply template to User prompt
    prompt_formatted: str = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Apply template to assistant responses
    chosen_formatted = tokenizer.apply_chat_template(
        chosen_messages, tokenize=False, add_generation_prompt=False)
    rejected_formatted = tokenizer.apply_chat_template(
        rejected_messages, tokenize=False, add_generation_prompt=False)

    return DpoPair(
        system=item.get('system'),
        prompt=prompt_formatted,
        chosen=chosen_formatted,
        rejected=rejected_formatted,
    )


def apply_string_chat_template(
    item: Dict[str, Any],
    prompt_template: Optional[str] = None,
    assistant_template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
) -> DpoPair:
    # Use .get() with a default to avoid KeyError if fields are missing
    user_prompt = item.get('prompt', '')
    chosen_cot_text = item.get('chosen', '')
    rejected_cot_text = item.get('rejected', '')

    if additional_prompt:
        user_prompt += '\n' + additional_prompt

    # 使用 .format() 格式化 Prompt
    prompt_formatted = prompt_template.format(
        system_prompt=system_prompt,
        user_question=user_prompt,
        additional_prompt=additional_prompt)

    # 使用 .format() 格式化 Assistant Response
    chosen_formatted = assistant_template.format(
        assistant_response=chosen_cot_text)

    rejected_formatted = assistant_template.format(
        assistant_response=rejected_cot_text)

    return DpoPair(
        system=item.get('system'),
        prompt=prompt_formatted,
        chosen=chosen_formatted,
        rejected=rejected_formatted,
    )


def generate_dpo_pairs(
    item: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str] = None,
    max_cot_len: int = 32768,
) -> Dict[str, List[DpoPair]]:
    """
    Processes a single data item to generate DPO pairs (prompt, chosen, rejected).

    This function formats the prompt, filters out CoT responses that are too long,
    and then creates pairs of correct vs. incorrect responses. If only correct
    responses are available, it pairs the shortest with the longest.

    Args:
        item (Dict[str, Any]): The input data sample, expected to contain "question" and "cots".
        tokenizer (PreTrainedTokenizerBase): Tokenizer for length calculation and prompt formatting.
        system_prompt (Optional[str]): System prompt template.
        max_cot_len (int): The maximum allowed token length for a CoT response.

    Returns:
        Dict[str, List[DpoPair]]: A dictionary containing
        the generated DPO pairs under the key "pairs".
    """
    question: Optional[str] = item.get('question')
    if not isinstance(question, str) or not question.strip():
        logger.warning("Skipping item without a valid 'question' field.")
        return {'pairs': []}

    raw_cots: List[Dict[str, Any]] = list(get_iter_cots(item.get('cots')))
    if not raw_cots:
        return {'pairs': []}

    # Filter out CoTs that exceed the maximum length
    filtered_cots: List[Dict[str, Any]] = [
        cot for cot in raw_cots
        if get_token_len(cot.get('cot', ''), tokenizer) <= max_cot_len
    ]

    # Differentiate between correct and incorrect CoTs based on the boolean value
    correct_cots: List[Dict[str, Any]] = [
        cot for cot in filtered_cots if item_as_bool(cot.get('is_correct'))
    ]
    incorrect_cots: List[Dict[str, Any]] = [
        cot for cot in filtered_cots if not item_as_bool(cot.get('is_correct'))
    ]

    dpo_pairs: List[DpoPair] = []

    # Case 1: Both correct and incorrect CoTs are available
    if correct_cots and incorrect_cots:
        for chosen_cot, rejected_cot in product(correct_cots, incorrect_cots):
            ch_cot = str(chosen_cot.get('cot', '')).strip()
            re_cot = str(rejected_cot.get('cot', '')).strip()
            if ch_cot and re_cot and ch_cot != re_cot:
                dpo_pairs.append(
                    DpoPair(
                        system=system_prompt,
                        prompt=question,
                        chosen=ch_cot,
                        rejected=re_cot,
                    ))

    # Case 2: Only correct CoTs are available, pair shortest vs. longest
    elif correct_cots:
        cots_with_len: List[Dict[str, Any]] = [{
            **cot, 'cot_token_len':
            get_token_len(cot.get('cot', ''), tokenizer)
        } for cot in correct_cots if cot.get('cot')]

        sorted_by_len: List[Dict[str, Any]] = sorted(
            cots_with_len, key=lambda x: x['cot_token_len'])
        num_cots: int = len(sorted_by_len)

        # We need at least 4 CoTs to get two chosen and two rejected candidates.
        # This prevents edge cases where the shortest and longest might be the same.
        if num_cots < 4:
            return {'pairs': []}

        chosen_candidates: List[Dict[str, Any]] = sorted_by_len[:2]
        rejected_candidates: List[Dict[str, Any]] = sorted_by_len[-2:]

        for chosen_cot, rejected_cot in product(chosen_candidates,
                                                rejected_candidates):
            ch_cot = str(chosen_cot.get('cot', '')).strip()
            re_cot = str(rejected_cot.get('cot', '')).strip()
            if ch_cot and re_cot and ch_cot != re_cot:
                dpo_pairs.append(
                    DpoPair(
                        system=system_prompt,
                        prompt=question,
                        chosen=ch_cot,
                        rejected=re_cot,
                    ))

    return {'pairs': dpo_pairs}


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description='Generate DPO pairs from reasoning data.')
    # --- Argument Definitions ---
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to the input JSONL file.')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Path for the output JSONL file.')
    parser.add_argument('--model_name_or_path',
                        type=str,
                        required=True,
                        help='Model name or path for loading the tokenizer.')
    parser.add_argument('--cache_dir',
                        type=str,
                        default='/root/llmtuner/hfhub/cache_dir',
                        help='HuggingFace cache directory.')
    parser.add_argument(
        '--num_proc',
        type=int,
        default=32,
        help='Number of processes for parallel data processing.')
    parser.add_argument('--max_cot_len',
                        type=int,
                        default=32768,
                        help='Maximum token length for a CoT response.')
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt template. Default value is built-in.')
    parser.add_argument('--math_cot_prompt',
                        type=str,
                        default=None,
                        help='Math CoT prompt. Default value is built-in.')
    parser.add_argument(
        '--add_generation_prompt',
        action='store_true',
        help=
        'Whether to add a generation prompt token (e.g., `<|im_start|>assistant`). This option is only effective when using the "tokenizer" method.'
    )
    parser.add_argument(
        '--apply_chat_template_method',
        type=str,
        choices=['tokenizer', 'formated'],
        default='formated',
        help=
        'Method for applying chat templates. "tokenizer" uses the HuggingFace tokenizer, while "formated" uses custom string templates.'
    )
    parser.add_argument('--debug',
                        action='store_true',
                        help='Whether to use debug mode.')
    parser.add_argument('--save_subset',
                        action='store_true',
                        help='Whether to save a smaller subset of the output.')
    parser.add_argument('--subset_size',
                        type=int,
                        default=256,
                        help='Size of the subset to save.')
    parser.add_argument('--subset_output_path',
                        type=str,
                        default='dpo_output_subset.jsonl',
                        help='Path for the subset output file.')
    return parser


def main() -> None:
    """
    Main function to orchestrate the DPO dataset generation process.

    This function performs the following steps:
    1. Parses command-line arguments.
    2. Initializes the tokenizer based on the provided model path.
    3. Loads the raw dataset from the input file.
    4. Maps a `generate_dpo_pairs` function over the dataset to create and format DPO pairs.
    5. Flattens the resulting list of pairs into a single, unified dataset.
    6. Saves the final DPO dataset to the specified output file.
    7. Optionally saves a smaller subset of the dataset for quick inspection or testing.
    """
    args = build_arg_parser().parse_args()

    # Resolve prompts (use defaults only if user didn't pass them)
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    math_cot_prompt = args.math_cot_prompt or DEFAULT_MATH_COT_PROMPT

    # --- Step 1: Load tokenizer ---
    logger.info(f'Loading tokenizer from model: {args.model_name_or_path}')
    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=args.cache_dir)
        if not tokenizer.chat_template:
            logger.warning(
                f'Model {args.model_name_or_path} does not have a chat template. '
                'Generated output may not be optimally formatted for chat models.'
            )
    except Exception as e:
        logger.error(f'Failed to load tokenizer: {e}')
        return

    # --- Step 2: Load the raw dataset ---
    logger.info(f'Loading dataset from {args.input_path}')
    try:
        dataset: Dataset = load_dataset(
            'json',
            data_files=args.input_path,
            split='train',
            cache_dir=args.cache_dir,
        )
    except Exception as e:
        logger.error(f'Failed to load dataset: {e}')
        return

    # --- Step 3: Generate DPO pairs ---
    logger.info('Generating DPO pairs...')
    generate_func = partial(
        generate_dpo_pairs,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_cot_len=args.max_cot_len,
    )

    mapped_dataset: Dataset = dataset.map(
        generate_func,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        batched=False,
        desc='Building DPO pairs',
    )

    # --- Step 4: Flatten the data into a single DPO dataset ---
    logger.info('Flattening DPO pairs...')
    flat_dpo_data: List[Dict[str, Any]] = list(
        chain.from_iterable(row['pairs'] for row in mapped_dataset))
    logger.info(f'Total raw DPO pairs: {len(flat_dpo_data)}')
    if not flat_dpo_data:
        logger.warning('No DPO pairs generated; exiting early.')
        return
    dpo_dataset: Dataset = Dataset.from_list(flat_dpo_data)

    # --- Step 5: Post-process DPO pairs to apply templates to responses ---
    logger.info(
        'Post-processing DPO pairs: applying chat template to chosen and rejected fields...'
    )

    if args.apply_chat_template_method == 'tokenizer':
        apply_chat_template_func = partial(
            apply_model_chat_template,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            additional_prompt=math_cot_prompt,
            add_generation_prompt=args.add_generation_prompt,
        )
    elif args.apply_chat_template_method == 'formated':
        apply_chat_template_func = partial(
            apply_string_chat_template,
            prompt_template=PROMPT_FORMAT_TEMPLATE,
            assistant_template=RESPONSE_FORMAT_TEMPLATE,
            system_prompt=system_prompt,
            additional_prompt=math_cot_prompt,
        )
    print(apply_model_chat_template)
    final_dpo_dataset: Dataset = dpo_dataset.map(
        apply_chat_template_func,
        num_proc=args.num_proc,
        desc='Applying model chat templates')

    # --- Step 6: Save the final dataset ---
    logger.info(f'Saving DPO dataset to {args.output_path}')
    final_dpo_dataset.to_json(args.output_path, lines=True)

    if args.save_subset:
        subset_size: int = args.subset_size
        logger.info(
            f'Saving subset of size {subset_size} to {args.subset_output_path}'
        )
        subset_dpo_dataset: Dataset = final_dpo_dataset.select(
            range(subset_size))
        subset_dpo_dataset.to_json(args.subset_output_path, lines=True)

    logger.info('DPO dataset generation completed.')


if __name__ == '__main__':
    main()
