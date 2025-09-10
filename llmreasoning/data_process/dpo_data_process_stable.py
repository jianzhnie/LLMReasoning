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
from pathlib import Path
from tkinter import N
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

# String templates for formatting
# These are fallback templates if the tokenizer doesn't have a chat template.
PROMPT_FORMAT_TEMPLATE: Final[str] = (
    '<|im_start|>system\n{system_prompt}<|im_end|>\n'
    '<|im_start|>user\n{user_question}\n{additional_prompt}<|im_end|>\n')

RESPONSE_FORMAT_TEMPLATE: Final[str] = (
    '<|im_start|>assistant\n{assistant_response}<|im_end|>\n')

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


# Internal type for CoT data with added token length
class CotWithLength(TypedDict):
    """
    Internal type for a CoT entry with its token length.

    Attributes:
        cot (str): The Chain of Thought reasoning text.
        is_correct (bool): True if the reasoning is correct, False otherwise.
        cot_token_len (int): The number of tokens in the CoT.
    """
    cot: str
    is_correct: bool
    cot_token_len: int


class MetaData(TypedDict):
    """
    Metadata about the DPO pair, primarily for analysis.

    Attributes:
        chosen_cot_len (int): Token length of the chosen response's CoT.
        rejected_cot_len (int): Token length of the rejected response's CoT.
        chosen_is_correct (bool): True if the chosen CoT was correct.
        rejected_is_correct (bool): True if the rejected CoT was correct.
    """
    chosen_cot_len: int
    rejected_cot_len: int
    chosen_is_correct: bool
    rejected_is_correct: bool


class DpoPair(TypedDict):
    """
    Defines the structure of a DPO training pair.

    Attributes:
        system (Optional[str]): An optional system prompt.
        prompt (str): The formatted user prompt.
        ground_truth (str): The ground truth from the dataset.
        chosen (str): The preferred (correct) response.
        rejected (str): The dis-preferred (incorrect) response.
        metadata (MetaData): A dictionary with metadata about the pair.
    """
    system: Optional[str]
    prompt: str
    ground_truth: str
    chosen: str
    rejected: str
    metadata: MetaData


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
        s: str = value.strip().lower()
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
    if not text:
        return 0
    try:
        # Use tokenizer.__call__ for better handling of special tokens and padding
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
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


def format_chat_messages(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> str:
    """Helper function to format messages using a tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def apply_chat_template(
    item: Dict[str, Any],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    system_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
    apply_chat_template_method: str = 'formated',
    add_generation_prompt: bool = True,
    prompt_template: Optional[str] = None,
    assistant_template: Optional[str] = None,
) -> DpoPair:
    """
    Formats a DPO pair's prompt and responses using either a tokenizer's chat
    template, custom string templates, or no template at all.

    Args:
        item (Dict[str, Any]): A dictionary containing 'chosen' and 'rejected' fields.
        tokenizer (Optional[PreTrainedTokenizerBase]): Tokenizer with a chat template.
        system_prompt (Optional[str]): The system prompt.
        additional_prompt (Optional[str]): An optional prompt to append to the user's text.
                                           For example, "Please reason step by step...".
        apply_chat_template_method (str): The method to use ('tokenizer', 'formated', or 'none').
        add_generation_prompt (bool): If True, the template will include a prompt for
                                      the assistant's turn, such as '<|im_start|>assistant\n'.
        prompt_template (Optional[str]): The string template for the prompt.
        assistant_template (Optional[str]): The string template for the assistant's response.

    Returns:
        DpoPair: The formatted DPO pair.
    """
    # Use .get() with a default to avoid KeyError if fields are missing
    user_prompt: str = item.get('prompt', '')
    ground_truth: str = item.get('ground_truth', '')
    chosen_cot_text: str = item.get('chosen', '')
    rejected_cot_text: str = item.get('rejected', '')
    metadata: MetaData = item.get(
        'metadata',
        MetaData(chosen_cot_len=0,
                 rejected_cot_len=0,
                 chosen_is_correct=False,
                 rejected_is_correct=False))

    if apply_chat_template_method == 'tokenizer':
        if not tokenizer:
            raise ValueError(
                "Tokenizer must be provided for 'tokenizer' method.")
        # Apply the HuggingFace tokenizer's chat template
        prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # Apply the additional prompt if it exists
        full_user_prompt = f'{user_prompt}\n{additional_prompt}' if additional_prompt else user_prompt
        # Generate the prompt message
        prompt_messages.append({'role': 'user', 'content': full_user_prompt})

        chosen_messages = [{'role': 'assistant', 'content': chosen_cot_text}]
        rejected_messages = [{
            'role': 'assistant',
            'content': rejected_cot_text
        }]
        # Apply template to User prompt
        # The prompt_formatted should end with the start of the assistant's turn
        prompt_formatted = format_chat_messages(prompt_messages, tokenizer,
                                                add_generation_prompt)
        # Apply template to assistant responses. The generation prompt is not needed here.
        chosen_formatted = format_chat_messages(chosen_messages,
                                                tokenizer,
                                                add_generation_prompt=False)
        rejected_formatted = format_chat_messages(rejected_messages,
                                                  tokenizer,
                                                  add_generation_prompt=False)
    elif apply_chat_template_method == 'formated':

        if not prompt_template or not assistant_template:
            raise ValueError(
                "Templates must be provided for 'formated' method.")

        # Format the prompt using a custom string template
        prompt_formatted = prompt_template.format(
            system_prompt=system_prompt,
            user_question=user_prompt,
            additional_prompt=additional_prompt,
        )
        # Format the chosen and rejected responses using a custom string template
        chosen_formatted = assistant_template.format(
            assistant_response=chosen_cot_text)
        rejected_formatted = assistant_template.format(
            assistant_response=rejected_cot_text)

    else:  # 'none'
        # Return the raw, unformatted text
        prompt_formatted = user_prompt
        chosen_formatted = chosen_cot_text
        rejected_formatted = rejected_cot_text

    return DpoPair(
        system=system_prompt,
        prompt=prompt_formatted,
        ground_truth=ground_truth,
        chosen=chosen_formatted,
        rejected=rejected_formatted,
        metadata=metadata,
    )


def apply_model_chat_template(
    item: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
    apply_chat_template: bool = False,
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
        DpoPair: The dictionary with 'chosen' and 'rejected' fields now formatted,
                 with a TypedDict signature.
    """
    # Use .get() with a default to avoid KeyError if fields are missing
    user_prompt: str = item.get('prompt', '')
    ground_truth: str = item.get('ground_truth', '')
    chosen_cot_text: str = item.get('chosen', '')
    rejected_cot_text: str = item.get('rejected', '')
    metadata: MetaData = item.get(
        'metadata',
        MetaData(chosen_cot_len=0,
                 rejected_cot_len=0,
                 chosen_is_correct=False,
                 rejected_is_correct=False))

    # Apply the additional prompt to the user question
    if additional_prompt:
        user_prompt += f'\n{additional_prompt}'

    if apply_chat_template:
        # Use the HuggingFace tokenizer's chat template
        prompt_messages: List[Dict[str, str]] = []
        if system_prompt:
            prompt_messages.append({
                'role': 'system',
                'content': system_prompt
            })

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
        # The prompt_formatted should end with the start of the assistant's turn
        prompt_formatted: str = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        # Apply template to assistant responses. This should NOT include the generation prompt
        # as it's already part of the `prompt_formatted` string.
        chosen_formatted = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False)
        rejected_formatted = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False)

        return DpoPair(
            system=system_prompt,
            prompt=prompt_formatted,
            ground_truth=ground_truth,
            chosen=chosen_formatted,
            rejected=rejected_formatted,
            metadata=metadata,
        )

    return DpoPair(
        system=system_prompt,
        prompt=user_prompt,
        ground_truth=ground_truth,
        chosen=chosen_cot_text,
        rejected=rejected_cot_text,
        metadata=metadata,
    )


def apply_string_chat_template(
    item: Dict[str, Any],
    prompt_template: str,
    assistant_template: str,
    system_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
) -> DpoPair:
    """
    Formats the user's prompt and chosen/rejected responses using custom string templates.

    Args:
        item (Dict[str, Any]): A dictionary containing 'chosen' and 'rejected' fields with raw text.
        prompt_template (str): The string template for the full user prompt.
        assistant_template (str): The string template for the assistant's response.
        system_prompt (Optional[str]): The system prompt to be included.
        additional_prompt (Optional[str]): An optional prompt to append to the user's text.

    Returns:
        DpoPair: The dictionary with formatted prompts and responses.
    """
    # Use .get() with a default to avoid KeyError if fields are missing
    user_prompt: str = item.get('prompt', '')
    ground_truth: str = item.get('ground_truth', '')
    chosen_cot_text: str = item.get('chosen', '')
    rejected_cot_text: str = item.get('rejected', '')
    metadata: MetaData = item.get(
        'metadata',
        MetaData(chosen_cot_len=0,
                 rejected_cot_len=0,
                 chosen_is_correct=False,
                 rejected_is_correct=False))

    # Format the prompt
    prompt_formatted: str = prompt_template.format(
        system_prompt=system_prompt,
        user_question=user_prompt,
        additional_prompt=additional_prompt,
    )
    # Format the chosen and rejected responses
    chosen_formatted: str = assistant_template.format(
        assistant_response=chosen_cot_text)
    rejected_formatted: str = assistant_template.format(
        assistant_response=rejected_cot_text)

    return DpoPair(
        system=system_prompt,
        prompt=prompt_formatted,
        ground_truth=ground_truth,
        chosen=chosen_formatted,
        rejected=rejected_formatted,
        metadata=metadata,
    )


def generate_dpo_pairs(
    item: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: Optional[str] = None,
    max_cot_len: int = 32768,
    min_cot_len: int = 1024,
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
        min_cot_len (int): The minimum allowed token length for a CoT response.


    Returns:
        Dict[str, List[DpoPair]]: A dictionary containing
        the generated DPO pairs under the key "pairs".
    """
    question: Optional[str] = item.get('question')
    ground_truth: Optional[str] = item.get('answer')

    if not isinstance(question, str) or not question.strip():
        logger.warning("Skipping item without a valid 'question' field.")
        return {'pairs': []}

    raw_cots: List[Dict[str, Any]] = list(get_iter_cots(item.get('cots')))
    if not raw_cots:
        logger.info(f"No CoTs found for question: '{question[:50]}...'")
        return {'pairs': []}

    # Filter and add token lengths in one pass
    cots_with_len: List[CotWithLength] = []
    seen_cots = set()
    for cot in raw_cots:
        cot_text = str(cot.get('cot', '')).strip()
        if not cot_text or cot_text in seen_cots:
            continue
        seen_cots.add(cot_text)
        is_correct = item_as_bool(cot.get('is_correct'))
        cot_token_len = get_token_len(cot_text, tokenizer)
        if min_cot_len < cot_token_len <= max_cot_len:
            cots_with_len.append(
                CotWithLength(
                    cot=cot_text,
                    is_correct=is_correct,
                    cot_token_len=cot_token_len,
                ))

    # Differentiate between correct and incorrect CoTs
    correct_cots = [cot for cot in cots_with_len if cot['is_correct']]
    incorrect_cots = [cot for cot in cots_with_len if not cot['is_correct']]

    dpo_pairs: List[DpoPair] = []

    # Case 1: Both correct and incorrect CoTs are available
    if correct_cots and incorrect_cots:
        for chosen_cot, rejected_cot in product(correct_cots, incorrect_cots):
            meta_data = MetaData(
                chosen_cot_len=chosen_cot['cot_token_len'],
                rejected_cot_len=rejected_cot['cot_token_len'],
                chosen_is_correct=chosen_cot['is_correct'],
                rejected_is_correct=rejected_cot['is_correct'],
            )
            dpo_pairs.append(
                DpoPair(system=system_prompt,
                        prompt=question,
                        ground_truth=ground_truth,
                        chosen=chosen_cot['cot'],
                        rejected=rejected_cot['cot'],
                        metadata=meta_data))
    # Case 2: Only correct CoTs are available, pair shortest vs. longest
    elif correct_cots and not incorrect_cots:
        # Require at least 4 correct CoTs to form a pair to ensure
        # chosen and rejected are distinct.
        if len(correct_cots) < 4:
            return {'pairs': []}

        sorted_by_len: List[CotWithLength] = sorted(
            correct_cots, key=lambda x: x['cot_token_len'])
        # Select the 2 shortest as chosen candidates and the 2 longest as rejected
        chosen_candidates: List[Dict[str, Any]] = sorted_by_len[:2]
        rejected_candidates: List[Dict[str, Any]] = sorted_by_len[-2:]
        for chosen_cot, rejected_cot in product(chosen_candidates,
                                                rejected_candidates):
            meta_data = MetaData(
                chosen_cot_len=chosen_cot['cot_token_len'],
                rejected_cot_len=rejected_cot['cot_token_len'],
                chosen_is_correct=chosen_cot['is_correct'],
                rejected_is_correct=rejected_cot['is_correct'],
            )
            dpo_pairs.append(
                DpoPair(system=system_prompt,
                        prompt=question,
                        ground_truth=ground_truth,
                        chosen=chosen_cot['cot'],
                        rejected=rejected_cot['cot'],
                        metadata=meta_data))

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
                        default='/home/jianzhnie/llmtuner/hfhub/cache_dir',
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
    parser.add_argument('--min_cot_len',
                        type=int,
                        default=1024,
                        help='Minimum token length for a CoT response.')
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt template. Default value is built-in.')
    parser.add_argument(
        '--math-cot-prompt',
        type=str,
        default=None,
        help='An additional prompt to append to the user question, e.g., '
        '"Please reason step by step...". Default value is built-in.')
    parser.add_argument(
        '--apply_model_chat_template',
        action='store_true',
        help='Whether to use the tokenizer\'s chat template for formatting.')
    parser.add_argument(
        '--add_generation_prompt',
        action='store_true',
        help=
        'Whether to add a generation prompt token (e.g., `<|im_start|>assistant`). This option is only effective when using the "tokenizer" method.'
    )
    parser.add_argument(
        '--apply_chat_template_method',
        type=str,
        choices=['tokenizer', 'formated', 'none'],  # Add 'none' to the choices
        default='None',
        help=
        'Method for applying chat templates. "tokenizer" uses the HuggingFace tokenizer, "formated" uses custom string templates, and "none" returns unformatted text.'
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
    6. Applies the final chat templates to the chosen and rejected responses.
    7. Saves the final DPO dataset to the specified output file.
    8. Optionally saves a smaller subset of the dataset for quick inspection or testing.
    """
    args = build_arg_parser().parse_args()

    # Configure logging level if in debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Resolve prompts (use defaults only if user didn't pass them)
    system_prompt: str = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    math_cot_prompt: str = args.math_cot_prompt or DEFAULT_MATH_COT_PROMPT

    # --- Step 1: Load tokenizer ---
    logger.info(f'Loading tokenizer from model: {args.model_name_or_path}')
    try:
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        # Check for chat template existence for the 'tokenizer' method
        if args.apply_chat_template_method == 'tokenizer' and not tokenizer.chat_template:
            logger.warning(
                f'Model {args.model_name_or_path} does not have a chat template. '
                'Falling back to "formated" method.')
            args.apply_chat_template_method = 'formated'
    except Exception as e:
        logger.error(f'Failed to load tokenizer: {e}')
        return

    # --- Step 2: Load the raw dataset ---
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f'Input file not found: {input_path}')
        return
    logger.info(f'Loading dataset from {input_path}')
    try:
        dataset: Dataset = load_dataset(
            'json',
            data_files=str(input_path),
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
        min_cot_len=args.min_cot_len,
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
    # The `mapped_dataset` contains a list of pairs for each input row.
    # We flatten this list to create a single dataset of DPO pairs.
    # Flatten the data and filter out empty pairs
    flat_dpo_data: List[DpoPair] = list(
        chain.from_iterable(row['pairs'] for row in mapped_dataset
                            if row['pairs']))
    logger.info(f'Total raw DPO pairs: {len(flat_dpo_data)}')
    if not flat_dpo_data:
        logger.warning('No DPO pairs generated; exiting early.')
        return
    dpo_dataset: Dataset = Dataset.from_list(flat_dpo_data)

    # --- Step 5: Post-process DPO pairs to apply chat templates ---
    logger.info(
        f'Post-processing DPO pairs using {args.apply_chat_template_method} method...'
    )
    apply_chat_template_func = partial(
        apply_chat_template,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        additional_prompt=math_cot_prompt,
        apply_chat_template_method=args.apply_chat_template_method,
        add_generation_prompt=args.add_generation_prompt,
        prompt_template=
        PROMPT_FORMAT_TEMPLATE,  # Pass templates for 'formated' method
        assistant_template=RESPONSE_FORMAT_TEMPLATE,
    )

    dpo_dataset: Dataset = dpo_dataset.map(
        apply_chat_template_func,
        num_proc=args.num_proc,
        desc=
        f'Applying chat templates using {args.apply_chat_template_method} method',
    )
    # --- Step 6: Save the final dataset ---
    output_path = Path(args.output_path)
    logger.info(f'Saving final DPO dataset to {output_path}')
    dpo_dataset.to_json(str(output_path), lines=True)

    if args.save_subset:
        subset_size: int = args.subset_size
        subset_output_path = Path(args.subset_output_path)
        logger.info(
            f'Saving subset of size {subset_size} to {subset_output_path}')
        subset_dpo_dataset: Dataset = dpo_dataset.select(
            range(min(subset_size, len(dpo_dataset))))
        subset_dpo_dataset.to_json(str(subset_output_path), lines=True)

    logger.info('DPO dataset generation completed successfully. ✨')


if __name__ == '__main__':
    main()
