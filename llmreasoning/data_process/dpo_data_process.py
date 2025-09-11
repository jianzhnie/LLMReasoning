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
import sys
from itertools import chain, product
from pathlib import Path
from typing import (Any, Dict, Final, Iterable, List, MutableMapping, Optional,
                    TypedDict, Union)

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------

# Configure logging for better visibility and control
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
# Type Definitions
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
        total_answers (int): Total number of CoTs for the question.
        correct_count (int): Number of correct CoTs.
        cots_token_len (List[int]): A list of all CoT token lengths.
        avg_cot_token_len (float): Average token length of all CoTs.
        max_cot_token_len (int): Maximum token length.
        min_cot_token_len (int): Minimum token length.
        chosen_cot_token_len (int): Token length of the chosen response's CoT.
        rejected_cot_token_len (int): Token length of the rejected response's CoT.
        chosen_is_correct (bool): True if the chosen CoT was correct.
        rejected_is_correct (bool): True if the rejected CoT was correct.
    """
    total_answers: int
    correct_count: int
    cots_token_len: List[int]
    avg_cot_token_len: float
    max_cot_token_len: int
    min_cot_token_len: int
    chosen_cot_token_len: int
    rejected_cot_token_len: int
    chosen_is_correct: bool
    rejected_is_correct: bool


class DpoPair(TypedDict):
    """
    Defines the structure of a DPO training pair.

    Attributes:
        prompt (str): The formatted user prompt.
        ground_truth (str): The ground truth from the dataset.
        chosen (str): The preferred (correct) response.
        rejected (str): The dis-preferred (incorrect) response.
        metadata (MetaData): A dictionary with metadata about the pair.
    """
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
    return False


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
        # Using `tokenizer.tokenize` for a direct token count.
        tokens = tokenizer.tokenize(text)
        return len(tokens)
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


class DataProcessor:
    """
    A class to handle the entire DPO pair generation process for a dataset.
    """

    def __init__(self, args: argparse.Namespace):
        """Initializes the DataProcessor.

        Args:
            args (argparse.Namespace): The parsed command-line arguments.
        """
        self.args = args
        self.tokenizer: PreTrainedTokenizerBase = self._load_tokenizer()

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Loads the tokenizer from the specified model name or path.

        Returns:
            PreTrainedTokenizerBase: The loaded tokenizer instance.
        """
        logger.info(
            f'Loading tokenizer from model: {self.args.model_name_or_path}')
        try:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path,
                use_fast=True,
                trust_remote_code=True,
                cache_dir=self.args.cache_dir,
            )
            # Check for chat template existence for the 'tokenizer' method.
            if (self.args.apply_chat_template_method == 'tokenizer'
                    and not tokenizer.chat_template):
                logger.warning(
                    f'Model {self.args.model_name_or_path} does not have a chat template. '
                    'Falling back to "formatted" method.')
                # ERROR FIX: Assigning the fallback method directly to the args.
                self.args.apply_chat_template_method = 'formatted'
            return tokenizer
        except Exception:
            logger.exception('Failed to load tokenizer.')
            sys.exit(1)

    def format_chat_messages(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        """Helper function to format messages using a tokenizer's chat template."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt)

    def _apply_chat_template(
        self,
        question: str,
        chosen_response: str,
        rejected_response: str,
    ) -> tuple[str, str, str]:
        """
        Formats a DPO pair's prompt and responses using a selected method.

        Args:
            question (str): The user's original question.
            chosen_response (str): The preferred assistant response.
            rejected_response (str): The dis-preferred assistant response.

        Returns:
            tuple[str, str, str]: A tuple containing the formatted prompt, chosen
                                    response, and rejected response.
        """
        # Use .get() with a default to avoid KeyError if fields are missing
        system_prompt: Optional[str] = self.args.system_prompt
        math_cot_prompt: Optional[str] = self.args.math_cot_prompt

        if self.args.apply_chat_template_method == 'tokenizer':

            # Prepare messages for prompt and responses
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})

            user_question_with_prompt = question
            if math_cot_prompt:
                user_question_with_prompt = f'{question}\n{math_cot_prompt}'

            messages.append({
                'role': 'user',
                'content': user_question_with_prompt
            })

            chosen_messages = [{
                'role': 'assistant',
                'content': chosen_response
            }]
            rejected_messages = [{
                'role': 'assistant',
                'content': rejected_response
            }]
            # Apply template to User prompt
            # The prompt_formatted should end with the start of the assistant's turn
            formatted_prompt = self.format_chat_messages(
                messages, self.args.add_generation_prompt)
            # Apply template to assistant responses. The generation prompt is not needed here.
            formatted_chosen = self.format_chat_messages(
                chosen_messages, add_generation_prompt=False)
            formatted_rejected = self.format_chat_messages(
                rejected_messages, add_generation_prompt=False)
        elif self.apply_chat_template_method == 'formatted':

            formatted_prompt = PROMPT_FORMAT_TEMPLATE.format(
                system_prompt=system_prompt,
                user_question=question,
                additional_prompt=math_cot_prompt,
            )
            formatted_chosen = RESPONSE_FORMAT_TEMPLATE.format(
                assistant_response=chosen_response)
            # Format the chosen and rejected responses using a custom string template
            formatted_rejected = RESPONSE_FORMAT_TEMPLATE.format(
                assistant_response=rejected_response)

        else:  # 'none'
            # Return the raw, unformatted text
            logger.warning(
                f'Invalid apply_chat_template_method: {self.args.apply_chat_template_method}. '
                'Falling back to "formatted" method.')
            logger.warning('Using unformatted raw text.')

            formatted_prompt, formatted_chosen, formatted_rejected = question, chosen_response, rejected_response

        return formatted_prompt, formatted_chosen, formatted_rejected

    def generate_dpo_pairs(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, List[DpoPair]]:
        """
        Processes a single data item to generate DPO pairs (prompt, chosen, rejected).

        This function formats the prompt, filters out CoT responses that are too long,
        and then creates pairs of correct vs. incorrect responses. If only correct
        responses are available, it pairs the shortest with the longest.

        Args:
            item (Dict[str, Any]): The input data sample, expected to contain "question" and "cots".

        Returns:
            Dict[str, List[DpoPair]]: A dictionary containing
            the generated DPO pairs under the key "pairs".
        """
        question: Optional[str] = item.get('question', '').strip()
        ground_truth: Optional[str] = item.get('answer', '').strip()

        if not question or not ground_truth:
            logger.warning('Skipping item with missing question or answer.')
            return {'pairs': []}

        # Normalize the 'cots' field to an iterable list of dicts.
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

            # Check if token length is already provided; if not, calculate it
            cot_token_len = cot.get('cot_token_len')
            if cot_token_len is None:
                cot_token_len = get_token_len(cot_text, self.tokenizer)

            # Apply length filtering
            if self.args.min_cot_len <= cot_token_len <= self.args.max_cot_len:
                cots_with_len.append(
                    CotWithLength(
                        cot=cot_text,
                        is_correct=is_correct,
                        cot_token_len=cot_token_len,
                    ))

        if not cots_with_len:
            logger.debug('No valid CoTs found after filtering.')
            return {'pairs': []}

        # Calculate metadata before creating pairs
        total_answers = len(cots_with_len)
        correct_count = sum(1 for cot in cots_with_len if cot['is_correct'])
        cot_lengths = [cot['cot_token_len'] for cot in cots_with_len]

        # Handle division by zero for avg_cot_token_len
        avg_cot_token_len = sum(
            cot_lengths) / total_answers if total_answers > 0 else 0
        max_cot_token_len = max(cot_lengths, default=0)
        min_cot_token_len = min(cot_lengths, default=0)

        # Differentiate between correct and incorrect CoTs
        correct_cots = [cot for cot in cots_with_len if cot['is_correct']]
        incorrect_cots = [
            cot for cot in cots_with_len if not cot['is_correct']
        ]

        dpo_pairs: List[DpoPair] = []

        # Case 1: Both correct and incorrect CoTs are available
        if correct_cots and incorrect_cots:
            for chosen_cot, rejected_cot in product(correct_cots,
                                                    incorrect_cots):
                meta_data = MetaData(
                    total_answers=total_answers,
                    correct_count=correct_count,
                    cots_token_len=cot_lengths,
                    avg_cot_token_len=avg_cot_token_len,
                    max_cot_token_len=max_cot_token_len,
                    min_cot_token_len=min_cot_token_len,
                    chosen_cot_token_len=chosen_cot['cot_token_len'],
                    rejected_cot_token_len=rejected_cot['cot_token_len'],
                    chosen_is_correct=chosen_cot['is_correct'],
                    rejected_is_correct=rejected_cot['is_correct'],
                )

                formatted_prompt, formatted_chosen, formatted_rejected = self._apply_chat_template(
                    question, chosen_cot['cot'], rejected_cot['cot'])

                dpo_pairs.append(
                    DpoPair(
                        prompt=formatted_prompt,
                        ground_truth=ground_truth,
                        chosen=formatted_chosen,
                        rejected=formatted_rejected,
                        metadata=meta_data,
                    ))
        # Case 2: Only correct CoTs are available, pair shortest vs. longest
        elif correct_cots:
            # Require at least 4 correct CoTs to form a pair to ensure
            # chosen and rejected are distinct.
            if len(correct_cots) < 4:
                return {'pairs': []}

            # Sort by length for the shortest vs longest strategy
            sorted_by_len: List[CotWithLength] = sorted(
                correct_cots, key=lambda x: x['cot_token_len'])
            # Select the 2 shortest as chosen candidates and the 2 longest as rejected
            chosen_candidates: List[Dict[str, Any]] = sorted_by_len[:2]
            rejected_candidates: List[Dict[str, Any]] = sorted_by_len[-2:]
            for chosen_cot, rejected_cot in product(chosen_candidates,
                                                    rejected_candidates):
                meta_data = MetaData(
                    total_answers=total_answers,
                    correct_count=correct_count,
                    cots_token_len=cot_lengths,
                    avg_cot_token_len=avg_cot_token_len,
                    max_cot_token_len=max_cot_token_len,
                    min_cot_token_len=min_cot_token_len,
                    chosen_cot_token_len=chosen_cot['cot_token_len'],
                    rejected_cot_token_len=rejected_cot['cot_token_len'],
                    chosen_is_correct=chosen_cot['is_correct'],
                    rejected_is_correct=rejected_cot['is_correct'],
                )
                formatted_prompt, formatted_chosen, formatted_rejected = self._apply_chat_template(
                    question, chosen_cot['cot'], rejected_cot['cot'])

                dpo_pairs.append(
                    DpoPair(
                        prompt=formatted_prompt,
                        ground_truth=ground_truth,
                        chosen=formatted_chosen,
                        rejected=formatted_rejected,
                        metadata=meta_data,
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
        help='System prompt template. If not provided, a default will be used.'
    )
    parser.add_argument(
        '--math-cot-prompt',
        type=str,
        default=None,
        help='An additional prompt to append to the user question, e.g., '
        '"Please reason step by step...". If not provided, a default will be used.'
    )
    parser.add_argument(
        '--apply_chat_template_method',
        type=str,
        choices=['tokenizer', 'formatted'],
        default='formatted',
        help=
        'Method for applying chat templates. "tokenizer" uses the HuggingFace tokenizer, "formatted" uses custom string templates, and "none" returns unformatted text.'
    )
    parser.add_argument(
        '--add_generation_prompt',
        action='store_true',
        help=
        'Whether to add a generation prompt token (e.g., `<|im_start|>assistant`). This option is only effective when using the "tokenizer" method.'
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

    # Configure logging level if in debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode enabled. Verbose logging will be active.')

    # Set prompts (use defaults if user didn't pass them)
    # No need to re-declare variables as they are properties of args.
    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.math_cot_prompt is None:
        args.math_cot_prompt = DEFAULT_MATH_COT_PROMPT

    # 1. Initialize data processor and handle tokenizer loading errors
    # The constructor handles tokenizer loading and validation.
    data_processor = DataProcessor(args)

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
    # Use map with batched=False, as generate_dpo_pairs processes one item at a time.
    mapped_dataset: Dataset = dataset.map(
        data_processor.generate_dpo_pairs,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        batched=False,
        desc='Building DPO data',  # Updated desc for clarity
    )

    # --- Step 4: Flatten the data into a single DPO dataset ---
    logger.info('Flattening DPO pairs...')
    # The `mapped_dataset` contains a list of pairs for each input row.
    # We flatten this list to create a single dataset of DPO pairs.
    flat_dpo_data: List[DpoPair] = list(
        chain.from_iterable(row['pairs'] for row in mapped_dataset
                            if row['pairs']))

    if not flat_dpo_data:
        logger.warning('No DPO pairs generated; exiting early.')
        return

    logger.info(f'Total raw DPO pairs generated: {len(flat_dpo_data)}')
    dpo_dataset: Dataset = Dataset.from_list(flat_dpo_data)

    dpo_dataset = dpo_dataset.shuffle(seed=42)

    # --- Step 5: Save the final dataset ---
    output_path = Path(args.output_path)
    logger.info(
        f'Saving final DPO dataset ({len(dpo_dataset)} pairs) to {output_path}'
    )
    try:
        dpo_dataset.to_json(str(output_path), lines=True)
    except Exception as e:
        logger.error(f'Failed to save final dataset: {e}')
        return

    if args.save_subset:
        subset_size: int = args.subset_size
        subset_output_path = Path(args.subset_output_path)
        logger.info(
            f'Saving subset of size {subset_size} to {subset_output_path}')
        try:
            subset_dpo_dataset: Dataset = dpo_dataset.select(
                range(min(subset_size, len(dpo_dataset))))
            subset_dpo_dataset.to_json(str(subset_output_path), lines=True)
        except Exception as e:
            logger.error(f'Failed to save subset dataset: {e}')

    logger.info('DPO dataset generation completed successfully. ✨')


if __name__ == '__main__':
    main()
