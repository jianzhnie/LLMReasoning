import argparse
import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import (Any, Dict, Final, Iterable, List, MutableMapping, Optional,
                    TypedDict, Union)

from datasets import Dataset, load_dataset
from LLamaTuner.llamatuner.train import sft
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
# Data Structures & Helper Functions
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
    Defines the structure of a DPO training pair.

    Attributes:
        prompt (str): The formatted user prompt.
        ground_truth (str): The ground truth from the dataset.
    """
    total_answers: int
    correct_count: int
    cots_len_list: List[int]
    avg_cot_token_len: float
    max_cot_token_len: int
    min_cot_token_len: int


class SFTCOTData(TypedDict):
    """SFT CoT Data structure.

    A TypedDict representing SFT CoT data.

    Args:
        prompt (str): The prompt text.
        cot (str): The chain-of-thought reasoning.
        ground_truth (str): The ground truth answer.
        is_correct (bool): Whether the CoT is correct.
        cot_token_len (int): The token length of the CoT.
        metadata (MeataData): Metadata about the CoT statistics.
    """
    prompt: str
    cot: str
    ground_truth: str
    is_correct: bool
    cot_token_len: int
    metadata: MetaData


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


class DataProcessor:

    def __init__(self, args):
        self.args = args
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        logger.info(
            f'Loading tokenizer from model: {self.args.model_name_or_path}')
        try:
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path,
                use_fast=True,
                trust_remote_code=True,
                cache_dir=self.args.cache_dir,
            )
            # Check for chat template existence for the 'tokenizer' method
            if (self.args.apply_chat_template_method == 'tokenizer'
                    and not tokenizer.chat_template):
                self.logger.warning(
                    f'Model {self.args.model_name_or_path} does not have a chat template. '
                    'Falling back to "formated" method.')
                self.args.apply_chat_template_method = 'formated'
        except Exception as e:
            logger.error(f'Failed to load tokenizer: {e}')
            return

    def generate_sft_data(
        self,
        item: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        max_cot_len: int = 32768,
        min_cot_len: int = 1024,
    ) -> Dict[str, Any]:
        """
        Process a single example to extract CoT statistics.
        """
        # Input validation
        if not isinstance(item, dict):
            raise ValueError('Input item must be a dictionary')

        question: Optional[str] = item.get('question', '').strip()
        ground_truth: Optional[str] = item.get('answer', '').strip()

        if not question or not ground_truth:
            logger.warning('Skipping item with missing question or answer')
            return {}

        raw_cots: List[Dict[str, Any]] = list(get_iter_cots(item.get('cots')))
        if not raw_cots:
            logger.info(f"No CoTs found for question: '{question[:50]}...'")
            return {}

        cots_with_len: List[CotWithLength] = []
        seen_cots = set()
        for cot in raw_cots:
            cot_text = str(cot.get('cot', '')).strip()
            if not cot_text or cot_text in seen_cots:
                continue
            seen_cots.add(cot_text)
            is_correct = item_as_bool(cot.get('is_correct'))
            cot_token_len = get_token_len(cot_text, tokenizer)
            if min_cot_len <= cot_token_len <= max_cot_len:
                cots_with_len.append(
                    CotWithLength(
                        cot=cot_text,
                        is_correct=is_correct,
                        cot_token_len=cot_token_len,
                    ))

        total_answers = len(cots_with_len)
        correct_count = sum(1 for cot in cots_with_len if cot['is_correct'])
        avg_cot_token_len = (sum(cot['cot_token_len']
                                 for cot in cots_with_len) /
                             total_answers) if total_answers > 0 else 0
        max_cot_token_len = max(
            (cot['cot_token_len'] for cot in cots_with_len), default=0)
        min_cot_token_len = min(
            (cot['cot_token_len'] for cot in cots_with_len), default=0)

        metadata: MetaData = MetaData(
            total_answers=total_answers,
            correct_count=correct_count,
            cots_len_list=[cot['cot_token_len'] for cot in cots_with_len],
            avg_cot_token_len=avg_cot_token_len,
            max_cot_token_len=max_cot_token_len,
            min_cot_token_len=min_cot_token_len,
        )
        # Generate SFT data entries
        sft_data_list: List[SFTCOTData] = []
        if correct_count > 0:
            for cot_entry in cots_with_len:
                if cot_entry['is_correct']:
                    sft_data_list.append(
                        SFTCOTData(prompt=question,
                                   cot=cot_entry['cot'],
                                   ground_truth=ground_truth,
                                   is_correct=cot_entry['is_correct'],
                                   cot_token_len=cot_entry['cot_token_len'],
                                   metadata=metadata))

        return sft_data_list

    def validate_cot_data(self, cot_data: SFTCOTData) -> bool:
        """Validate the generated CoT data"""
        required_fields = ['prompt', 'cot', 'ground_truth', 'is_correct']

        # Check required fields
        if not all(field in cot_data for field in required_fields):
            return False

        # Validate token lengths
        if not self.args.min_cot_len <= cot_data[
                'cot_token_len'] <= self.args.max_cot_len:
            return False

        return True


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
        choices=['tokenizer', 'formated', 'none'],
        default='formated',
        help=
        'Method for applying chat templates. "tokenizer" uses the HuggingFace tokenizer, "formated" uses custom string templates, and "none" returns unformatted text.'
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


def main():
    args = build_arg_parser().parse_args()

    # Configure logging level if in debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode enabled. Verbose logging will be active.')

    # Initialize data processor and tokenizer
    data_processor = DataProcessor(args)
    tokenizer = data_processor.tokenizer

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
        data_processor.generate_sft_data,
        tokenizer=tokenizer,
        max_cot_len=args.max_cot_len,
        min_cot_len=args.min_cot_len,
    )

    mapped_dataset: Dataset = dataset.map(
        generate_func,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        batched=False,
        desc='Building SFT CoT data',
    )
    # --- Step 4: Flatten the data into a single DPO dataset ---
    logger.info('Flattening SFT CoT data...')
    # The `mapped_dataset` contains a list of pairs for each input row.
    # We flatten this list to create a single dataset of SFT CoT data.
    # Flatten the data and filter out empty pairs
    flat_sft_data: List[SFTCOTData] = list((sft_cot
                                            for sublist in mapped_dataset
                                            for sft_cot in sublist if sft_cot))
    logger.info(f'Total raw SFT CoT data generated: {len(flat_sft_data)}')
    if not flat_sft_data:
        logger.warning('No SFT CoT data generated; exiting early.')
        return
    sft_dataset: Dataset = Dataset.from_list(flat_sft_data)

    output_path = Path(args.output_path)
    logger.info(
        f'Saving final SFT dataset ({len(sft_dataset)} pairs) to {output_path}'
    )
    try:
        sft_dataset.to_json(str(output_path), lines=True)
    except Exception as e:
        logger.error(f'Failed to save final dataset: {e}')
        return

    if args.save_subset:
        subset_size: int = args.subset_size
        subset_output_path = Path(args.subset_output_path)
        logger.info(
            f'Saving subset of size {subset_size} to {subset_output_path}')
        try:
            subset_sft_dataset: Dataset = sft_dataset.select(
                range(min(subset_size, len(sft_dataset))))
            subset_sft_dataset.to_json(str(subset_output_path), lines=True)
        except Exception as e:
            logger.error(f'Failed to save subset dataset: {e}')

    logger.info('SFT dataset generation completed successfully. ✨')


if __name__ == '__main__':
    main()
