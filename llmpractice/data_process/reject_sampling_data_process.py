import argparse
import json
import logging
import os
import re  # Import the regex library
import sys
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Final, Iterable, List, Optional, TypedDict, Union

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
logger = logging.getLogger(__name__)  # Use __name__ to get the module's name

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
    '<|im_start|>user\n{user_question}\n{additional_prompt}<|im_end|>\n'
    '<|im_start|>assistant\n')

RESPONSE_FORMAT_TEMPLATE: Final[str] = ('{assistant_response}<|im_end|>\n')

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
    Defines the structure of statistical metadata for CoT entries of a question.

    Attributes:
        total_answers (int): The total number of CoTs for a given question.
        correct_count (int): The number of correct CoTs.
        cots_token_len (List[int]): A list of token lengths for each valid CoT.
        avg_cot_token_len (float): The average token length of all valid CoTs.
        max_cot_token_len (int): The maximum token length among all valid CoTs.
        min_cot_token_len (int): The minimum token length among all valid CoTs.
    """
    total_answers: int
    correct_count: int
    cots_token_len: List[int]
    avg_cot_token_len: float
    max_cot_token_len: int
    min_cot_token_len: int


class SFTCOTData(TypedDict):
    """
    TypedDict representing a single SFT (Supervised Fine-Tuning) Chain of Thought data sample.

    This structure is used for the final processed dataset, where each entry represents
    a correct CoT paired with its corresponding prompt, response, ground truth, and metadata.

    Attributes:
        prompt (str): The formatted user prompt.
        response (str): The formatted assistant response containing the CoT.
        ground_truth (str): The ground truth answer.
        is_correct (bool): Always True, as this dataset only contains correct CoTs.
        response_token_len (int): The token length of the correct CoT response.
        metadata (MetaData): Statistical metadata about all CoTs for the prompt.
    """
    prompt: str
    response: str
    ground_truth: str
    is_correct: bool
    response_token_len: int
    metadata: MetaData


def is_integer(s: str) -> bool:
    """
    Checks if a string represents a valid integer using a regular expression.
    This method is more robust than a simple try-except block for non-integer
    strings like floats ("3.0") or scientific notation ("3e4").

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is a valid integer, False otherwise.
    """
    if not s:
        return False
    # Use re.fullmatch to ensure the pattern matches the entire string.
    return bool(re.fullmatch(r'[+-]?\d+', s.strip()))


def item_as_bool(value: Any) -> bool:
    """
    Convert various truthy/falsey encodings to a boolean.

    Accepts bool, int, float, or a string representation (e.g., "true", "1", "yes").
    Unrecognized strings and other types default to False.

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


def get_iter_cots(
    cots_field: Union[Dict[str, Any], List[Any],
                      Any]) -> Iterable[Dict[str, Any]]:
    """
    Normalizes the 'cots' field from a dataset item to an iterable of dictionaries.

    Args:
        cots_field (Union[Dict, List, Any]): The raw 'cots' data from a dataset item.
                                             Can be a dictionary, a list of dictionaries, or other types.

    Returns:
        Iterable[Dict[str, Any]]: An iterator over the valid CoT entries.
    """
    if isinstance(cots_field, dict):
        # Handle cases where 'cots' is a dictionary of CoTs, e.g., {'key1': {'cot': '...'}, ...}
        return (v for v in cots_field.values() if isinstance(v, dict))
    if isinstance(cots_field, list):
        # Handle cases where 'cots' is a list of CoT dictionaries, e.g., [{'cot': '...'}, ...]
        return (v for v in cots_field if isinstance(v, dict))
    # Return an empty iterator for any other type
    return iter([])


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


class DataProcessor:
    """
    A class to handle the processing of raw dataset items into a formatted SFT dataset.
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
            # If not found, fall back to the custom 'formatted' method.
            if (self.args.apply_chat_template_method == 'tokenizer'
                    and not tokenizer.chat_template):
                logger.warning(
                    f'Model {self.args.model_name_or_path} does not have a chat template. '
                    'Falling back to "formatted" method.')
                self.args.apply_chat_template_method = 'formatted'
            return tokenizer
        except Exception:
            logger.exception('Failed to load tokenizer.')
            sys.exit(1)

    def _apply_chat_template(
        self,
        question: str,
        response: str,
    ) -> tuple[str, str]:
        """
        Applies the chat template to format the prompt and response for SFT.

        This method handles both the HuggingFace tokenizer's `apply_chat_template`
        and a custom string-based formatting method.

        Args:
            question (str): The user's question.
            response (str): The assistant's response.

        Returns:
            tuple[str, str]: The formatted prompt and response strings.
        """
        system_prompt: Optional[str] = self.args.system_prompt
        math_cot_prompt: Optional[str] = self.args.math_cot_prompt

        # Use the tokenizer's built-in chat template if available.
        if self.args.apply_chat_template_method == 'tokenizer':
            # Use the tokenizer's built-in chat template if available.
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            user_question_with_prompt = question
            if math_cot_prompt:
                user_question_with_prompt = f'{question}\n{math_cot_prompt}'

            messages.append({
                'role': 'user',
                'content': user_question_with_prompt
            })

            assistant_response = []
            # Format the assistant's response with a specific tag format
            formatted_response_content = f'<think>{response}'
            # Format the assistant's response with a specific tag format
            assistant_response.append({
                'role': 'assistant',
                'content': formatted_response_content
            })
            # The `response` part will have the assistant's start tag at the beginning, so we prepend it back
            # or simply use the full formatted text and split it in the main function. Let's make this more robust.
            # A cleaner way is to create the prompt and response strings separately and return them.
            # Apply the chat template using the tokenizer
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=self.args.add_generation_prompt)
            formatted_response = self.tokenizer.apply_chat_template(
                assistant_response,
                tokenize=False,
                add_generation_prompt=self.args.add_generation_prompt)

            return formatted_prompt, formatted_response

        elif self.args.apply_chat_template_method == 'formatted':
            # Use custom templates if the tokenizer method is not chosen or available.
            formatted_prompt = PROMPT_FORMAT_TEMPLATE.format(
                system_prompt=system_prompt,
                user_question=question,
                additional_prompt=math_cot_prompt,
            )
            formatted_response = RESPONSE_FORMAT_TEMPLATE.format(
                assistant_response=(f'<think>{response}'))
            return formatted_prompt, formatted_response
        else:
            logger.warning(
                f'Invalid apply_chat_template_method: {self.args.apply_chat_template_method}. '
                'Falling back to "formatted" method.')
            logger.warning('Using unformatted raw text.')
            return question, response

    def generate_sft_data(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, List[SFTCOTData]]:
        """
        Processes a single dataset item to generate a list of SFTCOTData entries.

        This function performs the following steps:
        1. Extracts and validates the question, answer, and CoTs from the item.
        2. Filters out duplicate and invalid CoTs.
        3. Calculates token lengths for each valid CoT.
        4. Compiles statistical metadata about the CoTs.
        5. Generates `SFTCOTData` entries for all correct CoTs that meet the length criteria.
        6. Returns a dictionary containing the list of generated entries.

        Args:
            item (Dict[str, Any]): A single row from the raw dataset.

        Returns:
            Dict[str, List[SFTCOTData]]: A dictionary containing the list of
                                         SFTCOTData entries, where each entry corresponds
                                         to a correct Chain of Thought. Returns an empty
                                         list if no valid correct CoTs are found.
        """
        question: Optional[str] = item.get('question', '').strip()
        ground_truth: Optional[str] = item.get('answer', '').strip()

        if not question or not ground_truth:
            logger.warning('Skipping item with missing question or answer.')
            return {'sft_cots': []}
        # New filtering condition: skip if the ground truth is not an integer
        if not is_integer(ground_truth):
            logger.debug(
                f"Skipping item because the answer '{ground_truth}' is not an integer."
            )
            return {'sft_cots': []}

        raw_cots: Iterable[Dict[str, Any]] = get_iter_cots(item.get('cots'))
        if not raw_cots:
            logger.debug(f"No CoTs found for question: '{question[:50]}...'")
            return {'sft_cots': []}

        # Filter and add token lengths in one pass
        cots_with_len: List[CotWithLength] = []
        seen_cots = set()
        for cot in raw_cots:
            cot_text = str(cot.get('cot', '')).strip()
            # Skip empty or duplicate CoTs
            if not cot_text or cot_text in seen_cots:
                continue
            seen_cots.add(cot_text)
            is_correct = item_as_bool(cot.get('is_correct'))

            # Check if token length is already provided; if not, calculate it
            cot_token_len = cot.get('cot_token_len')
            if cot_token_len is None:
                cot_token_len = get_token_len(cot_text, self.tokenizer)
            # Filter based on min/max token length
            if self.args.min_cot_len <= cot_token_len <= self.args.max_cot_len:
                cots_with_len.append(
                    CotWithLength(
                        cot=cot_text,
                        is_correct=is_correct,
                        cot_token_len=cot_token_len,
                    ))

        if not cots_with_len:
            logger.debug('No valid CoTs found after filtering.')
            return {'sft_cots': []}

        # Calculate metadata
        total_answers = len(cots_with_len)
        correct_count = sum(1 for cot in cots_with_len if cot['is_correct'])
        cot_lengths = [cot['cot_token_len'] for cot in cots_with_len]

        # Handle division by zero for avg_cot_token_len
        avg_cot_token_len = sum(
            cot_lengths) / total_answers if total_answers > 0 else 0
        max_cot_token_len = max(cot_lengths, default=0)
        min_cot_token_len = min(cot_lengths, default=0)

        metadata: MetaData = MetaData(
            total_answers=total_answers,
            correct_count=correct_count,
            cots_token_len=cot_lengths,
            avg_cot_token_len=avg_cot_token_len,
            max_cot_token_len=max_cot_token_len,
            min_cot_token_len=min_cot_token_len,
        )
        # Generate SFT data entries for all correct CoTs
        sft_data_list: List[SFTCOTData] = []
        for cot_entry in cots_with_len:
            if cot_entry['is_correct']:
                formatted_prompt, formatted_response = self._apply_chat_template(
                    question, cot_entry['cot'])

                sft_data_list.append(
                    SFTCOTData(
                        prompt=formatted_prompt,
                        response=formatted_response,
                        ground_truth=ground_truth,
                        is_correct=cot_entry['is_correct'],
                        response_token_len=cot_entry['cot_token_len'],
                        metadata=metadata,
                    ))
        return {'sft_cots': sft_data_list}

    def validate_cot_data(self, cot_data: SFTCOTData) -> bool:
        """
        Validates the generated CoT data to ensure all required fields are present and valid.

        Args:
            cot_data (SFTCOTData): The data sample to validate.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        required_fields = ['prompt', 'cot', 'ground_truth', 'is_correct']

        # Check for presence of required fields
        if not all(field in cot_data for field in required_fields):
            return False

        # Validate token lengths against the configured min/max
        if not self.args.min_cot_len <= cot_data[
                'cot_token_len'] <= self.args.max_cot_len:
            return False

        return True


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description=
        'Generate SFT (Supervised Fine-Tuning) CoT data from a raw dataset.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
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
        '--math_cot_prompt',
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
        help='Method for applying chat templates.\n'
        '"tokenizer" uses the HuggingFace tokenizer\'s built-in chat template.\n'
        '"formatted" uses custom string templates defined in the script.')
    parser.add_argument(
        '--add_generation_prompt',
        action='store_true',
        help=
        'Whether to add a generation prompt token (e.g., `<|im_start|>assistant`). '
        'This option is only effective when using the "tokenizer" method.')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to use debug mode. Enables verbose logging.')
    parser.add_argument('--save_subset',
                        action='store_true',
                        help='Whether to save a smaller subset of the output.')
    parser.add_argument('--subset_size',
                        type=int,
                        default=256,
                        help='Size of the subset to save.')
    parser.add_argument('--subset_output_path',
                        type=str,
                        default='sft_cot_subset.jsonl',
                        help='Path for the subset output file.')
    return parser


def main() -> None:
    """
    Main function to orchestrate the dataset processing pipeline.

    Steps:
    1. Parse command-line arguments.
    2. Initialize the `DataProcessor` to handle tokenization and data transformation.
    3. Load the raw dataset from the specified input path.
    4. Map the `generate_sft_data` function over the dataset in parallel to process each item.
    5. Flatten the resulting list of lists into a single list of SFT data samples.
    6. Convert the flattened list into a HuggingFace `Dataset`.
    7. Save the final processed dataset to the specified output JSONL file.
    8. (Optional) Save a smaller subset of the dataset if specified by the arguments.
    """
    # 1. Parse arguments and configure logging
    args = build_arg_parser().parse_args()

    args.system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    args.math_cot_prompt = args.math_cot_prompt or DEFAULT_MATH_COT_PROMPT

    # Configure logging level if in debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Debug mode enabled. Verbose logging will be active.')

    # 2. Initialize data processor and handle tokenizer loading errors
    data_processor = DataProcessor(args)

    # 3. Load the raw dataset
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f'Input file not found: {input_path}')
        sys.exit(1)

    logger.info(f'Loading dataset from {input_path}')
    try:
        dataset: Dataset = load_dataset(
            'json',
            data_files=str(input_path),
            split='train',
            cache_dir=args.cache_dir,
        )
    except Exception:
        logger.exception('Failed to load dataset.')
        sys.exit(1)

    # 4. Generate SFT data entries in parallel
    logger.info('Generating SFT data...')
    # Use map with batched=False, as generate_sft_data processes one item at a time.
    # The output will be a dataset where each row contains a list of dictionaries.
    mapped_dataset: Dataset = dataset.map(
        data_processor.generate_sft_data,
        num_proc=args.num_proc,
        # Remove original columns to create a clean, new dataset.
        remove_columns=dataset.column_names,
        # Set batched to False as we are processing one item at a time.
        batched=False,
        desc='Building SFT CoT data',
    )
    # The `map` function with batched=False returns a dataset of lists
    # even when the function returns a single list. We need to flatten it.

    # 5. Flatten the data into a single SFT dataset
    logger.info('Flattening SFT data...')
    # The mapped dataset is a list of lists (e.g., [[{...}, {...}], [...],...]).
    # We need to flatten it into a single list of dictionaries.
    flat_sft_data: List[SFTCOTData] = list(
        chain.from_iterable(row['sft_cots'] for row in mapped_dataset
                            if row['sft_cots']))

    logger.info(f'Total SFT CoT data generated: {len(flat_sft_data)}')
    if not flat_sft_data:
        logger.warning('No SFT CoT data generated; exiting early.')
        return
    sft_dataset: Dataset = Dataset.from_list(flat_sft_data)

    # 对数据集进行随机打乱
    sft_dataset = sft_dataset.shuffle(seed=42)  # 使用种子保证可重复性

    # 6. Save the final dataset
    output_path = Path(args.output_path)
    logger.info(
        f'Saving final SFT dataset ({len(sft_dataset)} pairs) to {output_path}'
    )
    try:
        sft_dataset.to_json(str(output_path), lines=True)
    except Exception:
        logger.exception('Failed to save final dataset.')
        sys.exit(1)

    # 7. (Optional) Save a subset if requested
    if args.save_subset:
        subset_size: int = args.subset_size
        subset_output_path = Path(args.subset_output_path)
        logger.info(
            f'Saving subset of size {subset_size} to {subset_output_path}')
        try:
            subset_sft_dataset: Dataset = sft_dataset.select(
                range(min(subset_size, len(sft_dataset))))
            subset_sft_dataset.to_json(str(subset_output_path), lines=True)
        except Exception:
            logger.exception('Failed to save subset dataset.')
            sys.exit(1)

    logger.info('SFT dataset generation completed successfully. ✨')


if __name__ == '__main__':
    main()
