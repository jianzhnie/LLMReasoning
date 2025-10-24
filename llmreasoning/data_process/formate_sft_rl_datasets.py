import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

# Third-party library imports
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
# System prompts used for formatting conversations in a chat-based format.
# These prompts are model-specific and guide the model's behavior.
amthinking_system_prompt: Final[str] = (
    "You are a helpful assistant. To answer the user's question, you first think "
    'about the reasoning process and then provide the user with the answer. '
    'The reasoning process and answer are enclosed within <think> </think> and '
    '<answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think> <answer> answer here </answer>.')

deepseek_r1_system_prompt: Final[str] = (
    'A conversation between User and Assistant. The User asks a question, '
    'and the Assistant solves it. The Assistant first thinks about the '
    'reasoning process in the mind and then provides the User with the '
    'answer. The reasoning process is enclosed within <think> </think> '
    'and the answer is enclosed within <answer> </answer>.')

openr1_system_prompt: Final[str] = (
    'You are a helpful AI Assistant that provides well-reasoned and detailed responses. '
    'You first think about the reasoning process as an internal monologue and then '
    'provide the user with the answer. Respond in the following format: '
    '<think>\n...\n</think>\n<answer>\n...\n</answer>')

qwen_math_cot_prompt: Final[str] = (
    'Please reason step by step, and put your final answer within \\boxed{}.')

default_system_prompt: Final[str] = 'You are a helpful AI assistant.'

# A factory for different types of system prompts.
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'deepseek_r1': deepseek_r1_system_prompt,
    'amthinking': amthinking_system_prompt,
    'openr1': openr1_system_prompt,
    'default': default_system_prompt,
    'empty': None
}

# String templates for formatting
# These are fallback templates if the tokenizer doesn't have a chat template.
PROMPT_FORMAT_TEMPLATE: Final[str] = (
    '<|im_start|>system\n{system_prompt}<|im_end|>\n'
    '<|im_start|>user\n{user_question}\n{additional_prompt}<|im_end|>\n'
    '<|im_start|>assistant\n')

RESPONSE_FORMAT_TEMPLATE: Final[str] = ('{assistant_response}<|im_end|>\n')

# --- Helper Functions ---


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    response: str,
    system_prompt: Optional[str] = None,
    math_cot_prompt: Optional[str] = None,
    apply_chat_template_method: str = 'tokenizer',
    add_generation_prompt: bool = False,
) -> Tuple[str, str]:
    """
    Applies the chat template to format the prompt and response for SFT.

    This method handles both the HuggingFace tokenizer's `apply_chat_template`
    and a custom string-based formatting method.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for formatting.
        question (str): The user's question.
        response (str): The assistant's response.
        system_prompt (Optional[str]): System prompt to include.
        math_cot_prompt (Optional[str]): Math CoT prompt to append.
        apply_chat_template_method (str): Method to use ('tokenizer' or 'formatted').
        add_generation_prompt (bool): Whether to add generation prompt.

    Returns:
        tuple[str, str]: The formatted prompt and response strings.
    """

    # Use the tokenizer's built-in chat template if available.
    if apply_chat_template_method == 'tokenizer':
        # Use the tokenizer's built-in chat template if available.
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        user_question_with_prompt = question
        if math_cot_prompt:
            user_question_with_prompt = f'{question}\n{math_cot_prompt}'

        messages.append({'role': 'user', 'content': user_question_with_prompt})

        assistant_response = []
        # Format the assistant's response with a specific tag format
        formatted_response_content = f'{response}'
        # Format the assistant's response with a specific tag format
        assistant_response.append({
            'role': 'assistant',
            'content': formatted_response_content
        })
        # Apply the chat template using the tokenizer
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt)
        formatted_response = tokenizer.apply_chat_template(
            assistant_response,
            tokenize=False,
            add_generation_prompt=add_generation_prompt)

        return formatted_prompt, formatted_response

    elif apply_chat_template_method == 'formatted':
        # Use custom templates if the tokenizer method is not chosen or available.
        formatted_prompt = PROMPT_FORMAT_TEMPLATE.format(
            system_prompt=system_prompt,
            user_question=question,
            additional_prompt=math_cot_prompt,
        )
        formatted_response = RESPONSE_FORMAT_TEMPLATE.format(
            assistant_response=(f'{response}'))
        return formatted_prompt, formatted_response
    else:
        logger.warning(
            f'Invalid apply_chat_template_method: {apply_chat_template_method}. '
            'Falling back to "formatted" method.')
        logger.warning('Using unformatted raw text.')
        return question, response


def load_custom_dataset(data_path: str) -> Dataset:
    """
    Load a dataset from either a local JSON/JSONL file or from the Hugging Face Hub.

    Args:
        data_path (str): A string path to a local JSON/JSONL file or a Hugging Face
                         dataset name (e.g., 'gemini/math_qa_datasets').

    Returns:
        Dataset: A loaded Hugging Face `Dataset` object, specifically the 'train' split.

    Raises:
        FileNotFoundError: If the local data file does not exist.
        ValueError: If the file format is not supported or if the dataset from
                    the Hub cannot be loaded.
    """
    data_path_obj = Path(data_path)

    try:
        if data_path_obj.suffix in ['.json', '.jsonl']:
            logger.info(
                f'🔍 Detected local file format: {data_path_obj.suffix}, using JSON loader.'
            )
            # The 'json' loader supports both .json and .jsonl formats.
            dataset = load_dataset('json', data_files=data_path, split='train')
        else:
            logger.info(
                '🌐 Detected dataset name, loading from Hugging Face Hub.')
            dataset = load_dataset(data_path, split='train')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f'Failed to find the data file at: {data_path}') from e
    except Exception as e:
        # Catch any other loading-related errors.
        raise ValueError(
            f'Failed to load dataset from {data_path}: {e}') from e

    logger.info(f'✅ Successfully loaded dataset with {len(dataset)} samples.')
    return dataset


# --- DatasetProcessor Class ---


class DatasetProcessor:
    """
    A class to handle the end-to-end processing of a dataset:
    loading, applying chat templates, and saving the formatted data.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the processor with arguments from the command line.
        """
        self.args = args
        self.tokenizer: Optional[
            PreTrainedTokenizerBase] = self.load_tokenizer()

        # Determine prompts based on arguments
        self.system_prompt: Optional[str] = SYSTEM_PROMPT_FACTORY.get(
            args.system_prompt_type, None)
        self.qwen_math_cot: Optional[
            str] = qwen_math_cot_prompt if args.use_qwen_math_cot else None

        # Validation for chat template method
        if self.args.apply_chat_template_method not in [
                'tokenizer', 'formatted'
        ]:
            logger.warning(
                f"Invalid chat template method '{self.args.apply_chat_template_method}' detected. Falling back to 'formatted'."
            )
            self.args.apply_chat_template_method = 'formatted'

    def load_tokenizer(self) -> bool:
        """
        Loads the tokenizer from the specified model path.
        Returns True on success, False otherwise.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(
                f'✅ Tokenizer loaded successfully from {self.args.model_name_or_path}'
            )
            return True
        except Exception as e:
            logger.error(
                f"❌ Error loading tokenizer from '{self.args.model_name_or_path}': {e}"
            )
            return False

    def preprocess_data_entry(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Preprocess a single data entry, applying chat templates or custom prompts.

        Args:
            example (Dict[str, Any]): A dictionary representing a single example from the dataset.

        Returns:
            Dict[str, str]: A dictionary with two keys:
                            - 'problem': The formatted prompt text.
                            - 'answer': The raw label/answer text.
        """
        if not self.tokenizer:
            raise RuntimeError('Tokenizer is not loaded.')

        question: str = example.get(self.args.input_key, '')
        gen_response: str = example.get(self.args.response_key, '')
        ground_truth: str = example.get(self.args.label_key, '')
        if isinstance(gen_response, list):
            gen_response = gen_response[0] if gen_response else ''

        # Create the chat history list.
        formatted_prompt, formatted_response = apply_chat_template(
            tokenizer=self.tokenizer,
            question=question,
            response=gen_response,
            system_prompt=self.system_prompt,
            math_cot_prompt=self.qwen_math_cot,
            apply_chat_template_method=self.args.apply_chat_template_method,
            add_generation_prompt=self.args.add_generation_prompt,
        )
        return {
            'question': formatted_prompt,
            'response': formatted_response,
            'ground_truth': ground_truth
        }

    def process_and_save_dataset(
        self,
        dataset: Dataset,
    ) -> None:
        """
        Process an entire dataset and save the results to a JSONL file.

        Each example is preprocessed and written as a single line JSON object
        (JSONL format), making it easy to load for model training.

        Args:
            dataset (Dataset): The Hugging Face Dataset object to process.
        """
        output_path = Path(self.args.output_path)
        # Ensure the parent directory exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use functools.partial to wrap the preprocessing function with its arguments
        # This is the cleanest way to pass additional arguments to the map function.
        processing_fn = self.preprocess_data_entry

        # Determine columns to remove (original keys)
        keep_columns = [
            self.args.input_key, self.args.response_key, self.args.label_key
        ]
        columns_to_remove: List[str] = [
            col for col in dataset.column_names if col not in keep_columns
        ]

        # Use multiprocessing to speed up processing
        logger.info(
            f'🚀 Starting multi-process data processing with {self.args.num_proc} cores...'
        )
        # Use multiprocessing to speed up processing
        processed_dataset = dataset.map(
            processing_fn,
            num_proc=self.args.num_proc,
            remove_columns=columns_to_remove,  # Remove original columns
        )

        # Save as JSONL file
        processed_dataset.to_json(
            output_path,
            orient='records',
            lines=True,
            force_ascii=False,
        )
        logger.info('🎉 Finished processing all examples.')
        logger.info(f'💾 Dataset saved to {output_path}')


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=
        'Preprocess datasets for model training using a specified chat template.'
    )

    # Required Arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help=
        'Path to the local dataset file (.json, .jsonl) or Hugging Face dataset name.'
    )
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Path to save the processed .jsonl output file.')
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='Hugging Face model name or path, used to load the tokenizer.')

    # Optional Arguments
    parser.add_argument(
        '--input_key',
        type=str,
        default='prompt',
        help=
        'Key in the dataset for the input text (e.g., "prompt" or "Problem").')
    parser.add_argument(
        '--response_key',
        type=str,
        default='response',
        help=
        'Key in the dataset for the response text (e.g., "response" or "Response").'
    )
    parser.add_argument(
        '--label_key',
        type=str,
        default='answer',
        help=
        'Key in the dataset for the label/answer text (e.g., "answer" or "Answer").'
    )
    parser.add_argument(
        '--system_prompt_type',
        type=str,
        choices=list(SYSTEM_PROMPT_FACTORY.keys()),
        default='empty',
        help='Type of system prompt to use. Available choices: ' +
        ', '.join(SYSTEM_PROMPT_FACTORY.keys()))
    parser.add_argument(
        '--use_qwen_math_cot',
        action='store_true',
        help=
        'Use the Qwen math Chain-of-Thought prompt. Overrides --system_prompt_type.'
    )
    parser.add_argument('--add_generation_prompt',
                        action='store_true',
                        help='Add a generation prompt to the response.')
    parser.add_argument('--apply_chat_template_method',
                        type=str,
                        choices=['tokenizer', 'formatted'],
                        default='formatted',
                        help='Method to use for applying chat template.')
    parser.add_argument(
        '--num_proc',
        type=int,
        default=16,
        help=
        'Number of processes to use for parallel processing of the dataset.')

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main execution function, parses arguments and runs the DatasetProcessor.
    """

    args = parse_arguments()
    processor = DatasetProcessor(args)

    # === Run Main Processing Logic ===
    dataset = load_custom_dataset(args.data_path)
    processor.process_and_save_dataset(dataset)


if __name__ == '__main__':
    main()
