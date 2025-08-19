import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party library imports
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# === Constants and Configuration ===
# System prompts used for formatting conversations in a chat-based format.
# These prompts are model-specific and guide the model's behavior.
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'deepseek_r1':
    ('A conversation between User and Assistant. The User asks a question, '
     'and the Assistant solves it. The Assistant first thinks about the '
     'reasoning process in the mind and then provides the User with the '
     'answer. The reasoning process is enclosed within <think> </think> '
     'and the answer is enclosed within <answer> </answer>.'),
    'openr1_prompt':
    ('You are a helpful AI Assistant that provides well-reasoned and detailed responses. '
     'You first think about the reasoning process as an internal monologue and then '
     'provide the user with the answer. Respond in the following format: '
     '<think>\n...\n</think>\n<answer>\n...\n</answer>'),
    'none':
    None,
}

# Specific prompt for Qwen models related to math Chain-of-Thought (COT).
QWEN_MATH_COT: str = (
    'Please reason step by step, and put your final answer within \\boxed{}.')

# --- Helper Functions ---


def create_chat_messages(
        user_message: str,
        assistant_response: Optional[str],
        system_prompt: Optional[str] = None,
        qwen_math_cot: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Creates a chat history list with a system prompt and a user-assistant turn.

    Args:
        user_message (str): The user's message in the conversation.
        assistant_response (Optional[str]): The assistant's response to the user.
                                            If None, the assistant's turn is omitted.
        system_prompt (Optional[str]): The system-level instruction for the chat.
                                        If None, the system role is omitted.
        qwen_math_cot (Optional[str]): A specific prompt for Qwen math reasoning.
                                       If provided, it will be appended to the user message.

    Returns:
        List[Dict[str, str]]: A list representing the chat history,
                              formatted for `tokenizer.apply_chat_template`.
    """
    chat: List[Dict[str, str]] = []
    if system_prompt:
        chat.append({'role': 'system', 'content': system_prompt})

    if qwen_math_cot:
        user_message = f'{user_message}\n{qwen_math_cot}'

    chat.append({'role': 'user', 'content': user_message})

    # The assistant_response should only be added if it is not None.
    if assistant_response:
        chat.append({'role': 'assistant', 'content': assistant_response})
    return chat


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
            print(
                f'🔍 Detected local file format: {data_path_obj.suffix}, using JSON loader.'
            )
            # The 'json' loader supports both .json and .jsonl formats.
            dataset = load_dataset('json', data_files=data_path, split='train')
        else:
            print('🌐 Detected dataset name, loading from Hugging Face Hub.')
            dataset = load_dataset(data_path, split='train')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f'Failed to find the data file at: {data_path}') from e
    except Exception as e:
        # Catch any other loading-related errors.
        raise ValueError(
            f'Failed to load dataset from {data_path}: {e}') from e

    print(f'✅ Successfully loaded dataset with {len(dataset)} samples.')
    return dataset


def preprocess_data(
    data: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    input_key: str = 'input',
    label_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    qwen_math_cot: Optional[str] = None,
) -> Dict[str, str]:
    """
    Preprocess a single data entry, applying chat templates or custom prompts.

    Args:
        data (Dict[str, Any]): A dictionary representing a single example from the dataset.
        tokenizer (PreTrainedTokenizerBase): The tokenizer object, required for `apply_chat_template`.
        input_key (str): The key to retrieve the input text (e.g., 'Problem' or 'prompt').
        label_key (Optional[str]): The key to retrieve the label/answer text (e.g., 'Answer').
        system_prompt (Optional[str]): Optional system prompt for chat-based formatting.
        qwen_math_cot (Optional[str]): Optional CoT prompt to append to the user's input.

    Returns:
        Dict[str, str]: A dictionary with two keys:
                        - 'problem': The formatted prompt text.
                        - 'answer': The raw label/answer text.
    """
    input_prompt: str = str(data.get(input_key, ''))
    label_text: str = str(data.get(label_key, '')) if label_key else ''

    # Create the chat history list.
    chat_messages = create_chat_messages(user_message=input_prompt,
                                         assistant_response=None,
                                         system_prompt=system_prompt,
                                         qwen_math_cot=qwen_math_cot)

    # Apply the model's specific chat template. `tokenize=False` returns a string.
    template_message = tokenizer.apply_chat_template(chat_messages,
                                                     tokenize=False)

    return {'problem': template_message, 'answer': label_text}


def process_and_save_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    output_path: str,
    input_key: str = 'Problem',
    label_key: str = 'Answer',
    system_prompt: Optional[str] = None,
    qwen_math_cot: Optional[str] = None,
    num_proc: int = 4,
) -> None:
    """
    Process an entire dataset and save the results to a JSONL file.

    Each example is preprocessed and written as a single line JSON object
    (JSONL format), making it easy to load for model training.

    Args:
        dataset (Dataset): The Hugging Face Dataset object to process.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for formatting.
        output_path (str): The path to the output .jsonl file.
        input_key (str): The key to access the input text in the dataset.
        label_key (str): The key to access the labels/answers in the dataset.
        system_prompt (Optional[str]): Optional system prompt to pass to `preprocess_data`.
        qwen_math_cot (Optional[str]): Optional Chain-of-Thought prompt to pass to `preprocess_data`.
    """
    output_file = Path(output_path)
    # Ensure the parent directory exists.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use functools.partial to wrap the preprocessing function with its arguments
    # This is the cleanest way to pass additional arguments to the map function.
    processing_fn = partial(
        preprocess_data,
        tokenizer=tokenizer,
        input_key=input_key,
        label_key=label_key,
        system_prompt=system_prompt,
        qwen_math_cot=qwen_math_cot,
    )

    # Use multiprocessing to speed up processing
    print(f'🚀 Starting multi-process data processing with {num_proc} cores...')
    processed_dataset = dataset.map(
        processing_fn,
        num_proc=num_proc,
    )

    # Save as JSONL file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    processed_dataset.to_json(
        output_path,
        orient='records',
        lines=True,
        force_ascii=False,
    )
    print('🎉 Finished processing all examples.')
    print(f'💾 Dataset saved to {output_file}')


def chat_template_example(tokenizer: PreTrainedTokenizerBase) -> None:
    """
    An example of how to use `apply_chat_template` for debugging purposes.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to apply the chat template with.
    """
    # The `QWEN_MATH_COT` prompt is a specific user-level instruction, not a system prompt.
    user_message = (
        'Natalia sold clips to 48 of her friends in April, and then she sold '
        'half as many clips in May. How many clips did Natalia sell '
        'altogether in April and May?')

    assistant_response = (
        'Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips '
        'altogether in April and May.\n\\boxed{72}')

    chat_messages = create_chat_messages(
        user_message=user_message,
        assistant_response=assistant_response,
        system_prompt=SYSTEM_PROMPT_FACTORY['none'],
        qwen_math_cot=QWEN_MATH_COT)

    # This function applies the model's specific chat format.
    results = tokenizer.apply_chat_template(chat_messages, tokenize=False)
    print('\n--- Example of Chat Template Output ---')
    print(results)
    print('---------------------------------------')


def main() -> None:
    """
    Main execution function.
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
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='Hugging Face model name or path, used to load the tokenizer.')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Path to save the processed .jsonl output file.')

    # Optional Arguments
    parser.add_argument(
        '--input_key',
        type=str,
        default='prompt',
        help=
        'Key in the dataset for the input text (e.g., "prompt" or "Problem").')
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
        default='none',
        help='Type of system prompt to use. Available choices: ' +
        ', '.join(SYSTEM_PROMPT_FACTORY.keys()))
    parser.add_argument(
        '--use_qwen_math_cot',
        action='store_true',
        help=
        'Use the Qwen math Chain-of-Thought prompt. Overrides --system_prompt_type.'
    )
    parser.add_argument(
        '--run_example',
        action='store_true',
        help=
        'Run only the chat template example without processing the dataset.')

    args = parser.parse_args()

    # Determine which prompt to use based on args
    system_prompt = SYSTEM_PROMPT_FACTORY.get(args.system_prompt_type, None)
    qwen_math_cot = QWEN_MATH_COT if args.use_qwen_math_cot else None

    # === Load Resources ===
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer from '{args.model_name_or_path}': {e}")
        return

    # === Run Main Processing Logic ===
    if args.run_example:
        chat_template_example(tokenizer)
    else:
        dataset = load_custom_dataset(args.data_path)
        process_and_save_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            output_path=args.output_path,
            input_key=args.input_key,
            label_key=args.label_key,
            qwen_math_cot=qwen_math_cot,
            system_prompt=system_prompt,
        )


if __name__ == '__main__':
    main()
