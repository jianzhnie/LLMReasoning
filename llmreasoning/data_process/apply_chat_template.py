"""
A utility script to test and verify the chat templates of various
Hugging Face-based language models.

This script iterates through a predefined list of models and system prompts,
applying the chat template to a sample conversation and printing the result.
It serves as a quick tool to check how different models format their
conversational input for inference.
"""

import argparse
import os
from typing import Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

# --- Constants for Model and Prompt Definitions ---
# These constants define the models and prompts to be tested.

# Model names and their corresponding Hugging Face hub paths.
MODEL_PATHS: Dict[str, str] = {
    'Qwen2.5-7B': 'Qwen/Qwen2.5-7B',
    'QwQ-32B': 'Qwen/QwQ-32B',
    'Qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'AM-Thinking-v1': 'a-m-team/AM-Thinking-v1',
    'DeepSeek-R1-Distill-Qwen-32B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'Skywork-OR1-32B': 'Skywork/Skywork-OR1-32B',
    'DeepSeek-R1': 'deepseek-ai/DeepSeek-R1',
    'OpenThinker3-7B': 'open-thoughts/OpenThinker3-7B'
}

# A factory for different types of system prompts.
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
    'default':
    'You are a helpful assistant.',
    'none':
    None,
}

# Specific prompt for Qwen models related to math Chain-of-Thought (COT).
QWEN_MATH_COT: str = (
    'Please reason step by step, and put your final answer within \\boxed{}.')

# --- Helper Functions ---


def create_chat_messages(
        user_message: str,
        assistant_response: str,
        system_prompt: Optional[str] = None,
        qwen_math_cot: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Creates a chat history list with a system prompt and a user-assistant turn.

    Args:
        user_message (str): The user's message in the conversation.
        assistant_response (str): The assistant's response to the user.
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
    chat.append({'role': 'assistant', 'content': assistant_response})
    return chat


def get_tokenizer(model_path: str) -> Optional[PreTrainedTokenizer]:
    """
    Loads and returns a tokenizer from the specified path.

    Handles potential exceptions if the tokenizer cannot be loaded.

    Args:
        model_path (str): The local or hub path to the model's tokenizer.

    Returns:
        Optional[PreTrainedTokenizer]: The loaded tokenizer, or None if an error occurs.
    """
    try:
        # We set `trust_remote_code=True` to allow loading custom tokenizers.
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer from '{model_path}':\n{e}")
        return None


def apply_and_print_template(model_name: str, prompt_name: str,
                             tokenizer: PreTrainedTokenizer,
                             chat_history: List[Dict[str, str]]) -> None:
    """
    Applies a chat template and prints the result, handling potential errors.

    Args:
        model_name (str): The display name of the model.
        prompt_name (str): The display name of the system prompt type.
        tokenizer (PreTrainedTokenizer): The tokenizer with the chat template.
        chat_history (List[Dict[str, str]]): The conversation to format.
    """
    print(f"\n{'='*60}\nModel: {model_name}\nPrompt Type: {prompt_name}\n")
    try:
        result = tokenizer.apply_chat_template(chat_history, tokenize=False)
        print(result)
    except Exception as e:
        print(
            f"[ERROR] Failed to apply template for {model_name} with prompt '{prompt_name}':\n{e}"
        )


# --- Main Logic ---


def main() -> None:
    """
    Parses command-line arguments and orchestrates the chat template testing.
    """
    parser = argparse.ArgumentParser(
        description='Test chat templates for various Hugging Face models.')
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.environ.get('HF_MODEL_DIR',
                               '/home/jianzhnie/llmtuner/hfhub/models/'),
        help=
        'The base directory where models are stored. Can be overridden by HF_MODEL_DIR env var.'
    )
    parser.add_argument(
        '--use-qwen-math-cot',
        action='store_true',
        help='A flag to append the Qwen math COT prompt to the user message.')

    args = parser.parse_args()
    model_dir = args.model_dir

    qwen_math_cot = QWEN_MATH_COT if args.use_qwen_math_cot else None

    user_message = (
        'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.'
        'How many clips did Natalia sell altogether in April and May?')
    assistant_response = (
        'Natalia sold 48/2 = 24 clips in May.\n'
        'Natalia sold 48+24 = 72 clips altogether in April and May.\n\\boxed{72}'
    )

    for model_name, model_path_suffix in MODEL_PATHS.items():
        full_model_path = os.path.join(model_dir, model_path_suffix)

        if not os.path.exists(full_model_path):
            print(
                f"Skipping {model_name}: Local path not found at '{full_model_path}'."
            )
            continue

        tokenizer = get_tokenizer(full_model_path)
        if not tokenizer:
            continue

        for prompt_name, system_prompt in SYSTEM_PROMPT_FACTORY.items():
            chat_messages = create_chat_messages(user_message,
                                                 assistant_response,
                                                 system_prompt=system_prompt,
                                                 qwen_math_cot=qwen_math_cot)
            apply_and_print_template(model_name, prompt_name, tokenizer,
                                     chat_messages)


if __name__ == '__main__':
    main()
