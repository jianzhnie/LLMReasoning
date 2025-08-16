"""
A utility script to test and verify the chat templates of various
Hugging Face-based language models.

This script iterates through a predefined list of models and system prompts,
applying the chat template to a sample conversation and printing the result.
It serves as a quick tool to check how different models format their
conversational input for inference.
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer

# Model names and their corresponding Hugging Face hub paths.
# These models are used to demonstrate chat template application.
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
# These prompts guide the model's behavior and tone.
SYSTEM_PROMPT_FACTORY: Dict[str, Optional[str]] = {
    'qwen_math_cot':
        'Please reason step by step, and put your final answer within \\boxed{}.',
    'deepseek_r1': (
        'A conversation between User and Assistant. The User asks a question, '
        'and the Assistant solves it. The Assistant first thinks about the '
        'reasoning process in the mind and then provides the User with the '
        'answer. The reasoning process is enclosed within <think> </think> '
        'and the answer is enclosed within <answer> </answer>.'
    ),
    'default': 'You are a helpful assistant.',
    'none': None,
    'openr1_prompt': (
        'You are a helpful AI Assistant that provides well-reasoned and detailed responses. '
        'You first think about the reasoning process as an internal monologue and then '
        'provide the user with the answer. Respond in the following format: '
        '<think>\n...\n</think>\n<answer>\n...\n</answer>'
    )
}


def create_chat_messages(system_prompt: Optional[str],
                       user_message: str,
                       assistant_response: str) -> List[Dict[str, str]]:
    """
    Creates a chat history list with a system prompt and a user-assistant turn.

    Args:
        system_prompt (Optional[str]): The system-level instruction for the chat.
                                        If None, the system role is omitted.
        user_message (str): The user's message in the conversation.
        assistant_response (str): The assistant's response to the user.

    Returns:
        List[Dict[str, str]]: A list representing the chat history,
                              formatted for `tokenizer.apply_chat_template`.
    """
    chat: List[Dict[str, str]] = []
    if system_prompt is not None:
        chat.append({'role': 'system', 'content': system_prompt})
    
    chat.append({'role': 'user', 'content': user_message})
    chat.append({'role': 'assistant', 'content': assistant_response})
    
    return chat


def apply_and_print_template(model_name: str,
                             model_path: str,
                             prompt_name: str,
                             chat_history: List[Dict[str, str]]) -> None:
    """
    Attempts to apply a chat template to a given conversation and prints the result.

    Handles potential exceptions during the process, providing clear error messages.

    Args:
        model_name (str): The display name of the model.
        model_path (str): The local or hub path to the model's tokenizer.
        prompt_name (str): The display name of the system prompt type.
        chat_history (List[Dict[str, str]]): The conversation to format.
    """
    print(f"\n{'='*60}\nModel: {model_name}\nPrompt Type: {prompt_name}\n")
    try:
        # Load the tokenizer from the specified path.
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Apply the chat template to the conversation and get the formatted string.
        result = tokenizer.apply_chat_template(chat_history, tokenize=False)
        print(result)
    except Exception as e:
        # Catch and print any errors that occur during loading or templating.
        print(f"[ERROR] Failed for {model_name} with prompt '{prompt_name}':\n{e}")


def main():
    """
    Main function to handle argument parsing and orchestrate the template application.
    """
    parser = argparse.ArgumentParser(
        description='Test chat templates for various Hugging Face models.')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='/root/llmtuner/hfhub/models/',
        help='The base directory where models are stored. Defaults to a local path.')
    
    args = parser.parse_args()
    model_dir = args.model_dir

    # Sample conversation to use for all models and prompts.
    user_message = (
        'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.'
        'How many clips did Natalia sell altogether in April and May?'
    )
    assistant_response = (
        'Natalia sold 48/2 = 24 clips in May.\n'
        'Natalia sold 48+24 = 72 clips altogether in April and May.\n\\boxed{72}'
    )

    # Iterate through each model and each system prompt, then apply the template.
    for model_name, model_path_suffix in MODEL_PATHS.items():
        full_model_path = os.path.join(model_dir, model_path_suffix)
        
        # Check if the local model path exists before trying to load it.
        if not os.path.exists(full_model_path):
            print(f"Skipping {model_name}: Local path not found at '{full_model_path}'")
            continue

        for prompt_name, system_prompt in SYSTEM_PROMPT_FACTORY.items():
            # For the 'qwen_math_cot' prompt, we need to add a special instruction
            # to the user message content.
            if prompt_name == 'qwen_math_cot':
                full_user_message = user_message + '\n' + SYSTEM_PROMPT_FACTORY['qwen_math_cot']
                chat_history = create_chat_messages(system_prompt, full_user_message, assistant_response)
            else:
                chat_history = create_chat_messages(system_prompt, user_message, assistant_response)
            
            # Call the main function to apply the template
            apply_and_print_template(model_name, full_model_path, prompt_name, chat_history)


if __name__ == '__main__':
    main()
