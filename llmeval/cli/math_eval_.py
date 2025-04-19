import json
import time
from typing import Callable, Dict, List, Tuple

import fire
import numpy as np
import vllm
from datasets import load_from_disk
from jinja2 import Template
from understand_r1_zero.math_grader import (answer_tag_reward_fn,
                                            answer_tag_reward_fn_for_orz,
                                            boxed_reward_fn)


def apply_qwen_math_template(question: str) -> str:
    """Formats a question using Qwen-style chat template."""
    return (
        '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n'
        + question + '<|im_end|>\n<|im_start|>assistant\n')


def apply_r1_template(question: str) -> str:
    """Formats a question using R1-style template with <think> and <answer> tags."""
    return (
        'A conversation between User and Assistant. The User asks a question, and the Assistant solves it. '
        'The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. '
        'The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags.\nUser: '
        + question + '\nAssistant: <think>')


def apply_prime_zero_template(question: str) -> str:
    """Formats a question using PRIME-Zero template."""
    question += '\n\nPresent the answer in LaTex format: \\boxed{Your answer}'
    return (
        'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
        'The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. '
        'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. '
        f'User: {question}. Assistant:')


def apply_open_reasoner_zero_template(question: str) -> str:
    """Formats a question using Open-Reasoner-Zero prompt template."""
    prompt_template_jinja = Template(
        """{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags. "
    "User: {{prompt}}\nAssistant: <think>""")

    prompt_instruction_template_jinja = Template("""
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
""")

    prompt_instruction = prompt_instruction_template_jinja.render(
        prompt=question)
    return prompt_template_jinja.render(bos_token='',
                                        prompt=prompt_instruction)


def main(
    model_name: str = 'Qwen/Qwen2.5-Math-1.5B',
    tasks: List[str] = ['aime', 'amc', 'math', 'minerva', 'olympiad_bench'],
    template: str = 'qwen_math',
    dataset_name: str = './datasets/evaluation_suite',
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 3000,
    max_model_len: int = 4096,
    n_samples: int = 1,
    max_test: int = 999999,
    save: bool = False,
):
    """Run evaluation over selected math datasets using specified model and prompt template."""

    # Set up sampling parameters
    sampling_params = vllm.SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=2,
        seed=int(time.time_ns()),
    )

    # Load the model
    model = vllm.LLM(
        model_name,
        swap_space=32,
        max_model_len=max_model_len,
        dtype='bfloat16',
        enable_prefix_caching=True,
    )

    # Select the correct prompt template and reward function
    if 'prime' in model_name.lower():
        template = 'prime-zero'
    elif 'open-reasoner-zero' in model_name.lower():
        template = 'open-reasoner-zero'

    apply_template: Callable[[str], str]
    if template in ['qwen_math', 'no']:
        math_reward_fn = boxed_reward_fn
        apply_template = (apply_qwen_math_template
                          if template == 'qwen_math' else lambda x: x)

    elif template == 'r1':
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ['</answer>']
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_r1_template

    elif template == 'prime-zero':
        math_reward_fn = boxed_reward_fn
        apply_template = apply_prime_zero_template

    elif template == 'open-reasoner-zero':
        math_reward_fn = answer_tag_reward_fn_for_orz
        apply_template = apply_open_reasoner_zero_template

    elif template in ['llama-instruct', 'r1d']:
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question: str) -> str:
            return tokenizer.apply_chat_template(
                [{
                    'content': question,
                    'role': 'user'
                }],
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        raise ValueError(f'Unsupported template: {template}')

    results: Dict[str, float] = {}
    avg_lens: Dict[str, float] = {}
    max_lens: Dict[str, int] = {}
    formatted: Dict[str, float] = {}
    to_be_saved = []

    # Load datasets and evaluate
    for task_name, dataset in load_from_disk(dataset_name).items():
        if task_name not in tasks:
            continue

        prompts = list(map(apply_template, dataset['problem'][:max_test]))
        targets = dataset['answer'][:max_test]

        print(f'Running inference on task: {task_name}')
        outputs = model.generate(prompts, sampling_params)

        batch_scores, batch_formatted, batch_lengths = [], [], []

        for i, output in enumerate(outputs):
            gt_repeated = [targets[i]] * sampling_params.n
            rewards, infos = [], []

            for o, gt in zip(output.outputs, gt_repeated):
                info, r = math_reward_fn(o.text, gt, fast=False)
                rewards.append(r)
                infos.append(info)

            rewards_np = np.array(rewards)
            batch_lengths.append([len(o.token_ids) for o in output.outputs])
            batch_scores.append(rewards_np.mean())

            if infos[0]:
                batch_formatted.append(
                    np.mean(
                        [i['formatted'] for i in infos if 'formatted' in i]))

            to_be_saved.append({
                'task_name':
                task_name,
                'prompt':
                output.prompt,
                'gt':
                gt_repeated,
                'model_output': [o.text for o in output.outputs],
                'model_output_token_ids':
                [o.token_ids for o in output.outputs],
                'reward':
                rewards,
            })

        results[task_name] = float(np.mean(batch_scores))
        avg_lens[task_name] = float(np.mean(batch_lengths))
        max_lens[task_name] = int(np.max(batch_lengths))

        if batch_formatted:
            formatted[task_name] = float(np.mean(batch_formatted))

    # Print summary
    print('Results:', results)
    print('Average score:', np.mean(list(results.values())))
    print('Average output lengths:', avg_lens)
    print('Max output lengths:', max_lens)
    print('Formatted stats:', formatted)

    if save:
        timestamp = int(time.time())
        filename = f"model_eval_out_{model_name.replace('/', '_')}_{timestamp}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
        print(f'Saving model outputs to {filename}')
        with open(filename, 'w') as f:
            json.dump(to_be_saved, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
