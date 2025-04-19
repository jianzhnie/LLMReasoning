import argparse
import importlib.util
import json
import os
import pickle
import random
import re
import time
from datetime import datetime
from math import comb
from typing import Any, Dict, List

import vllm.envs as envs
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.data_loader import load_data
from utils.grader import check_is_correct
from utils.math_normalization import extract_answer
from utils.parser import parse_ground_truth, parse_question
from utils.utils import construct_prompt, load_jsonl, save_jsonl, set_seed
from vllm import LLM, SamplingParams


def parse_list(arg: str) -> List[str]:
    """Parses a comma-separated string into a list of strings."""
    return arg.split(',')


def save_completions(completions: Any, filepath: str) -> None:
    """Pickles the completions to a file."""
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='./',
                        help='Model directory')
    parser.add_argument('--n_sampling',
                        type=int,
                        default=1,
                        help='Number of samples')
    parser.add_argument('--k',
                        type=int,
                        default=1,
                        help='Value of k for pass@k calculation')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--data_name',
                        type=str,
                        default='math',
                        help='Dataset identifier')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--prompt_type', type=str, default='qwen-base')
    parser.add_argument('--prompt_file_path', type=str, default='./prompts')
    parser.add_argument('--surround_with_messages', action='store_true')
    parser.add_argument('--use_few_shot', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', type=str, default='auto')
    parser.add_argument('--completions_save_dir',
                        type=str,
                        default='./completions')

    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    print(f'Current stop list: {args.stop}')
    return args


def get_conversation_prompt_by_messages(tokenizer: AutoTokenizer,
                                        messages: List[Dict[str, str]]) -> str:
    """Converts messages to a conversation-style prompt."""
    return tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)


def get_three_prompt(prompt_type: str, data_name: str) -> tuple:
    """Dynamically imports the prompt module and returns system, few-shot, and question format prompts."""
    file_path = os.path.join('./prompts', prompt_type, f'{data_name}.py')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Prompt file not found: {file_path}')

    spec = importlib.util.spec_from_file_location('dynamic_module', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return module.system_prompt, module.few_shot_prompt, module.question_format
    except AttributeError as e:
        raise AttributeError(f'Missing required prompt format: {e}')


def infer(args: argparse.Namespace) -> None:
    """Runs the main inference pipeline."""
    print(f'Evaluating model: {args.model_name_or_path}')
    n_sampling = args.n_sampling
    factor = max(i for i in range(2, 65) if n_sampling % i == 0)
    generation_epoch = n_sampling // factor
    print(f'Sampling factor = {factor}, Epochs = {generation_epoch}')

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=factor,
        top_p=args.top_p,
    )

    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]

    dt_string = datetime.now().strftime('%m-%d_%H-%M')
    model_name = '/'.join(args.model_name_or_path.split('/')[-3:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'

    if os.path.exists(out_file):
        print(f'Output already exists: {out_file}, skipping generation.')
        return

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}',
                exist_ok=True)

    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
    envs.VLLM_HOST_IP = '0.0.0.0' if len(
        available_gpus) == 1 else envs.VLLM_HOST_IP

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              trust_remote_code=True)

    prompt_batch = []
    for example in tqdm(examples, desc='Generating prompts'):
        question = parse_question(example, args.data_name)
        system_prompt, few_shot_prompt, question_format = get_three_prompt(
            args.prompt_type, args.data_name)
        cur_prompt = (few_shot_prompt if args.use_few_shot else
                      '') + question_format.format(question=question)

        if args.surround_with_messages:
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': cur_prompt
                },
            ]
            cur_prompt = get_conversation_prompt_by_messages(
                tokenizer, messages)

        prompt_batch.append(cur_prompt)

    print('Example prompt:', prompt_batch[0])

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus),
        trust_remote_code=True,
        gpu_memory_utilization=0.96,
    )

    file_outputs = [{} for _ in examples]
    correct_cnt = 0
    for epoch in range(generation_epoch):
        completions = llm.generate(prompt_batch, sampling_params)
        save_file = f'{args.completions_save_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}_gen_round{epoch}.pkl'
        save_completions(completions, save_file)

        for i, (example, completion) in enumerate(zip(examples, completions)):
            question = parse_question(example, args.data_name)
            generated = [output.text for output in completion.outputs]
            file_outputs[i].setdefault('question', question)
            file_outputs[i].setdefault('generated_responses',
                                       []).extend(generated)
            file_outputs[i]['id'] = example.get('id')
            file_outputs[i]['source'] = example.get('source')

    pass_at_k_list = []
    for i, example in enumerate(tqdm(examples, desc='Evaluating correctness')):
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        responses = file_outputs[i]['generated_responses']
        answers = [extract_answer(r, args.data_name) for r in responses]
        correctness = [check_is_correct(ans, gt_ans) for ans in answers]
        is_correct = any(correctness)

        file_outputs[i].update({
            'generated_answers': answers,
            'gold_answer': gt_ans,
            'is_correct': is_correct,
            'answers_correctness': correctness,
        })

        correct_cnt += int(is_correct)
        if len(correctness) > 1:
            corrects = sum(correctness)
            n = len(answers)
            pass_k = (1 - (comb(n - corrects, args.k) / comb(n, args.k))
                      if corrects else 0)
            pass_at_k_list.append(pass_k)

    with open(out_file + '.tmp', 'w', encoding='utf-8') as f:
        for output in tqdm(file_outputs, desc='Writing results'):
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
    os.rename(out_file + '.tmp', out_file)

    print(
        f'Accuracy: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}'
    )
    if pass_at_k_list:
        avg_pass_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f'Pass@{args.k}: {avg_pass_k:.4f}')


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    infer(args)
