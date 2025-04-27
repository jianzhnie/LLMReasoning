import json
import os
from datetime import datetime
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.dataset_utils import PromptDataset, load_data
from llmeval.utils.eval_config import EvaluationArguments
from llmeval.utils.eval_metrics import EvaluationMetrics
from llmeval.utils.llm_template import TEMPLATE_FACTORY
from llmeval.utils.logger import init_logger
from llmeval.utils.model_utils import load_hf_lm_and_tokenizer

logger = init_logger(__name__)


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze evaluation results and generate insights."""
    analysis = {}
    for task_name, task_results in results.items():
        task_analysis = {
            'performance_summary': task_results['metrics'],
            'error_patterns': {},
            'common_mistakes': [],
            'reasoning_quality': {},
        }

        # Analyze samples for error patterns
        samples = task_results['samples']
        error_count = 0
        reasoning_scores = []

        for sample in samples:
            if not sample.get('exact_match'):
                error_count += 1
                if 'error_analysis' in sample:
                    error_type = sample['error_analysis']
                    task_analysis['error_patterns'][error_type] = (
                        task_analysis['error_patterns'].get(error_type, 0) + 1)

            if 'reasoning_score' in sample:
                reasoning_scores.append(sample['reasoning_score'])

        # Calculate reasoning quality statistics
        if reasoning_scores:
            task_analysis['reasoning_quality'] = {
                'mean': sum(reasoning_scores) / len(reasoning_scores),
                'min': min(reasoning_scores),
                'max': max(reasoning_scores),
            }

        analysis[task_name] = task_analysis

    return analysis


def save_results(results: Dict[str, Any], output_dir: str, task_name: str):
    """Save evaluation results and analysis to files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results
    results_file = os.path.join(output_dir,
                                f'{task_name}_{timestamp}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate and save analysis
    analysis = analyze_results({task_name: results})
    analysis_file = os.path.join(output_dir,
                                 f'{task_name}_{timestamp}_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f'Results saved to {results_file}')
    logger.info(f'Analysis saved to {analysis_file}')

    return analysis


class MathEvaluator:

    def __init__(self, config: EvaluationArguments) -> None:
        self.args = config

    def evaluate(self) -> Dict[str, Any]:
        logger.info(f'Evaluating model: {self.args.model_name_or_path}')
        os.makedirs(self.args.output_dir, exist_ok=True)

        try:
            if self.args.infer_backend == 'vllm':
                sampling_params = SamplingParams(
                    n=self.args.n_samples,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    self.args.model_name_or_path, trust_remote_code=True)

                model = LLM(
                    model=self.args.model_name_or_path,
                    trust_remote_code=True,
                    dtype='bfloat16',
                    max_model_len=self.args.max_model_len,
                    gpu_memory_utilization=0.96,
                    enable_prefix_caching=True,
                    tensor_parallel_size=self.args.tensor_parallel_size,
                )
            else:
                tokenizer, model = load_hf_lm_and_tokenizer(
                    model_name_or_path=self.args.model_name_or_path,
                    use_fast_tokenizer=True,
                )

                generation_config = {
                    'max_new_tokens': self.args.max_tokens,
                    'temperature': self.args.temperature,
                    'top_p': self.args.top_p,
                    'do_sample': self.args.temperature > 0,
                }
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')
            raise

        all_results = {}
        template = TEMPLATE_FACTORY.get(self.args.systerm_template, '')

        for task_name in self.args.tasks:
            logger.info(f'Evaluating task: {task_name}')
            metrics = EvaluationMetrics()

            try:
                raw_data = load_data(
                    data_name=task_name,
                    split=self.args.eval_dataset_split,
                    data_dir=self.args.eval_dataset_dir,
                )[:self.args.max_test]

                prompt_eval_dataset = PromptDataset(
                    raw_data,
                    tokenizer=tokenizer,
                    input_key=self.args.input_key,
                    label_key=self.args.label_key,
                    systerm_template=template,
                    input_template=self.args.input_template,
                    apply_chat_template=self.args.apply_chat_template,
                )

                prompt_eval_dataloader = DataLoader(
                    prompt_eval_dataset,
                    batch_size=self.args.generation_batch_size,
                    shuffle=False,
                    collate_fn=lambda x: x,
                )

                for batch_inputs in tqdm(prompt_eval_dataloader,
                                         desc=f'Evaluating {task_name}'):
                    prompts = batch_inputs['prompt']
                    labels = batch_inputs['label']

                    try:
                        if self.args.infer_backend == 'vllm':
                            outputs = model.generate(prompts, sampling_params)
                            generated_texts = [
                                output.outputs[0].text for output in outputs
                            ]
                        else:
                            inputs = tokenizer(
                                prompts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.args.max_model_len,
                            )
                            inputs = {
                                k: v.to(model.device)
                                for k, v in inputs.items()
                            }

                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs, **generation_config)
                            generated_texts = tokenizer.batch_decode(
                                outputs, skip_special_tokens=True)

                        # Update metrics with prompts included
                        metrics.update(generated_texts, labels, prompts)

                    except Exception as e:
                        logger.error(f'Error during generation: {str(e)}')
                        continue

                task_results = {
                    'metrics': metrics.get_metrics(),
                    'samples': metrics.results[:10],
                    'config': {
                        'model': self.args.model_name_or_path,
                        'task': task_name,
                        'timestamp': datetime.now().isoformat(),
                        'generation_config': {
                            'temperature': self.args.temperature,
                            'top_p': self.args.top_p,
                            'max_tokens': self.args.max_tokens,
                            'n_samples': self.args.n_samples,
                        },
                    },
                }

                if self.args.save_predictions:
                    task_results['all_predictions'] = metrics.results

                all_results[task_name] = task_results
                analysis = save_results(task_results, self.args.output_dir,
                                        task_name)
                all_results[task_name]['analysis'] = analysis[task_name]

            except Exception as e:
                logger.error(f'Failed to evaluate task {task_name}: {str(e)}')
                continue

        return all_results


def main():
    # Initialize the HfArgumentParser with EvaluationArguments
    parser = HfArgumentParser(EvaluationArguments)

    # Parse arguments from command line
    # This will automatically handle all the arguments defined in EvaluationArguments
    args = parser.parse_args_into_dataclasses()[0]

    # Now you can use the parsed arguments
    print(f'Model path: {args.model_name_or_path}')
    print(f'Task: {args.task}')
    # ... use other arguments as needed

    evaluator = MathEvaluator(args)
    results = evaluator.evaluate()
    print(results)


if __name__ == '__main__':
    # Example usage
    main()
