import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llmeval.utils.dataset_utils import PromptDataset, load_data
from llmeval.utils.llm_template import TEMPLATE_FACTORY
from llmeval.utils.math_grader import MathAccuracyReward
from llmeval.utils.model_utils import load_hf_lm_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Enhanced evaluation metrics with answer normalization and custom comparison"""

    def __init__(self):
        self.total = 0
        self.correct = 0
        self.partial_correct = 0
        self.results = []
        self.math_grader = MathAccuracyReward()

    def normalize_answer(self, text: str) -> str:
        """Normalize answer text by removing spaces and converting to lowercase."""
        return ''.join(text.lower().split())

    def update(self,
               prediction: str,
               label: str,
               metadata: Dict[str, Any] = None):
        self.total += 1

        # Use math grader for detailed evaluation
        eval_result = self.math_grader(prediction, label)

        if eval_result['exact_match']:
            self.correct += 1
        elif eval_result['partial_match']:
            self.partial_correct += 1

        result = {
            'prediction': prediction,
            'label': label,
            'metadata': metadata,
            'exact_match': eval_result['exact_match'],
            'partial_match': eval_result['partial_match'],
            'reasoning_score': eval_result.get('reasoning_score', 0.0),
            'error_analysis': eval_result.get('error_analysis', ''),
        }
        self.results.append(result)

    def get_metrics(self) -> Dict[str, float]:
        metrics = {
            'total_samples':
            self.total,
            'exact_accuracy':
            self.correct / self.total if self.total > 0 else 0,
            'partial_accuracy':
            self.partial_correct / self.total if self.total > 0 else 0,
            'combined_accuracy': (self.correct + 0.5 * self.partial_correct) /
            self.total if self.total > 0 else 0,
        }

        # Calculate average reasoning scores
        if self.results:
            reasoning_scores = [
                r['reasoning_score'] for r in self.results
                if 'reasoning_score' in r
            ]
            if reasoning_scores:
                metrics['avg_reasoning_score'] = sum(reasoning_scores) / len(
                    reasoning_scores)

        return metrics


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


def evaluate(
    model_name_or_path: str,
    eval_dataset_dir: str,
    output_dir: str = 'eval_results',
    eval_dataset_split: str = 'test',
    infer_backend: str = 'hf',
    generation_batch_size: int = 16,
    tasks: List[str] = ['aime', 'amc', 'math', 'minerva', 'olympiad_bench'],
    systerm_template: str = 'qwen_math_cot',
    input_template: str = '',
    input_key: str = 'question',
    label_key: str = 'solution',
    apply_chat_template: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 3000,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 4,
    n_samples: int = 1,
    max_test: int = 10000,
    save_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model using the specified datasets and generation settings.

    Args:
        model_name_or_path (str): Path to the model or model hub name.
        eval_dataset_dir (str): Path to the evaluation dataset directory.
        output_dir (str): Directory to save evaluation results.
        tasks (List[str]): Task names to evaluate on.
        systerm_template (str): Template name for system prompt.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling.
        max_tokens (int): Maximum generation length.
        max_model_len (int): Maximum model input length.
        tensor_parallel_size (int): Tensor parallelism for vLLM.
        n_samples (int): Number of samples per prompt.
        max_test (int): Maximum number of test examples per task.
        save_predictions (bool): Whether to save all predictions.

    Returns:
        Dict[str, Any]: Evaluation results for all tasks.
    """
    logger.info(f'Evaluating model: {model_name_or_path}')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and tokenizer
    try:
        if infer_backend == 'vllm':
            sampling_params = SamplingParams(
                n=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True)

            model = LLM(
                model=model_name_or_path,
                trust_remote_code=True,
                dtype='bfloat16',
                max_model_len=max_model_len,
                gpu_memory_utilization=0.96,
                enable_prefix_caching=True,
                tensor_parallel_size=tensor_parallel_size,
            )
        else:
            tokenizer, model = load_hf_lm_and_tokenizer(
                model_name_or_path=model_name_or_path,
                use_fast_tokenizer=True,
            )
            # Initialize HF-specific generation parameters
            generation_config = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': temperature > 0,
            }
    except Exception as e:
        logger.error(f'Failed to load model: {str(e)}')
        raise

    all_results = {}
    template = TEMPLATE_FACTORY.get(systerm_template, '')

    for task_name in tasks:
        logger.info(f'Evaluating task: {task_name}')
        metrics = EvaluationMetrics()

        try:
            # Load dataset
            raw_data = load_data(
                data_name=task_name,
                split=eval_dataset_split,
                data_dir=eval_dataset_dir,
            )[:max_test]

            # Create dataset and dataloader
            prompt_eval_dataset = PromptDataset(
                raw_data,
                tokenizer=tokenizer,
                input_key=input_key,
                label_key=label_key,
                systerm_template=template,
                input_template=input_template,
                apply_chat_template=apply_chat_template,
            )

            prompt_eval_dataloader = DataLoader(
                prompt_eval_dataset,
                batch_size=generation_batch_size,
                shuffle=False,
                collate_fn=lambda x: x,  # Custom collate handled in dataset
            )

            # Evaluation loop
            for batch_inputs in tqdm(prompt_eval_dataloader,
                                     desc=f'Evaluating {task_name}'):
                prompts = batch_inputs['prompt']
                labels = batch_inputs['label']

                try:
                    if infer_backend == 'vllm':
                        outputs = model.generate(prompts, sampling_params)
                        generated_texts = [
                            output.outputs[0].text for output in outputs
                        ]
                        for pred, label in zip(generated_texts, labels):
                            metrics.update(pred, label)
                    else:
                        inputs = tokenizer(
                            prompts,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=max_model_len,
                        )
                        inputs = {
                            k: v.to(model.device)
                            for k, v in inputs.items()
                        }

                        with torch.no_grad():
                            outputs = model.generate(**inputs,
                                                     **generation_config)
                        generated_texts = tokenizer.batch_decode(
                            outputs, skip_special_tokens=True)

                    # Update metrics
                    for pred, label in zip(generated_texts, labels):
                        metrics.update(pred, label)

                except Exception as e:
                    logger.error(f'Error during generation: {str(e)}')
                    metrics.total += len(
                        prompts)  # Mark failed batch as errors
                    continue

            # Save task results with enhanced analysis
            task_results = {
                'metrics': metrics.get_metrics(),
                'samples': metrics.results[:10],  # Save first 10 examples
                'config': {
                    'model': model_name_or_path,
                    'task': task_name,
                    'timestamp': datetime.now().isoformat(),
                    'generation_config': {
                        'temperature': temperature,
                        'top_p': top_p,
                        'max_tokens': max_tokens,
                        'n_samples': n_samples,
                    },
                },
            }

            # Save all predictions if requested
            if save_predictions:
                task_results['all_predictions'] = metrics.results

            all_results[task_name] = task_results
            analysis = save_results(task_results, output_dir, task_name)
            all_results[task_name]['analysis'] = analysis[task_name]

        except Exception as e:
            logger.error(f'Failed to evaluate task {task_name}: {str(e)}')
            continue

    return all_results


if __name__ == '__main__':
    # Example usage
    results = evaluate(
        model_name_or_path='Qwen/Qwen1.5-7B',
        eval_dataset_dir='./datasets',
        tasks=['math'],
        max_test=100,
    )
    print(json.dumps(results, indent=2))
