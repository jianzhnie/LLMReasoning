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
from llmeval.utils.model_utils import load_hf_lm_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Enhanced evaluation metrics with answer normalization and custom comparison"""

    def __init__(self):
        self.total = 0
        self.correct = 0
        self.results = []

    def update(self,
               prediction: str,
               label: str,
               metadata: Dict[str, Any] = None):
        self.total += 1
        # TODO: Implement task-specific accuracy metrics
        result = {
            'prediction': prediction,
            'label': label,
            'metadata': metadata
        }
        self.results.append(result)

    def get_metrics(self) -> Dict[str, float]:
        return {
            'total_samples': self.total,
            'accuracy': self.correct / self.total if self.total > 0 else 0,
        }


def save_results(results: Dict[str, Any], output_dir: str, task_name: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{task_name}_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {output_file}')


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
                datat_name=task_name,
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

            # Save task results
            task_results = {
                'metrics': metrics.get_metrics(),
                'samples': metrics.results[:10],  # Save first 10 examples
                'config': {
                    'model': model_name_or_path,
                    'task': task_name,
                    'timestamp': datetime.now().isoformat(),
                },
            }
            all_results[task_name] = task_results
            save_results(task_results, output_dir, task_name)

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
