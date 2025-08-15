import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from torch.utils.data import DataLoader
from tqdm import tqdm

from llmeval.utils.dataset_utils import PromptDataset, load_data
from llmeval.utils.eval_config import EvaluationArguments
from llmeval.utils.eval_metrics import EvaluationMetrics
from llmeval.utils.llm_template import TEMPLATE_FACTORY
from llmeval.utils.logger import init_logger

logger = init_logger(__name__)


class BaseEvaluator(ABC):

    def __init__(self, config: EvaluationArguments) -> None:
        self.args = config

    @abstractmethod
    def load_model_and_tokenizer(self):
        """Load model and tokenizer based on specific backend"""
        pass

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for given prompts"""
        pass

    def evaluate(self) -> Dict[str, Any]:
        logger.info(f'Evaluating model: {self.args.model_name_or_path}')
        os.makedirs(self.args.output_dir, exist_ok=True)

        try:
            self.load_model_and_tokenizer()
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
                    tokenizer=self.tokenizer,
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
                        generated_texts = self.generate(prompts)
                        metrics.update(generated_texts, labels, prompts)
                    except Exception as e:
                        logger.error(f'Error during generation: {str(e)}')
                        continue

                task_results = self._process_results(metrics, task_name)
                all_results[task_name] = task_results

            except Exception as e:
                logger.error(f'Failed to evaluate task {task_name}: {str(e)}')
                continue

        return all_results

    def _process_results(self, metrics: EvaluationMetrics,
                         task_name: str) -> Dict[str, Any]:
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

        analysis = self._save_results(task_results, task_name)
        task_results['analysis'] = analysis[task_name]
        return task_results

    def _save_results(self, results: Dict[str, Any], task_name: str):
        os.makedirs(self.args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_file = os.path.join(self.args.output_dir,
                                    f'{task_name}_{timestamp}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        analysis = self._analyze_results({task_name: results})
        analysis_file = os.path.join(self.args.output_dir,
                                     f'{task_name}_{timestamp}_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f'Results saved to {results_file}')
        logger.info(f'Analysis saved to {analysis_file}')

        return analysis

    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze evaluation results and generate insights."""
        analysis = {}
        for task_name, task_results in results.items():
            task_analysis = {
                'performance_summary': task_results['metrics'],
                'error_patterns': {},
                'common_mistakes': [],
                'reasoning_quality': {},
            }
            analysis[task_name] = task_analysis
        return analysis
