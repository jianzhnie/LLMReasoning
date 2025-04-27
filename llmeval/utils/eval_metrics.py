from typing import Dict, List

from llmeval.utils.math_grader import MathAccuracyReward


class EvaluationMetrics:
    """Enhanced evaluation metrics with answer normalization and custom comparison"""

    def __init__(self):
        self.results = []
        self.math_grader = MathAccuracyReward()
        self.correct_count = 0  # 添加正确答案计数
        self.total_count = 0  # 添加总样本计数

    def update(
            self,
            predictions: List[str],
            solutions: List[str],
            prompts: List[str] = None,  # Add prompts parameter
    ):
        # Use math grader for detailed evaluation
        # scores is list such as [0,1,1,0, ...]
        scores = self.math_grader(predictions, solutions)

        # Update accuracy statistics
        self.correct_count += sum(scores)  # scores中1表示正确，0表示错误
        self.total_count += len(scores)

        # Create results in the desired format
        for idx, (prompt, prediction, solution, score) in enumerate(
                zip(prompts, predictions, solutions, scores)):
            result = {
                'id': idx,
                'prompt': prompt,
                'answer': solution,
                'model_gen': prediction,
                'score': score,
            }
            self.results.append(result)

    def get_metrics(self) -> Dict[str, float]:
        metrics = {
            'total_samples':
            self.total_count,
            'correct_samples':
            self.correct_count,
            'accuracy':
            self.correct_count /
            self.total_count if self.total_count > 0 else 0.0,
        }
        return metrics

    def get_formatted_metrics(self) -> str:
        """返回格式化的指标字符串，方便打印"""
        metrics = self.get_metrics()
        return (f"Total Samples: {metrics['total_samples']}\n"
                f"Correct Samples: {metrics['correct_samples']}\n"
                f"Accuracy: {metrics['accuracy']:.2%}")
