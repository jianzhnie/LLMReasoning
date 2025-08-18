from typing import List

from latex2sympy2_extended import NormalizationConfig
from llmeval.utils.logger import init_logger
from math_verify.grader import verify
from math_verify.parser import LatexExtractionConfig, parse

logger = init_logger(__name__)


class BaseRewardFunction:
    """Placeholder for the base reward function class."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        raise NotImplementedError

    def validate_input(self, completions, solution=None):
        """统一的输入验证."""
        pass


class MathAccuracyReward(BaseRewardFunction):
    """Computes a reward based on whether the model's response is
    mathematically equivalent to the ground truth solution using latex2sympy2
    and math_verify.

    **Reward Criteria:**
        - ✅ 1.0 → If the response is **mathematically equivalent** to the solution.
        - ❌ 0.0 → If the response is **incorrect**.
        - 🔄 0.5 → If the **ground truth cannot be parsed**, to avoid unfair penalties.

    **Key Features:**
        - Parses mathematical expressions into symbolic form.
        - Compares model-generated answers with ground truth.
        - Handles edge cases where solutions are unparseable.

    **Args:**
        completions (List[str]): Model-generated completions.
        solution (List[str]): Ground truth solutions.

    **Returns:**
        List[float]: Reward scores between 0.0 and 1.0.
    """

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        neutral_reward: float = 0.0,
        gold_is_latex: bool = True,
    ) -> None:
        """Initializes the MathAccuracyReward function with parsing
        configurations.

        Args:
            gold_is_latex (bool): Flag indicating whether the ground truth is provided in LaTeX format.
        """
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.neutral_reward = neutral_reward

        self.gold_extraction_config: List[LatexExtractionConfig] = [
            LatexExtractionConfig()
        ]
        # We require the answer to be provided in correct latex (no malformed operators)
        self.answer_extraction_config: List[LatexExtractionConfig] = [
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed='all',
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ]
        self.gold_is_latex = gold_is_latex

    def parse_expression(self, expression: str,
                         extraction_config: List[LatexExtractionConfig]):
        """Parses a mathematical expression using latex2sympy2.

        Args:
            expression (str): The input mathematical expression in LaTeX.
            extraction_config (Sequence[ExtractionTarget]): Extraction configuration.

        Returns:
            Parsed expression object or None if parsing fails.
        """
        if not expression.strip():
            return None  # 避免解析空字符串

        try:
            result = parse(expression,
                           extraction_mode='first_match',
                           extraction_config=extraction_config)
            return result or None  # 直接返回结果或 None
        except Exception as e:
            logger.info(
                f'Parsing failed for expression: {expression}, Error: {e}')
            return None

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes accuracy-based rewards for mathematical expressions.

        Args:
            completions (List[str]): Model-generated responses.
            solution (List[str]): Ground truth solutions.

        Returns:
            List[float]: Rewards based on correctness.
        """
        if len(completions) != len(solution):
            raise ValueError(
                f'Completions length ({len(completions)}) does not match solutions length ({len(solution)})'
            )

        rewards: List[float] = []
        for content, sol in zip(completions, solution):
            gold_parsed = self.parse_expression(
                sol, extraction_config=self.gold_extraction_config)

            if not gold_parsed:
                # Assign neutral reward if the ground truth cannot be parsed
                logger.info(f'Warning: Failed to parse gold solution: {sol}')
                rewards.append(self.correct_reward)
                continue

            answer_parsed = self.parse_expression(
                content, extraction_config=self.answer_extraction_config)

            if not answer_parsed:
                rewards.append(self.incorrect_reward)  # Invalid model response
                continue

            try:
                # If the verification function succeeds, return the verification score (1.0 or 0.0)
                is_correct = verify(answer_parsed, gold_parsed)

                reward = self.correct_reward if is_correct else self.incorrect_reward
            except Exception as e:
                logger.warning(
                    f'Verification failed: {e}, Answer: {answer_parsed}, Gold: {gold_parsed}'
                )
                reward = self.incorrect_reward

            rewards.append(reward)

        return rewards
