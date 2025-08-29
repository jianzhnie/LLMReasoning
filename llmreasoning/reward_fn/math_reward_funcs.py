import json
import logging
import math
import re
from typing import Any, List, Optional, Set, Tuple, Union

from latex2sympy2_extended import NormalizationConfig

try:
    from math_verify.errors import TimeoutException
    from math_verify.grader import verify
    from math_verify.metric import math_metric
    from math_verify.parser import (ExprExtractionConfig,
                                    LatexExtractionConfig, parse)
except ImportError:
    print(
        'To use Math-Verify, please install it first by running `pip install math-verify`.'
    )

from transformers import PreTrainedTokenizer

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------

# Configure logging for better visibility and control
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger('Relu Based Reward Function')


def extract_solution(solution_str: str,
                     method: str = 'strict') -> Optional[str]:
    """
    Extracts the final numerical solution from a given string.

    Args:
        solution_str (str): The string containing the solution.
        method (str): Extraction method ('strict' or 'flexible').
                      - 'strict': Looks for a specific pattern (#### <number>).
                      - 'flexible': Extracts the last valid number from the string.

    Returns:
        Optional[str]: Extracted solution as a string if found, otherwise None.
    """
    if method not in {'strict', 'flexible'}:
        raise ValueError("Method must be either 'strict' or 'flexible'.")

    if method == 'strict':
        # Look for a number prefixed by '#### '
        match = re.search(r'#### (-?[0-9.,]+)', solution_str)
        if match:
            # Extract the number and clean formatting (remove commas, dollar signs)
            return match.group(1).replace(',', '').replace('$', '')
        return None  # No valid match found

    elif method == 'flexible':
        # Find all numeric values in the string
        numbers = re.findall(r'-?[0-9.,]+', solution_str)

        # Filter out invalid numbers (e.g., '.' alone)
        numbers = [num for num in numbers if num not in {'', '.'}]

        # Return the last valid number found, or None if none exist
        return numbers[-1] if numbers else None

    return None  # Redundant but keeps function structure clear


class BaseRewardFunction:
    """
    Abstract base class for reward functions.

    Subclasses should implement the `__call__` method to return a list of reward scores
    for each completion.
    """

    def __init__(self, **kwargs):
        """
        Initialize the base reward function with optional configuration parameters.
        """
        self.config = kwargs

    def __call__(self,
                 completions: List[str],
                 solution: Optional[str] = None,
                 **kwargs: Any) -> List[float]:
        """
        Compute reward scores for the provided completions.

        Args:
            completions (List[str]): A list of generated completion strings.
            solution (Optional[str]): An optional ground truth or reference solution.
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: A list of reward scores corresponding to each completion.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    def validate_input(self,
                       completions: List[str],
                       solution: Optional[str] = None) -> None:
        """
        Perform input validation for completions and optional solution.

        Args:
            completions (List[str]): A list of completion strings.
            solution (Optional[str]): A reference solution, if applicable.
        """
        if not isinstance(completions, list) or not all(
                isinstance(c, str) for c in completions):
            raise ValueError('`completions` must be a list of strings.')
        if solution is not None and not isinstance(solution, str):
            raise ValueError('`solution` must be a string if provided.')


class GSM8KAccuracyReward:
    """
    A reward model for evaluating GSM8K solutions.

    Reference: Trung, Luong, et al.
    "Reft: Reasoning with reinforced fine-tuning."
    Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.
    """

    def __init__(self, method: str = 'strict', correct_reward: float = 1.0):
        """
        Initializes the reward model.

        Args:
            method (str): The extraction method ('strict' or 'flexible').
            correct_reward (float): Reward value for correct answers.
        """
        if method not in {'strict', 'flexible'}:
            raise ValueError("Method must be 'strict' or 'flexible'.")

        self.method = method
        self.correct_reward = correct_reward  # Store the reward value

    def __call__(self, completions: List[str],
                 solutions: List[str]) -> List[float]:
        """
        Computes rewards for a batch of completions.

        Args:
            completions (List[str]): List of model-generated answers.
            solutions (List[str]): List of ground-truth solutions.

        Returns:
            List[float]: A list of rewards (1.0 for correct answers, 0.0 otherwise).
        """
        if len(completions) != len(solutions):
            raise ValueError(
                f'Completions length ({len(completions)}) does not match solutions length ({len(solutions)})'
            )

        rewards: List[float] = []

        for completion, ground_truth in zip(completions, solutions):
            extracted_answer = extract_solution(solution_str=completion,
                                                method=self.method)

            if extracted_answer is None:
                reward = 0.0
            else:
                reward = self.correct_reward if extracted_answer == ground_truth else 0.0

            rewards.append(reward)

        return rewards


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
            # Wrap the ground truth in \boxed{} format for verification
            sol_boxed = '\\boxed{' + sol + '}'
            gold_parsed = self.parse_expression(
                sol_boxed, extraction_config=self.gold_extraction_config)

            if not gold_parsed:
                # Assign neutral reward if the ground truth cannot be parsed
                logger.info(
                    f'Warning: Failed to parse gold solution: {sol_boxed}')
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

            except ValueError as e:
                error_msg = str(e)
                # Handle specific ValueError for nan/zoo evaluation
                if "Can't evaluate nan or zoo" in error_msg:
                    logger.warning(
                        f'Mathematical expression contains nan/zoo : {error_msg}'
                    )
                    reward = self.incorrect_reward
                elif 'division by zero' in error_msg.lower():
                    logger.warning(f'Division by zero detected: {error_msg}')
                    reward = self.incorrect_reward
                elif 'undefined' in error_msg.lower():
                    logger.warning(
                        f'Undefined operation detected: {error_msg}')
                    reward = self.incorrect_reward
                else:
                    # Re-raise other ValueError exceptions
                    logger.error(f'Unexpected ValueError: {error_msg}')
                    reward = self.incorrect_reward
            except TimeoutException as e:
                logger.warning(f'Verification timeout: {e}')
                reward = self.incorrect_reward
            except Exception as e:
                logger.warning(
                    f'Verification failed: {e}, Answer: {answer_parsed}, Gold: {gold_parsed}'
                )
                reward = self.incorrect_reward

            rewards.append(reward)

        return rewards


class MathVerifyReward(BaseRewardFunction):
    """Computes accuracy-based rewards for mathematical expressions using
    latex2sympy2.

    **Key Enhancements:**
        - Supports both **LaTeX** and **symbolic expressions**.
        - Uses an **aggregation function** to handle multiple parsing strategies.
        - Adds **robust error handling** and logging.

    **Args:**
        completions (List[str]): Model-generated completions.
        solution (List[str]): Ground truth solutions.

    **Returns:**
        List[float]: Reward scores between 0.0 and 1.0.
    """

    def __init__(self, gold_is_latex: bool = True, **kwargs):
        super().__init__(**kwargs)

        # Ensure extraction config is a list (not a tuple)
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig() if gold_is_latex
                                    else ExprExtractionConfig(), ),
            pred_extraction_target=(ExprExtractionConfig(),
                                    LatexExtractionConfig()),
            aggregation_function=max,
            precision=6,
        )
        self.timeout_score = 0

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes accuracy-based rewards for mathematical expressions.

        Args:
            completions (List[str]): Model-generated responses.
            solution (List[str]): Ground truth solutions.

        Returns:
            List[float]: Rewards based on correctness.
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Wrap the ground truth in \boxed{} format for verification
            sol_boxed = '\\boxed{' + sol + '}'
            try:
                reward, _ = self.verify_func([sol_boxed], [content])
            except Exception:
                pass
            except TimeoutException:
                reward = self.timeout_score

            rewards.append(reward)

        return rewards


class FormatReward(BaseRewardFunction):
    """Reward function to check if the completion follows the correct format:

    Expected format:
    ```
    <think>...</think> <answer>...</answer>
    ```

    - Matches two possible formats:
        1. Multi-line format:
            ```
            <think>
            ...
            </think>
            <answer>
            ...
            </answer>
            ```
        2. Single-line format:
            ```
            <think>...</think> <answer>...</answer>
            ```

    A reward of `1.0` is given for correctly formatted responses, otherwise `0.0`.
    """

    def __init__(self):
        # Regular expression patterns for checking format validity
        self.patterns = [
            re.compile(r'^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$',
                       re.DOTALL | re.MULTILINE),
            re.compile(r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])',
                       re.DOTALL | re.MULTILINE),
        ]

    def is_valid_format(self, content: str) -> bool:
        """Checks if the given content matches at least one of the valid
        formats.

        Args:
            content (str): The completion text to validate.

        Returns:
            bool: True if the format is valid, False otherwise.
        """
        return any(pattern.match(content) for pattern in self.patterns)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Evaluates whether each completion follows the expected format.

        Args:
            completions (List[str]): List of generated completions.

        Returns:
            List[float]: List of rewards (1.0 if formatted correctly, else 0.0).
        """
        return [
            1.0 if self.is_valid_format(content) else 0.0
            for content in completions
        ]


class ReActFormat(BaseRewardFunction):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific
        format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completions
        ]
        return [1.0 if match else 0.0 for match in matches]


class TagCountReward(BaseRewardFunction):
    """Reward function that checks if the generated text includes the correct
    number of `<think>` and `<answer>` tags, ensuring proper structure.

    - Each correctly placed tag contributes `0.25` to the reward.
    - Maximum reward = `1.0` (when all tags are correctly placed).

    Reference:
    Adapted from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90

    Args:
        completions (List[str]): List of model-generated completions.

    Returns:
        List[float]: List of rewards, where `1.0` is the highest score.
    """

    @staticmethod
    def count_tags(text: str) -> float:
        """Counts the number of correctly placed `<think>` and `<answer>` tags.

        Args:
            text (str): The generated text to analyze.

        Returns:
            float: A reward score between `0.0` and `1.0`, based on tag correctness.
        """
        reward = 0.0
        if text.count('<think>\n') == 1:
            reward += 0.25
        if text.count('\n</think>\n') == 1:
            reward += 0.25
        if text.count('\n<answer>\n') == 1:
            reward += 0.25
        if text.count('\n</answer>') == 1:
            reward += 0.25
        return reward

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Evaluates the presence of properly structured `<think>` and
        `<answer>` tags.

        Args:
            completions (List[str]): List of model-generated completions.

        Returns:
            List[float]: List of rewards, each ranging from `0.0` to `1.0`.
        """
        return [self.count_tags(content) for content in completions]


class ReasoningStepReward(BaseRewardFunction):
    r"""
    Reward function that checks for clear step-by-step reasoning.

    This function scans for structural elements indicating stepwise logical reasoning,
    such as "Step 1:", numbered lists, bullet points, and transition words.

    **Regex Pattern Matches:**
    - `Step \d+:` → Matches "Step 1:", "Step 2:", etc.
    - `^\d+\.` → Matches numbered lists like "1.", "2.", at the start of a line.
    - `\n-` → Matches bullet points using hyphens.
    - `\n\*` → Matches bullet points using asterisks.
    - `First,|Second,|Next,|Finally,` → Matches common transition words.

    **Reward Calculation:**
    - The function counts the number of reasoning step indicators in each completion.
    - A target of **3 or more indicators** yields the maximum reward (`1.0`).
    - If fewer than 3 indicators are found, a **proportional reward** (`count / 3`) is assigned.

    **Example Rewards:**
    - 3 or more indicators → `1.0`
    - 2 indicators → `0.67`
    - 1 indicator → `0.33`
    - 0 indicators → `0.0`
    """

    # Regex pattern to detect reasoning step indicators
    REASONING_PATTERN = re.compile(
        r'(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)',
        re.MULTILINE)

    @staticmethod
    def count_reasoning_steps(text: str) -> int:
        """Counts occurrences of reasoning step indicators in the given text.

        Args:
            text (str): The generated completion text.

        Returns:
            int: Number of matched reasoning steps.
        """
        return len(ReasoningStepReward.REASONING_PATTERN.findall(text))

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Computes rewards based on the presence of reasoning step indicators.

        Args:
            completions (List[str]): List of generated text completions.

        Returns:
            List[float]: Reward scores between `0.0` and `1.0` for each completion.
        """
        return [
            min(1.0,
                self.count_reasoning_steps(content) / 3)
            for content in completions
        ]


class LengthReward(BaseRewardFunction):
    """Computes length-based rewards to discourage overthinking and encourage
    concise responses.

    **Reference:** Kimi 1.5 tech report (https://arxiv.org/abs/2501.12599)

    **Reward Calculation:**
        - Correct answers: `reward = 0.5 - (length - min_length) / (max_length - min_length)`
        - Incorrect answers: `reward = min(0, 0.5 - (length - min_length) / (max_length - min_length))`

    **Args:**
        tokenizer (PreTrainedTokenizer): Tokenizer for measuring response length.
        accuracy_orm (Union[BaseRewardFunction, None], optional): Function for computing accuracy.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        accuracy_orm: Union[BaseRewardFunction, None] = None,
    ) -> None:
        """Initializes the LengthReward function.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer for measuring response length.
            accuracy_orm (Union[BaseRewardFunction, None], optional): Accuracy computation module.
        """
        self.tokenizer = tokenizer
        self.accuracy_orm = accuracy_orm or MathAccuracyReward()

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes length-based rewards for each completion.

        Args:
            completions (List[str]): List of model-generated completions.
            solution (List[str]): List of ground truth solutions.
            **kwargs: Additional parameters for accuracy computation.

        Returns:
            List[float]: Computed rewards for each completion.
        """
        # Compute correctness scores using accuracy function
        correctness_scores = self.accuracy_orm(completions, solution, **kwargs)

        # Calculate response lengths using tokenizer
        lengths = [
            len(self.tokenizer.encode(content, add_special_tokens=False))
            for content in completions
        ]

        # Determine min and max lengths for scaling
        min_len, max_len = min(lengths), max(lengths)

        # If all responses have the same length, return zero rewards
        if min_len == max_len:
            return [0.0] * len(completions)

        # Compute rewards
        rewards = []
        for length, correctness in zip(lengths, correctness_scores):
            lambda_val = 0.5 - (length - min_len) / (max_len - min_len)
            reward = lambda_val if correctness >= 0.5 else min(0, lambda_val)
            rewards.append(float(reward))

        return rewards


class CosineScaledReward(BaseRewardFunction):
    """Reward function that scales rewards based on completion length using a
    cosine schedule.

    **Reference**: https://arxiv.org/abs/2502.03373

    **Key Behavior**:
        - ✅ Shorter **correct** completions receive **higher** rewards.
        - ❌ Longer **incorrect** completions receive **lower** penalties.

    **Args:**
        - `cosine_min_value_wrong` (float): Minimum reward for incorrect answers.
        - `cosine_max_value_wrong` (float): Maximum reward for incorrect answers.
        - `cosine_min_value_correct` (float): Minimum reward for correct answers.
        - `cosine_max_value_correct` (float): Maximum reward for correct answers.
        - `cosine_max_len` (int): Maximum length for scaling.
        - `cosine_accuracy_orm` (BaseRewardFunction, optional): Accuracy computation module.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        cosine_min_value_wrong: float = -0.5,
        cosine_max_value_wrong: float = 0.0,
        cosine_min_value_correct: float = 0.5,
        cosine_max_value_correct: float = 1.0,
        cosine_max_len: int = 1000,
        accuracy_orm: Union[BaseRewardFunction, None] = None,
    ) -> None:

        self.tokenizer = tokenizer
        self.min_value_wrong = cosine_min_value_wrong
        self.max_value_wrong = cosine_max_value_wrong
        self.min_value_correct = cosine_min_value_correct
        self.max_value_correct = cosine_max_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracyReward()

        if cosine_min_value_wrong > 0 or cosine_max_value_wrong > 0:
            raise ValueError('Wrong values should not be positive')

        logger.info(
            'Initialized math rule cosine scaled reward with'
            f' min_value_wrong: {cosine_min_value_wrong}, max_value_wrong: {cosine_max_value_wrong},'
            f' min_value_correct: {cosine_min_value_correct}, max_value_correct: {cosine_max_value_correct},'
            f' max_len: {cosine_max_len}')

    @staticmethod
    def cosine_scaled_reward(t: int, T: int, min_value: float,
                             max_value: float) -> float:
        """Computes a cosine-scaled reward value based on response length.

        Args:
            t (int): Current length of the completion.
            T (int): Maximum length for scaling.
            min_value (float): Minimum reward value.
            max_value (float): Maximum reward value.

        Returns:
            float: Scaled reward value.
        """
        cosine_value = math.cos(t * math.pi / T)
        return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs) -> List[float]:
        """Computes cosine-scaled rewards for a list of model completions.

        Args:
            completions (List[str]): List of generated completions.
            solution (List[str]): List of ground truth solutions.
            **kwargs: Additional arguments for the accuracy function.

        Returns:
            List[float]: List of computed rewards.
        """
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []

        for content, acc_reward in zip(completions, acc_rewards):
            gen_len = len(
                self.tokenizer.encode(content, add_special_tokens=False))

            if gen_len == 0:
                logger.warning(f'Skipping empty completion: {content}')
                rewards.append(self.min_value_wrong)
                # Assign minimum penalty for empty responses
                continue

            is_correct = acc_reward >= 1.0

            # Correct answers get higher rewards for being concise
            if is_correct:
                min_value, max_value = self.min_value_correct, self.max_value_correct
            else:
                min_value, max_value = self.max_value_wrong, self.min_value_wrong  # Fixed logic

            # Compute scaled reward
            reward = self.cosine_scaled_reward(gen_len, self.max_len,
                                               min_value, max_value)
            rewards.append(reward)

        return rewards


class RepetitionPenalty(BaseRewardFunction):
    """Computes an N-gram repetition penalty as described in Appendix C.2 of:
    https://arxiv.org/abs/2502.03373.

    This function penalizes responses that contain repeated sequences of `n` words.

    **Reference Implementation:**
    - Adapted from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    **Reward Calculation:**
    - If an `n`-gram appears more than once, it contributes to a repetition penalty.
    - The penalty is scaled based on the proportion of repeated n-grams.
    - The maximum penalty (`max_penalty`) is applied when all n-grams are repeated.

    **Args:**
        repetition_n_grams (int): The size of the n-grams to check for repetition.
        repetition_max_penalty (float): The maximum negative penalty for highly repetitive text.

    **Raises:**
        ValueError: If `repetition_max_penalty` is positive.

    **Example Usage:**
        ```python
        penalty_fn = RepetitionPenalty(repetition_n_grams=3, repetition_max_penalty=-1.0)
        rewards = penalty_fn(["The cat sat on the mat. The cat sat on the mat."])
        ```
    """

    def __init__(self,
                 repetition_n_grams: int = 3,
                 repetition_max_penalty: float = -1.0):
        if repetition_max_penalty > 0:
            raise ValueError(
                f'`repetition_max_penalty` should not be positive: {repetition_max_penalty}'
            )

        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def extract_ngrams(text: str, ngram_size: int) -> Set[Tuple[str, ...]]:
        """Extracts all n-grams from the given text.

        Args:
            text (str): The input text.
            ngram_size (int): The size of n-grams to extract.

        Returns:
            Set[Tuple[str, ...]]: A set of unique n-grams found in the text.
        """
        words = text.lower().split()
        if len(words) < ngram_size:
            return set()
        return set(
            tuple(words[i:i + ngram_size])
            for i in range(len(words) - ngram_size + 1))

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Computes repetition penalties for a list of completions.

        Args:
            completions (List[str]): List of generated text completions.

        Returns:
            List[float]: Penalty scores (negative values), with `0.0` meaning no penalty.
        """
        rewards = []
        for completion in completions:
            words = completion.split()

            # Handle empty or short completions (no penalty)
            if not completion or len(words) < self.ngram_size:
                rewards.append(0.0)
                continue

            # Compute total n-grams and unique n-grams
            total_ngrams = len(words) - self.ngram_size + 1
            unique_ngrams = len(
                self.extract_ngrams(completion, self.ngram_size))

            # Compute repetition scaling factor
            repetition_ratio = 1 - (unique_ngrams / total_ngrams)
            reward = repetition_ratio * self.max_penalty

            rewards.append(reward)

        return rewards


class SoftOverlongPenalty(BaseRewardFunction):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)
    The penalty is proportional to how much the completion length exceeds the expected maximum length (max_completion_length - soft_cache_length).

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for measuring response length.
        max_completion_length (int): The soft maximum allowed length.
        soft_cache_length (int): The length of the cached prompt.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer = None,
                 max_completion_length: int = 8000,
                 soft_cache_length: int = 2000) -> None:
        """
        Initialize the SoftOverlong reward function.

        Args:
            tokenizer (Any): A tokenizer with an `encode` method.
            max_completion_length (int): The soft maximum allowed length.
            soft_cache_length (int): The length of the cached prompt.
        """
        super().__init__()
        assert soft_cache_length < max_completion_length, 'soft_cache_length must be less than max_completion_length'
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        self.soft_cache_length = soft_cache_length
        self.expected_len = self.max_completion_length - self.soft_cache_length

    def __call__(self, completions: List[str], solution: List[str],
                 **kwargs: Any) -> List[float]:
        """
        Compute the reward for each completion based on its length.

        Args:
            completions (List[str]): A list of generated completion strings.
            solution (List[str]): List of ground truth solutions.
            **kwargs: Additional keyword arguments (unused here).

        Returns:
            List[float]: A list of reward values where more negative scores indicate
                         a greater length violation.
        """

        rewards: List[float] = []
        # 批量计算所有completions的长度
        completion_lengths = [
            len(ids) for ids in self.tokenizer(
                completions, add_special_tokens=False)['input_ids']
        ]

        # 计算惩罚值
        for completion_length in completion_lengths:
            exceed_len = completion_length - self.expected_len
            penalty = min(-exceed_len / self.soft_cache_length, 0.0)
            rewards.append(penalty)

        return rewards


relu_based_reward_func_mapping = {
    'gsm8k': GSM8KAccuracyReward,
    'accuracy': MathAccuracyReward,
    'mathverify_accuracy': MathVerifyReward,
    'format': FormatReward,
    'react_format': ReActFormat,
    'tag_reward': TagCountReward,
    'reasoning_steps': ReasoningStepReward,
    'length': LengthReward,
    'cosine': CosineScaledReward,
    'repetition': RepetitionPenalty,
    'softoverlong': SoftOverlongPenalty,
}
