import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from transformers import GenerationConfig

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationArguments:
    """
    Data class to encapsulate all configuration parameters required for evaluation tasks.
    """

    # Core evaluation parameters
    task: str = field(metadata={'help': 'Name of the evaluation task.'})
    task_dir: str = field(
        default='evaluation',
        metadata={'help': 'Directory containing the evaluation datasets.'},
    )
    batch_size: int = field(
        default=4, metadata={'help': 'Batch size per GPU for evaluation.'})
    seed: int = field(default=42,
                      metadata={'help': 'Random seed for data loaders.'})
    lang: Literal['en', 'zh'] = field(
        default='en', metadata={'help': 'Language used in evaluation.'})
    n_shot: int = field(
        default=5,
        metadata={'help': 'Number of exemplars for few-shot learning.'})
    save_dir: Optional[str] = field(
        default=None, metadata={'help': 'Path to save evaluation results.'})

    # Model and data configuration
    model_name_or_path: str = field(
        default='./', metadata={'help': 'Path to the model directory.'})
    n_sampling: int = field(
        default=1, metadata={'help': 'Number of output samples to generate.'})
    k: int = field(default=1,
                   metadata={'help': 'Top-k value for pass@k calculation.'})
    data_dir: str = field(
        default='./data',
        metadata={'help': 'Directory containing the dataset.'})
    data_name: str = field(default='math',
                           metadata={'help': 'Dataset identifier.'})
    split: str = field(
        default='test',
        metadata={'help': 'Dataset split to use (e.g., train, test).'})
    start_idx: int = field(default=0,
                           metadata={'help': 'Start index for evaluation.'})
    end_idx: int = field(default=-1,
                         metadata={'help': 'End index for evaluation.'})

    # Generation parameters
    temperature: float = field(default=0.0,
                               metadata={'help': 'Sampling temperature.'})
    top_p: float = field(
        default=1.0, metadata={'help': 'Top-p (nucleus) sampling parameter.'})
    max_tokens: int = field(
        default=2048,
        metadata={'help': 'Maximum number of tokens to generate.'})
    stop: Optional[Union[List[str], str]] = field(
        default=None, metadata={'help': 'List of stop tokens.'})

    # Prompt configuration
    prompt_type: str = field(default='qwen-base',
                             metadata={'help': 'Type of prompt format used.'})
    prompt_file_path: str = field(
        default='./prompts',
        metadata={'help': 'Directory with prompt templates.'})
    surround_with_messages: bool = field(
        default=False, metadata={'help': 'Wrap prompts using message format.'})
    use_few_shot: bool = field(default=False,
                               metadata={'help': 'Use few-shot prompting.'})

    # Output directories
    output_dir: str = field(
        default='./outputs',
        metadata={'help': 'Directory to save output results.'})
    completions_save_dir: str = field(
        default='./completions',
        metadata={'help': 'Directory to save completions.'})
    dtype: str = field(
        default='auto',
        metadata={
            'help': 'Data type for model execution (e.g., "fp16", "auto").'
        },
    )

    def __post_init__(self) -> None:
        """
        Post-initialization to handle default behavior and validation.
        """
        # Ensure top_p is 1.0 when temperature is 0
        if self.temperature == 0:
            self.top_p = 1.0

        # Check if save directory exists to avoid overwriting
        if self.save_dir is not None and Path(self.save_dir).exists():
            raise ValueError(
                f'`save_dir` "{self.save_dir}" already exists. Please use a different directory.'
            )

        # Log the current stop list
        logger.info(f'Current stop list: {self.stop}')


@dataclass
class GeneratingArguments:
    """Arguments pertaining to specify the model generation parameters."""

    # Generation strategy
    # 是否采样
    do_sample: Optional[bool] = field(
        default=True,
        metadata={
            'help':
            'Whether or not to use sampling, use greedy decoding otherwise.'
        },
    )
    # Hyperparameters for logit manipulation
    # softmax 函数的温度因子，来调节输出token的分布
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            'help': 'The value used to modulate the next token probabilities.'
        },
    )
    # 核采样参数，top_p最高的前n个（n是变化）概率和为p，从这些n个候选token中随机采样
    top_p: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept.'
        },
    )
    # top_k随机搜索中的k个最高概率选择
    top_k: Optional[int] = field(
        default=50,
        metadata={
            'help':
            'The number of highest probability vocabulary tokens to keep for top-k filtering.'
        },
    )
    # 集束搜索的数量
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            'help': 'Number of beams for beam search. 1 means no beam search.'
        },
    )
    # 最大的token数量，会被 max_new_tokens 覆盖
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'The maximum length the generated tokens can have. It can be overridden by max_new_tokens.'
        },
    )
    # 最大的新生成的token数量
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'Maximum number of new tokens to be generated in evaluation or prediction loops'
            'if predict_with_generate is set.'
        },
    )
    # 重复性惩罚因子
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'The parameter for repetition penalty. 1.0 means no penalty.'
        },
    )
    # 长度惩罚因子
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            'help':
            'Exponential penalty to the length that is used with beam-based generation.'
        },
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={'help': 'Default system message to use in chat completion.'},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={
            'help': 'Whether or not to remove special tokens in the decoding.'
        },
    )

    def to_dict(self, obey_generation_config: bool = False) -> Dict[str, Any]:
        args = asdict(self)
        if args.get('max_new_tokens', -1) > 0:
            args.pop('max_length', None)
        else:
            args.pop('max_new_tokens', None)

        if obey_generation_config:
            generation_config = GenerationConfig()
            for key in list(args.keys()):
                if not hasattr(generation_config, key):
                    args.pop(key)

        return args
