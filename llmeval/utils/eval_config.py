from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from transformers import GenerationConfig

from llmeval.utils.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DataArguments:
    """Arguments for dataset configuration and loading."""
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
    batch_size: int = field(
        default=4, metadata={'help': 'Batch size per GPU for evaluation.'})


@dataclass
class PromptArguments:
    """Arguments for prompt configuration and formatting."""
    prompt_type: str = field(default='qwen-base',
                             metadata={'help': 'Type of prompt format used.'})
    prompt_file_path: str = field(
        default='./prompts',
        metadata={'help': 'Directory with prompt templates.'})
    surround_with_messages: bool = field(
        default=False, metadata={'help': 'Wrap prompts using message format.'})
    use_few_shot: bool = field(default=False,
                               metadata={'help': 'Use few-shot prompting.'})
    n_shot: int = field(
        default=5,
        metadata={'help': 'Number of exemplars for few-shot learning.'})


@dataclass
class ModelArguments:
    """Arguments related to model configuration and loading."""
    model_name_or_path: str = field(
        default='./', metadata={'help': 'Path to the model directory.'})
    dtype: str = field(
        default='auto',
        metadata={
            'help': 'Data type for model execution (e.g., "fp16", "auto").'
        })


@dataclass
class GenerationArguments:
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
    n_sampling: int = field(
        default=1, metadata={'help': 'Number of output samples to generate.'})
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
    # 最大的新生成的token数量
    max_tokens: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'Maximum number of new tokens to be generated in evaluation or prediction loops'
            'if predict_with_generate is set.'
        },
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


@dataclass
class VLLMArguments:
    """Arguments for vLLM-specific configuration and optimization.

    This class contains parameters specific to vLLM's distributed inference
    and optimization features.
    """
    # Engine Configuration
    tensor_parallel_size: int = field(
        default=1,
        metadata={'help': 'Number of GPUs to use for tensor parallelism.'})
    max_model_len: int = field(
        default=4096,
        metadata={'help': 'Maximum sequence length for the model.'})
    gpu_memory_utilization: float = field(
        default=0.96,
        metadata={
            'help': 'Target GPU memory utilization for vLLM (0.0 to 1.0).'
        })

    # Performance Optimization
    enable_prefix_caching: bool = field(
        default=True,
        metadata={
            'help':
            'Enable KV cache prefix optimization for better performance.'
        })
    swap_space: int = field(
        default=4,
        metadata={'help': 'Size of CPU swap space in GiB (0 to disable).'})
    block_size: int = field(
        default=16,
        metadata={'help': 'Size of blocks to use for tensor parallelism.'})

    # Quantization and Precision
    dtype: str = field(
        default='auto',
        metadata={
            'help':
            'Data type for model weights. Options: auto, float16, bfloat16, float32'
        })
    quantization: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Quantization method. Options: awq, squeezellm, None'
        })

    # Distributed Settings
    worker_use_ray: bool = field(
        default=False, metadata={'help': 'Use Ray for distributed inference.'})
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'URL for distributed initialization (tcp://MASTER_IP:PORT).'
        })

    # Model Loading
    trust_remote_code: bool = field(
        default=True,
        metadata={
            'help': 'Trust remote code when loading models from Hugging Face.'
        })
    # Request Processing
    max_num_batched_tokens: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Maximum number of tokens to process in a single batch.'
        })
    max_num_seqs: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Maximum number of sequences to process in parallel.'
        })
    disable_custom_kernels: bool = field(
        default=False,
        metadata={
            'help':
            'Disable custom CUDA kernels and use PyTorch implementations.'
        })

    def __post_init__(self):
        """Validate and adjust parameters after initialization."""
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError('gpu_memory_utilization must be between 0 and 1')

        if self.tensor_parallel_size < 1:
            raise ValueError('tensor_parallel_size must be at least 1')

        if self.dtype not in ['auto', 'float16', 'bfloat16', 'float32']:
            raise ValueError(
                'dtype must be one of: auto, float16, bfloat16, float32')

        if self.quantization not in [None, 'awq', 'squeezellm']:
            raise ValueError(
                'quantization must be one of: None, awq, squeezellm')

    def to_dict(self) -> Dict[str, Any]:
        """Convert the arguments to a dictionary for vLLM initialization."""
        config_dict = {
            'tensor_parallel_size': self.tensor_parallel_size,
            'max_model_len': self.max_model_len,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'enable_prefix_caching': self.enable_prefix_caching,
            'trust_remote_code': self.trust_remote_code,
            'dtype': self.dtype,
        }

        # Only add optional parameters if they are set
        optional_params = [
            'swap_space', 'block_size', 'quantization',
            'distributed_init_method', 'download_dir',
            'max_num_batched_tokens', 'max_num_seqs'
        ]

        for param in optional_params:
            value = getattr(self, param)
            if value is not None:
                config_dict[param] = value

        return config_dict


@dataclass
class EvaluationArguments:
    """Master configuration class to store all evaluation arguments."""
    # Core evaluation settings
    task: str = field(metadata={'help': 'Name of the evaluation task.'})
    task_dir: str = field(
        default='evaluation',
        metadata={'help': 'Directory containing the evaluation datasets.'})
    seed: int = field(default=42,
                      metadata={'help': 'Random seed for data loaders.'})
    lang: Literal['en', 'zh'] = field(
        default='en', metadata={'help': 'Language used in evaluation.'})
    k: int = field(default=1,
                   metadata={'help': 'Top-k value for pass@k calculation.'})

    # Output settings
    output_dir: str = field(
        default='./outputs',
        metadata={'help': 'Directory to save output results.'})
    completions_save_dir: str = field(
        default='./completions',
        metadata={'help': 'Directory to save completions.'})
    save_dir: Optional[str] = field(
        default=None, metadata={'help': 'Path to save evaluation results.'})

    # Nested configurations
    model_args: ModelArguments = field(default_factory=ModelArguments)
    generation_args: GenerationArguments = field(
        default_factory=GenerationArguments)
    data_args: DataArguments = field(default_factory=DataArguments)
    prompt_args: PromptArguments = field(default_factory=PromptArguments)

    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        # Ensure top_p is 1.0 when temperature is 0
        if self.generation_args.temperature == 0:
            self.generation_args.top_p = 1.0

        # Check if save directory exists to avoid overwriting
        if self.save_dir is not None and Path(self.save_dir).exists():
            raise ValueError(
                f'`save_dir` "{self.save_dir}" already exists. Please use a different directory.'
            )
