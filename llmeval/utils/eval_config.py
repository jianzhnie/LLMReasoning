from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from llmeval.utils.logger import init_logger

logger = init_logger(__name__)


@dataclass
class DataArguments:
    """Arguments for dataset configuration and loading.

    Attributes:
        data_dir: Directory containing the dataset files
        data_name: Identifier for the dataset (e.g., 'math', 'aime')
        split: Dataset split to use (e.g., 'train', 'test', 'validation')
        start_idx: Starting index for data loading (for partial evaluation)
        end_idx: Ending index for data loading (-1 for all data)
        batch_size: Number of samples to process in each batch
    """

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

    def __post_init__(self) -> None:
        """Validate data arguments after initialization."""
        if self.start_idx < 0:
            raise ValueError('start_idx must be non-negative')
        if self.end_idx != -1 and self.end_idx < self.start_idx:
            raise ValueError('end_idx must be -1 or greater than start_idx')
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')


@dataclass
class PromptArguments:
    """Arguments for prompt configuration and formatting.

    Attributes:
        prompt_type: Type of prompt template to use
        prompt_file_path: Directory containing prompt template files
        surround_with_messages: Whether to wrap prompts in chat format
        use_few_shot: Whether to use few-shot examples
        n_shot: Number of few-shot examples to include
        input_key: Key for input text in the dataset
        label_key: Key for label/target text in the dataset
        input_template: Optional template for formatting input text
    """

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
    input_key: str = field(default='question',
                           metadata={'help': 'Key for input text in dataset.'})
    label_key: str = field(
        default='solution',
        metadata={'help': 'Key for target/label text in dataset.'})
    input_template: str = field(
        default='',
        metadata={'help': 'Optional template for formatting input text.'})

    def __post_init__(self) -> None:
        """Validate prompt arguments after initialization."""
        if self.n_shot < 0:
            raise ValueError('n_shot must be non-negative')
        if not Path(self.prompt_file_path).exists():
            logger.warning(
                f'Prompt directory {self.prompt_file_path} does not exist')


@dataclass
class ModelArguments:
    """Arguments related to model configuration and loading.

    Attributes:
        model_name_or_path: Path to model or model identifier from huggingface.co
        dtype: Data type for model weights and computation
        infer_backend: Inference backend to use ('vllm' or 'hf')
        tensor_parallel_size: Number of GPUs for tensor parallelism
        trust_remote_code: Whether to trust remote code when loading models
    """

    model_name_or_path: str = field(
        default='./', metadata={'help': 'Path to the model directory.'})
    dtype: str = field(
        default='auto',
        metadata={
            'help': 'Data type for model execution (e.g., "fp16", "auto").'
        },
    )
    infer_backend: Literal['vllm', 'hf'] = field(
        default='hf', metadata={'help': 'Inference backend to use.'})
    trust_remote_code: bool = field(
        default=True,
        metadata={'help': 'Trust remote code when loading models.'})

    def __post_init__(self) -> None:
        """Validate model arguments after initialization."""
        if self.dtype not in ['auto', 'float16', 'bfloat16', 'float32']:
            raise ValueError(
                'dtype must be one of: auto, float16, bfloat16, float32')
        if self.infer_backend not in ['vllm', 'hf']:
            raise ValueError("infer_backend must be either 'vllm' or 'hf'")


@dataclass
class GenerationArguments:
    """Arguments for controlling text generation.

    Attributes:
        do_sample: Whether to use sampling instead of greedy decoding
        n_sampling: Number of sequences to generate for each prompt
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum number of tokens to generate
        skip_special_tokens: Whether to remove special tokens from output
    """

    do_sample: bool = field(
        default=True,
        metadata={'help': 'Whether to use sampling vs greedy decoding.'})
    n_sampling: int = field(
        default=1,
        metadata={'help': 'Number of sequences to generate per prompt.'})
    temperature: float = field(default=1.0,
                               metadata={'help': 'Sampling temperature.'})
    top_p: float = field(
        default=1.0,
        metadata={'help': 'Nucleus sampling probability threshold.'})
    top_k: int = field(default=50,
                       metadata={'help': 'Top-k sampling parameter.'})
    max_tokens: int = field(
        default=1024,
        metadata={'help': 'Maximum number of tokens to generate.'})
    skip_special_tokens: bool = field(
        default=True, metadata={'help': 'Remove special tokens from output.'})

    def __post_init__(self) -> None:
        """Validate generation arguments after initialization."""
        if self.temperature < 0:
            raise ValueError('temperature must be non-negative')
        if not 0 <= self.top_p <= 1:
            raise ValueError('top_p must be between 0 and 1')
        if self.top_k < 0:
            raise ValueError('top_k must be non-negative')
        if self.max_tokens < 1:
            raise ValueError('max_tokens must be positive')
        if self.n_sampling < 1:
            raise ValueError('n_sampling must be positive')


@dataclass
class VLLMArguments:
    """Arguments specific to vLLM inference backend.

    Attributes:
        gpu_memory_utilization: Target GPU memory usage (0-1)
        enable_prefix_caching: Whether to cache KV prefix for efficiency
        swap_space: Size of CPU swap space in GB
        block_size: Size of tensor parallel blocks
        quantization: Quantization method to use
        max_num_batched_tokens: Max tokens per batch
        max_num_seqs: Max sequences to process in parallel
        disable_custom_kernels: Whether to disable CUDA kernels
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
        metadata={'help': 'Target GPU memory utilization (0-1).'})
    enable_prefix_caching: bool = field(
        default=True,
        metadata={'help': 'Enable KV cache prefix optimization.'})
    swap_space: int = field(default=4,
                            metadata={'help': 'CPU swap space in GB.'})
    block_size: int = field(default=16,
                            metadata={'help': 'Tensor parallel block size.'})
    quantization: Optional[str] = field(
        default=None,
        metadata={'help': 'Quantization method (awq, squeezellm).'})
    max_num_batched_tokens: Optional[int] = field(
        default=None, metadata={'help': 'Max tokens per batch.'})
    max_num_seqs: Optional[int] = field(
        default=None, metadata={'help': 'Max parallel sequences.'})

    def __post_init__(self) -> None:
        """Validate vLLM arguments after initialization."""
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError('gpu_memory_utilization must be between 0 and 1')
        if self.swap_space < 0:
            raise ValueError('swap_space must be non-negative')
        if self.block_size < 1:
            raise ValueError('block_size must be positive')
        if self.quantization not in [None, 'awq', 'squeezellm']:
            raise ValueError(
                "quantization must be None, 'awq', or 'squeezellm'")


@dataclass
class EvaluationArguments(DataArguments, ModelArguments, GenerationArguments,
                          VLLMArguments):
    """Master configuration class to store all evaluation arguments."""

    # Core evaluation settings
    task: str = field(metadata={'help': 'Name of the evaluation task.'})
    task_dir: str = field(
        default='evaluation',
        metadata={'help': 'Directory containing the evaluation datasets.'},
    )
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
    save_predictions: bool = field(
        default=True, metadata={'help': 'Save model predictions.'})

    def __post_init__(self) -> None:
        """Validate and adjust evaluation arguments after initialization."""
        # Basic validation
        if self.k < 1:
            raise ValueError('k must be positive')
        if self.seed < 0:
            raise ValueError('seed must be non-negative')

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Adjust generation parameters based on temperature
        if self.temperature == 0:
            self.top_p = 1.0
            self.do_sample = False
