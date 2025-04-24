from typing import List

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from llmeval.utils.dataset_utils import PromptDataset, load_data
from llmeval.utils.llm_template import TEMPLATE_FACTORY
from llmeval.utils.model_utils import load_hf_lm_and_tokenizer


def evaluate(
    model_name_or_path: str,
    eval_dataset_dir: str,
    eval_dataset_split: str = 'test',
    infer_backend: str = 'hf',
    generation_batch_size: int = 16,
    tasks: List[str] = ['aime', 'amc', 'math', 'minerva', 'olympiad_bench'],
    systerm_template: str = 'qwen_math',  # Unused parameter
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
) -> None:
    """
    Evaluate a model using the specified datasets and generation settings.

    Args:
        model_name_or_path (str): Path to the model or model hub name.
        eval_dataset_dir (str): Path to the evaluation dataset directory.
        tasks (List[str]): Task names to evaluate on.
        template (str): Template name (currently unused).
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling.
        max_tokens (int): Maximum generation length.
        max_model_len (int): Maximum model input length.
        tensor_parallel_size (int): Tensor parallelism for vLLM.
        n_samples (int): Number of samples per prompt.
        max_test (int): Maximum number of test examples per task.
    """
    print(f'Evaluating model: {model_name_or_path}')
    model_name_or_path
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
    systerm_template = TEMPLATE_FACTORY[systerm_template]
    for task_name in tasks:
        prompt_eval_dataset = load_data(
            datat_name=task_name,
            split='test',
            data_dir=eval_dataset_dir,
        )
        prompt_eval_dataset = PromptDataset(
            prompt_eval_dataset,
            tokenizer=tokenizer,
            input_key=input_key,
            label_key=label_key,
            systerm_template=systerm_template,
            input_template=input_template,
            apply_chat_template=apply_chat_template,
        )
        prompt_eval_dataloader = DataLoader(
            prompt_eval_dataset,
            batch_size=generation_batch_size,
        )

        print(f'Generating outputs for task: {task_name}')
        for batch_inputs in prompt_eval_dataloader:
            prompts = batch_inputs['prompt']
            labels = batch_inputs['label']
            inputs = tokenizer(prompts, device='cuda')
            outputs = model.generate(prompts, sampling_params)
            print(inputs, outputs, labels)
