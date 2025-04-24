from typing import List, Optional, Tuple

import torch
import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, StoppingCriteria,
                          StoppingCriteriaList)
"""
https://github.com/allenai/open-instruct
"""


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    A stopping criterion that halts generation when any of the provided keywords appear in the decoded output.
    """

    def __init__(self, keywords: List[str], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.current_context: List[List[int]] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        if not self.current_context:
            self.current_context = [[] for _ in range(input_ids.shape[0])]

        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            token_id = input_ids[i, -1].item()
            self.current_context[i].append(token_id)
            current_text = self.tokenizer.decode(self.current_context[i])
            should_stop = any(keyword in current_text
                              for keyword in self.keywords)
            sequences_should_be_stopped.append(should_stop)

        return all(sequences_should_be_stopped)


class KeyWordsCriteriaTrunc(StoppingCriteria):
    """
    A stopping criterion that checks for stop sequences within the portion of the output excluding the prompt.
    """

    def __init__(self, stop_id_sequences: List[List[int]], prompt_length: int):
        assert isinstance(
            stop_id_sequences[0],
            list), ('stop_id_sequences should be a list of lists')
        self.stop_sequences = stop_id_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][self.prompt_length:].tolist()
            should_stop = any(ids[-len(seq):] == seq
                              for seq in self.stop_sequences)
            sequences_should_be_stopped.append(should_stop)

        return all(sequences_should_be_stopped)


class KeyWordsCriteria(StoppingCriteria):
    """
    A stopping criterion that checks whether the most recent tokens match any of the stop sequences.
    """

    def __init__(self, stop_id_sequences: List[List[int]]):
        assert isinstance(
            stop_id_sequences[0],
            list), ('stop_id_sequences should be a list of lists')
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            should_stop = any(input_ids[i][-len(seq):].tolist() == seq
                              for seq in self.stop_sequences)
            sequences_should_be_stopped.append(should_stop)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    batch_size: int = 1,
    stop_id_sequences: Optional[List[List[int]]] = None,
    add_special_tokens: bool = True,
    disable_tqdm: bool = False,
    **generation_kwargs,
) -> List[str]:
    """
    Generate text completions for a list of prompts using a Hugging Face model.

    Args:
        model (PreTrainedModel): The causal language model.
        tokenizer (PreTrainedTokenizer): The corresponding tokenizer.
        prompts (List[str]): A list of input prompts.
        batch_size (int): Number of prompts per batch.
        stop_id_sequences (Optional[List[List[int]]]): Token ID sequences to use as stopping criteria.
        add_special_tokens (bool): Whether to add special tokens during tokenization.
        disable_tqdm (bool): Whether to disable the progress bar.
        **generation_kwargs: Additional keyword arguments for `model.generate`.

    Returns:
        List[str]: A list of generated completions.
    """
    generations: List[str] = []

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc='Generating Completions')

    num_return_sequences = generation_kwargs.get('num_return_sequences', 1)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized = tokenizer(
            batch_prompts,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=add_special_tokens,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        stopping_criteria = None
        if stop_id_sequences:
            stopping_criteria = StoppingCriteriaList(
                [KeywordsStoppingCriteria(stop_id_sequences, tokenizer)])

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            **generation_kwargs,
        )

        decoded_outputs = tokenizer.batch_decode(outputs,
                                                 skip_special_tokens=True)
        decoded_prompts = tokenizer.batch_decode(input_ids,
                                                 skip_special_tokens=True)
        decoded_prompts = [
            prompt for prompt in decoded_prompts
            for _ in range(num_return_sequences)
        ]

        completions = [
            output[len(prompt):]
            for prompt, output in zip(decoded_prompts, decoded_outputs)
        ]

        if stop_id_sequences:
            for idx, prediction in enumerate(completions):
                for stop_sequence in stop_id_sequences:
                    stop_text = tokenizer.decode(stop_sequence,
                                                 skip_special_tokens=True)
                    completions[idx] = prediction.split(stop_text)[0]

        generations.extend(completions)

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, (
        'Mismatch in number of generated outputs.')

    return generations


def load_hf_lm_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    use_fast_tokenizer: bool = False,
    padding_side: str = 'left',
    use_safetensors: bool = False,
    device_map: str = 'auto',
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a Hugging Face causal language model and tokenizer.

    Args:
        model_name_or_path (str): Path or name of the pre-trained model to load.
        tokenizer_name_or_path (Optional[str]): Path or name of the tokenizer. If None, uses model_name_or_path.
        device_map (str): Device mapping strategy for model loading (e.g., "auto", "cpu", "cuda").
        use_fast_tokenizer (bool): Whether to use a fast tokenizer if available.
        padding_side (str): Padding side for tokenizer ("left" or "right").
        use_safetensors (bool): Whether to use safetensors for model loading.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer objects.
    """
    # Use model path as tokenizer path if not explicitly provided
    tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path

    # Load tokenizer with specified options
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=use_fast_tokenizer,
        padding_side=padding_side,
        trust_remote_code=True,
    )

    # Ensure tokenizer has a pad token; fallback to unk or eos tokens if necessary
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError(
                'Tokenizer is missing a pad token, and neither unk nor eos tokens are available as a fallback.'
            )

    # Load the causal language model with specified device and dtype settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        use_safetensors=use_safetensors,
    )

    # Move model to CUDA if available and not handled by device_map
    if torch.cuda.is_available() and device_map == 'auto':
        model = model.cuda()

    # Set model to evaluation mode
    model.eval()

    return model, tokenizer
