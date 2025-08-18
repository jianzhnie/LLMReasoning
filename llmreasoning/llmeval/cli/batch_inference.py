import json
from typing import Dict, List, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from llmeval.utils.eval_config import EvaluationArguments
from llmeval.utils.logger import init_logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer)

logger = init_logger(__name__)


def main(config: EvaluationArguments) -> None:
    """
    Main function to run inference using a pretrained causal language model.
    """
    # Initialize Accelerator for distributed training/inference
    accelerator: Accelerator = Accelerator()
    device: torch.device = accelerator.device

    # Load model and tokenizer
    model_name: str = 'meta-llama/Llama-2-7b-chat-hf'
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True)

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16,  # Use 16-bit precision if supported
    )

    # Prepare model with Accelerator (handles multi-device setup)
    model = accelerator.prepare(model)

    # Load dataset (adjust split and dataset name as needed)
    dataset: Dataset = load_dataset('your_dataset_name', split='validation')

    def collate_fn(
            batch: List[Dict[str,
                             str]]) -> Tuple[Dict[str, Tensor], List[str]]:
        """
        Collate function to tokenize a batch of examples.

        Args:
            batch (List[Dict[str, str]]): A batch of samples from the dataset.

        Returns:
            Tuple containing tokenized inputs (Dict[str, Tensor]) and original texts (List[str]).
        """
        texts: List[str] = [example['text'] for example in batch]
        tokenized = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        return tokenized, texts

    # Define DataLoader with custom collate function
    batch_size: int = 32
    dataloader: DataLoader = DataLoader(dataset,
                                        batch_size=batch_size,
                                        collate_fn=collate_fn)

    # Prepare DataLoader with Accelerator
    dataloader = accelerator.prepare(dataloader)

    # Run inference
    all_results: List[Dict[str, str]] = infer(model, tokenizer, dataloader,
                                              device)

    # Save results (only on the main process)
    if accelerator.is_main_process:
        save_results(all_results, output_file='inference_outputs.json')

        print(f'Inference complete. Saved {len(all_results)} results.')


def infer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> List[Dict[str, str]]:
    """
    Perform inference using the given model and dataloader.

    Args:
        model (PreTrainedModel): The loaded language model.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        dataloader (DataLoader): The prepared dataloader.
        device (torch.device): Device to run inference on.

    Returns:
        List of dictionaries containing input text and generated output text.
    """
    model.eval()
    all_results: List[Dict[str, str]] = []

    with torch.no_grad():
        for batch_inputs, raw_texts in dataloader:
            # Move inputs to the correct device
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

            # Generate outputs
            generated_tokens = model.generate(
                **batch_inputs,
                max_new_tokens=100,
                do_sample=False,  # Deterministic decoding
            )

            # Decode generated tokens
            decoded_outputs: List[str] = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True)

            # Collect input-output pairs
            for input_text, output_text in zip(raw_texts, decoded_outputs):
                all_results.append({
                    'input': input_text,
                    'output': output_text
                })

    return all_results


def save_results(results: List[Dict[str, str]], output_file: str) -> None:
    """
    Save inference results to a JSON file.

    Args:
        results (List[Dict[str, str]]): List of input-output dictionaries.
        output_file (str): Path to save the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
