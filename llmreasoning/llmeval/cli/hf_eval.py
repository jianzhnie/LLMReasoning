from typing import List

import torch
from llmeval.utils.eval_config import EvaluationArguments
from transformers import HfArgumentParser

from ..utils.model_utils import load_hf_lm_and_tokenizer
from .base_eval import BaseEvaluator


class HFEvaluator(BaseEvaluator):

    def load_model_and_tokenizer(self):
        self.tokenizer, self.model = load_hf_lm_and_tokenizer(
            model_name_or_path=self.args.model_name_or_path,
            use_fast_tokenizer=True,
        )

        self.generation_config = {
            'max_new_tokens': self.args.max_tokens,
            'temperature': self.args.temperature,
            'top_p': self.args.top_p,
            'do_sample': self.args.temperature > 0,
        }

    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.args.max_model_len,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_config)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    evaluator = HFEvaluator(args)
    results = evaluator.evaluate()
    print(results)


if __name__ == '__main__':
    main()
