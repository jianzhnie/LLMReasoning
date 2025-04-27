from typing import List

from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

from llmeval.utils.eval_config import EvaluationArguments

from .base_eval import BaseEvaluator


class VLLMEvaluator(BaseEvaluator):

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, trust_remote_code=True)

        self.model = LLM(
            model=self.args.model_name_or_path,
            trust_remote_code=True,
            dtype='bfloat16',
            max_model_len=self.args.max_model_len,
            gpu_memory_utilization=0.96,
            enable_prefix_caching=True,
            tensor_parallel_size=self.args.tensor_parallel_size,
        )

        self.sampling_params = SamplingParams(
            n=self.args.n_samples,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_tokens,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


def main():
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    evaluator = VLLMEvaluator(args)
    results = evaluator.evaluate()
    print(results)


if __name__ == '__main__':
    main()
