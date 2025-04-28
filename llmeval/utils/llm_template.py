from datasets import load_dataset
from transformers import AutoTokenizer
from trl.data_utils import maybe_apply_chat_template

qwen_math_cot = (
    'Please reason step by step, and put your final answer within \\boxed{}.')

deepseek_r1 = (
    'A conversation between User and Assistant. The User asks a question, and the Assistant solves it.'
    'The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. '
    'The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, '
    'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.'
)

TEMPLATE_FACTORY = {'qwen_math_cot': qwen_math_cot, 'deepseek_r1': deepseek_r1}

# DeepSeek system prompt for GRPO based training
SYSTEM_TEMPLATE = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
    'first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning '
    'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think><answer> answer here </answer>')

INPUT_TEMPLATE = {
    'prompt':
    ('{instruction}. \n\n Please reason step by step, and put your final answer within \\boxed{{}}.'
     )
}


def preprocess_data(
    data,
    input_key: str = 'input',
    label_key: str = None,
    system_prompt: str = None,
    input_template: str = None,
) -> str:
    input_prompt = data[input_key]
    if system_prompt:
        prompt = []
        if system_prompt is not None:
            prompt.append({'role': 'system', 'content': system_prompt})

        prompt.append({'role': 'user', 'content': input_prompt})

        example = {'prompt': prompt}
        prompt_text = maybe_apply_chat_template(example, tokenizer)['prompt']
    elif input_template:
        prompt_text = INPUT_TEMPLATE.format(input_prompt)

    # for Reinforced Fine-tuning
    label_text = '' if label_key is None else data[label_key]
    processed_input = {'prompt': prompt_text, 'answer': label_text}
    return processed_input


if __name__ == '__main__':
    data_path = '/root/llmtuner/hfhub/datasets/Maxwell-Jia/AIME_2024'
    model_name_or_path = 'Qwen/Qwen2.5-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load the dataset from the Hugging Face Hub
    dataset = load_dataset(data_path, split='train')
    formated_data = []
    for example in dataset:
        formated_example = preprocess_data(
            example,
            input_key='Problem',
            label_key='Answer',
            system_prompt=SYSTEM_TEMPLATE,
            input_template=INPUT_TEMPLATE,
        )
        formated_data.append(formated_example)
        print(formated_example)
    print(dataset)
    # 查看第一条数据
    print(dataset[0])
