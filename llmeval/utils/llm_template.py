qwen_math_cot = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

deepseek_r1 = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

systerm_factory = {"qwen_math_cot": qwen_math_cot, "deepseek_r1": deepseek_r1}
