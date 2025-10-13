#!/usr/bin/env python3
"""
多轮对话数据转换为Alpaca格式的脚本
使用Hugging Face datasets库处理数据转换
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from datasets import Dataset, load_dataset
import pandas as pd


def convert_multi_turn_to_alpaca(
    conversations: List[Dict[str, Any]], 
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    将多轮对话数据转换为Alpaca格式
    
    Args:
        conversations: 多轮对话数据列表，每个对话包含messages字段
        system_prompt: 可选的系统提示词
    
    Returns:
        转换后的Alpaca格式数据列表
    """
    alpaca_data = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        if not messages:
            continue
            
        # 构建完整的对话文本
        full_conversation = ""
        if system_prompt:
            full_conversation += f"System: {system_prompt}\n"
        
        for i, message in enumerate(messages):
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                full_conversation += f"Human: {content}\n"
            elif role == 'assistant':
                full_conversation += f"Assistant: {content}\n"
            elif role == 'system':
                full_conversation += f"System: {content}\n"
        
        # 将对话转换为Alpaca格式
        # 取第一个用户消息作为instruction
        first_user_msg = None
        for msg in messages:
            if msg.get('role') == 'user':
                first_user_msg = msg.get('content', '')
                break
        
        if first_user_msg:
            # 构建完整的对话上下文作为input
            context = full_conversation.replace(f"Human: {first_user_msg}\n", "", 1)
            context = context.strip()
            
            # 构建assistant的完整回复作为output
            assistant_responses = []
            for msg in messages:
                if msg.get('role') == 'assistant':
                    assistant_responses.append(msg.get('content', ''))
            
            output = "\n".join(assistant_responses) if assistant_responses else ""
            
            alpaca_data.append({
                "instruction": first_user_msg,
                "input": context if context else "",
                "output": output
            })
    
    return alpaca_data


def convert_single_turn_to_alpaca(
    conversations: List[Dict[str, Any]], 
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    将多轮对话数据转换为单轮Alpaca格式（每个用户-助手对作为一条记录）
    
    Args:
        conversations: 多轮对话数据列表
        system_prompt: 可选的系统提示词
    
    Returns:
        转换后的Alpaca格式数据列表
    """
    alpaca_data = []
    
    for conv in conversations:
        messages = conv.get('messages', [])
        if not messages:
            continue
        
        # 构建对话历史
        conversation_history = []
        if system_prompt:
            conversation_history.append(f"System: {system_prompt}")
        
        for i, message in enumerate(messages):
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                # 当遇到用户消息时，将之前的对话作为context
                context = "\n".join(conversation_history) if conversation_history else ""
                
                # 寻找对应的助手回复
                assistant_response = ""
                for j in range(i + 1, len(messages)):
                    if messages[j].get('role') == 'assistant':
                        assistant_response = messages[j].get('content', '')
                        break
                
                if assistant_response:  # 只有当有助手回复时才添加记录
                    alpaca_data.append({
                        "instruction": content,
                        "input": context,
                        "output": assistant_response
                    })
                
                # 将用户消息添加到历史中
                conversation_history.append(f"Human: {content}")
            elif role == 'assistant':
                conversation_history.append(f"Assistant: {content}")
            elif role == 'system':
                conversation_history.append(f"System: {content}")
    
    return alpaca_data


def load_conversation_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载对话数据文件
    
    Args:
        file_path: 数据文件路径
    
    Returns:
        对话数据列表
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError("只支持.json和.jsonl格式的文件")
    
    return data


def save_alpaca_data(data: List[Dict[str, str]], output_path: str):
    """
    保存Alpaca格式数据
    
    Args:
        data: Alpaca格式数据
        output_path: 输出文件路径
    """
    if output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif output_path.endswith('.jsonl'):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    elif output_path.endswith('.csv'):
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    else:
        raise ValueError("输出格式只支持.json、.jsonl和.csv")


def main():
    parser = argparse.ArgumentParser(description="将多轮对话数据转换为Alpaca格式")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出文件路径")
    parser.add_argument("--system-prompt", "-s", help="系统提示词")
    parser.add_argument("--mode", "-m", choices=["full", "single"], default="single",
                       help="转换模式: full(完整对话) 或 single(单轮对话)")
    parser.add_argument("--from-hf", help="从Hugging Face Hub加载数据集")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.from_hf:
        print(f"从Hugging Face Hub加载数据集: {args.from_hf}")
        dataset = load_dataset(args.from_hf)
        # 假设使用train split，可以根据需要调整
        conversations = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
        conversations = conversations.to_list()
    else:
        print(f"从本地文件加载数据: {args.input}")
        conversations = load_conversation_data(args.input)
    
    print(f"加载了 {len(conversations)} 条对话数据")
    
    # 转换数据
    if args.mode == "full":
        alpaca_data = convert_multi_turn_to_alpaca(conversations, args.system_prompt)
    else:
        alpaca_data = convert_single_turn_to_alpaca(conversations, args.system_prompt)
    
    print(f"转换完成，生成了 {len(alpaca_data)} 条Alpaca格式数据")
    
    # 保存数据
    save_alpaca_data(alpaca_data, args.output)
    print(f"数据已保存到: {args.output}")
    
    # 显示示例
    if alpaca_data:
        print("\n转换示例:")
        print("=" * 50)
        example = alpaca_data[0]
        print(f"Instruction: {example['instruction']}")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")

    {
    "uid": "1b6d912-8135-4f23-b704-2ceea8675617",
    "license": "CC BY 4.0",
    "generator": "Owen3-235B-A22B",
    "version": "v1",
    "category": "chat",
    "reasoning": "off",
    "messages": [
        {
        "role": "user",
        "content": "**.",
        "tool_calls": []
        },
        {
        "role": "assistant",
        "content": "Understood. I'm ready to proceed with the activity. Please ask the first question.",
        "tool_calls": []
        }
    ],
    "metadata": {
        "conversation_id": "8e31a02d8f1d49f48f6563a86d5fbd2f",
        "source": "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
    }
    }


if __name__ == "__main__":
    main()