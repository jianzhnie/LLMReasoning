#!/usr/bin/env python3
"""
使用示例：演示如何使用convert_to_alpaca.py脚本
"""

import json

from convert_to_alpaca import (convert_multi_turn_to_alpaca,
                               convert_single_turn_to_alpaca)

# 示例多轮对话数据
sample_conversations = [{
    'messages': [{
        'role': 'system',
        'content': '你是一个有用的AI助手。'
    }, {
        'role': 'user',
        'content': '你好，请介绍一下Python编程语言。'
    }, {
        'role': 'assistant',
        'content': 'Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。'
    }, {
        'role': 'user',
        'content': 'Python有哪些主要特点？'
    }, {
        'role':
        'assistant',
        'content':
        'Python的主要特点包括：1. 语法简洁易读 2. 跨平台兼容 3. 丰富的库生态系统 4. 支持多种编程范式'
    }]
}, {
    'messages': [{
        'role': 'user',
        'content': '如何学习机器学习？'
    }, {
        'role': 'assistant',
        'content': '学习机器学习的建议：1. 掌握数学基础 2. 学习编程 3. 实践项目 4. 阅读论文'
    }, {
        'role': 'user',
        'content': '推荐一些学习资源'
    }, {
        'role': 'assistant',
        'content': '推荐资源：Coursera的机器学习课程、Kaggle竞赛、scikit-learn文档等'
    }]
}]


def demo_conversion():
    print('=== 单轮对话转换示例 ===')
    single_turn_data = convert_single_turn_to_alpaca(sample_conversations)

    for i, item in enumerate(single_turn_data):
        print(f'\n--- 示例 {i+1} ---')
        print(f"Instruction: {item['instruction']}")
        print(f"Input: {item['input']}")
        print(f"Output: {item['output']}")

    print('\n=== 完整对话转换示例 ===')
    full_conversation_data = convert_multi_turn_to_alpaca(sample_conversations)

    for i, item in enumerate(full_conversation_data):
        print(f'\n--- 示例 {i+1} ---')
        print(f"Instruction: {item['instruction']}")
        print(f"Input: {item['input']}")
        print(f"Output: {item['output']}")


if __name__ == '__main__':
    demo_conversion()
