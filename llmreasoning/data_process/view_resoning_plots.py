import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt


def read_json_file(file_path):
    """读取普通 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl_file(file_path):
    """逐行读取 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data.append(json.loads(line.strip()))
    return data


def auto_read_json(file_path):
    """
    根据文件扩展名自动选择读取方式
    支持 .json 和 .jsonl
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.json':
        return read_json_file(file_path)
    elif ext == '.jsonl':
        return read_jsonl_file(file_path)
    else:
        raise ValueError(
            f'Unsupported file extension: {ext}. Only .json and .jsonl are supported.'
        )


def plot_bar_chart(data, labels, title, xlabel, ylabel, output_path):
    """
    Plot a bar chart and save it to a file.

    Args:
        data (list): Values for the bars.
        labels (list): Labels for the bars.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, data, color='skyblue')

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 yval + 0.3,
                 int(yval),
                 ha='center',
                 va='bottom',
                 fontsize=8)

    # Set titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_histogram(data, bins, title, xlabel, ylabel, output_path):
    """
    Plot a histogram and save it to a file.

    Args:
        data (list): Data points for the histogram.
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')

    # Set titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate plots from reasoning statistics')
    parser.add_argument('--stats-file',
                        type=str,
                        required=True,
                        help='Path to the statistics JSON/JSONL file')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./plots',
        help='Directory to save the generated plots (default: ./plots)')
    parser.add_argument('--bins',
                        type=int,
                        default=30,
                        help='Number of bins for histograms (default: 30)')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # === 配置路径 ===
    stats_file = args.stats_file
    output_dir = args.output_dir
    bins = args.bins

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # === 读取数据 ===
    try:
        stats_data = auto_read_json(stats_file)
    except Exception as e:
        print(f'❌ Error reading stats file: {e}')
        return

    # 提取字段
    correct_counts = [item['correct_count'] for item in stats_data]
    avg_lengths = [item['avg_cot_length'] for item in stats_data]
    max_lengths = [item['max_cot_length'] for item in stats_data]

    # 统计频次
    count_freq = Counter(correct_counts)

    print(count_freq)
    # 准备绘图数据
    labels, values = zip(*sorted(count_freq.items()))

    # === 2. Barplot - 正确数 ===
    plot_bar_chart(data=values,
                   labels=labels,
                   title='Correct CoT Count per Question',
                   xlabel='Question',
                   ylabel='Correct Count',
                   output_path=os.path.join(output_dir,
                                            'cot_correct_count_barplot.png'))

    # === 3. Histogram - 平均 CoT 长度分布 ===
    plot_histogram(data=avg_lengths,
                   bins=bins,
                   title='Distribution of Average CoT Length',
                   xlabel='Average CoT Token Length',
                   ylabel='Frequency',
                   output_path=os.path.join(output_dir,
                                            'avg_cot_length_histogram.png'))

    # === 4. Histogram - 最大 CoT 长度分布 ===
    plot_histogram(data=max_lengths,
                   bins=bins,
                   title='Distribution of Max CoT Length',
                   xlabel='Max CoT Token Length',
                   ylabel='Frequency',
                   output_path=os.path.join(output_dir,
                                            'max_cot_length_histogram.png'))

    print(f'📊 Plots saved to: {output_dir}')


if __name__ == '__main__':
    main()
