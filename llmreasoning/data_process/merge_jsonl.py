# merge_jsonl_glob.py
import argparse
import glob
import json
import os

keys = ['question', 'answer', 'gen', 'accuracy']


def is_valid_field_content(field_name, content):
    """
    验证字段内容是否有效

    :param field_name: 字段名
    :param content: 字段内容
    :return: (is_valid, error_message)
    """
    # 检查内容是否为空
    if content is None or (isinstance(content, str) and content.strip() == ''):
        return False, '内容为空'

    # 对于特定字段的验证
    if field_name == 'accuracy':
        # accuracy 应该是布尔值或可以转换为布尔值的值
        if not isinstance(content, bool):
            if isinstance(content, (int, float)):
                if content not in [0, 1]:
                    return False, '数值型accuracy必须是0或1'
            elif isinstance(content, str):
                if content.lower() not in ['true', 'false', '0', '1']:
                    return False, "字符串型accuracy必须是'true', 'false', '0', '1'之一"
            else:
                return False, "accuracy必须是布尔值、数值(0/1)或字符串('true'/'false'/'0'/'1')"

    # question 和 answer 字段应该是非空字符串
    if field_name in ['question', 'answer']:
        if not isinstance(content, str):
            return False, f'{field_name}必须是字符串类型'
        if content.strip() == '':
            return False, f'{field_name}内容不能为空'

    # gen 字段应该是一个列表
    if field_name == 'gen':
        if not isinstance(content, list):
            return False, 'gen必须是列表类型'
        if len(content) == 0:
            return False, 'gen列表不能为空'

    return True, ''


def merge_jsonl_files(patterns, output_file, recursive=False):
    """
    根据通配符模式匹配并合并多个 JSONL 文件。

    :param patterns: 文件路径模式列表，如 ['data/*.jsonl', 'logs/**/*.jsonl']
    :param output_file: 输出文件路径
    :param recursive: 是否支持递归匹配（即 **）
    """
    matched_files = []
    seen_files = set()

    print('🔍 正在搜索匹配的文件...')
    for pattern in patterns:
        # 使用 glob 查找匹配的文件
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if os.path.isfile(file_path) and file_path not in seen_files:
                matched_files.append(file_path)
                seen_files.add(file_path)

    if not matched_files:
        print('❌ 没有找到匹配的文件。')
        return

    # 按文件名排序，确保合并顺序一致
    matched_files.sort()

    print(f'✅ 找到 {len(matched_files)} 个匹配的文件：')
    for f in matched_files:
        print(f'   - {f}')

    # 开始合并
    merged_count = 0
    skipped_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in matched_files:
            print(f'📌 正在处理: {file_path}')
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue  # 跳过空行
                    try:
                        data = json.loads(line)  # 解析 JSON 格式

                        # 验证必需字段是否存在
                        missing_keys = [key for key in keys if key not in data]
                        if missing_keys:
                            print(
                                f'❌ 文件 {file_path} 第 {line_num} 行缺少字段: {missing_keys}'
                            )
                            skipped_count += 1
                            continue

                        # 验证字段内容是否正确
                        invalid_fields = []
                        for key in keys:
                            is_valid, error_msg = is_valid_field_content(
                                key, data[key])
                            if not is_valid:
                                invalid_fields.append(f'{key}({error_msg})')

                        if invalid_fields:
                            print(
                                f'❌ 文件 {file_path} 第 {line_num} 行字段内容无效: {invalid_fields}'
                            )
                            skipped_count += 1
                            continue

                        outfile.write(line + '\n')
                        merged_count += 1
                    except json.JSONDecodeError as e:
                        print(
                            f'❌ 文件 {file_path} 第 {line_num} 行 JSON 解析错误: {e}')
                        skipped_count += 1

    print(f'\n✅ 合并完成！共写入 {merged_count} 条记录，跳过 {skipped_count} 条记录。')
    print(f'📁 输出文件: {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='合并模糊匹配路径下的所有 JSONL 文件（支持通配符 * 和 **）')
    parser.add_argument('--patterns',
                        nargs='+',
                        help='文件路径匹配模式，如: data/*.jsonl 或 logs/**/*.jsonl')
    parser.add_argument('--output',
                        default='merged_output.jsonl',
                        help='输出文件路径（默认: merged_output.jsonl）')

    args = parser.parse_args()
    merge_jsonl_files(args.patterns, args.output)


if __name__ == '__main__':
    main()
