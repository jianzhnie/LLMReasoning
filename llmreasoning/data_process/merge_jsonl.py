# merge_jsonl.py
import argparse
import glob
import json
import os
from typing import Any, List, Tuple

keys = ['question', 'answer', 'gen', 'accuracy']


def is_valid_field_content(field_name: str, content: Any) -> Tuple[bool, str]:
    """
    Validate if field content is valid

    :param field_name: Field name
    :param content: Field content
    :return: (is_valid, error_message)
    """
    # Check if content is empty
    if content is None or (isinstance(content, str) and content.strip() == ''):
        return False, 'Content is empty'

    # Validation for specific fields
    if field_name == 'accuracy':
        # accuracy should be a boolean or a value that can be converted to boolean
        if not isinstance(content, bool):
            if isinstance(content, (int, float)):
                if content not in [0, 1]:
                    return False, 'Numeric accuracy must be 0 or 1'
            elif isinstance(content, str):
                if content.lower() not in ['true', 'false', '0', '1']:
                    return False, "String accuracy must be one of 'true', 'false', '0', '1'"
            else:
                return False, "accuracy must be boolean, numeric (0/1) or string ('true'/'false'/'0'/'1')"

    # question and answer fields should be non-empty strings
    if field_name in ['question', 'answer']:
        if not isinstance(content, str):
            return False, f'{field_name} must be a string'
        if content.strip() == '':
            return False, f'{field_name} content cannot be empty'

    # gen field should be a list
    if field_name == 'gen':
        if not isinstance(content, list):
            return False, 'gen must be a list'
        if len(content) == 0:
            return False, 'gen list cannot be empty'

    return True, ''


def merge_jsonl_files(patterns: List[str],
                      output_file: str,
                      recursive: bool = True) -> None:
    """
    Match and merge multiple JSONL files based on wildcard patterns.

    :param patterns: List of file path patterns, e.g. ['data/*.jsonl', 'logs/**/*.jsonl']
    :param output_file: Output file path
    :param recursive: Whether to support recursive matching (i.e. **)
    """
    matched_files = []
    seen_files = set()

    print('🔍 Searching for matching files...')
    for pattern in patterns:
        # Use glob to find matching files
        files = glob.glob(pattern, recursive=recursive)
        for file_path in files:
            if os.path.isfile(file_path) and file_path not in seen_files:
                matched_files.append(file_path)
                seen_files.add(file_path)

    if not matched_files:
        print('❌ No matching files found.')
        return

    # Sort by filename to ensure consistent merge order
    matched_files.sort()

    print(f'✅ Found {len(matched_files)} matching files:')
    for f in matched_files:
        print(f'   - {f}')

    # Start merging
    merged_count = 0
    skipped_count = 0

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in matched_files:
                print(f'📌 Processing: {file_path}')
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line_num, line in enumerate(infile, 1):
                            line = line.strip()
                            if not line:
                                continue  # Skip empty lines
                            try:
                                data = json.loads(line)  # Parse JSON format

                                # Verify required fields exist
                                missing_keys = [
                                    key for key in keys if key not in data
                                ]
                                if missing_keys:
                                    print(
                                        f'❌ Line {line_num} in file {file_path} missing fields: {missing_keys}'
                                    )
                                    skipped_count += 1
                                    continue

                                # Validate field content
                                invalid_fields = []
                                for key in keys:
                                    is_valid, error_msg = is_valid_field_content(
                                        key, data[key])
                                    if not is_valid:
                                        invalid_fields.append(
                                            f'{key}({error_msg})')

                                if invalid_fields:
                                    print(
                                        f'❌ Invalid field content in line {line_num} of file {file_path}: {invalid_fields}'
                                    )
                                    skipped_count += 1
                                    continue

                                outfile.write(line + '\n')
                                merged_count += 1
                            except json.JSONDecodeError as e:
                                print(
                                    f'❌ JSON parsing error on line {line_num} of file {file_path}: {e}'
                                )
                                skipped_count += 1
                except IOError as e:
                    print(f'❌ Error reading file {file_path}: {e}')
                    skipped_count += 1

        print(
            f'\n✅ Merge completed! Written {merged_count} records, skipped {skipped_count} records.'
        )
        print(f'📁 Output file: {output_file}')

    except IOError as e:
        print(f'❌ Error writing to output file {output_file}: {e}')


def main():
    parser = argparse.ArgumentParser(
        description=
        'Merge all JSONL files under fuzzy matched paths (supports wildcards * and **)'
    )
    parser.add_argument(
        '--patterns',
        nargs='+',
        required=True,
        help=
        'File path matching patterns, e.g.: data/*.jsonl or logs/**/*.jsonl')
    parser.add_argument('--output',
                        default='merged_output.jsonl',
                        help='Output file path (default: merged_output.jsonl)')
    parser.add_argument('--recursive',
                        action='store_true',
                        help='Enable recursive matching with ** patterns')

    args = parser.parse_args()
    merge_jsonl_files(args.patterns, args.output, args.recursive)


if __name__ == '__main__':
    main()
