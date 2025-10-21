# merge_jsonl.py
"""
A utility script to merge multiple JSONL (JSON Lines) files based on wildcard
patterns, performing validation on each record before merging.
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, Final, List, Tuple

# Define the mandatory fields expected in each JSONL record.
# The `is_valid_field_content` function relies on this list for field-specific validation.
EXPECTED_KEYS: Final[List[str]] = ['prompt', 'answer', 'gen', 'accuracy']


def is_valid_field_content(field_name: str, content: Any) -> Tuple[bool, str]:
    """
    Validates the content of a specific field within a JSONL record against
    predefined rules based on the field name.

    :param field_name: The name of the field being validated (e.g., 'prompt', 'accuracy').
    :param content: The content/value of the field.
    :return: A tuple where the first element is True if valid, False otherwise,
             and the second element is an empty string for valid content or an
             error message for invalid content.
    """
    # Check if content is empty (None, or an empty/whitespace-only string)
    if content is None or (isinstance(content, str) and content.strip() == ''):
        return False, 'Content is empty'

    # --- Validation for specific fields ---

    # 'prompt' and 'answer' fields must be non-empty strings.
    if field_name in ['prompt', 'answer']:
        if not isinstance(content, str):
            return False, f"'{field_name}' must be a string"
        # The empty check is already covered above, but kept this block for type safety.

    # 'accuracy' field validation
    elif field_name == 'accuracy':
        # Accuracy should be convertible to a boolean (True/False or 1/0)
        if isinstance(content, bool):
            return True, ''  # Boolean is always valid
        elif isinstance(content, (int, float)):
            # Numeric accuracy must be exactly 0 or 1
            if content not in [0, 1]:
                return False, 'Numeric accuracy must be 0 or 1'
        elif isinstance(content, str):
            # String accuracy must be one of the specified case-insensitive values
            if content.lower() not in ['true', 'false', '0', '1']:
                return False, "String accuracy must be one of 'true', 'false', '0', '1'"
        else:
            return False, "Accuracy must be boolean, numeric (0/1) or string ('true'/'false'/'0'/'1')"

    # 'gen' (generation) field validation
    elif field_name == 'gen':
        if not isinstance(content, list):
            return False, "'gen' must be a list"
        if not content:  # Checks if the list is empty
            return False, "'gen' list cannot be empty"
        if not all(isinstance(item, str) for item in content):
            return False, "'gen' list items must be strings"
        # Custom rule: checks for a specific placeholder value in the first item
        if '999, 999' in content[0]:
            return False, "'gen' list cannot contain the placeholder \"999, 999\" in the first element"

    return True, ''


# ---
def merge_jsonl_files(patterns: List[str],
                      output_file: str,
                      recursive: bool = True) -> None:
    """
    Matches JSONL files based on wildcard patterns, validates their contents,
    and merges valid records into a single output JSONL file.

    :param patterns: List of file path patterns (e.g., ['data/*.jsonl', 'logs/**/*.jsonl']).
    :param output_file: The path to the output JSONL file.
    :param recursive: Whether to enable recursive directory matching (using **).
    """
    matched_files: List[str] = []
    seen_files: set = set()

    print('🔍 Searching for matching files...')
    for pattern in patterns:
        # Use glob to find files matching the pattern, respecting the recursive flag.
        files = glob.glob(pattern, recursive=recursive)
        for file_path in files:
            # Check if it's a file and hasn't been added yet (in case of overlapping patterns)
            if os.path.isfile(file_path) and file_path not in seen_files:
                matched_files.append(file_path)
                seen_files.add(file_path)

    if not matched_files:
        print('❌ No matching files found.')
        return

    # Sort by filename to ensure a predictable and consistent merge order
    matched_files.sort()

    print(f'✅ Found {len(matched_files)} matching files:')
    for f in matched_files:
        print(f'   - {f}')

    # --- Start merging process ---
    merged_count: int = 0
    skipped_count: int = 0
    expected_keys: List[str] = EXPECTED_KEYS

    try:
        # Open the output file for writing (will overwrite if exists)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in matched_files:
                print(f'\n📌 Processing: {file_path}')
                try:
                    # Open the input file for reading
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line_num, line in enumerate(infile, 1):
                            line_strip = line.strip()
                            if not line_strip:
                                continue  # Skip empty lines

                            try:
                                # Attempt to parse the line as a JSON object
                                data: Dict[str, Any] = json.loads(line_strip)

                                # 1. Verify required fields exist
                                missing_keys = [
                                    key for key in expected_keys
                                    if key not in data
                                ]
                                if missing_keys:
                                    print(
                                        f'❌ Skipped: Line {line_num} in file {file_path} missing fields: {missing_keys}'
                                    )
                                    skipped_count += 1
                                    continue

                                # 2. Validate field content
                                invalid_fields = []
                                for key in expected_keys:
                                    is_valid, error_msg = is_valid_field_content(
                                        key, data[key])
                                    if not is_valid:
                                        invalid_fields.append(
                                            f'{key}({error_msg})')

                                if invalid_fields:
                                    print(
                                        f'❌ Skipped: Invalid field content in line {line_num} of file {file_path}: {"; ".join(invalid_fields)}'
                                    )
                                    skipped_count += 1
                                    continue

                                # 3. Record is valid, write the original line to the output file
                                outfile.write(line_strip + '\n')
                                merged_count += 1

                            except json.JSONDecodeError as e:
                                print(
                                    f'❌ Skipped: JSON parsing error on line {line_num} of file {file_path}: {e}'
                                )
                                skipped_count += 1

                except IOError as e:
                    print(f'❌ Error reading file {file_path}: {e}')
                    skipped_count += 1
                    # Continue to the next file if one fails to read

        # --- Final summary ---
        print('\n\n✅ Merge completed!')
        print(f'   - Written {merged_count} valid records.')
        print(f'   - Skipped {skipped_count} invalid or problematic records.')
        print(f'📁 Output file: {output_file}')

    except IOError as e:
        print(
            f'❌ Critical Error: Could not write to output file {output_file}: {e}'
        )


# ---
def main() -> None:
    """
    Sets up the command-line argument parser and initiates the merge process.
    """
    parser = argparse.ArgumentParser(description=(
        'Merge all JSONL files matched by fuzzy paths (supports wildcards * and **), '
        'validating each record before writing.'))
    parser.add_argument(
        '--patterns',
        nargs='+',
        required=True,
        help=
        'File path matching patterns, e.g.: data/*.jsonl or logs/**/*.jsonl')
    parser.add_argument('--output',
                        default='merged_output.jsonl',
                        help='Output file path (default: merged_output.jsonl)')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Enable recursive directory matching with ** patterns')

    args = parser.parse_args()
    merge_jsonl_files(args.patterns, args.output, args.recursive)


if __name__ == '__main__':
    main()
