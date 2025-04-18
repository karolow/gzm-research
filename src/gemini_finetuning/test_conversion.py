import json
from pathlib import Path
from typing import Any


def validate_jsonl_file(file_path: str) -> tuple[bool, list[str], int]:
    """
    Validate that a JSONL file is properly formatted for Gemini fine-tuning.

    Args:
        file_path: Path to the JSONL file to validate

    Returns:
        A tuple of (is_valid, error_messages, example_count)
    """
    error_messages = []
    example_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                example_count += 1
                try:
                    example = json.loads(line)
                    validate_example(example, line_num, error_messages)
                except json.JSONDecodeError as e:
                    error_messages.append(f"Line {line_num}: Invalid JSON format - {e}")
    except Exception as e:
        error_messages.append(f"Error reading file: {e}")

    return len(error_messages) == 0, error_messages, example_count


def validate_example(
    example: dict[str, Any], line_num: int, error_messages: list[str]
) -> None:
    """
    Validate an individual example has the correct structure.

    Args:
        example: The example to validate
        line_num: The line number in the file for error reporting
        error_messages: List to append error messages to
    """
    # Check required top-level keys
    if "contents" not in example:
        error_messages.append(f"Line {line_num}: Missing 'contents' key")
        return

    # Check contents is a list with at least 2 elements (user and model)
    contents = example["contents"]
    if not isinstance(contents, list):
        error_messages.append(f"Line {line_num}: 'contents' is not a list")
        return

    if len(contents) < 2:
        error_messages.append(
            f"Line {line_num}: 'contents' must have at least 2 elements (user and model)"
        )
        return

    # Check each content item
    for i, content in enumerate(contents):
        # Check required content keys
        if "role" not in content:
            error_messages.append(f"Line {line_num}, content {i}: Missing 'role' key")
            continue

        if "parts" not in content:
            error_messages.append(f"Line {line_num}, content {i}: Missing 'parts' key")
            continue

        # Check role value
        role = content["role"]
        if role not in ["user", "model"]:
            error_messages.append(
                f"Line {line_num}, content {i}: 'role' must be 'user' or 'model', got '{role}'"
            )

        # Check parts is a list
        parts = content["parts"]
        if not isinstance(parts, list):
            error_messages.append(
                f"Line {line_num}, content {i}: 'parts' is not a list"
            )
            continue

        if len(parts) == 0:
            error_messages.append(f"Line {line_num}, content {i}: 'parts' is empty")
            continue

        # Check each part
        for j, part in enumerate(parts):
            if "text" not in part:
                error_messages.append(
                    f"Line {line_num}, content {i}, part {j}: Missing 'text' key"
                )


def test_conversion() -> None:
    """Test that the JSONL files are correctly formatted for Gemini fine-tuning."""
    script_dir = Path(__file__).parent

    # Files to validate
    files_to_validate = [
        script_dir / "train.jsonl",
        script_dir / "test.jsonl",
        script_dir / "validation.jsonl",
    ]

    # Check if files exist
    missing_files = [f for f in files_to_validate if not f.exists()]
    if missing_files:
        print("Error: The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nRun convert_dataset.py first to generate these files.")
        return

    # Validate each file
    all_valid = True
    total_examples = 0

    print("Validating Gemini fine-tuning data files...\n")

    for file_path in files_to_validate:
        print(f"Checking {file_path.name}...")
        is_valid, errors, count = validate_jsonl_file(str(file_path))
        total_examples += count

        if is_valid:
            print(f"  ✓ Valid with {count} examples")
        else:
            all_valid = False
            print(f"  ✗ Found {len(errors)} errors in {count} examples:")
            for error in errors:
                print(f"    - {error}")
        print()

    if all_valid:
        print(f"All files are valid! Total examples: {total_examples}")
        print(
            "\nYou can now upload these files to Google Cloud Storage and start fine-tuning:"
        )
        print("python upload_to_gcs.py --bucket YOUR_BUCKET_NAME")
    else:
        print("Please fix the errors before proceeding with fine-tuning.")


if __name__ == "__main__":
    test_conversion()
