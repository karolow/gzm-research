#!/usr/bin/env python3
"""
Script for generating data in the required format for fine-tuning Gemma3 model.
Takes data from JSON schema definition and generates JSONL files for training, validation, and testing.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, TypeVar

# Type variable for generic typing
T = TypeVar("T")


def load_json_file(file_path: str) -> Any:
    """Load JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file {file_path}: {e}")


def create_column_list(metadata: list[dict[str, Any]]) -> list[str]:
    """Create a list of column names from metadata."""
    columns: list[str] = []

    for item in metadata:
        # Skip items that don't have all required fields
        if not all(
            key in item for key in ["semantic_name", "data_type", "description"]
        ):
            continue

        name: str = item["semantic_name"]
        columns.append(name)

    return columns


def create_fine_tuning_example(
    question: str, columns: list[str], sql_query: str
) -> dict[str, str]:
    """Create a single fine-tuning example in the required format."""
    # column_list = ", ".join(columns)

    user_prompt = (
        "Translate the following natural language query to SQL "
        f"Question: {question}\n\n"
        "Generate only the SQL query."
    )

    return {
        "text": f"<start_of_turn>user\n{user_prompt}\n<end_of_turn>\n<start_of_turn>model\n{sql_query}<end_of_turn>\n"
    }


def sample_examples(examples: list[T], max_examples: int | None = None) -> list[T]:
    """Randomly sample a subset of examples if max_examples is specified."""
    if max_examples is None or max_examples >= len(examples):
        return examples

    return random.sample(examples, max_examples)


def split_dataset(
    items: list[T], train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[list[T], list[T], list[T]]:
    """Split a list of items into training, validation, and test sets."""
    # Validate ratios
    if not 0.99 <= train_ratio + val_ratio + test_ratio <= 1.01:
        raise ValueError("Split ratios must sum to approximately 1.0")

    # Shuffle the data
    random.shuffle(items)

    # Calculate split indices
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)

    # Split the data
    train_set = items[:train_end]
    val_set = items[train_end:val_end]
    test_set = items[val_end:]

    return train_set, val_set, test_set


def save_jsonl(items: list[dict[str, str]], output_file: str) -> None:
    """Save a list of dictionaries as JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    """Main function for the script."""
    parser = argparse.ArgumentParser(
        description="Generate data for fine-tuning Gemma3 model"
    )
    parser.add_argument(
        "--eval-dataset",
        required=True,
        type=str,
        help="Path to the evaluation dataset JSON file",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=str,
        help="Path to the survey metadata JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save the output JSONL files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of training data (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation data (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of test data (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting (default: 42)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to randomly select (default: use all available examples)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    eval_dataset = load_json_file(args.eval_dataset)
    metadata = load_json_file(args.metadata)

    # Create column list from metadata
    columns = create_column_list(metadata)

    # Create fine-tuning examples
    examples: list[dict[str, str]] = []
    for case in eval_dataset.get("cases", []):
        if (
            "inputs" in case
            and "expected_output" in case
            and "sql_query" in case["expected_output"]
        ):
            question = case["inputs"]
            sql_query = case["expected_output"]["sql_query"]
            examples.append(create_fine_tuning_example(question, columns, sql_query))

    # Sample examples if max_examples is specified
    if args.max_examples:
        original_count = len(examples)
        examples = sample_examples(examples, args.max_examples)
        print(
            f"Randomly selected {len(examples)} examples from {original_count} total examples"
        )

    # Split dataset
    train_set, val_set, test_set = split_dataset(
        examples, args.train_ratio, args.val_ratio, args.test_ratio
    )

    # Save datasets
    output_dir = Path(args.output_dir)
    save_jsonl(train_set, str(output_dir / "train.jsonl"))
    save_jsonl(val_set, str(output_dir / "valid.jsonl"))
    save_jsonl(test_set, str(output_dir / "test.jsonl"))

    print("Generated datasets:")
    print(f"  Training:   {len(train_set)} examples")
    print(f"  Validation: {len(val_set)} examples")
    print(f"  Test:       {len(test_set)} examples")
    print(f"Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
