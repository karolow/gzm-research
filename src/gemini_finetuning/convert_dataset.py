import json
import os
import random
from pathlib import Path
from typing import Any

# Set random seed for reproducibility
random.seed(42)


def create_gemini_format_example(
    item: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Convert a single dataset item into Gemini's fine-tuning format.

    Args:
        item: A dictionary with the case data

    Returns:
        A dictionary formatted according to Gemini's fine-tuning requirements
    """
    # Extract the input and expected output
    query = item["inputs"]
    sql_query = item["expected_output"]["sql_query"]
    result = item["expected_output"]["result"]

    # Create Gemini format example
    gemini_example = {
        "contents": [
            {"role": "user", "parts": [{"text": query}]},
            {
                "role": "model",
                "parts": [{"text": f"SQL Query: {sql_query}\n\nResult:\n{result}"}],
            },
        ]
    }

    return gemini_example


def convert_and_split_dataset(
    input_file: str, output_dir: str, split_ratios: list[float] = [0.7, 0.15, 0.15]
) -> None:
    """
    Convert the dataset to Gemini format and split it into train/test/validation sets.

    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to save the output files
        split_ratios: List of ratios for train/test/validation splits (must sum to 1)
    """
    # Validate split ratios
    if sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} cases from {input_file}")

    # Convert all examples to Gemini format
    gemini_examples = [create_gemini_format_example(case) for case in cases]

    # Shuffle the examples
    random.shuffle(gemini_examples)

    # Calculate split indices
    total_examples = len(gemini_examples)
    train_end = int(total_examples * split_ratios[0])
    test_end = train_end + int(total_examples * split_ratios[1])

    # Split the dataset
    train_examples = gemini_examples[:train_end]
    test_examples = gemini_examples[train_end:test_end]
    val_examples = gemini_examples[test_end:]

    # Save the splits to JSONL files
    splits = {
        "train": train_examples,
        "test": test_examples,
        "validation": val_examples,
    }

    for split_name, examples in splits.items():
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Saved {len(examples)} examples to {output_file}")


if __name__ == "__main__":
    # Paths
    script_dir = Path(__file__).parent
    input_file = script_dir.parent / "eval_dataset.json"
    output_dir = script_dir

    # Convert and split the dataset
    convert_and_split_dataset(
        input_file=str(input_file),
        output_dir=str(output_dir),
        split_ratios=[0.7, 0.15, 0.15],  # 70% train, 15% test, 15% validation
    )

    print("Conversion and splitting completed successfully!")
