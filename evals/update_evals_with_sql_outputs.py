"""
Script to run all SQL queries in a JSON file and update the result field
in eval dataset with the actual results from the database.
"""

import json
import sys
import time
from pathlib import Path

import click
import pandas as pd

from research.config import get_config
from research.db.operations import query_duckdb


def format_result(df: pd.DataFrame) -> str:
    """Format the DataFrame result to match the expected format in the JSON file."""
    if df.empty:
        return ""

    # Get column headers
    header = "|".join(str(col) for col in df.columns)

    # Format each row
    rows: list[str] = []
    for _, row in df.iterrows():
        # Convert all values to strings and join with pipes
        row_values = [str(val) for val in row.values]
        rows.append("|".join(row_values))

    # Join header and rows with newlines
    return header + "\n" + "\n".join(rows)


@click.command()
@click.option(
    "--database",
    "--db",
    required=False,
    default=lambda: get_config().db.default_path,
    help="Path to DuckDB database",
)
@click.option(
    "--json-file",
    "-j",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation JSON file",
)
def main(database: str, json_file: str) -> None:
    db_path = database
    json_path = Path(json_file)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} cases from {json_path}")

    # Process each query
    for i, case in enumerate(cases):
        case_name = case.get("name", str(i + 1))
        expected_output = case.get("expected_output", {})
        sql = expected_output.get("sql_query", "")

        if not sql:
            print(f"Skipping case {case_name}: No SQL found")
            continue

        try:
            print(f"Running case {case_name}...")

            # Execute the query
            result_df = query_duckdb(db_path, sql)

            # Format the result
            formatted_result = format_result(result_df)

            # Update the entry
            expected_output["result"] = formatted_result

            print(f"Case {case_name} completed successfully")

            # Wait to be gentle on the server
            time.sleep(0.5)

        except Exception as e:
            print(f"Error executing case {case_name}: {e}")

    # Save the updated JSON file
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Updated {json_path} with query results")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
