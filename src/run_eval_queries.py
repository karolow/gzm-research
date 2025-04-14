#!/usr/bin/env python3
"""
Script to run all queries in eval_sql.json and update the expected_result field
with the actual results from the database.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, cast

import pandas as pd
from pandas import Series

from db_operations import query_duckdb


def format_result(df: pd.DataFrame) -> str:
    """Format the DataFrame result to match the expected format in the JSON file."""
    if df.empty:
        return ""

    # Get column headers
    header = "|".join(str(col) for col in df.columns)

    # Format each row
    rows: List[str] = []
    for _, row in df.iterrows():
        # Convert all values to strings and join with pipes
        row_str = cast(Series, row)
        row_values = [str(val) for val in row_str.values]
        rows.append("|".join(row_values))

    # Join header and rows with newlines
    return header + "\n" + "\n".join(rows)


def main():
    # Check if database path is provided
    if len(sys.argv) < 2:
        print("Usage: python run_eval_queries.py <path_to_database>")
        sys.exit(1)

    db_path = sys.argv[1]

    # Load the eval_sql.json file
    json_path = Path(__file__).parent / "eval_sql.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

    print(f"Loaded {len(data)} queries from {json_path}")

    # Process each query
    for i, entry in enumerate(data):
        query_id = entry.get("id", i + 1)
        sql = entry.get("expected_sql", "")

        if not sql:
            print(f"Skipping query {query_id}: No SQL found")
            continue

        try:
            print(f"Running query {query_id}...")

            # Execute the query
            result_df = query_duckdb(db_path, sql)

            # Format the result
            formatted_result = format_result(result_df)

            # Update the entry
            entry["expected_result"] = formatted_result

            print(f"Query {query_id} completed successfully")

            # Wait to be gentle on the server
            time.sleep(0.5)

        except Exception as e:
            print(f"Error executing query {query_id}: {e}")

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
