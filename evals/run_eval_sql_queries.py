#!/usr/bin/env python3
"""
Utility script to run SQL queries from a dataset following eval_dataset_schema.json format.
"""

import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, TypedDict

import click
import pandas as pd

from research.config import get_config
from research.db.operations import query_duckdb


class SuccessResult(TypedDict):
    """Type for successful query results."""

    id: int
    success: bool
    rows: int


class ErrorResult(TypedDict):
    """Type for failed query results."""

    id: int
    success: bool
    error: str


QueryResult = SuccessResult | ErrorResult


class EvalCase(TypedDict):
    """Type for a transformed evaluation case."""

    id: int
    question: str
    expected_sql: Optional[str]
    expected_result: str
    metadata: Dict[str, Any]


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_eval_data(path: str) -> List[EvalCase]:
    """Load evaluation data from JSON file following eval_dataset_schema.json format."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract cases from the dataset
        cases = data.get("cases", [])

        # Transform cases to match script expectations
        transformed_cases: List[EvalCase] = []
        for i, case in enumerate(cases):
            case_id = i + 1
            if "name" in case and case["name"]:
                try:
                    case_id = int(case["name"])
                except ValueError:
                    # Keep the auto-generated ID if name is not an integer
                    pass

            expected_sql = None
            if (
                case.get("expected_output")
                and isinstance(case["expected_output"], dict)
                and "sql_query" in case["expected_output"]
            ):
                expected_sql = case["expected_output"]["sql_query"]

            transformed_case: EvalCase = {
                "id": case_id,
                "question": case.get("inputs", ""),
                "expected_sql": expected_sql,
                "expected_result": case.get("expected_output", {}).get("result", ""),
                "metadata": case.get("metadata", {}),
            }
            transformed_cases.append(transformed_case)

        return transformed_cases
    except Exception as e:
        logging.error(f"Failed to load evaluation data: {e}")
        sys.exit(1)


def run_sql_query(database: str, query: str) -> pd.DataFrame:
    """Run SQL query against database."""
    try:
        return query_duckdb(database, query)
    except Exception as e:
        logging.error(f"Query execution failed: {e}")
        sys.exit(1)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Utility to run SQL queries from dataset following eval_dataset_schema.json format."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option(
    "--database", "--db", required=True, type=str, help="Path to DuckDB database"
)
@click.option(
    "--eval-file",
    "-e",
    type=click.Path(exists=True),
    default=lambda: get_config().eval.dataset_path,
    help="Path to evaluation data JSON file",
)
@click.option(
    "--query-id", "-id", type=int, help="ID of specific query to run from eval data"
)
@click.option(
    "--range",
    "-r",
    type=str,
    help="Range of query IDs to run (e.g., '1-5')",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    help="Directory to save query files and results",
)
@click.option(
    "--compare-expected/--no-compare-expected",
    default=False,
    help="Compare results with expected output",
)
@click.pass_context
def run(
    ctx: click.Context,
    database: str,
    eval_file: str,
    query_id: Optional[int] = None,
    range: Optional[str] = None,
    output_dir: Optional[str] = None,
    compare_expected: bool = False,
) -> None:
    """Run SQL queries from evaluation dataset."""
    if query_id and range:
        raise click.UsageError("Cannot specify both --query-id and --range.")
    # Load evaluation data
    eval_data = load_eval_data(eval_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter by ID if specified
    if query_id is not None:
        queries = [q for q in eval_data if q["id"] == query_id]
        if not queries:
            logging.error(f"No query with ID {query_id} found in evaluation data")
            sys.exit(1)
    elif range is not None:
        try:
            start, end = map(int, range.split("-"))
            queries = [q for q in eval_data if start <= q["id"] <= end]
            if not queries:
                logging.error(f"No queries found in range {range}")
                sys.exit(1)
        except ValueError:
            logging.error("Invalid range format. Use 'start-end' (e.g., '1-5')")
            sys.exit(1)
    else:
        queries = eval_data

    results: List[QueryResult] = []

    # Process each query
    for query_item in queries:
        query_id = int(query_item["id"])  # Ensure ID is an integer
        question = query_item["question"]
        sql = query_item["expected_sql"]

        if not sql:
            logging.warning(f"Query ID {query_id} has no SQL query defined, skipping")
            continue

        logging.info(f"Processing query ID {query_id}: {question}")

        # Save query to file if output directory is specified
        if output_dir:
            query_file = os.path.join(output_dir, f"query_{query_id}.sql")
            with open(query_file, "w", encoding="utf-8") as f:
                f.write(sql)
            logging.info(f"Saved query to {query_file}")

        # Run the query
        try:
            # Execute the query
            result_df = run_sql_query(database, sql)

            # Save result if output directory is specified
            if output_dir:
                result_file = os.path.join(output_dir, f"result_{query_id}.csv")
                result_df.to_csv(result_file, index=False)
                logging.info(f"Saved result to {result_file}")

            # Display result
            click.echo(f"\nResults for query ID {query_id}:")
            click.echo(result_df)

            # Compare with expected result if requested
            if (
                compare_expected
                and "expected_result" in query_item
                and query_item["expected_result"]
            ):
                expected_result = query_item["expected_result"]
                click.echo("\nExpected result:")
                click.echo(expected_result)

                # Convert result to string format for comparison
                result_str = result_df.to_csv(sep="|", index=False).strip()

                # Simple comparison (could be enhanced for more robust comparison)
                matches = expected_result.strip() == result_str
                click.echo(f"\nResults match expected output: {matches}")

            # Add success result
            success_result: SuccessResult = {
                "id": query_id,
                "success": True,
                "rows": len(result_df),
            }
            results.append(success_result)

        except Exception as e:
            logging.error(f"Failed to execute query ID {query_id}: {e}")
            # Add error result
            error_result: ErrorResult = {
                "id": query_id,
                "success": False,
                "error": str(e),
            }
            results.append(error_result)

    # Summary
    success_count = sum(1 for r in results if r["success"])
    click.echo(
        f"\nSummary: {success_count}/{len(results)} queries executed successfully"
    )


@cli.command()
@click.option(
    "--database", "--db", required=True, type=str, help="Path to DuckDB database"
)
@click.option("--query", "-q", type=str, help="SQL query to execute directly")
@click.pass_context
def direct(ctx: click.Context, database: str, query: Optional[str] = None) -> None:
    """Run a SQL query directly from command line or stdin."""
    sql_query = query

    if not sql_query and not sys.stdin.isatty():
        # Read from stdin if available
        sql_query = sys.stdin.read()

    if not sql_query:
        logging.error("No query provided via argument or stdin")
        sys.exit(1)

    # Create a temporary file for the query
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sql", delete=False, encoding="utf-8"
    ) as temp:
        temp.write(sql_query)
        temp_path = temp.name

    try:
        # Run the query
        result = run_sql_query(database, sql_query)

        # Display result
        click.echo("\nResults:")
        click.echo(result)

        logging.info(f"Query executed successfully, returned {len(result)} rows")

    except Exception as e:
        logging.error(f"Query execution failed: {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    cli()
