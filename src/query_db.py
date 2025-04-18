#!/usr/bin/env python3
"""
Query tool for the research database.
Allows executing SQL queries against the DuckDB database.
"""

import logging
import sys
from typing import Optional

import click
import pandas as pd

from db_operations import query_duckdb


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Query tool for the research database."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option(
    "--database",
    "--db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the DuckDB database file",
)
@click.option(
    "--query", "-q", required=True, type=str, help="SQL query to execute (use quotes)"
)
@click.option(
    "--output", "-o", type=click.Path(dir_okay=False), help="Save results to CSV file"
)
@click.option(
    "--limit", type=int, default=None, help="Limit the number of results to return"
)
@click.option(
    "--show-rows", type=int, default=20, help="Number of rows to display (0 for all)"
)
@click.pass_context
def query(
    ctx: click.Context,
    database: str,
    query: str,
    output: Optional[str] = None,
    limit: Optional[int] = None,
    show_rows: int = 20,
) -> None:
    """Execute SQL query against the database."""
    try:
        # Add LIMIT clause if requested and not already in the query
        if limit is not None and "limit" not in query.lower():
            query = f"{query} LIMIT {limit}"

        logging.info(f"Executing query: {query}")
        result = query_duckdb(database, query)

        if output:
            result.to_csv(output, index=False)
            logging.info(f"Results saved to {output}")

        # Always print a summary
        row_count = len(result)
        if row_count == 0:
            click.echo("Query returned no results")
        else:
            col_count = len(result.columns)
            click.echo(f"Query returned {row_count} rows with {col_count} columns")

            # Determine max rows to display - 0 means all rows
            display_rows = None if show_rows <= 0 else show_rows

            # Print results if not saved to file or if verbose
            if not output or ctx.obj["verbose"]:
                with pd.option_context(
                    "display.max_rows",
                    display_rows,
                    "display.max_columns",
                    None,
                    "display.width",
                    1000,
                ):
                    click.echo("\nResults:")
                    click.echo(result)

    except Exception as e:
        logging.error(f"Error executing query: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--database",
    "--db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the DuckDB database file",
)
def list_tables(database: str) -> None:
    """List all tables in the database."""
    try:
        tables = query_duckdb(
            database, "SELECT name FROM sqlite_master WHERE type='table'"
        )

        if tables.empty:
            click.echo("No tables found in the database")
        else:
            click.echo("Tables in the database:")
            # Extract table names as strings to avoid linter errors
            table_names = tables["name"].astype(str).tolist()
            for table_name in table_names:
                click.echo(f"- {table_name}")

    except Exception as e:
        logging.error(f"Error listing tables: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--database",
    "--db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the DuckDB database file",
)
@click.option("--table", required=True, type=str, help="Name of the table to describe")
@click.option(
    "--show-rows",
    type=int,
    default=5,
    help="Number of sample rows to display (0 for none)",
)
def describe_table(database: str, table: str, show_rows: int = 5) -> None:
    """Describe a table's structure."""
    try:
        # Check if table exists
        tables = query_duckdb(
            database,
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'",
        )

        if tables.empty:
            click.echo(f"Table '{table}' does not exist")
            sys.exit(1)

        # Get table info
        columns = query_duckdb(database, f"PRAGMA table_info({table})")

        if columns.empty:
            click.echo(f"No columns found in table '{table}'")
        else:
            click.echo(f"Columns in table '{table}':")
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                click.echo(columns[["name", "type"]])

        # Get row count
        count = query_duckdb(database, f"SELECT COUNT(*) as count FROM {table}")
        # Get value directly and convert to int
        count_value = int(count["count"].values[0])
        click.echo(f"\nTotal rows: {count_value}")

        # Show sample data if requested
        if count_value > 0 and show_rows > 0:
            sample = query_duckdb(database, f"SELECT * FROM {table} LIMIT {show_rows}")
            click.echo("\nSample data:")
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                1000,
            ):
                click.echo(sample)

    except Exception as e:
        logging.error(f"Error describing table: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
