#!/usr/bin/env python3
"""
CLI tool for generating SQL queries using LLM and executing them against DuckDB.
"""

import logging
import sys
from typing import Optional

import click
import pandas as pd

from db_operations import query_duckdb
from src.llm_sql import natural_language_to_sql


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
    """CLI tool for generating SQL queries with LLM and executing them against DuckDB."""
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
    "--question", "-q", required=True, type=str, help="Natural language question"
)
@click.option(
    "--metadata",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default="src/survey_metadata_queries.json",
    help="Path to survey metadata JSON file",
)
@click.option(
    "--template",
    "-t",
    type=click.Path(exists=True, dir_okay=False),
    default="src/sql_query_prompt.jinja2",
    help="Path to prompt template file",
)
@click.option(
    "--output", "-o", type=click.Path(dir_okay=False), help="Save results to CSV file"
)
@click.option(
    "--show-rows", type=int, default=20, help="Number of rows to display (0 for all)"
)
@click.option(
    "--show-sql",
    is_flag=True,
    default=True,
    help="Display the generated SQL query",
)
@click.option(
    "--execute/--no-execute",
    default=True,
    help="Execute the generated SQL query (default: True)",
)
@click.pass_context
def ask(
    ctx: click.Context,
    database: str,
    question: str,
    metadata: str,
    template: str,
    output: Optional[str] = None,
    show_rows: int = 20,
    show_sql: bool = True,
    execute: bool = True,
) -> None:
    """Ask a natural language question and convert it to SQL using LLM."""
    try:
        # Generate SQL query
        sql_query = natural_language_to_sql(
            question=question,
            metadata_path=metadata,
            template_path=template,
            verbose=ctx.obj["verbose"],
        )

        if show_sql:
            click.echo("\nGenerated SQL query:")
            click.echo(f"{sql_query}\n")

        if not execute:
            return

        # Execute the query
        logging.info("Executing generated SQL query")
        result = query_duckdb(database, sql_query)

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

            # Print results
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
        logging.error(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--question", "-q", required=True, type=str, help="Natural language question"
)
@click.option(
    "--metadata",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default="src/survey_metadata_queries.json",
    help="Path to survey metadata JSON file",
)
@click.option(
    "--template",
    "-t",
    type=click.Path(exists=True, dir_okay=False),
    default="src/sql_query_prompt.jinja2",
    help="Path to prompt template file",
)
@click.pass_context
def generate(ctx: click.Context, question: str, metadata: str, template: str) -> None:
    """Generate SQL from natural language without executing it."""
    try:
        # Generate SQL query
        sql_query = natural_language_to_sql(
            question=question,
            metadata_path=metadata,
            template_path=template,
            verbose=ctx.obj["verbose"],
        )

        click.echo("\nGenerated SQL query:")
        click.echo(sql_query)

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--database",
    "--db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the DuckDB database file",
)
@click.option(
    "--query",
    "-q",
    type=str,
    help="SQL query to execute (use quotes for complex queries)",
)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    help="SQL file to execute",
)
@click.option(
    "--output", "-o", type=click.Path(dir_okay=False), help="Save results to CSV file"
)
@click.option(
    "--show-rows", type=int, default=20, help="Number of rows to display (0 for all)"
)
@click.pass_context
def execute(
    ctx: click.Context,
    database: str,
    query: Optional[str] = None,
    file: Optional[str] = None,
    output: Optional[str] = None,
    show_rows: int = 20,
) -> None:
    """Execute SQL query directly against the database."""
    if not query and not file:
        click.echo("Error: Either --query or --file must be provided")
        sys.exit(1)

    if query and file:
        click.echo("Error: Only one of --query or --file can be provided, not both")
        sys.exit(1)

    try:
        # Get SQL from file if specified
        if file:
            with open(file, "r", encoding="utf-8") as f:
                sql_query = f.read()
            logging.info(f"Loaded SQL query from {file}")
        else:
            sql_query = query

        # Ensure sql_query is not None before proceeding
        if sql_query is None:
            raise ValueError("SQL query is empty or not provided correctly")

        # Execute the query
        logging.info("Executing SQL query")
        result = query_duckdb(database, sql_query)

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

            # Print results
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


if __name__ == "__main__":
    cli()
