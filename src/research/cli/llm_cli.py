#!/usr/bin/env python3
"""
CLI tool for generating SQL queries using LLM and executing them against DuckDB.
"""

import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from openai import OpenAI

from research.db.operations import query_duckdb
from research.llm.sql_generator import natural_language_to_sql
from research.utils.logger import setup_logger

logger = setup_logger(__name__)


def natural_language_to_sql_openai(
    question: str,
    base_url: str = "http://127.0.0.1:8001/v1",
    api_key: str = "test-key",
    model: str = "gzm-research",
    temperature: float = 0.0,
    verbose: bool = False,
) -> str:
    """
    Convert natural language to SQL using OpenAI-compatible API endpoint.

    Args:
        question: Natural language question
        base_url: Base URL for the OpenAI-compatible API
        api_key: API key for authentication
        model: Model name to use
        temperature: Model temperature
        verbose: Enable verbose logging

    Returns:
        Generated SQL query
    """
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        if verbose:
            logger.info(f"Using OpenAI-compatible endpoint: {base_url}")
            logger.info(f"Model: {model}")

        # Create a simple system message for SQL generation
        system_message = """You are a SQL expert. Convert natural language questions to SQL queries.
Return only the SQL query without any explanations or markdown formatting."""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=1000,
        )

        sql_query = response.choices[0].message.content.strip()

        if verbose:
            logger.info(f"Generated SQL via OpenAI API: {sql_query}")

        return sql_query

    except Exception as e:
        logger.error(f"Error generating SQL via OpenAI API: {e}")
        return ""


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CLI tool for generating SQL queries with LLM and executing them against DuckDB."""
    # Set up logging
    if verbose:
        logger.setLevel("DEBUG")

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
    help="Path to survey metadata JSON file",
)
@click.option(
    "--template",
    "-t",
    type=click.Path(exists=True, dir_okay=False),
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
@click.option("--temperature", type=float, help="Model temperature (0.0-1.0)")
@click.option(
    "--use-api",
    is_flag=True,
    help="Use OpenAI-compatible API endpoint instead of direct Gemini",
)
@click.option(
    "--api-url",
    default="http://127.0.0.1:8001/v1",
    help="OpenAI-compatible API base URL",
)
@click.option(
    "--api-key",
    default="test-key",
    help="API key for OpenAI-compatible endpoint",
)
@click.option(
    "--model",
    default="gzm-research",
    help="Model name for OpenAI-compatible endpoint",
)
@click.pass_context
def ask(
    ctx: click.Context,
    database: str,
    question: str,
    metadata: Optional[str] = None,
    template: Optional[str] = None,
    output: Optional[str] = None,
    show_rows: int = 20,
    show_sql: bool = True,
    execute: bool = True,
    temperature: Optional[float] = None,
    use_api: bool = False,
    api_url: str = "http://127.0.0.1:8001/v1",
    api_key: str = "test-key",
    model: str = "gzm-research",
) -> None:
    """Ask a natural language question and convert it to SQL using LLM."""
    try:
        # Generate SQL query using either API endpoint or direct Gemini
        if use_api:
            if ctx.obj["verbose"]:
                click.echo(f"Using OpenAI-compatible API at: {api_url}")
            sql_query = natural_language_to_sql_openai(
                question=question,
                base_url=api_url,
                api_key=api_key,
                model=model,
                temperature=temperature or 0.0,
                verbose=ctx.obj["verbose"],
            )
        else:
            sql_query = natural_language_to_sql(
                question=question,
                metadata_path=metadata,
                template_path=template,
                temperature=temperature,
                verbose=ctx.obj["verbose"],
            )

        if not sql_query:
            logger.error("Failed to generate SQL query")
            sys.exit(1)

        if show_sql:
            click.echo("\nGenerated SQL query:")
            click.echo(f"{sql_query}\n")

        if not execute:
            return

        # Execute the query
        logger.info("Executing generated SQL query")
        result = query_duckdb(database, sql_query)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False)
            logger.info(f"Results saved to {output}")

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
        logger.error(f"Error: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--question", "-q", required=True, type=str, help="Natural language question"
)
@click.option(
    "--metadata",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to survey metadata JSON file",
)
@click.option(
    "--template",
    "-t",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to prompt template file",
)
@click.option("--temperature", type=float, help="Model temperature (0.0-1.0)")
@click.option(
    "--use-api",
    is_flag=True,
    help="Use OpenAI-compatible API endpoint instead of direct Gemini",
)
@click.option(
    "--api-url",
    default="http://127.0.0.1:8001/v1",
    help="OpenAI-compatible API base URL",
)
@click.option(
    "--api-key",
    default="test-key",
    help="API key for OpenAI-compatible endpoint",
)
@click.option(
    "--model",
    default="gzm-research",
    help="Model name for OpenAI-compatible endpoint",
)
@click.pass_context
def generate(
    ctx: click.Context,
    question: str,
    metadata: Optional[str] = None,
    template: Optional[str] = None,
    temperature: Optional[float] = None,
    use_api: bool = False,
    api_url: str = "http://127.0.0.1:8001/v1",
    api_key: str = "test-key",
    model: str = "gzm-research",
) -> None:
    """Generate SQL from natural language without executing it."""
    try:
        # Generate SQL query using either API endpoint or direct Gemini
        if use_api:
            if ctx.obj["verbose"]:
                click.echo(f"Using OpenAI-compatible API at: {api_url}")
            sql_query = natural_language_to_sql_openai(
                question=question,
                base_url=api_url,
                api_key=api_key,
                model=model,
                temperature=temperature or 0.0,
                verbose=ctx.obj["verbose"],
            )
        else:
            sql_query = natural_language_to_sql(
                question=question,
                metadata_path=metadata,
                template_path=template,
                temperature=temperature,
                verbose=ctx.obj["verbose"],
            )

        if not sql_query:
            logger.error("Failed to generate SQL query")
            sys.exit(1)

        click.echo("\nGenerated SQL query:")
        click.echo(sql_query)

    except Exception as e:
        logger.error(f"Error: {e}")
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
            logger.info(f"Loaded SQL query from {file}")
        else:
            sql_query = query or ""

        # Ensure sql_query is not empty
        if not sql_query:
            raise ValueError("SQL query is empty")

        # Execute the query
        logger.info("Executing SQL query")
        result = query_duckdb(database, sql_query)

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output, index=False)
            logger.info(f"Results saved to {output}")

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
        logger.error(f"Error executing query: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
