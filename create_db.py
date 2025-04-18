#!/usr/bin/env python3
"""
Create a new DuckDB database.
This script initializes an empty DuckDB database file that can be used with other tools.
"""

import logging
import sys
from pathlib import Path

import click

from db_operations import init_db


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


@click.command()
@click.option(
    "--database",
    "--db",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path where the new database file will be created",
)
@click.option("--force", is_flag=True, help="Overwrite existing database")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.option(
    "--metadata-table/--no-metadata-table",
    default=True,
    help="Create metadata table (default: yes)",
)
def create_database(
    database: str,
    force: bool = False,
    verbose: bool = False,
    metadata_table: bool = True,
) -> None:
    """Create a new empty DuckDB database."""
    setup_logging(verbose)
    db_file = Path(database)

    # Check if file exists and handle accordingly
    if db_file.exists() and not force:
        logging.error(
            f"Database file already exists at {database}. Use --force to overwrite."
        )
        sys.exit(1)

    # Create parent directory if it doesn't exist
    db_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize the database
        logging.info(f"Creating new DuckDB database at {database}")
        conn = init_db(database)

        # Create a simple metadata table to track database info if requested
        if metadata_table:
            logging.info("Creating metadata table")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _metadata (
                    key VARCHAR,
                    value VARCHAR,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Add creation timestamp
            conn.execute("""
                INSERT INTO _metadata (key, value) VALUES 
                ('created_at', CURRENT_TIMESTAMP),
                ('version', '1.0')
            """)

        conn.close()
        logging.info(f"Database successfully created at {database}")
        click.echo(f"Database created at {database}")

    except Exception as e:
        logging.error(f"Failed to create database: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    create_database()
