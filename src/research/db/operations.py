"""
Core database operations for DuckDB.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from duckdb import DuckDBPyConnection

from research.db.connection import init_db
from research.utils.logging import setup_logger

logger = setup_logger(__name__)


def table_exists(conn: DuckDBPyConnection, table_name: str) -> bool:
    """
    Check if a table exists in the database.

    Args:
        conn: DuckDB connection
        table_name: Name of the table to check

    Returns:
        True if the table exists, False otherwise
    """
    result = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", [table_name]
    ).fetchall()
    return result[0][0] > 0


def save_to_duckdb(
    df: pd.DataFrame,
    db_path: str,
    table_name: str,
    replace: bool = False,
) -> bool:
    """
    Save DataFrame to DuckDB.

    Args:
        df: DataFrame to save
        db_path: Path to DuckDB database
        table_name: Name of the table to save to
        replace: Whether to replace the table if it exists

    Returns:
        Success status
    """
    try:
        # Create parent directory if it doesn't exist
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        conn = init_db(db_path)

        # Register the DataFrame as a temporary view
        conn.register("temp_df", df)

        # Check if table exists and handle accordingly
        if replace or not table_exists(conn, table_name):
            logger.info(f"Creating/replacing table '{table_name}'")
            conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df"
            )
        else:
            logger.info(f"Appending to existing table '{table_name}'")
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")

        conn.unregister("temp_df")
        conn.close()

        logger.info(f"Successfully saved {len(df)} rows to table '{table_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to save to DuckDB: {e}", exc_info=True)
        return False


def query_duckdb(
    db_path: str,
    sql_query: str,
    params: Optional[list[Any]] = None,
    read_only: bool = True,
) -> pd.DataFrame:
    """
    Execute SQL query against DuckDB database.

    Args:
        db_path: Path to DuckDB database
        sql_query: SQL query to execute
        params: Optional parameters for the query
        read_only: Whether to open the database in read-only mode

    Returns:
        Query result as DataFrame

    Raises:
        RuntimeError: If the query fails
    """
    try:
        conn = init_db(db_path, read_only=read_only)

        if params:
            result = conn.execute(sql_query, params).df()
        else:
            result = conn.execute(sql_query).df()

        conn.close()
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise RuntimeError(f"Database query failed: {e}") from e


def list_tables(db_path: str) -> list[str]:
    """
    List all tables in the database.

    Args:
        db_path: Path to DuckDB database

    Returns:
        List of table names
    """
    try:
        tables_df = query_duckdb(
            db_path, "SELECT name FROM sqlite_master WHERE type='table'"
        )
        if tables_df.empty:
            return []
        return tables_df["name"].astype(str).tolist()
    except Exception as e:
        logger.error(f"Failed to list tables: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list tables: {e}") from e


def get_table_schema(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Get the schema of a table.

    Args:
        db_path: Path to DuckDB database
        table_name: Name of the table

    Returns:
        DataFrame with column information
    """
    try:
        return query_duckdb(db_path, f"PRAGMA table_info({table_name})")
    except Exception as e:
        logger.error(
            f"Failed to get schema for table '{table_name}': {e}", exc_info=True
        )
        raise RuntimeError(f"Failed to get schema for table '{table_name}': {e}") from e
