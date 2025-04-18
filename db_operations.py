import logging
from pathlib import Path

import duckdb
from duckdb import DuckDBPyConnection
from pandas import DataFrame


def init_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection."""
    logging.info(f"Initializing DuckDB at {db_path}")
    try:
        # Connect to database
        conn: DuckDBPyConnection = duckdb.connect(db_path)  # type: ignore[no-untyped-call]
        logging.info("DuckDB connection established")
        return conn
    except Exception as e:
        logging.error(f"Failed to initialize DuckDB: {e}", exc_info=True)
        raise


def save_to_duckdb(
    df: DataFrame,
    db_path: str,
    table_name: str,
    replace: bool = False,
) -> bool:
    """Save DataFrame to DuckDB.

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
            logging.info(f"Creating/replacing table '{table_name}'")
            conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df"
            )
        else:
            logging.info(f"Appending to existing table '{table_name}'")
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")

        conn.unregister("temp_df")
        conn.close()

        logging.info(f"Successfully saved {len(df)} rows to table '{table_name}'")
        return True
    except Exception as e:
        logging.error(f"Failed to save to DuckDB: {e}", exc_info=True)
        return False


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    # Direct check without using fetchone to avoid typing issues
    result = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", [table_name]
    ).fetchall()
    # If count > 0, table exists
    return result[0][0] > 0


def query_duckdb(db_path: str, sql_query: str) -> DataFrame:
    """Execute SQL query against DuckDB database.

    Args:
        db_path: Path to DuckDB database
        sql_query: SQL query to execute

    Returns:
        Query result as DataFrame
    """
    try:
        conn = init_db(db_path)
        result = conn.execute(sql_query).df()
        conn.close()
        return result
    except Exception as e:
        logging.error(f"Query failed: {e}", exc_info=True)
        raise
