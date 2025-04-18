"""
Database connection management for DuckDB.
"""

from pathlib import Path

import duckdb
from duckdb import DuckDBPyConnection

from research.utils.logging import setup_logger

logger = setup_logger(__name__)


def init_db(db_path: str, read_only: bool = False) -> DuckDBPyConnection:
    """Initialize DuckDB connection.

    Args:
        db_path: Path to the DuckDB database file
        read_only: Whether to open the database in read-only mode

    Returns:
        DuckDB connection

    Raises:
        ValueError: If the database file doesn't exist and read_only is True
        RuntimeError: If database connection fails
    """
    db_file = Path(db_path)

    # For read-only mode, verify the file exists
    if read_only and not db_file.exists():
        raise ValueError(f"Database file not found: {db_path}")

    # Create parent directory if needed and not in read-only mode
    if not read_only:
        db_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Connect to database with appropriate read_only setting
        logger.info(f"Initializing DuckDB at {db_path} (read_only={read_only})")
        conn: DuckDBPyConnection = duckdb.connect(db_path, read_only=read_only)
        logger.debug("DuckDB connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to initialize DuckDB: {e}", exc_info=True)
        raise RuntimeError(f"Database connection failed: {e}") from e
