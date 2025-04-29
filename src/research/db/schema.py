"""
Database schema management for DuckDB.
"""

from research.db.connection import init_db
from research.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_metadata_table(db_path: str) -> bool:
    """
    Create a metadata table to track database information.

    Args:
        db_path: Path to DuckDB database

    Returns:
        Success status
    """
    try:
        # Initialize the database connection
        conn = init_db(db_path)

        # Create the metadata table
        logger.info("Creating metadata table")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _metadata (
                key VARCHAR,
                value VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add creation timestamp if it doesn't exist
        result = conn.execute("""
            SELECT COUNT(*) FROM _metadata WHERE key = 'created_at'
        """).fetchone()[0]

        if result == 0:
            # Add creation timestamp and version
            conn.execute("""
                INSERT INTO _metadata (key, value) VALUES 
                ('created_at', CURRENT_TIMESTAMP),
                ('version', '1.0')
            """)

        conn.close()
        logger.info("Metadata table successfully created or updated")
        return True

    except Exception as e:
        logger.error(f"Failed to create metadata table: {e}", exc_info=True)
        return False


def set_metadata(db_path: str, key: str, value: str) -> bool:
    """
    Set or update a metadata value.

    Args:
        db_path: Path to DuckDB database
        key: Metadata key
        value: Metadata value

    Returns:
        Success status
    """
    try:
        # Initialize the database
        conn = init_db(db_path)

        # Ensure metadata table exists
        create_metadata_table(db_path)

        # Insert or update the metadata value
        conn.execute(
            """
            INSERT INTO _metadata (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (key) DO UPDATE SET 
                value = EXCLUDED.value,
                updated_at = CURRENT_TIMESTAMP
        """,
            [key, value],
        )

        conn.close()
        logger.info(f"Metadata key '{key}' set to '{value}'")
        return True

    except Exception as e:
        logger.error(f"Failed to set metadata: {e}", exc_info=True)
        return False


def get_metadata(db_path: str, key: str) -> str | None:
    """
    Get a metadata value.

    Args:
        db_path: Path to DuckDB database
        key: Metadata key

    Returns:
        Metadata value or None if not found
    """
    try:
        # Initialize the database
        conn = init_db(db_path, read_only=True)

        # Check if metadata table exists
        result = conn.execute("""
            SELECT COUNT(*) FROM sqlite_master 
            WHERE type='table' AND name='_metadata'
        """).fetchone()[0]

        if result == 0:
            conn.close()
            return None

        # Query the metadata value
        result = conn.execute(
            """
            SELECT value FROM _metadata WHERE key = ?
        """,
            [key],
        ).fetchone()

        conn.close()

        if result is None:
            return None

        return result[0]

    except Exception as e:
        logger.error(f"Failed to get metadata: {e}", exc_info=True)
        return None
