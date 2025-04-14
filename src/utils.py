import logging
import sys


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup and configure a logger.

    Args:
        name: The name of the logger
        log_level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        A configured logger instance
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure logger
    logger = logging.getLogger(name)

    # If logger already has handlers, just set the level and return it
    # This prevents duplicate handlers being added
    if logger.hasHandlers():
        logger.setLevel(numeric_level)
        return logger

    # Clear any existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(numeric_level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger
