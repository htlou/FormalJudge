"""
Logging configuration for iterative refinement pipeline.

Provides structured logging with file and console handlers.
"""
import logging
import os
from datetime import datetime


def setup_logger(
    output_dir: str,
    log_level: str = "INFO",
    log_filename: str = "pipeline.log"
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Args:
        output_dir: Directory where log file will be saved
        log_level: Logging level for console handler (DEBUG, INFO, WARNING, ERROR)
        log_filename: Name of the log file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("iterative_refinement")
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )

    # File handler - DEBUG level (captures everything)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - configurable level
    console_handler = logging.StreamHandler()
    console_level = getattr(logging, log_level.upper(), logging.INFO)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the existing logger instance.

    Returns:
        Logger instance (may not be configured if setup_logger wasn't called)
    """
    return logging.getLogger("iterative_refinement")
