"""
Centralized logging configuration for Shutterstock AI Metadata Generator.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ShutterstockAI", level: int = logging.DEBUG, log_file: Optional[Path] = None, console: bool = True
) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name
        level: Logging level (default: DEBUG)
        log_file: Optional path to log file
        console: Whether to log to console (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ShutterstockAI") -> logging.Logger:
    """
    Get an existing logger or create one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


# Create default application logger
app_logger = get_logger("ShutterstockAI")
