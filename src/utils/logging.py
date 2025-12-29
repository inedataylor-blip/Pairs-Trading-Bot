"""Logging configuration for the trading bot."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/trading_bot.log",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # Create log directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # File handler
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Trade-specific log
    logger.add(
        log_path.parent / "trades.log",
        level="INFO",
        filter=lambda record: "TRADE" in record["message"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="1 day",
        retention="1 month",
    )

    logger.info(f"Logging initialized. Level: {log_level}, File: {log_file}")
