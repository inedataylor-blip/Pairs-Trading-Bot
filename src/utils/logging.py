"""Logging configuration for the trading bot."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ZoneInfo("America/New_York"))


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
) -> None:
    """
    Set up logging configuration with daily log files.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """
    # Remove default handler
    logger.remove()

    # Create log directory using absolute path relative to project root
    if not Path(log_dir).is_absolute():
        # Get project root (parent of src/)
        project_root = Path(__file__).parent.parent.parent
        log_path = project_root / log_dir
    else:
        log_path = Path(log_dir)

    log_path.mkdir(parents=True, exist_ok=True)

    # Console handler with Eastern Time
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

    # Daily log file - rotates at midnight
    logger.add(
        log_path / "pairs_bot_{time:YYYY-MM-DD}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="zip",
    )

    # Trade-specific log (daily)
    logger.add(
        log_path / "trades_{time:YYYY-MM-DD}.log",
        level="INFO",
        filter=lambda record: "TRADE" in record["message"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="00:00",
        retention="90 days",
    )

    # Error log (persistent)
    logger.add(
        log_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="90 days",
        compression="zip",
    )

    logger.info(f"Logging initialized. Level: {log_level}, Directory: {log_dir}")
