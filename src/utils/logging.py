"""Logging configuration for the trading bot."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Bot identification
BOT_NAME = "PAIRS-COI"  # Short identifier for log prefix
BOT_STRATEGY = "Cointegration Pairs Trading"
BOT_VERSION = "1.0.0"


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

    # Console handler with Eastern Time and bot identifier
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            f"<blue>[{BOT_NAME}]</blue> "
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )

    # Daily log file - rotates at midnight
    logger.add(
        log_path / f"{BOT_NAME.lower()}_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        format=f"[{BOT_NAME}] {{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {{name}}:{{function}}:{{line}} | {{message}}",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
    )

    # Trade-specific log (daily)
    logger.add(
        log_path / f"{BOT_NAME.lower()}_trades_{{time:YYYY-MM-DD}}.log",
        level="INFO",
        filter=lambda record: "TRADE" in record["message"],
        format=f"[{BOT_NAME}] {{time:YYYY-MM-DD HH:mm:ss}} | {{message}}",
        rotation="00:00",
        retention="90 days",
    )

    # Error log (persistent)
    logger.add(
        log_path / f"{BOT_NAME.lower()}_errors.log",
        level="ERROR",
        format=f"[{BOT_NAME}] {{time:YYYY-MM-DD HH:mm:ss}} | {{level: <8}} | {{name}}:{{function}}:{{line}} | {{message}}",
        rotation="10 MB",
        retention="90 days",
    )

    logger.info(f"Logging initialized. Level: {log_level}, Directory: {log_path}")
