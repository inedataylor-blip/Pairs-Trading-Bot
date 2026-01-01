"""Main entry point for the pairs trading bot."""

import argparse
import signal
import sys
import time
from datetime import datetime

from loguru import logger

from .bot import PairsTradingBot
from .config import Config
from .utils.logging import setup_logging

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Eastern timezone
ET = ZoneInfo("America/New_York")


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def is_weekday() -> bool:
    """Check if today is a weekday (Mon-Fri)."""
    return get_eastern_time().weekday() < 5


def is_sunday() -> bool:
    """Check if today is Sunday."""
    return get_eastern_time().weekday() == 6


class EasternTimeScheduler:
    """Scheduler that runs on Eastern Time."""

    def __init__(self, bot: PairsTradingBot, config: Config):
        """
        Initialize scheduler.

        Args:
            bot: PairsTradingBot instance
            config: Configuration object
        """
        self.bot = bot
        self.config = config
        self.running = False

        # Track last execution to avoid duplicates
        self._last_executed: dict[str, str] = {}

        # Load schedule times from config
        self.signal_time = self.config.get("schedule.signal_time", "09:25")
        self.exec_time = self.config.get("schedule.execution_time", "09:35")
        self.midday_time = self.config.get("schedule.midday_check", "12:00")
        self.close_time = self.config.get("schedule.close_check", "15:50")
        self.pair_scan_time = "18:00"  # Sunday evening

    def _should_run(self, job_name: str, target_time: str) -> bool:
        """Check if a job should run now (Eastern Time)."""
        now = get_eastern_time()
        current_time = now.strftime("%H:%M")
        current_date = now.strftime("%Y-%m-%d")

        # Check if we're at the target time (within the minute)
        if current_time != target_time:
            return False

        # Check if already executed today
        execution_key = f"{job_name}_{current_date}"
        if execution_key in self._last_executed:
            return False

        return True

    def _mark_executed(self, job_name: str) -> None:
        """Mark a job as executed for today."""
        now = get_eastern_time()
        current_date = now.strftime("%Y-%m-%d")
        execution_key = f"{job_name}_{current_date}"
        self._last_executed[execution_key] = now.isoformat()

        # Clean up old entries (keep only last 7 days)
        keys_to_remove = []
        for key in self._last_executed:
            if len(keys_to_remove) > 100:  # Safety limit
                break
            try:
                key_date = key.split("_")[-1]
                if key_date < (now.strftime("%Y-%m-%d")):
                    days_old = (now - datetime.fromisoformat(self._last_executed[key]).replace(tzinfo=ET)).days
                    if days_old > 7:
                        keys_to_remove.append(key)
            except (ValueError, IndexError):
                pass

        for key in keys_to_remove:
            del self._last_executed[key]

    def _run_pair_scan(self) -> None:
        """Run weekly pair scan."""
        try:
            logger.info("Running weekly pair scan...")
            self.bot.scan_pairs()
        except Exception as e:
            logger.error(f"Pair scan failed: {e}")

    def _run_signal_generation(self) -> None:
        """Run pre-market signal generation."""
        try:
            logger.info("Generating signals...")
            for pair in self.bot.active_pairs:
                self.bot.update_pair_signals(pair)
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")

    def _run_daily_cycle(self) -> None:
        """Run daily trading cycle."""
        try:
            self.bot.run_daily_cycle()
        except Exception as e:
            logger.error(f"Daily cycle failed: {e}")

    def _run_midday_check(self) -> None:
        """Run midday check."""
        try:
            self.bot.run_midday_check()
        except Exception as e:
            logger.error(f"Midday check failed: {e}")

    def _run_end_of_day(self) -> None:
        """Run end of day processing."""
        try:
            self.bot.run_end_of_day()
        except Exception as e:
            logger.error(f"End of day processing failed: {e}")

    def check_and_run_jobs(self) -> None:
        """Check all scheduled jobs and run if it's time."""
        # Sunday: pair scan
        if is_sunday() and self._should_run("pair_scan", self.pair_scan_time):
            logger.info("Starting scheduled job: pair_scan")
            self._run_pair_scan()
            self._mark_executed("pair_scan")
            logger.info("Completed scheduled job: pair_scan")

        # Weekdays only
        if is_weekday():
            # Pre-market signal generation
            if self._should_run("signal_gen", self.signal_time):
                logger.info("Starting scheduled job: signal_generation")
                self._run_signal_generation()
                self._mark_executed("signal_gen")
                logger.info("Completed scheduled job: signal_generation")

            # Trading execution
            if self._should_run("daily_cycle", self.exec_time):
                logger.info("Starting scheduled job: daily_cycle")
                self._run_daily_cycle()
                self._mark_executed("daily_cycle")
                logger.info("Completed scheduled job: daily_cycle")

            # Midday check
            if self._should_run("midday_check", self.midday_time):
                logger.info("Starting scheduled job: midday_check")
                self._run_midday_check()
                self._mark_executed("midday_check")
                logger.info("Completed scheduled job: midday_check")

            # End of day
            if self._should_run("end_of_day", self.close_time):
                logger.info("Starting scheduled job: end_of_day")
                self._run_end_of_day()
                self._mark_executed("end_of_day")
                logger.info("Completed scheduled job: end_of_day")

    def run(self) -> None:
        """Run the scheduler loop."""
        self.running = True

        logger.info("=" * 60)
        logger.info("Starting Pairs Trading Bot Scheduler")
        logger.info("=" * 60)
        logger.info(f"All times are in Eastern Time (ET)")
        logger.info(f"Current ET time: {get_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Scheduled jobs:")
        logger.info(f"  - Signal generation: {self.signal_time} ET (Mon-Fri)")
        logger.info(f"  - Trade execution: {self.exec_time} ET (Mon-Fri)")
        logger.info(f"  - Midday check: {self.midday_time} ET (Mon-Fri)")
        logger.info(f"  - End of day: {self.close_time} ET (Mon-Fri)")
        logger.info(f"  - Pair scan: {self.pair_scan_time} ET (Sunday)")
        logger.info("=" * 60)

        last_heartbeat = get_eastern_time()

        while self.running:
            try:
                self.check_and_run_jobs()

                # Hourly heartbeat
                now = get_eastern_time()
                if (now - last_heartbeat).total_seconds() >= 3600:
                    logger.info(f"Scheduler heartbeat - bot is running (ET: {now.strftime('%H:%M')})")
                    last_heartbeat = now

            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            time.sleep(30)  # Check every 30 seconds

    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        logger.info("Scheduler stopped")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cointegration Pairs Trading Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trades)",
    )

    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run pair scan only and exit",
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest mode",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging with daily files
    setup_logging(log_level=args.log_level, log_dir="logs")

    # Load configuration
    config = Config(args.config)

    logger.info("=" * 60)
    logger.info("Cointegration Pairs Trading Bot")
    logger.info(f"Current Eastern Time: {get_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Initialize bot
    bot = PairsTradingBot(config=config, dry_run=args.dry_run)

    # Handle specific modes
    if args.status:
        status = bot.get_status()
        print("\nBot Status:")
        print(f"  Current ET Time: {get_eastern_time().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Market Open: {status['market_open']}")
        print(f"  Account Equity: ${status['account']['equity']:,.2f}")
        print(f"  Active Pairs: {status['active_pairs']}")
        print(f"  Open Positions: {status['positions']['num_positions']}")
        print(f"  Trading Halted: {status['risk']['trading_halted']}")
        return

    if args.scan:
        logger.info("Running pair scan...")
        pairs = bot.scan_pairs()
        print("\nCointegrated Pairs:")
        for i, pair in enumerate(pairs, 1):
            print(f"  {i}. {pair}")
        return

    if args.backtest:
        logger.info("Backtest mode - use backtest module directly")
        print("\nTo run backtest, use:")
        print("  python examples/run_backtest.py")
        return

    # Set up signal handlers
    scheduler = None

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal...")
        if scheduler:
            scheduler.stop()
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initial pair scan
    logger.info("Running initial pair scan...")
    bot.scan_pairs()

    # Start scheduler with Eastern Time
    scheduler = EasternTimeScheduler(bot, config)

    try:
        scheduler.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
