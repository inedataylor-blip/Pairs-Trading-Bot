"""Main entry point for the pairs trading bot."""

import argparse
import signal
import sys
import time
from datetime import datetime

import schedule
from loguru import logger

from .bot import PairsTradingBot
from .config import Config
from .utils.logging import setup_logging


class TradingScheduler:
    """Scheduler for the trading bot."""

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

    def setup_schedule(self) -> None:
        """Set up the daily trading schedule."""
        # Weekly pair scan (Sundays at 6 PM)
        schedule.every().sunday.at("18:00").do(self._run_pair_scan)

        # Pre-market preparation
        signal_time = self.config.get("schedule.signal_time", "09:25")
        schedule.every().monday.at(signal_time).do(self._run_signal_generation)
        schedule.every().tuesday.at(signal_time).do(self._run_signal_generation)
        schedule.every().wednesday.at(signal_time).do(self._run_signal_generation)
        schedule.every().thursday.at(signal_time).do(self._run_signal_generation)
        schedule.every().friday.at(signal_time).do(self._run_signal_generation)

        # Trading execution
        exec_time = self.config.get("schedule.execution_time", "09:35")
        schedule.every().monday.at(exec_time).do(self._run_daily_cycle)
        schedule.every().tuesday.at(exec_time).do(self._run_daily_cycle)
        schedule.every().wednesday.at(exec_time).do(self._run_daily_cycle)
        schedule.every().thursday.at(exec_time).do(self._run_daily_cycle)
        schedule.every().friday.at(exec_time).do(self._run_daily_cycle)

        # Midday check
        midday_time = self.config.get("schedule.midday_check", "12:00")
        schedule.every().monday.at(midday_time).do(self._run_midday_check)
        schedule.every().tuesday.at(midday_time).do(self._run_midday_check)
        schedule.every().wednesday.at(midday_time).do(self._run_midday_check)
        schedule.every().thursday.at(midday_time).do(self._run_midday_check)
        schedule.every().friday.at(midday_time).do(self._run_midday_check)

        # End of day
        close_time = self.config.get("schedule.close_check", "15:50")
        schedule.every().monday.at(close_time).do(self._run_end_of_day)
        schedule.every().tuesday.at(close_time).do(self._run_end_of_day)
        schedule.every().wednesday.at(close_time).do(self._run_end_of_day)
        schedule.every().thursday.at(close_time).do(self._run_end_of_day)
        schedule.every().friday.at(close_time).do(self._run_end_of_day)

        logger.info("Trading schedule configured")

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

    def run(self) -> None:
        """Run the scheduler loop."""
        self.running = True
        logger.info("Starting scheduler...")

        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

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

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Load configuration
    config = Config(args.config)

    logger.info("=" * 60)
    logger.info("Cointegration Pairs Trading Bot")
    logger.info("=" * 60)

    # Initialize bot
    bot = PairsTradingBot(config=config, dry_run=args.dry_run)

    # Handle specific modes
    if args.status:
        status = bot.get_status()
        print("\nBot Status:")
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
        print("  from src.backtest import BacktestEngine")
        print("  engine = BacktestEngine()")
        print("  result = engine.run(prices1, prices2, pair)")
        return

    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal...")
        scheduler.stop()
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initial pair scan
    logger.info("Running initial pair scan...")
    bot.scan_pairs()

    # Start scheduler
    scheduler = TradingScheduler(bot, config)
    scheduler.setup_schedule()

    try:
        scheduler.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
