"""Main trading bot orchestration."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .config import Config
from .data import AlpacaClient, Database, PriceCache, StateStore
from .execution import OrderRouter, PositionManager, RiskManager
from .pair_discovery import CointegrationTester, PairSelector, PairUniverse
from .signal_generation import (
    KalmanFilterHedgeRatio,
    SignalGenerator,
    SignalType,
    TradingSignal,
    ZScoreCalculator,
)


class PairsTradingBot:
    """
    Main pairs trading bot orchestration class.

    Coordinates all components for pair discovery, signal generation,
    and trade execution.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        dry_run: bool = False,
    ):
        """
        Initialize the pairs trading bot.

        Args:
            config: Configuration object (uses default if not provided)
            dry_run: If True, simulate trades without executing
        """
        self.config = config or Config()
        self.dry_run = dry_run

        # Initialize data layer
        self.alpaca = AlpacaClient(
            api_key=self.config.alpaca_api_key,
            secret_key=self.config.alpaca_secret_key,
            paper=self.config.is_paper_trading,
        )
        self.cache = PriceCache()
        self.state_store = StateStore()
        self.database = Database(self.config.get("database.path", "data/trading.db"))

        # Initialize pair discovery
        self.universe = PairUniverse()
        self.coint_tester = CointegrationTester(
            p_value_threshold=self.config.get("cointegration.p_value_threshold", 0.05),
            min_half_life=self.config.get("cointegration.min_half_life", 5),
            max_half_life=self.config.get("cointegration.max_half_life", 60),
        )
        self.pair_selector = PairSelector(
            cointegration_tester=self.coint_tester,
            max_pairs=self.config.get("trading.max_pairs", 3),
        )

        # Initialize signal generation (per-pair)
        self.kalman_filters: Dict[Tuple[str, str], KalmanFilterHedgeRatio] = {}
        self.zscore_calculators: Dict[Tuple[str, str], ZScoreCalculator] = {}
        self.signal_generator = SignalGenerator(
            entry_threshold=self.config.get("trading.entry_zscore", 2.0),
            exit_threshold=self.config.get("trading.exit_zscore", 0.5),
            stop_threshold=self.config.get("trading.stop_zscore", 3.5),
        )

        # Initialize execution
        self.position_manager = PositionManager()
        self.order_router = OrderRouter(self.alpaca, dry_run=dry_run)
        self.risk_manager = RiskManager(
            max_pairs=self.config.get("trading.max_pairs", 3),
            max_pair_exposure=self.config.get("risk.max_pair_exposure", 0.40),
            max_gross_exposure=self.config.get("risk.max_gross_exposure", 1.50),
            max_daily_loss=self.config.get("risk.max_daily_loss", 0.05),
            max_trade_loss=self.config.get("risk.max_trade_loss", 0.03),
        )

        # Active trading pairs
        self.active_pairs: List[Tuple[str, str]] = []

        logger.info(
            f"PairsTradingBot initialized (dry_run={dry_run}, "
            f"paper={self.config.is_paper_trading})"
        )

    def _get_kalman(self, pair: Tuple[str, str]) -> KalmanFilterHedgeRatio:
        """Get or create Kalman filter for a pair."""
        if pair not in self.kalman_filters:
            self.kalman_filters[pair] = KalmanFilterHedgeRatio(
                delta=self.config.get("kalman.delta", 0.0001)
            )
        return self.kalman_filters[pair]

    def _get_zscore_calc(self, pair: Tuple[str, str]) -> ZScoreCalculator:
        """Get or create z-score calculator for a pair."""
        if pair not in self.zscore_calculators:
            self.zscore_calculators[pair] = ZScoreCalculator(
                lookback=self.config.get("kalman.lookback_zscore", 20)
            )
        return self.zscore_calculators[pair]

    def scan_pairs(self) -> List[Tuple[str, str]]:
        """
        Scan universe for cointegrated pairs.

        Returns:
            List of cointegrated pairs ranked by quality
        """
        logger.info(f"Scanning {len(self.universe)} pairs for cointegration...")

        results = []
        lookback = self.config.get("cointegration.lookback_days", 252)

        for pair in self.universe:
            try:
                prices1, prices2 = self.alpaca.get_prices_for_pair(
                    pair[0], pair[1], days=lookback
                )

                result = self.coint_tester.test_pair(
                    prices1, prices2, pair, lookback
                )
                results.append(result)

                # Log diagnostic info for why pair passed/failed
                status = "PASS" if result.is_cointegrated else "FAIL"
                p_status = "✓" if result.p_value < 0.05 else "✗"
                hl_status = "✓" if 5 < result.half_life < 60 else "✗"
                logger.info(
                    f"  {pair[0]}/{pair[1]}: {status} | "
                    f"p={result.p_value:.4f} {p_status} | "
                    f"half_life={result.half_life:.1f}d {hl_status} | "
                    f"hurst={result.hurst_exponent:.3f}"
                )

                # Store result in database
                self.database.record_cointegration_result(
                    pair=pair,
                    p_value=result.p_value,
                    adf_statistic=result.adf_statistic,
                    half_life=result.half_life,
                    beta=result.beta,
                    is_cointegrated=result.is_cointegrated,
                    lookback_days=lookback,
                )

            except Exception as e:
                logger.warning(f"Failed to test {pair}: {e}")

        # Select top pairs
        selected = self.pair_selector.select_top_pairs(results)
        self.active_pairs = [r.pair for r in selected]

        logger.info(f"Selected {len(self.active_pairs)} pairs: {self.active_pairs}")

        return self.active_pairs

    def update_pair_signals(
        self,
        pair: Tuple[str, str],
    ) -> Optional[TradingSignal]:
        """
        Update signals for a single pair.

        Args:
            pair: Tuple of (asset1, asset2)

        Returns:
            TradingSignal if action needed, None otherwise
        """
        try:
            # Get current prices
            price1 = self.alpaca.get_latest_price(pair[0])
            price2 = self.alpaca.get_latest_price(pair[1])

            if price1 is None or price2 is None:
                logger.warning(f"Could not get prices for {pair}")
                return None

            # Update Kalman filter
            kalman = self._get_kalman(pair)
            hedge_ratio, spread, _ = kalman.update(price2, price1)

            # Update z-score
            zscore_calc = self._get_zscore_calc(pair)
            zscore, _, _ = zscore_calc.update(spread)

            # Generate signal
            signal = self.signal_generator.generate_signal(
                pair=pair,
                zscore=zscore,
                hedge_ratio=hedge_ratio,
                spread=spread,
            )

            # Update position tracking
            if self.position_manager.has_position(pair):
                self.position_manager.update_position(
                    pair, price1, price2, spread, zscore
                )

            return signal

        except Exception as e:
            logger.error(f"Error updating signals for {pair}: {e}")
            return None

    def execute_signal(
        self,
        signal: TradingSignal,
    ) -> bool:
        """
        Execute a trading signal.

        Args:
            signal: TradingSignal to execute

        Returns:
            True if executed successfully
        """
        if signal.signal_type == SignalType.HOLD:
            return True

        pair = signal.pair
        price1 = self.alpaca.get_latest_price(pair[0])
        price2 = self.alpaca.get_latest_price(pair[1])

        if price1 is None or price2 is None:
            logger.error(f"Cannot execute: prices unavailable for {pair}")
            return False

        # Get account info
        account = self.alpaca.get_account()
        account_value = account["equity"]

        # Calculate position sizes
        if signal.signal_type in (SignalType.LONG_SPREAD, SignalType.SHORT_SPREAD):
            # Calculate spread volatility (simplified)
            spread_vol = 0.10  # Default 10% annualized

            shares1, shares2 = self.order_router.calculate_position_sizes(
                pair=pair,
                hedge_ratio=signal.hedge_ratio,
                account_value=account_value,
                risk_per_trade=self.config.get("trading.risk_per_trade", 0.02),
                spread_volatility=spread_vol,
                price1=price1,
                price2=price2,
            )

            # Calculate proposed trade value
            trade_value = (shares1 * price1 + shares2 * price2)

            # Run risk checks
            risk_check = self.risk_manager.pre_trade_check(
                signal=signal,
                proposed_value=trade_value,
                current_positions=self.position_manager.num_positions,
                current_gross_exposure=self.position_manager.total_market_value,
                account_value=account_value,
                buying_power=account["buying_power"],
                current_equity=account["equity"],
            )

            if not risk_check.passed:
                logger.warning(f"Risk check failed: {risk_check.reasons}")
                return False

            # Adjust shares for direction
            if signal.signal_type == SignalType.LONG_SPREAD:
                shares2 = -shares2  # Short asset2
            else:  # SHORT_SPREAD
                shares1 = -shares1  # Short asset1

            # Execute trade
            result = self.order_router.execute_spread_trade(
                signal=signal,
                shares_asset1=abs(shares1),
                shares_asset2=abs(shares2),
            )

            if result.success:
                # Update position manager
                direction = 1 if signal.signal_type == SignalType.LONG_SPREAD else -1
                self.position_manager.open_position(
                    pair=pair,
                    direction=direction,
                    shares_asset1=shares1,
                    shares_asset2=shares2,
                    price1=price1,
                    price2=price2,
                    hedge_ratio=signal.hedge_ratio,
                    zscore=signal.zscore,
                    spread=signal.spread,
                )

                # Update signal generator state
                self.signal_generator.process_signal(signal)

                # Record trade
                self.database.record_trade(
                    pair=pair,
                    signal=signal.signal_type.value,
                    zscore=signal.zscore,
                    hedge_ratio=signal.hedge_ratio,
                    shares=(shares1, shares2),
                    prices=(price1, price2),
                    spread_value=signal.spread,
                    account_value=account_value,
                )

                logger.info(f"TRADE: Executed {signal.signal_type.value} for {pair}")
                return True

        elif signal.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
            # Close position
            result = self.order_router.execute_spread_trade(
                signal=signal,
                shares_asset1=0,
                shares_asset2=0,
            )

            if result.success:
                closed_info = self.position_manager.close_position(
                    pair, price1, price2
                )

                # Update signal generator state
                self.signal_generator.process_signal(signal)

                if closed_info:
                    # Record trade
                    self.database.record_trade(
                        pair=pair,
                        signal=signal.signal_type.value,
                        zscore=signal.zscore,
                        hedge_ratio=signal.hedge_ratio,
                        shares=(0, 0),
                        prices=(price1, price2),
                        spread_value=signal.spread,
                        account_value=account_value,
                        notes=f"P&L: {closed_info['realized_pnl']:.2f}",
                    )

                logger.info(f"TRADE: Closed position for {pair}")
                return True

        return False

    def run_daily_cycle(self) -> None:
        """Run a complete daily trading cycle."""
        logger.info("Starting daily trading cycle...")

        # Get account info
        account = self.alpaca.get_account()
        self.position_manager.set_equity(account["equity"], is_starting=True)
        self.risk_manager.set_daily_start(account["equity"])

        logger.info(f"Account equity: ${account['equity']:,.2f}")

        # Check if market is open
        if not self.alpaca.is_market_open():
            logger.info("Market is closed. Skipping trading cycle.")
            return

        # Update signals and execute trades for active pairs
        for pair in self.active_pairs:
            signal = self.update_pair_signals(pair)

            if signal and signal.signal_type != SignalType.HOLD:
                self.execute_signal(signal)

        # Log position summary
        summary = self.position_manager.get_summary()
        logger.info(
            f"Positions: {summary['num_positions']}, "
            f"Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}"
        )

    def run_midday_check(self) -> None:
        """Run midday position check and stop loss monitoring."""
        logger.info("Running midday check...")

        for pair in self.active_pairs:
            if self.position_manager.has_position(pair):
                signal = self.update_pair_signals(pair)

                # Check for stop loss
                if signal and signal.signal_type == SignalType.STOP_LOSS:
                    logger.warning(f"Stop loss triggered for {pair}")
                    self.execute_signal(signal)

        # Update account equity
        account = self.alpaca.get_account()
        self.position_manager.set_equity(account["equity"])

    def run_end_of_day(self) -> None:
        """Run end of day processing."""
        logger.info("Running end of day processing...")

        # Get final account state
        account = self.alpaca.get_account()
        self.position_manager.set_equity(account["equity"])

        summary = self.position_manager.get_summary()

        # Record daily P&L
        today = datetime.now().strftime("%Y-%m-%d")
        self.database.record_daily_pnl(
            date=today,
            starting_equity=summary["starting_equity"] or account["equity"],
            ending_equity=account["equity"],
            realized_pnl=summary["total_realized_pnl"],
            unrealized_pnl=summary["total_unrealized_pnl"],
            num_trades=summary["num_closed_trades"],
        )

        # Save state
        self.state_store.save_signals(self.signal_generator.get_all_positions())

        # Log summary
        daily_pnl = summary["daily_pnl"]
        logger.info(
            f"End of day summary:\n"
            f"  Starting equity: ${summary['starting_equity']:,.2f}\n"
            f"  Ending equity: ${account['equity']:,.2f}\n"
            f"  Daily P&L: ${daily_pnl:,.2f}\n"
            f"  Trades: {summary['num_closed_trades']}"
        )

        # Reset daily tracking
        self.position_manager.reset_daily()

    def shutdown(self) -> None:
        """Clean shutdown of the bot."""
        logger.info("Shutting down pairs trading bot...")

        # Save all state
        kalman_states = {}
        for pair, kf in self.kalman_filters.items():
            state = kf.get_state()
            if state:
                kalman_states[str(pair)] = state.to_dict()

        self.state_store.save_pickle("kalman_states", kalman_states)
        self.state_store.save_signals(self.signal_generator.get_all_positions())
        self.state_store.save_positions(
            [p.to_dict() for p in self.position_manager.get_all_positions().values()]
        )

        logger.info("Bot shutdown complete")

    def get_status(self) -> dict:
        """Get current bot status."""
        account = self.alpaca.get_account()

        return {
            "account": account,
            "active_pairs": self.active_pairs,
            "positions": self.position_manager.get_summary(),
            "risk": self.risk_manager.get_status(),
            "market_open": self.alpaca.is_market_open(),
        }
