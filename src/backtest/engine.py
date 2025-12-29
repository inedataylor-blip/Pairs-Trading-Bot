"""Backtest engine for pairs trading strategies."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..pair_discovery.cointegration import CointegrationTester
from ..signal_generation.kalman_filter import KalmanFilterHedgeRatio
from ..signal_generation.signals import SignalGenerator, SignalType
from ..signal_generation.zscore import ZScoreCalculator


@dataclass
class BacktestTrade:
    """Record of a backtest trade."""

    pair: Tuple[str, str]
    entry_date: datetime
    exit_date: Optional[datetime]
    direction: int  # 1 for long spread, -1 for short spread
    entry_zscore: float
    exit_zscore: Optional[float]
    entry_spread: float
    exit_spread: Optional[float]
    entry_hedge_ratio: float
    shares_asset1: int
    shares_asset2: int
    entry_price1: float
    entry_price2: float
    exit_price1: Optional[float] = None
    exit_price2: Optional[float] = None
    pnl: float = 0.0
    return_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""

    def close(
        self,
        exit_date: datetime,
        exit_price1: float,
        exit_price2: float,
        exit_zscore: float,
        exit_spread: float,
        exit_reason: str,
    ) -> None:
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price1 = exit_price1
        self.exit_price2 = exit_price2
        self.exit_zscore = exit_zscore
        self.exit_spread = exit_spread
        self.exit_reason = exit_reason

        # Calculate P&L
        pnl1 = self.shares_asset1 * (exit_price1 - self.entry_price1)
        pnl2 = self.shares_asset2 * (exit_price2 - self.entry_price2)
        self.pnl = pnl1 + pnl2

        # Calculate return
        entry_value = abs(self.shares_asset1 * self.entry_price1) + abs(
            self.shares_asset2 * self.entry_price2
        )
        self.return_pct = self.pnl / entry_value if entry_value > 0 else 0

        # Holding period
        self.holding_days = (exit_date - self.entry_date).days

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "direction": self.direction,
            "entry_zscore": self.entry_zscore,
            "exit_zscore": self.exit_zscore,
            "entry_spread": self.entry_spread,
            "exit_spread": self.exit_spread,
            "entry_hedge_ratio": self.entry_hedge_ratio,
            "shares": {
                "asset1": self.shares_asset1,
                "asset2": self.shares_asset2,
            },
            "entry_prices": {
                "asset1": self.entry_price1,
                "asset2": self.entry_price2,
            },
            "exit_prices": {
                "asset1": self.exit_price1,
                "asset2": self.exit_price2,
            },
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    pair: Tuple[str, str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    signals_df: pd.DataFrame
    parameters: dict

    @property
    def total_return(self) -> float:
        """Total return as decimal."""
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_trades(self) -> int:
        """Total number of completed trades."""
        return len([t for t in self.trades if t.exit_date is not None])

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_trades": self.total_trades,
            "parameters": self.parameters,
        }


class BacktestEngine:
    """
    Backtesting engine for pairs trading strategies.

    Simulates trading with historical data using the same
    signal generation logic as live trading.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        position_size_pct: float = 0.20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float = 3.5,
        kalman_delta: float = 0.0001,
        zscore_lookback: int = 20,
        commission_per_share: float = 0.0,
        slippage_pct: float = 0.0005,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            entry_zscore: Entry z-score threshold
            exit_zscore: Exit z-score threshold
            stop_zscore: Stop loss z-score threshold
            kalman_delta: Kalman filter delta parameter
            zscore_lookback: Z-score rolling window
            commission_per_share: Commission per share
            slippage_pct: Slippage as % of price
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        self.kalman_delta = kalman_delta
        self.zscore_lookback = zscore_lookback
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to price."""
        if side == "buy":
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def _calculate_commission(self, shares: int) -> float:
        """Calculate commission for a trade."""
        return abs(shares) * self.commission_per_share

    def run(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair: Tuple[str, str],
    ) -> BacktestResult:
        """
        Run backtest on a pair.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            pair: Tuple of (asset1, asset2)

        Returns:
            BacktestResult object
        """
        logger.info(f"Running backtest for {pair}")

        # Initialize components
        kalman = KalmanFilterHedgeRatio(delta=self.kalman_delta)
        zscore_calc = ZScoreCalculator(lookback=self.zscore_lookback)
        signal_gen = SignalGenerator(
            entry_threshold=self.entry_zscore,
            exit_threshold=self.exit_zscore,
            stop_threshold=self.stop_zscore,
        )

        # State tracking
        capital = self.initial_capital
        position: Optional[BacktestTrade] = None
        trades: List[BacktestTrade] = []
        equity_curve = []
        signals_data = []

        # Process each bar
        for idx, (date, p1) in enumerate(prices1.items()):
            p2 = prices2.iloc[idx]

            # Update Kalman filter
            hedge_ratio, spread, _ = kalman.update(float(p2), float(p1))

            # Update z-score
            zscore, _, _ = zscore_calc.update(spread)

            # Get current position state
            current_position = 0
            if position is not None:
                current_position = position.direction

            # Generate signal
            signal = signal_gen.generate_signal(
                pair=pair,
                zscore=zscore,
                hedge_ratio=hedge_ratio,
                spread=spread,
                timestamp=date,
            )

            # Track signals
            signals_data.append(
                {
                    "date": date,
                    "price1": p1,
                    "price2": p2,
                    "spread": spread,
                    "zscore": zscore,
                    "hedge_ratio": hedge_ratio,
                    "signal": signal.signal_type.value,
                    "position": current_position,
                }
            )

            # Execute trades
            if signal.signal_type == SignalType.LONG_SPREAD and position is None:
                # Open long spread position
                position_value = capital * self.position_size_pct
                shares1 = int(position_value / 2 / p1)
                shares2 = -int(position_value / 2 * hedge_ratio / p2)

                # Apply slippage
                exec_p1 = self._apply_slippage(p1, "buy")
                exec_p2 = self._apply_slippage(p2, "sell")

                # Commission
                commission = self._calculate_commission(
                    shares1
                ) + self._calculate_commission(shares2)
                capital -= commission

                position = BacktestTrade(
                    pair=pair,
                    entry_date=date,
                    exit_date=None,
                    direction=1,
                    entry_zscore=zscore,
                    exit_zscore=None,
                    entry_spread=spread,
                    exit_spread=None,
                    entry_hedge_ratio=hedge_ratio,
                    shares_asset1=shares1,
                    shares_asset2=shares2,
                    entry_price1=exec_p1,
                    entry_price2=exec_p2,
                )
                signal_gen.set_position(pair, 1)

            elif signal.signal_type == SignalType.SHORT_SPREAD and position is None:
                # Open short spread position
                position_value = capital * self.position_size_pct
                shares1 = -int(position_value / 2 / p1)
                shares2 = int(position_value / 2 * hedge_ratio / p2)

                # Apply slippage
                exec_p1 = self._apply_slippage(p1, "sell")
                exec_p2 = self._apply_slippage(p2, "buy")

                # Commission
                commission = self._calculate_commission(
                    shares1
                ) + self._calculate_commission(shares2)
                capital -= commission

                position = BacktestTrade(
                    pair=pair,
                    entry_date=date,
                    exit_date=None,
                    direction=-1,
                    entry_zscore=zscore,
                    exit_zscore=None,
                    entry_spread=spread,
                    exit_spread=None,
                    entry_hedge_ratio=hedge_ratio,
                    shares_asset1=shares1,
                    shares_asset2=shares2,
                    entry_price1=exec_p1,
                    entry_price2=exec_p2,
                )
                signal_gen.set_position(pair, -1)

            elif signal.signal_type in (
                SignalType.EXIT,
                SignalType.STOP_LOSS,
            ) and position is not None:
                # Close position
                # Apply slippage
                if position.direction > 0:
                    # Closing long spread: sell asset1, buy asset2
                    exec_p1 = self._apply_slippage(p1, "sell")
                    exec_p2 = self._apply_slippage(p2, "buy")
                else:
                    # Closing short spread: buy asset1, sell asset2
                    exec_p1 = self._apply_slippage(p1, "buy")
                    exec_p2 = self._apply_slippage(p2, "sell")

                # Commission
                commission = self._calculate_commission(
                    position.shares_asset1
                ) + self._calculate_commission(position.shares_asset2)
                capital -= commission

                position.close(
                    exit_date=date,
                    exit_price1=exec_p1,
                    exit_price2=exec_p2,
                    exit_zscore=zscore,
                    exit_spread=spread,
                    exit_reason=signal.signal_type.value,
                )

                capital += position.pnl
                trades.append(position)
                position = None
                signal_gen.set_position(pair, 0)

            # Calculate current equity (mark-to-market)
            equity = capital
            if position is not None:
                # Add unrealized P&L
                pnl1 = position.shares_asset1 * (p1 - position.entry_price1)
                pnl2 = position.shares_asset2 * (p2 - position.entry_price2)
                equity += pnl1 + pnl2

            equity_curve.append({"date": date, "equity": equity})

        # Close any remaining position at the end
        if position is not None:
            p1 = prices1.iloc[-1]
            p2 = prices2.iloc[-1]
            position.close(
                exit_date=prices1.index[-1],
                exit_price1=p1,
                exit_price2=p2,
                exit_zscore=zscore_calc.get_current_zscore(),
                exit_spread=kalman.calculate_spread(p2, p1),
                exit_reason="END_OF_BACKTEST",
            )
            capital += position.pnl
            trades.append(position)

        # Create result
        equity_df = pd.DataFrame(equity_curve).set_index("date")["equity"]
        signals_df = pd.DataFrame(signals_data).set_index("date")

        result = BacktestResult(
            pair=pair,
            start_date=prices1.index[0],
            end_date=prices1.index[-1],
            initial_capital=self.initial_capital,
            final_capital=capital,
            trades=trades,
            equity_curve=equity_df,
            signals_df=signals_df,
            parameters={
                "entry_zscore": self.entry_zscore,
                "exit_zscore": self.exit_zscore,
                "stop_zscore": self.stop_zscore,
                "kalman_delta": self.kalman_delta,
                "zscore_lookback": self.zscore_lookback,
                "position_size_pct": self.position_size_pct,
            },
        )

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"Return: {result.total_return:.2%}"
        )

        return result

    def run_walk_forward(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair: Tuple[str, str],
        train_window: int = 252,
        test_window: int = 63,
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtest.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            pair: Tuple of (asset1, asset2)
            train_window: Training window in days
            test_window: Test window in days

        Returns:
            List of BacktestResult for each test period
        """
        results = []
        total_len = len(prices1)

        for start in range(
            0, total_len - train_window - test_window + 1, test_window
        ):
            train_end = start + train_window
            test_start = train_end
            test_end = test_start + test_window

            # Use training data to calibrate (optional: optimize parameters)
            # For now, just use fixed parameters

            # Run backtest on test period
            test_p1 = prices1.iloc[test_start:test_end]
            test_p2 = prices2.iloc[test_start:test_end]

            if len(test_p1) > 0:
                result = self.run(test_p1, test_p2, pair)
                results.append(result)

        logger.info(
            f"Walk-forward complete: {len(results)} periods, "
            f"Avg return: {np.mean([r.total_return for r in results]):.2%}"
        )

        return results
