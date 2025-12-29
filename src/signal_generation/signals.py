"""Trading signal generation for pairs trading."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

from loguru import logger


class SignalType(Enum):
    """Types of trading signals."""

    LONG_SPREAD = "LONG_SPREAD"  # Buy asset1, sell asset2
    SHORT_SPREAD = "SHORT_SPREAD"  # Sell asset1, buy asset2
    EXIT = "EXIT"  # Close position
    STOP_LOSS = "STOP_LOSS"  # Stop loss triggered
    HOLD = "HOLD"  # No action


@dataclass
class TradingSignal:
    """Trading signal with metadata."""

    signal_type: SignalType
    pair: Tuple[str, str]
    zscore: float
    hedge_ratio: float
    spread: float
    timestamp: datetime
    reason: str
    current_position: int  # -1, 0, or 1

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Signal({self.signal_type.value}, {self.pair}, "
            f"z={self.zscore:.2f}, pos={self.current_position})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "pair": self.pair,
            "zscore": self.zscore,
            "hedge_ratio": self.hedge_ratio,
            "spread": self.spread,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "current_position": self.current_position,
        }


class SignalGenerator:
    """Generate trading signals based on z-score thresholds."""

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_threshold: float = 3.5,
        min_holding_periods: int = 0,
    ):
        """
        Initialize signal generator.

        Args:
            entry_threshold: Z-score threshold for entry (abs value)
            exit_threshold: Z-score threshold for exit (abs value)
            stop_threshold: Z-score threshold for stop loss (abs value)
            min_holding_periods: Minimum periods to hold before exit
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_threshold = stop_threshold
        self.min_holding_periods = min_holding_periods

        # Track position and holding time per pair
        self._positions: dict[Tuple[str, str], int] = {}
        self._holding_periods: dict[Tuple[str, str], int] = {}

    def get_position(self, pair: Tuple[str, str]) -> int:
        """
        Get current position for a pair.

        Returns:
            -1 for short spread, 0 for no position, 1 for long spread
        """
        return self._positions.get(pair, 0)

    def set_position(self, pair: Tuple[str, str], position: int) -> None:
        """Set position for a pair."""
        self._positions[pair] = position
        if position != 0:
            self._holding_periods[pair] = 0
        else:
            self._holding_periods.pop(pair, None)

    def generate_signal(
        self,
        pair: Tuple[str, str],
        zscore: float,
        hedge_ratio: float,
        spread: float,
        timestamp: Optional[datetime] = None,
    ) -> TradingSignal:
        """
        Generate trading signal based on current z-score.

        Args:
            pair: Tuple of (asset1, asset2)
            zscore: Current z-score
            hedge_ratio: Current hedge ratio
            spread: Current spread value
            timestamp: Signal timestamp (defaults to now)

        Returns:
            TradingSignal object
        """
        timestamp = timestamp or datetime.now()
        position = self.get_position(pair)

        signal_type = SignalType.HOLD
        reason = "No signal"

        # Update holding period
        if position != 0:
            self._holding_periods[pair] = self._holding_periods.get(pair, 0) + 1

        if position == 0:
            # No position - look for entry signals
            if zscore < -self.entry_threshold:
                signal_type = SignalType.LONG_SPREAD
                reason = f"Z-score {zscore:.2f} < -{self.entry_threshold}"
            elif zscore > self.entry_threshold:
                signal_type = SignalType.SHORT_SPREAD
                reason = f"Z-score {zscore:.2f} > {self.entry_threshold}"

        elif position > 0:
            # Long spread position
            holding_time = self._holding_periods.get(pair, 0)

            if zscore < -self.stop_threshold:
                signal_type = SignalType.STOP_LOSS
                reason = f"Stop loss: Z-score {zscore:.2f} < -{self.stop_threshold}"
            elif zscore > -self.exit_threshold and holding_time >= self.min_holding_periods:
                signal_type = SignalType.EXIT
                reason = f"Mean reversion: Z-score {zscore:.2f} > -{self.exit_threshold}"

        elif position < 0:
            # Short spread position
            holding_time = self._holding_periods.get(pair, 0)

            if zscore > self.stop_threshold:
                signal_type = SignalType.STOP_LOSS
                reason = f"Stop loss: Z-score {zscore:.2f} > {self.stop_threshold}"
            elif zscore < self.exit_threshold and holding_time >= self.min_holding_periods:
                signal_type = SignalType.EXIT
                reason = f"Mean reversion: Z-score {zscore:.2f} < {self.exit_threshold}"

        signal = TradingSignal(
            signal_type=signal_type,
            pair=pair,
            zscore=zscore,
            hedge_ratio=hedge_ratio,
            spread=spread,
            timestamp=timestamp,
            reason=reason,
            current_position=position,
        )

        if signal_type != SignalType.HOLD:
            logger.info(f"Generated signal: {signal}")

        return signal

    def process_signal(self, signal: TradingSignal) -> None:
        """
        Update internal state after signal is acted upon.

        Args:
            signal: The signal that was executed
        """
        pair = signal.pair

        if signal.signal_type == SignalType.LONG_SPREAD:
            self.set_position(pair, 1)
        elif signal.signal_type == SignalType.SHORT_SPREAD:
            self.set_position(pair, -1)
        elif signal.signal_type in (SignalType.EXIT, SignalType.STOP_LOSS):
            self.set_position(pair, 0)

    def get_all_positions(self) -> dict[Tuple[str, str], int]:
        """Get all current positions."""
        return self._positions.copy()

    def reset(self) -> None:
        """Reset all positions and state."""
        self._positions.clear()
        self._holding_periods.clear()


class RegimeFilter:
    """
    Filter signals based on market regime.

    Reduces exposure during trending markets where
    mean reversion strategies tend to underperform.
    """

    def __init__(
        self,
        volatility_threshold: float = 1.5,
        trend_threshold: float = 0.6,
    ):
        """
        Initialize regime filter.

        Args:
            volatility_threshold: Multiplier above average volatility
            trend_threshold: Hurst exponent threshold for trending
        """
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self._regime: str = "normal"

    def update_regime(
        self,
        current_volatility: float,
        avg_volatility: float,
        hurst_exponent: float,
    ) -> str:
        """
        Update and return current market regime.

        Args:
            current_volatility: Current volatility measure
            avg_volatility: Historical average volatility
            hurst_exponent: Current Hurst exponent

        Returns:
            Regime string: 'normal', 'high_volatility', or 'trending'
        """
        if current_volatility > avg_volatility * self.volatility_threshold:
            self._regime = "high_volatility"
        elif hurst_exponent > self.trend_threshold:
            self._regime = "trending"
        else:
            self._regime = "normal"

        return self._regime

    def should_trade(self) -> bool:
        """Check if trading is recommended in current regime."""
        return self._regime == "normal"

    def get_position_scalar(self) -> float:
        """
        Get position size multiplier based on regime.

        Returns:
            Scalar between 0 and 1 for position sizing
        """
        if self._regime == "normal":
            return 1.0
        elif self._regime == "high_volatility":
            return 0.5
        else:  # trending
            return 0.25

    @property
    def regime(self) -> str:
        """Get current regime."""
        return self._regime
