"""Z-score calculation for spread analysis."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class ZScoreCalculator:
    """Calculate rolling z-score of spread."""

    def __init__(
        self,
        lookback: int = 20,
        min_periods: Optional[int] = None,
    ):
        """
        Initialize z-score calculator.

        Args:
            lookback: Rolling window size for mean/std calculation
            min_periods: Minimum periods required for calculation
                         (defaults to lookback // 2)
        """
        self.lookback = lookback
        self.min_periods = min_periods or max(1, lookback // 2)
        self._spread_buffer: list[float] = []

    def update(self, spread: float) -> Tuple[float, float, float]:
        """
        Update with new spread value and calculate z-score.

        Args:
            spread: Current spread value

        Returns:
            Tuple of (zscore, mean, std)
        """
        self._spread_buffer.append(spread)

        # Keep only lookback + some extra for safety
        if len(self._spread_buffer) > self.lookback * 2:
            self._spread_buffer = self._spread_buffer[-self.lookback * 2 :]

        if len(self._spread_buffer) < self.min_periods:
            return 0.0, spread, 0.0

        # Use recent values for calculation
        recent = self._spread_buffer[-self.lookback :]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)

        if std < 1e-10:
            return 0.0, mean, std

        zscore = (spread - mean) / std
        return zscore, mean, std

    def calculate(self, spread_series: pd.Series) -> pd.DataFrame:
        """
        Calculate rolling z-score for a spread series.

        Args:
            spread_series: Series of spread values

        Returns:
            DataFrame with zscore, mean, std columns
        """
        rolling_mean = spread_series.rolling(
            window=self.lookback,
            min_periods=self.min_periods,
        ).mean()

        rolling_std = spread_series.rolling(
            window=self.lookback,
            min_periods=self.min_periods,
        ).std()

        zscore = (spread_series - rolling_mean) / rolling_std

        # Handle division by zero
        zscore = zscore.replace([np.inf, -np.inf], np.nan)

        return pd.DataFrame(
            {
                "zscore": zscore,
                "mean": rolling_mean,
                "std": rolling_std,
                "spread": spread_series,
            }
        )

    def get_current_zscore(self) -> float:
        """Get the most recent z-score."""
        if len(self._spread_buffer) < self.min_periods:
            return 0.0

        spread = self._spread_buffer[-1]
        recent = self._spread_buffer[-self.lookback :]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)

        if std < 1e-10:
            return 0.0

        return (spread - mean) / std

    def get_stats(self) -> dict:
        """Get current statistics."""
        if len(self._spread_buffer) < self.min_periods:
            return {
                "zscore": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "n_obs": len(self._spread_buffer),
            }

        recent = self._spread_buffer[-self.lookback :]
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        spread = self._spread_buffer[-1]

        zscore = (spread - mean) / std if std > 1e-10 else 0.0

        return {
            "zscore": zscore,
            "mean": mean,
            "std": std,
            "spread": spread,
            "n_obs": len(self._spread_buffer),
        }

    def reset(self) -> None:
        """Reset calculator state."""
        self._spread_buffer = []


class AdaptiveZScoreCalculator(ZScoreCalculator):
    """
    Z-score calculator with adaptive lookback based on half-life.

    Adjusts the lookback window based on the estimated half-life
    of mean reversion.
    """

    def __init__(
        self,
        base_lookback: int = 20,
        half_life_multiplier: float = 2.0,
        min_lookback: int = 10,
        max_lookback: int = 60,
    ):
        """
        Initialize adaptive z-score calculator.

        Args:
            base_lookback: Base lookback window
            half_life_multiplier: Multiplier for half-life to get lookback
            min_lookback: Minimum allowed lookback
            max_lookback: Maximum allowed lookback
        """
        super().__init__(lookback=base_lookback)
        self.base_lookback = base_lookback
        self.half_life_multiplier = half_life_multiplier
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback

    def set_half_life(self, half_life: float) -> None:
        """
        Adjust lookback based on half-life.

        Args:
            half_life: Estimated half-life in days
        """
        new_lookback = int(half_life * self.half_life_multiplier)
        new_lookback = max(self.min_lookback, min(self.max_lookback, new_lookback))

        if new_lookback != self.lookback:
            logger.debug(
                f"Adjusted z-score lookback from {self.lookback} to {new_lookback}"
            )
            self.lookback = new_lookback
            self.min_periods = max(1, new_lookback // 2)
