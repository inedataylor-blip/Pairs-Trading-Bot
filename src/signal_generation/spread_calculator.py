"""Spread calculation utilities."""

from typing import Optional, Tuple

import pandas as pd

from .kalman_filter import KalmanFilterHedgeRatio


class SpreadCalculator:
    """Calculate and manage spread for a trading pair."""

    def __init__(
        self,
        pair: Tuple[str, str],
        use_kalman: bool = True,
        kalman_delta: float = 0.0001,
        static_beta: Optional[float] = None,
    ):
        """
        Initialize spread calculator.

        Args:
            pair: Tuple of (asset1, asset2) symbols
            use_kalman: Whether to use Kalman filter for dynamic hedge ratio
            kalman_delta: Delta parameter for Kalman filter
            static_beta: Static hedge ratio to use if not using Kalman
        """
        self.pair = pair
        self.use_kalman = use_kalman
        self.static_beta = static_beta

        if use_kalman:
            self.kalman = KalmanFilterHedgeRatio(delta=kalman_delta)
        else:
            self.kalman = None

        self._spread_history: list[dict] = []

    def update(
        self,
        price1: float,
        price2: float,
    ) -> Tuple[float, float]:
        """
        Update with new prices and calculate spread.

        Args:
            price1: Price of first asset (dependent)
            price2: Price of second asset (independent)

        Returns:
            Tuple of (spread, hedge_ratio)
        """
        if self.use_kalman and self.kalman:
            hedge_ratio, spread, _ = self.kalman.update(price2, price1)
        else:
            hedge_ratio = self.static_beta or 1.0
            spread = price1 - hedge_ratio * price2

        self._spread_history.append(
            {
                "price1": price1,
                "price2": price2,
                "spread": spread,
                "hedge_ratio": hedge_ratio,
            }
        )

        return spread, hedge_ratio

    def calculate_spread(
        self,
        price1: float,
        price2: float,
    ) -> float:
        """
        Calculate spread without updating filter.

        Args:
            price1: Price of first asset
            price2: Price of second asset

        Returns:
            Spread value
        """
        if self.use_kalman and self.kalman:
            return self.kalman.calculate_spread(price2, price1)
        else:
            hedge_ratio = self.static_beta or 1.0
            return price1 - hedge_ratio * price2

    def get_hedge_ratio(self) -> float:
        """Get current hedge ratio."""
        if self.use_kalman and self.kalman:
            return self.kalman.get_beta()
        return self.static_beta or 1.0

    def get_spread_series(self) -> pd.Series:
        """Get historical spread as a Series."""
        if not self._spread_history:
            return pd.Series(dtype=float)

        return pd.Series([h["spread"] for h in self._spread_history])

    def get_history(self) -> pd.DataFrame:
        """Get full history as DataFrame."""
        return pd.DataFrame(self._spread_history)

    def reset(self) -> None:
        """Reset calculator state."""
        if self.kalman:
            self.kalman.reset()
        self._spread_history = []

    def process_batch(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> pd.DataFrame:
        """
        Process a batch of historical prices.

        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset

        Returns:
            DataFrame with spread and hedge ratio history
        """
        results = []

        for (idx, p1), (_, p2) in zip(prices1.items(), prices2.items()):
            spread, hedge_ratio = self.update(float(p1), float(p2))
            results.append(
                {
                    "timestamp": idx,
                    "price1": p1,
                    "price2": p2,
                    "spread": spread,
                    "hedge_ratio": hedge_ratio,
                }
            )

        df = pd.DataFrame(results)
        if not df.empty:
            df.set_index("timestamp", inplace=True)

        return df
