"""Cointegration testing for pairs trading."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class CointegrationResult:
    """Result of cointegration test."""

    pair: Tuple[str, str]
    is_cointegrated: bool
    p_value: float
    adf_statistic: float
    critical_values: dict
    beta: float
    half_life: float
    hurst_exponent: float
    spread_mean: float
    spread_std: float
    lookback_days: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "is_cointegrated": self.is_cointegrated,
            "p_value": self.p_value,
            "adf_statistic": self.adf_statistic,
            "critical_values": self.critical_values,
            "beta": self.beta,
            "half_life": self.half_life,
            "hurst_exponent": self.hurst_exponent,
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std,
            "lookback_days": self.lookback_days,
        }


class CointegrationTester:
    """Test pairs for cointegration relationship."""

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        min_half_life: float = 5,
        max_half_life: float = 60,
    ):
        """
        Initialize cointegration tester.

        Args:
            p_value_threshold: Maximum p-value for cointegration
            min_half_life: Minimum half-life in days
            max_half_life: Maximum half-life in days
        """
        self.p_value_threshold = p_value_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def calculate_beta(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> float:
        """
        Calculate hedge ratio (beta) using OLS regression.

        Args:
            prices1: Price series for asset 1 (dependent)
            prices2: Price series for asset 2 (independent)

        Returns:
            Hedge ratio (beta)
        """
        # Add constant for OLS
        X = prices2.values.reshape(-1, 1)
        X = np.column_stack([np.ones(len(X)), X])
        y = prices1.values

        # OLS regression
        model = OLS(y, X)
        result = model.fit()

        return result.params[1]  # Return slope (beta)

    def calculate_spread(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        beta: Optional[float] = None,
    ) -> pd.Series:
        """
        Calculate spread between two price series.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            beta: Hedge ratio (calculated if not provided)

        Returns:
            Spread series
        """
        if beta is None:
            beta = self.calculate_beta(prices1, prices2)

        spread = prices1 - beta * prices2
        return spread

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck model.

        Uses: spread_diff = theta * (spread_mean - spread_lag) + noise
        Half-life = -ln(2) / ln(1 + theta)

        Args:
            spread: Spread series

        Returns:
            Half-life in periods (days)
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align series
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]

        if len(spread_lag) < 20:
            return np.inf

        # OLS regression: spread_diff = theta * spread_lag
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values

        model = OLS(y, X)
        result = model.fit()
        theta = result.params[0]

        if theta >= 0:
            # No mean reversion
            return np.inf

        half_life = -np.log(2) / theta
        return max(0, half_life)

    def calculate_hurst_exponent(
        self,
        series: pd.Series,
        max_lag: int = 100,
    ) -> float:
        """
        Calculate Hurst exponent.

        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        Args:
            series: Time series
            max_lag: Maximum lag for calculation

        Returns:
            Hurst exponent
        """
        lags = range(2, min(max_lag, len(series) // 4))
        tau = []

        for lag in lags:
            diff = series.values[lag:] - series.values[:-lag]
            tau.append(np.std(diff))

        if len(tau) < 2:
            return 0.5

        # Linear regression on log-log plot
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        reg = np.polyfit(log_lags, log_tau, 1)
        hurst = reg[0] / 2

        return np.clip(hurst, 0, 1)

    def test_adf(self, spread: pd.Series) -> Tuple[float, float, dict]:
        """
        Run Augmented Dickey-Fuller test on spread.

        Args:
            spread: Spread series

        Returns:
            Tuple of (adf_statistic, p_value, critical_values)
        """
        result = adfuller(spread.dropna(), autolag="AIC")

        adf_stat = result[0]
        p_value = result[1]
        critical_values = {
            "1%": result[4]["1%"],
            "5%": result[4]["5%"],
            "10%": result[4]["10%"],
        }

        return adf_stat, p_value, critical_values

    def test_engle_granger(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
    ) -> Tuple[float, float]:
        """
        Run Engle-Granger cointegration test.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2

        Returns:
            Tuple of (t_statistic, p_value)
        """
        result = coint(prices1, prices2)
        return result[0], result[1]

    def test_pair(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair: Tuple[str, str],
        lookback_days: int = 252,
    ) -> CointegrationResult:
        """
        Perform full cointegration analysis on a pair.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            pair: Tuple of asset symbols
            lookback_days: Number of days used

        Returns:
            CointegrationResult object
        """
        # Calculate beta and spread
        beta = self.calculate_beta(prices1, prices2)
        spread = self.calculate_spread(prices1, prices2, beta)

        # ADF test on spread
        adf_stat, p_value, critical_values = self.test_adf(spread)

        # Calculate half-life
        half_life = self.calculate_half_life(spread)

        # Calculate Hurst exponent
        hurst = self.calculate_hurst_exponent(spread)

        # Determine if cointegrated
        is_cointegrated = (
            p_value < self.p_value_threshold
            and self.min_half_life < half_life < self.max_half_life
        )

        result = CointegrationResult(
            pair=pair,
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            adf_statistic=adf_stat,
            critical_values=critical_values,
            beta=beta,
            half_life=half_life,
            hurst_exponent=hurst,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            lookback_days=lookback_days,
        )

        logger.debug(
            f"Tested {pair}: p={p_value:.4f}, half_life={half_life:.1f}, "
            f"hurst={hurst:.3f}, cointegrated={is_cointegrated}"
        )

        return result

    def test_pair_rolling(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair: Tuple[str, str],
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Perform rolling cointegration test.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            pair: Tuple of asset symbols
            window: Rolling window size

        Returns:
            DataFrame with rolling p-values and betas
        """
        results = []

        for i in range(window, len(prices1)):
            p1_window = prices1.iloc[i - window : i]
            p2_window = prices2.iloc[i - window : i]

            beta = self.calculate_beta(p1_window, p2_window)
            spread = self.calculate_spread(p1_window, p2_window, beta)
            _, p_value, _ = self.test_adf(spread)

            results.append(
                {
                    "date": prices1.index[i],
                    "p_value": p_value,
                    "beta": beta,
                    "is_cointegrated": p_value < self.p_value_threshold,
                }
            )

        return pd.DataFrame(results).set_index("date")
