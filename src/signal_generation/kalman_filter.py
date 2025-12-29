"""Kalman Filter for dynamic hedge ratio estimation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class KalmanState:
    """State of the Kalman filter."""

    beta: float  # Current hedge ratio estimate
    intercept: float  # Current intercept estimate
    R: np.ndarray  # State covariance matrix
    Ve: float  # Observation variance estimate
    n_obs: int  # Number of observations processed

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "beta": self.beta,
            "intercept": self.intercept,
            "R": self.R.tolist(),
            "Ve": self.Ve,
            "n_obs": self.n_obs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KalmanState":
        """Create from dictionary."""
        return cls(
            beta=data["beta"],
            intercept=data["intercept"],
            R=np.array(data["R"]),
            Ve=data["Ve"],
            n_obs=data["n_obs"],
        )


class KalmanFilterHedgeRatio:
    """
    Kalman Filter for estimating dynamic hedge ratios.

    State Space Model:
        Observation: y_t = beta_t * x_t + alpha_t + epsilon_t
        State: [beta_t, alpha_t]' = [beta_{t-1}, alpha_{t-1}]' + omega_t

    Where:
        - y_t is the dependent asset price
        - x_t is the independent asset price
        - beta_t is the time-varying hedge ratio
        - alpha_t is the time-varying intercept
        - epsilon_t ~ N(0, Ve) is observation noise
        - omega_t ~ N(0, Vw) is state transition noise
    """

    def __init__(
        self,
        delta: float = 0.0001,
        Ve_init: float = 0.001,
        include_intercept: bool = True,
    ):
        """
        Initialize Kalman Filter.

        Args:
            delta: Transition covariance scaling factor.
                   Higher values allow faster adaptation but more noise.
                   Typical range: 0.00001 to 0.001
            Ve_init: Initial observation variance estimate
            include_intercept: Whether to include intercept in the model
        """
        self.delta = delta
        self.Ve_init = Ve_init
        self.include_intercept = include_intercept
        self.state: Optional[KalmanState] = None

    def _initialize_state(self, x: float, y: float) -> None:
        """Initialize filter state with first observation."""
        n_states = 2 if self.include_intercept else 1

        # Initial state estimates (simple OLS-like initialization)
        if self.include_intercept:
            beta_init = 0.0  # Will be updated quickly
            intercept_init = y  # Start with y as intercept
        else:
            beta_init = y / x if abs(x) > 1e-10 else 0.0
            intercept_init = 0.0

        # Initial state covariance (large uncertainty)
        R_init = np.eye(n_states) * 1.0

        self.state = KalmanState(
            beta=beta_init,
            intercept=intercept_init,
            R=R_init,
            Ve=self.Ve_init,
            n_obs=0,
        )

    def update(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float, float]:
        """
        Update Kalman filter with new observation.

        Args:
            x: Independent variable value (price of asset 2)
            y: Dependent variable value (price of asset 1)

        Returns:
            Tuple of (beta, spread, Ve) where:
                - beta: Updated hedge ratio estimate
                - spread: Prediction error (y - y_hat)
                - Ve: Current observation variance estimate
        """
        if self.state is None:
            self._initialize_state(x, y)
            return self.state.beta, 0.0, self.state.Ve

        # Build observation vector
        if self.include_intercept:
            obs = np.array([x, 1.0])
            theta = np.array([self.state.beta, self.state.intercept])
        else:
            obs = np.array([x])
            theta = np.array([self.state.beta])

        # Prediction step
        # State prediction: theta_pred = theta (random walk)
        # Covariance prediction: R_pred = R + Vw
        Vw = self.delta / (1 - self.delta) * self.state.R
        R_pred = self.state.R + Vw

        # Observation prediction
        y_pred = np.dot(obs, theta)
        error = y - y_pred  # Prediction error (spread)

        # Innovation covariance
        Q = np.dot(obs, np.dot(R_pred, obs)) + self.state.Ve

        # Kalman gain
        K = np.dot(R_pred, obs) / Q

        # Update step
        theta_new = theta + K * error
        R_new = R_pred - np.outer(K, K) * Q

        # Update observation variance using exponential smoothing
        alpha = 0.01  # Smoothing parameter
        Ve_new = (1 - alpha) * self.state.Ve + alpha * error**2

        # Update state
        if self.include_intercept:
            self.state.beta = theta_new[0]
            self.state.intercept = theta_new[1]
        else:
            self.state.beta = theta_new[0]

        self.state.R = R_new
        self.state.Ve = Ve_new
        self.state.n_obs += 1

        return self.state.beta, error, self.state.Ve

    def update_batch(
        self,
        prices_x: pd.Series,
        prices_y: pd.Series,
    ) -> pd.DataFrame:
        """
        Process a batch of price data.

        Args:
            prices_x: Price series for independent asset
            prices_y: Price series for dependent asset

        Returns:
            DataFrame with columns: beta, spread, Ve
        """
        results = []

        for (idx, x), (_, y) in zip(prices_x.items(), prices_y.items()):
            beta, spread, Ve = self.update(float(x), float(y))
            results.append(
                {
                    "timestamp": idx,
                    "beta": beta,
                    "spread": spread,
                    "Ve": Ve,
                }
            )

        df = pd.DataFrame(results)
        if not df.empty:
            df.set_index("timestamp", inplace=True)

        return df

    def get_beta(self) -> float:
        """Get current hedge ratio estimate."""
        return self.state.beta if self.state else 0.0

    def get_intercept(self) -> float:
        """Get current intercept estimate."""
        return self.state.intercept if self.state else 0.0

    def get_state(self) -> Optional[KalmanState]:
        """Get current filter state."""
        return self.state

    def set_state(self, state: KalmanState) -> None:
        """Set filter state (for loading saved state)."""
        self.state = state

    def reset(self) -> None:
        """Reset filter to initial state."""
        self.state = None

    def calculate_spread(self, x: float, y: float) -> float:
        """
        Calculate spread using current hedge ratio.

        Args:
            x: Price of independent asset
            y: Price of dependent asset

        Returns:
            Spread value
        """
        if self.state is None:
            return 0.0

        if self.include_intercept:
            return y - self.state.beta * x - self.state.intercept
        else:
            return y - self.state.beta * x


class KalmanFilterManager:
    """Manager for multiple Kalman filters (one per pair)."""

    def __init__(self, delta: float = 0.0001, include_intercept: bool = True):
        """
        Initialize filter manager.

        Args:
            delta: Transition covariance for all filters
            include_intercept: Whether to include intercept
        """
        self.delta = delta
        self.include_intercept = include_intercept
        self.filters: dict[Tuple[str, str], KalmanFilterHedgeRatio] = {}

    def get_filter(self, pair: Tuple[str, str]) -> KalmanFilterHedgeRatio:
        """Get or create filter for a pair."""
        if pair not in self.filters:
            self.filters[pair] = KalmanFilterHedgeRatio(
                delta=self.delta,
                include_intercept=self.include_intercept,
            )
        return self.filters[pair]

    def update(
        self,
        pair: Tuple[str, str],
        price_x: float,
        price_y: float,
    ) -> Tuple[float, float]:
        """
        Update filter for a pair.

        Args:
            pair: Tuple of asset symbols
            price_x: Price of independent asset (second in pair)
            price_y: Price of dependent asset (first in pair)

        Returns:
            Tuple of (beta, spread)
        """
        kf = self.get_filter(pair)
        beta, spread, _ = kf.update(price_x, price_y)
        return beta, spread

    def get_beta(self, pair: Tuple[str, str]) -> float:
        """Get current hedge ratio for a pair."""
        kf = self.filters.get(pair)
        return kf.get_beta() if kf else 0.0

    def get_all_states(self) -> dict:
        """Get states for all filters (for serialization)."""
        return {
            str(pair): kf.get_state().to_dict()
            for pair, kf in self.filters.items()
            if kf.get_state() is not None
        }

    def load_states(self, states: dict) -> None:
        """Load states from saved data."""
        for pair_str, state_dict in states.items():
            # Parse pair string back to tuple
            pair = tuple(pair_str.strip("()").replace("'", "").split(", "))
            kf = self.get_filter(pair)
            kf.set_state(KalmanState.from_dict(state_dict))

    def reset_all(self) -> None:
        """Reset all filters."""
        for kf in self.filters.values():
            kf.reset()
        logger.info(f"Reset {len(self.filters)} Kalman filters")
