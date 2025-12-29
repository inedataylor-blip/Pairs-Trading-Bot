"""Tests for Kalman filter hedge ratio estimation."""

import numpy as np
import pandas as pd
import pytest

from src.signal_generation.kalman_filter import KalmanFilterHedgeRatio, KalmanState


class TestKalmanFilterHedgeRatio:
    """Tests for KalmanFilterHedgeRatio class."""

    @pytest.fixture
    def kalman(self):
        """Create a Kalman filter instance."""
        return KalmanFilterHedgeRatio(delta=0.0001)

    @pytest.fixture
    def price_data(self):
        """Generate synthetic price data with known hedge ratio."""
        np.random.seed(42)
        n = 100

        # True hedge ratio
        true_beta = 1.5

        # Price 2 (independent)
        price2 = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # Price 1 (dependent) = beta * price2 + spread + noise
        spread = np.random.randn(n) * 2
        price1 = true_beta * price2 + spread

        return price1, price2, true_beta

    def test_initialization(self, kalman):
        """Test Kalman filter initialization."""
        assert kalman.delta == 0.0001
        assert kalman.state is None

    def test_first_update(self, kalman):
        """Test first update initializes state."""
        beta, spread, Ve = kalman.update(100.0, 150.0)

        assert kalman.state is not None
        assert kalman.state.n_obs == 0  # First observation doesn't count
        assert isinstance(beta, float)
        assert isinstance(spread, float)

    def test_update_convergence(self, kalman, price_data):
        """Test that beta converges to true value."""
        price1, price2, true_beta = price_data

        betas = []
        for p1, p2 in zip(price1, price2):
            beta, _, _ = kalman.update(p2, p1)
            betas.append(beta)

        # After enough observations, beta should be close to true value
        final_beta = betas[-1]
        assert abs(final_beta - true_beta) < 0.5  # Within 0.5 of true beta

    def test_spread_calculation(self, kalman, price_data):
        """Test spread calculation."""
        price1, price2, _ = price_data

        # Process all data
        for p1, p2 in zip(price1, price2):
            kalman.update(p2, p1)

        # Calculate spread at last point
        spread = kalman.calculate_spread(price2[-1], price1[-1])

        assert isinstance(spread, float)

    def test_state_serialization(self, kalman, price_data):
        """Test state can be serialized and deserialized."""
        price1, price2, _ = price_data

        # Process some data
        for p1, p2 in zip(price1[:50], price2[:50]):
            kalman.update(p2, p1)

        # Serialize state
        state = kalman.get_state()
        state_dict = state.to_dict()

        assert "beta" in state_dict
        assert "R" in state_dict
        assert "Ve" in state_dict

        # Deserialize
        new_state = KalmanState.from_dict(state_dict)
        assert new_state.beta == state.beta
        assert new_state.n_obs == state.n_obs

    def test_reset(self, kalman, price_data):
        """Test filter reset."""
        price1, price2, _ = price_data

        # Process some data
        for p1, p2 in zip(price1[:10], price2[:10]):
            kalman.update(p2, p1)

        assert kalman.state is not None

        # Reset
        kalman.reset()

        assert kalman.state is None

    def test_batch_update(self, kalman, price_data):
        """Test batch update processing."""
        price1, price2, _ = price_data

        dates = pd.date_range("2023-01-01", periods=len(price1), freq="D")
        prices_x = pd.Series(price2, index=dates)
        prices_y = pd.Series(price1, index=dates)

        result_df = kalman.update_batch(prices_x, prices_y)

        assert len(result_df) == len(price1)
        assert "beta" in result_df.columns
        assert "spread" in result_df.columns
        assert "Ve" in result_df.columns
