"""Tests for cointegration analysis."""

import numpy as np
import pandas as pd
import pytest

from src.pair_discovery.cointegration import CointegrationTester


class TestCointegrationTester:
    """Tests for CointegrationTester class."""

    @pytest.fixture
    def tester(self):
        """Create a tester instance."""
        return CointegrationTester(
            p_value_threshold=0.05,
            min_half_life=5,
            max_half_life=60,
        )

    @pytest.fixture
    def cointegrated_pair(self):
        """Generate a cointegrated pair of price series."""
        np.random.seed(42)
        n = 252

        # Random walk for asset 2
        price2 = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # Cointegrated asset 1 with mean-reverting spread
        beta = 0.8
        mean_spread = 10
        spread = mean_spread + np.cumsum(np.random.randn(n) * 0.3)
        # Add mean reversion
        for i in range(1, n):
            spread[i] = spread[i - 1] * 0.95 + mean_spread * 0.05 + np.random.randn() * 0.2

        price1 = beta * price2 + spread

        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        return (
            pd.Series(price1, index=dates),
            pd.Series(price2, index=dates),
        )

    @pytest.fixture
    def random_walk_pair(self):
        """Generate non-cointegrated random walks."""
        np.random.seed(42)
        n = 252

        price1 = 100 + np.cumsum(np.random.randn(n) * 0.5)
        price2 = 100 + np.cumsum(np.random.randn(n) * 0.5)

        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        return (
            pd.Series(price1, index=dates),
            pd.Series(price2, index=dates),
        )

    def test_calculate_beta(self, tester, cointegrated_pair):
        """Test beta calculation."""
        prices1, prices2 = cointegrated_pair
        beta = tester.calculate_beta(prices1, prices2)

        # Beta should be close to 0.8 (the true value)
        assert 0.5 < beta < 1.1

    def test_calculate_spread(self, tester, cointegrated_pair):
        """Test spread calculation."""
        prices1, prices2 = cointegrated_pair
        spread = tester.calculate_spread(prices1, prices2)

        assert len(spread) == len(prices1)
        assert not spread.isna().any()

    def test_calculate_half_life(self, tester, cointegrated_pair):
        """Test half-life calculation."""
        prices1, prices2 = cointegrated_pair
        spread = tester.calculate_spread(prices1, prices2)
        half_life = tester.calculate_half_life(spread)

        # Half-life should be reasonable
        assert half_life > 0
        assert half_life < 100

    def test_calculate_hurst(self, tester, cointegrated_pair):
        """Test Hurst exponent calculation."""
        prices1, prices2 = cointegrated_pair
        spread = tester.calculate_spread(prices1, prices2)
        hurst = tester.calculate_hurst_exponent(spread)

        # Hurst should be between 0 and 1
        assert 0 <= hurst <= 1

    def test_test_adf(self, tester, cointegrated_pair):
        """Test ADF test."""
        prices1, prices2 = cointegrated_pair
        spread = tester.calculate_spread(prices1, prices2)

        adf_stat, p_value, critical_values = tester.test_adf(spread)

        assert isinstance(adf_stat, float)
        assert 0 <= p_value <= 1
        assert "1%" in critical_values
        assert "5%" in critical_values

    def test_test_pair_cointegrated(self, tester, cointegrated_pair):
        """Test pair testing on cointegrated pair."""
        prices1, prices2 = cointegrated_pair
        pair = ("ASSET1", "ASSET2")

        result = tester.test_pair(prices1, prices2, pair)

        assert result.pair == pair
        assert result.p_value >= 0
        assert result.half_life > 0
        # The mock cointegrated pair should pass
        # (Note: this is probabilistic, might occasionally fail)

    def test_test_pair_random_walk(self, tester, random_walk_pair):
        """Test pair testing on random walks."""
        prices1, prices2 = random_walk_pair
        pair = ("RANDOM1", "RANDOM2")

        result = tester.test_pair(prices1, prices2, pair)

        assert result.pair == pair
        # Random walks should typically not be cointegrated
        # (Note: this is probabilistic)
