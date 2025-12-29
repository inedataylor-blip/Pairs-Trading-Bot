"""Validation of cointegration relationships."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .cointegration import CointegrationResult, CointegrationTester


@dataclass
class ValidationResult:
    """Result of out-of-sample validation."""

    pair: Tuple[str, str]
    train_result: CointegrationResult
    test_result: CointegrationResult
    is_valid: bool
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    beta_stability: float  # Ratio of test beta to train beta
    p_value_change: float  # Difference in p-values

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pair": self.pair,
            "is_valid": self.is_valid,
            "train_period": self.train_period,
            "test_period": self.test_period,
            "train_p_value": self.train_result.p_value,
            "test_p_value": self.test_result.p_value,
            "train_beta": self.train_result.beta,
            "test_beta": self.test_result.beta,
            "beta_stability": self.beta_stability,
            "p_value_change": self.p_value_change,
        }


class PairValidator:
    """Validate cointegration relationships using out-of-sample testing."""

    def __init__(
        self,
        cointegration_tester: CointegrationTester,
        train_ratio: float = 0.7,
        beta_tolerance: float = 0.3,
    ):
        """
        Initialize pair validator.

        Args:
            cointegration_tester: CointegrationTester instance
            train_ratio: Ratio of data for training (0.7 = 70% train, 30% test)
            beta_tolerance: Maximum allowed change in beta (e.g., 0.3 = 30%)
        """
        self.tester = cointegration_tester
        self.train_ratio = train_ratio
        self.beta_tolerance = beta_tolerance

    def validate_pair(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair: Tuple[str, str],
    ) -> ValidationResult:
        """
        Validate a pair using train/test split.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            pair: Tuple of asset symbols

        Returns:
            ValidationResult object
        """
        # Split data
        split_idx = int(len(prices1) * self.train_ratio)

        train_p1 = prices1.iloc[:split_idx]
        train_p2 = prices2.iloc[:split_idx]
        test_p1 = prices1.iloc[split_idx:]
        test_p2 = prices2.iloc[split_idx:]

        # Test on training period
        train_result = self.tester.test_pair(
            train_p1,
            train_p2,
            pair,
            lookback_days=len(train_p1),
        )

        # Test on testing period
        test_result = self.tester.test_pair(
            test_p1,
            test_p2,
            pair,
            lookback_days=len(test_p1),
        )

        # Calculate stability metrics
        if abs(train_result.beta) > 0.001:
            beta_stability = test_result.beta / train_result.beta
        else:
            beta_stability = 0.0

        p_value_change = test_result.p_value - train_result.p_value

        # Determine if valid
        is_valid = (
            train_result.is_cointegrated
            and test_result.is_cointegrated
            and (1 - self.beta_tolerance)
            < beta_stability
            < (1 + self.beta_tolerance)
        )

        # Get period strings
        train_period = (
            str(train_p1.index[0].date()),
            str(train_p1.index[-1].date()),
        )
        test_period = (
            str(test_p1.index[0].date()),
            str(test_p1.index[-1].date()),
        )

        result = ValidationResult(
            pair=pair,
            train_result=train_result,
            test_result=test_result,
            is_valid=is_valid,
            train_period=train_period,
            test_period=test_period,
            beta_stability=beta_stability,
            p_value_change=p_value_change,
        )

        logger.debug(
            f"Validated {pair}: is_valid={is_valid}, "
            f"beta_stability={beta_stability:.3f}"
        )

        return result

    def walk_forward_validate(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair: Tuple[str, str],
        train_window: int = 252,
        test_window: int = 63,
        step_size: Optional[int] = None,
    ) -> List[ValidationResult]:
        """
        Perform walk-forward validation.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            pair: Tuple of asset symbols
            train_window: Training window size in days
            test_window: Test window size in days
            step_size: Steps between windows (defaults to test_window)

        Returns:
            List of ValidationResult objects for each period
        """
        if step_size is None:
            step_size = test_window

        results = []
        total_len = len(prices1)

        for start in range(0, total_len - train_window - test_window + 1, step_size):
            train_end = start + train_window
            test_end = train_end + test_window

            # Extract windows
            train_p1 = prices1.iloc[start:train_end]
            train_p2 = prices2.iloc[start:train_end]
            test_p1 = prices1.iloc[train_end:test_end]
            test_p2 = prices2.iloc[train_end:test_end]

            # Test each period
            train_result = self.tester.test_pair(
                train_p1, train_p2, pair, len(train_p1)
            )
            test_result = self.tester.test_pair(test_p1, test_p2, pair, len(test_p1))

            # Calculate metrics
            if abs(train_result.beta) > 0.001:
                beta_stability = test_result.beta / train_result.beta
            else:
                beta_stability = 0.0

            p_value_change = test_result.p_value - train_result.p_value

            is_valid = (
                train_result.is_cointegrated
                and test_result.is_cointegrated
                and (1 - self.beta_tolerance)
                < beta_stability
                < (1 + self.beta_tolerance)
            )

            train_period = (
                str(train_p1.index[0].date()),
                str(train_p1.index[-1].date()),
            )
            test_period = (
                str(test_p1.index[0].date()),
                str(test_p1.index[-1].date()),
            )

            results.append(
                ValidationResult(
                    pair=pair,
                    train_result=train_result,
                    test_result=test_result,
                    is_valid=is_valid,
                    train_period=train_period,
                    test_period=test_period,
                    beta_stability=beta_stability,
                    p_value_change=p_value_change,
                )
            )

        return results

    def get_validation_summary(
        self,
        results: List[ValidationResult],
    ) -> dict:
        """
        Summarize walk-forward validation results.

        Args:
            results: List of ValidationResult objects

        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {"num_periods": 0, "valid_ratio": 0}

        num_valid = sum(1 for r in results if r.is_valid)
        valid_ratio = num_valid / len(results)

        avg_train_pvalue = np.mean([r.train_result.p_value for r in results])
        avg_test_pvalue = np.mean([r.test_result.p_value for r in results])

        beta_changes = [abs(1 - r.beta_stability) for r in results]
        avg_beta_change = np.mean(beta_changes)

        return {
            "pair": results[0].pair,
            "num_periods": len(results),
            "num_valid": num_valid,
            "valid_ratio": valid_ratio,
            "avg_train_pvalue": avg_train_pvalue,
            "avg_test_pvalue": avg_test_pvalue,
            "avg_beta_change": avg_beta_change,
            "is_stable": valid_ratio >= 0.7,  # At least 70% of periods valid
        }
