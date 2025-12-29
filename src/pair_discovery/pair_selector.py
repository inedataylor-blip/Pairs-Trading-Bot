"""Pair selection and ranking based on cointegration results."""

from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from .cointegration import CointegrationResult, CointegrationTester


class PairSelector:
    """Select and rank pairs based on cointegration and other criteria."""

    def __init__(
        self,
        cointegration_tester: CointegrationTester,
        max_pairs: int = 3,
        min_liquidity: Optional[float] = None,
    ):
        """
        Initialize pair selector.

        Args:
            cointegration_tester: CointegrationTester instance
            max_pairs: Maximum number of pairs to select
            min_liquidity: Minimum average daily volume (optional)
        """
        self.tester = cointegration_tester
        self.max_pairs = max_pairs
        self.min_liquidity = min_liquidity

    def score_pair(self, result: CointegrationResult) -> float:
        """
        Calculate a composite score for a cointegrated pair.

        Higher score = better pair.

        Scoring factors:
        - Lower p-value = stronger cointegration
        - Optimal half-life around 10-20 days
        - Lower Hurst exponent = stronger mean reversion

        Args:
            result: CointegrationResult object

        Returns:
            Composite score (higher is better)
        """
        if not result.is_cointegrated:
            return 0.0

        # P-value score (lower is better)
        # Scale from 0-1 based on p-value (0.05 threshold)
        p_value_score = max(0, 1 - result.p_value / 0.05)

        # Half-life score (optimal around 10-20 days)
        # Penalize very short (<5) or very long (>40) half-lives
        optimal_half_life = 15
        half_life_diff = abs(result.half_life - optimal_half_life)
        half_life_score = max(0, 1 - half_life_diff / 30)

        # Hurst score (lower is better, < 0.5 means mean-reverting)
        hurst_score = max(0, 1 - result.hurst_exponent)

        # Weighted composite score
        score = (
            0.4 * p_value_score + 0.3 * half_life_score + 0.3 * hurst_score
        )

        return score

    def rank_pairs(
        self,
        results: List[CointegrationResult],
    ) -> List[Tuple[CointegrationResult, float]]:
        """
        Rank pairs by composite score.

        Args:
            results: List of CointegrationResult objects

        Returns:
            Sorted list of (result, score) tuples
        """
        scored = []
        for result in results:
            if result.is_cointegrated:
                score = self.score_pair(result)
                scored.append((result, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def select_top_pairs(
        self,
        results: List[CointegrationResult],
    ) -> List[CointegrationResult]:
        """
        Select the top N pairs based on ranking.

        Args:
            results: List of CointegrationResult objects

        Returns:
            List of top N CointegrationResult objects
        """
        ranked = self.rank_pairs(results)
        top_pairs = [result for result, score in ranked[: self.max_pairs]]

        logger.info(f"Selected {len(top_pairs)} pairs from {len(results)} tested")
        for i, pair_result in enumerate(top_pairs, 1):
            logger.info(
                f"  {i}. {pair_result.pair}: p={pair_result.p_value:.4f}, "
                f"half_life={pair_result.half_life:.1f}d"
            )

        return top_pairs

    def filter_by_correlation(
        self,
        pairs: List[Tuple[str, str]],
        price_data: dict[str, pd.Series],
        max_correlation: float = 0.7,
    ) -> List[Tuple[str, str]]:
        """
        Filter pairs to reduce correlation between selected pairs.

        Args:
            pairs: List of pairs to filter
            price_data: Dictionary mapping symbols to price series
            max_correlation: Maximum allowed correlation between pairs

        Returns:
            Filtered list of pairs with reduced inter-correlation
        """
        if len(pairs) <= 1:
            return pairs

        # Calculate spreads for each pair
        spreads = {}
        for pair in pairs:
            p1 = price_data.get(pair[0])
            p2 = price_data.get(pair[1])
            if p1 is not None and p2 is not None:
                beta = self.tester.calculate_beta(p1, p2)
                spread = self.tester.calculate_spread(p1, p2, beta)
                spreads[pair] = spread

        # Greedily select pairs with low inter-correlation
        selected = [pairs[0]]

        for pair in pairs[1:]:
            if pair not in spreads:
                continue

            # Check correlation with already selected pairs
            max_corr = 0
            for selected_pair in selected:
                if selected_pair in spreads:
                    common_idx = spreads[pair].index.intersection(
                        spreads[selected_pair].index
                    )
                    if len(common_idx) > 20:
                        corr = abs(
                            spreads[pair].loc[common_idx].corr(
                                spreads[selected_pair].loc[common_idx]
                            )
                        )
                        max_corr = max(max_corr, corr)

            if max_corr < max_correlation:
                selected.append(pair)

        logger.info(
            f"Filtered from {len(pairs)} to {len(selected)} pairs "
            f"(max_corr={max_correlation})"
        )

        return selected

    def get_summary_dataframe(
        self,
        results: List[CointegrationResult],
    ) -> pd.DataFrame:
        """
        Create summary DataFrame of cointegration results.

        Args:
            results: List of CointegrationResult objects

        Returns:
            DataFrame with summary statistics
        """
        data = []
        for result in results:
            score = self.score_pair(result)
            data.append(
                {
                    "asset1": result.pair[0],
                    "asset2": result.pair[1],
                    "is_cointegrated": result.is_cointegrated,
                    "p_value": result.p_value,
                    "adf_statistic": result.adf_statistic,
                    "beta": result.beta,
                    "half_life": result.half_life,
                    "hurst_exponent": result.hurst_exponent,
                    "score": score,
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("score", ascending=False)
        return df
