#!/usr/bin/env python3
"""Example script for running a backtest."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import BacktestEngine, PerformanceMetrics, BacktestVisualizer
from src.backtest.metrics import calculate_metrics
from src.pair_discovery import CointegrationTester
from src.utils.logging import setup_logging


def generate_sample_data(n_days: int = 500) -> tuple:
    """
    Generate sample cointegrated price data for testing.

    Returns:
        Tuple of (prices1, prices2, pair_tuple)
    """
    np.random.seed(42)

    # Create date index
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")

    # True parameters
    true_beta = 0.85
    mean_spread = 5.0

    # Generate asset 2 (random walk with drift)
    returns2 = np.random.randn(n_days) * 0.015 + 0.0002
    price2 = 100 * np.cumprod(1 + returns2)

    # Generate mean-reverting spread
    spread = np.zeros(n_days)
    spread[0] = mean_spread
    theta = 0.1  # Mean reversion speed

    for i in range(1, n_days):
        # Ornstein-Uhlenbeck process
        spread[i] = spread[i - 1] + theta * (mean_spread - spread[i - 1]) + np.random.randn() * 1.5

    # Asset 1 = beta * asset2 + spread
    price1 = true_beta * price2 + spread

    prices1 = pd.Series(price1, index=dates, name="XLF")
    prices2 = pd.Series(price2, index=dates, name="KBE")

    return prices1, prices2, ("XLF", "KBE")


def main():
    """Run backtest example."""
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/backtest.log")

    logger.info("=" * 60)
    logger.info("Pairs Trading Backtest Example")
    logger.info("=" * 60)

    # Generate sample data
    logger.info("Generating sample cointegrated price data...")
    prices1, prices2, pair = generate_sample_data(n_days=500)

    # Test for cointegration first
    logger.info("Testing for cointegration...")
    tester = CointegrationTester()
    coint_result = tester.test_pair(prices1, prices2, pair)

    logger.info(f"Cointegration test results:")
    logger.info(f"  P-value: {coint_result.p_value:.4f}")
    logger.info(f"  Half-life: {coint_result.half_life:.1f} days")
    logger.info(f"  Hurst: {coint_result.hurst_exponent:.3f}")
    logger.info(f"  Is cointegrated: {coint_result.is_cointegrated}")

    # Run backtest
    logger.info("\nRunning backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        position_size_pct=0.20,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_zscore=3.5,
        kalman_delta=0.0001,
        zscore_lookback=20,
        commission_per_share=0.01,
        slippage_pct=0.001,
    )

    result = engine.run(prices1, prices2, pair)

    # Calculate metrics
    metrics = calculate_metrics(result)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)

    print(f"\nPair: {pair[0]} / {pair[1]}")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"\nReturns:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  CAGR: {metrics.cagr:.2%}")
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Avg Win/Loss: {metrics.avg_win_loss_ratio:.2f}")
    print(f"  Avg Holding: {metrics.avg_holding_period:.1f} days")
    print(f"  Expectancy: ${metrics.expectancy:,.2f}")

    # Generate visualization
    logger.info("\nGenerating charts...")
    visualizer = BacktestVisualizer(output_dir="output/charts")
    visualizer.create_report(result, metrics, save=True)

    logger.info(f"\nCharts saved to: output/charts/")
    logger.info("Backtest complete!")

    return result, metrics


if __name__ == "__main__":
    result, metrics = main()
