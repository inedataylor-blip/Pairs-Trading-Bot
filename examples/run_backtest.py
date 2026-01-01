#!/usr/bin/env python3
"""Backtest with real historical data from Yahoo Finance."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.backtest import BacktestEngine, BacktestVisualizer
from src.backtest.metrics import calculate_metrics, aggregate_walk_forward_metrics
from src.pair_discovery import CointegrationTester
from src.utils.logging import setup_logging


def fetch_pair_data(
    symbol1: str,
    symbol2: str,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-01",
) -> tuple:
    """
    Fetch real historical data from Yahoo Finance.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Tuple of (prices1, prices2, pair_tuple)
    """
    logger.info(f"Fetching data for {symbol1} and {symbol2}...")

    # Download data (auto_adjust=True is now default in yfinance)
    data1 = yf.download(symbol1, start=start_date, end=end_date, progress=False, auto_adjust=True)
    data2 = yf.download(symbol2, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if data1.empty or data2.empty:
        raise ValueError(f"Could not fetch data for {symbol1} or {symbol2}")

    # Get close prices (already adjusted when auto_adjust=True)
    # Handle both single-level and multi-level column indexes
    if isinstance(data1.columns, pd.MultiIndex):
        prices1 = data1["Close"][symbol1]
        prices2 = data2["Close"][symbol2]
    else:
        prices1 = data1["Close"].squeeze()
        prices2 = data2["Close"].squeeze()

    # Align on common dates
    common_idx = prices1.index.intersection(prices2.index)
    prices1 = prices1.loc[common_idx]
    prices2 = prices2.loc[common_idx]

    prices1.name = symbol1
    prices2.name = symbol2

    logger.info(f"  Fetched {len(prices1)} days of data")
    logger.info(f"  Date range: {prices1.index[0].date()} to {prices1.index[-1].date()}")

    return prices1, prices2, (symbol1, symbol2)


def run_single_pair_backtest(
    prices1: pd.Series,
    prices2: pd.Series,
    pair: tuple,
    position_size: float = 0.40,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
) -> tuple:
    """Run backtest on a single pair and return results."""

    # Test for cointegration
    tester = CointegrationTester()
    coint_result = tester.test_pair(prices1, prices2, pair)

    print(f"\n{'='*60}")
    print(f"PAIR: {pair[0]} / {pair[1]}")
    print(f"{'='*60}")
    print(f"Cointegration Test:")
    print(f"  P-value: {coint_result.p_value:.4f}")
    print(f"  Half-life: {coint_result.half_life:.1f} days")
    print(f"  Hurst: {coint_result.hurst_exponent:.3f}")
    print(f"  Cointegrated: {coint_result.is_cointegrated}")

    if not coint_result.is_cointegrated:
        print(f"  WARNING: Pair does not pass cointegration test!")

    # Run backtest
    engine = BacktestEngine(
        initial_capital=100000,
        position_size_pct=position_size,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        stop_zscore=3.5,
        kalman_delta=0.0001,
        zscore_lookback=20,
        commission_per_share=0.005,
        slippage_pct=0.0005,
    )

    result = engine.run(prices1, prices2, pair)
    metrics = calculate_metrics(result)

    # Print results
    print(f"\nBacktest Results ({result.start_date.date()} to {result.end_date.date()}):")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  CAGR: {metrics.cagr:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Avg Holding: {metrics.avg_holding_period:.1f} days")

    return result, metrics, coint_result


def main():
    """Run backtest on multiple real pairs."""
    # Setup logging
    setup_logging(log_level="WARNING", log_file="logs/backtest.log")

    print("=" * 60)
    print("PAIRS TRADING BACKTEST - REAL DATA")
    print("=" * 60)

    # Define pairs to test (from the universe)
    pairs_to_test = [
        ("GLD", "SLV"),     # Gold vs Silver - classic commodity pair
        ("XLF", "KBE"),     # Financials vs Regional Banks
        ("TLT", "IEF"),     # Long-term vs Intermediate Treasuries
        ("XLE", "XOP"),     # Energy vs Oil & Gas E&P
        ("EFA", "EEM"),     # Developed vs Emerging Markets
        ("SPY", "IWM"),     # S&P 500 vs Russell 2000
        ("GLD", "GDX"),     # Gold vs Gold Miners
        ("QQQ", "XLK"),     # Nasdaq vs Tech Sector
    ]

    # Backtest parameters
    start_date = "2019-01-01"
    end_date = "2024-12-01"
    position_size = 0.40      # 40% position size
    entry_zscore = 2.0        # Entry at 2 std dev
    exit_zscore = 0.0         # Exit at mean

    all_results = []
    all_metrics = []
    cointegrated_pairs = []

    print(f"\nBacktest Period: {start_date} to {end_date}")
    print(f"Position Size: {position_size:.0%}")
    print(f"Entry Z-score: ±{entry_zscore}")
    print(f"Exit Z-score: ±{exit_zscore}")

    # Test each pair
    for symbol1, symbol2 in pairs_to_test:
        try:
            prices1, prices2, pair = fetch_pair_data(
                symbol1, symbol2, start_date, end_date
            )

            result, metrics, coint_result = run_single_pair_backtest(
                prices1, prices2, pair,
                position_size=position_size,
                entry_zscore=entry_zscore,
                exit_zscore=exit_zscore,
            )

            all_results.append(result)
            all_metrics.append((pair, metrics, coint_result))

            if coint_result.is_cointegrated:
                cointegrated_pairs.append((pair, metrics))

        except Exception as e:
            print(f"\nError testing {symbol1}/{symbol2}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - ALL PAIRS")
    print("=" * 60)

    print(f"\n{'Pair':<12} {'Coint?':<8} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8}")
    print("-" * 70)

    for pair, metrics, coint_result in all_metrics:
        pair_str = f"{pair[0]}/{pair[1]}"
        coint_str = "Yes" if coint_result.is_cointegrated else "No"
        print(
            f"{pair_str:<12} {coint_str:<8} {metrics.cagr:>7.1%} "
            f"{metrics.sharpe_ratio:>8.2f} {metrics.max_drawdown:>7.1%} "
            f"{metrics.total_trades:>8} {metrics.win_rate:>7.1%}"
        )

    # Best performers
    print("\n" + "=" * 60)
    print("TOP PERFORMERS (by CAGR)")
    print("=" * 60)

    sorted_metrics = sorted(all_metrics, key=lambda x: x[1].cagr, reverse=True)
    for i, (pair, metrics, _) in enumerate(sorted_metrics[:5], 1):
        print(f"{i}. {pair[0]}/{pair[1]}: CAGR={metrics.cagr:.1%}, Sharpe={metrics.sharpe_ratio:.2f}")

    # Cointegrated pairs only
    if cointegrated_pairs:
        print("\n" + "=" * 60)
        print("COINTEGRATED PAIRS ONLY")
        print("=" * 60)

        coint_cagrs = [m.cagr for _, m in cointegrated_pairs]
        coint_sharpes = [m.sharpe_ratio for _, m in cointegrated_pairs]

        print(f"Number of cointegrated pairs: {len(cointegrated_pairs)}")
        print(f"Average CAGR: {np.mean(coint_cagrs):.1%}")
        print(f"Average Sharpe: {np.mean(coint_sharpes):.2f}")

    # Generate charts for best pair
    if all_results:
        best_idx = np.argmax([m.cagr for _, m, _ in all_metrics])
        best_result = all_results[best_idx]
        best_metrics = all_metrics[best_idx][1]

        print(f"\nGenerating charts for best pair: {best_result.pair}...")
        visualizer = BacktestVisualizer(output_dir="output/charts")
        visualizer.create_report(best_result, best_metrics, save=True)
        print("Charts saved to: output/charts/")

    return all_results, all_metrics


if __name__ == "__main__":
    results, metrics = main()
