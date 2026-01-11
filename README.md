# Cointegration Pairs Trading Bot

A systematic pairs trading bot using cointegration analysis with Kalman Filter dynamic hedge ratios. Designed for liquid ETF pairs on Alpaca, targeting mean reversion alpha uncorrelated with momentum strategies.

## Features

- **Pair Discovery**: Automated cointegration testing with Engle-Granger and ADF tests
- **Dynamic Hedge Ratios**: Kalman Filter for real-time hedge ratio estimation
- **Signal Generation**: Z-score based entry/exit signals with configurable thresholds
- **Risk Management**: Position limits, exposure controls, and daily loss limits
- **Backtesting**: Full backtesting framework with performance metrics and visualization
- **Paper Trading**: Safe testing via Alpaca paper trading API

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PAIRS TRADING BOT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   PAIR       │    │   SIGNAL     │    │   EXECUTION          │  │
│  │   DISCOVERY  │───▶│   GENERATION │───▶│   ENGINE             │  │
│  │   MODULE     │    │   MODULE     │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         ▼                   ▼                      ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Cointegration│    │ Kalman Filter│    │ Position Manager     │  │
│  │ Testing      │    │ Hedge Ratio  │    │ Order Router         │  │
│  │ Universe Scan│    │ Z-Score Calc │    │ Risk Management      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     DATA LAYER                               │   │
│  │   Alpaca API  │  Price Cache  │  State Persistence          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Pairs-Trading-Bot.git
cd Pairs-Trading-Bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Alpaca API credentials
```

## Configuration

Edit `config.yaml` to customize trading parameters:

```yaml
trading:
  entry_zscore: 2.0      # Z-score threshold for entry
  exit_zscore: 0.5       # Z-score threshold for exit
  stop_zscore: 3.5       # Z-score threshold for stop loss
  max_pairs: 3           # Maximum simultaneous pairs
  risk_per_trade: 0.02   # Risk 2% per trade

kalman:
  delta: 0.0001          # Kalman filter adaptation rate
  lookback_zscore: 20    # Z-score rolling window

risk:
  max_pair_exposure: 0.40    # Max 40% per pair
  max_gross_exposure: 1.50   # Max 150% total exposure
  max_daily_loss: 0.05       # Halt at 5% daily loss
```

## Usage

### Run the Bot (Paper Trading)

```bash
# Start the bot with paper trading (recommended for testing)
python -m src.main
```

The bot uses Alpaca's paper trading API by default. Set `ALPACA_PAPER=true` in your `.env` file.

### Run in Dry-Run Mode

```bash
# Dry-run mode: bot runs but no orders are submitted
python -m src.main --dry-run
```

### Run Pair Scan Only

```bash
python -m src.main --scan
```

### Check Status

```bash
python -m src.main --status
```

### Run Backtest

```bash
python examples/run_backtest.py
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Run without submitting orders |
| `--scan` | Run pair scan only and exit |
| `--status` | Show current status and exit |
| `--backtest` | Show backtest instructions |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR) |
| `--config` | Path to config file (default: config.yaml) |

## Project Structure

```
Pairs-Trading-Bot/
├── src/
│   ├── data/                 # Data layer
│   │   ├── alpaca_client.py  # Alpaca API wrapper
│   │   ├── database.py       # SQLite storage
│   │   ├── price_cache.py    # Price caching
│   │   └── state_store.py    # State persistence
│   │
│   ├── pair_discovery/       # Pair discovery module
│   │   ├── universe.py       # ETF pair universe
│   │   ├── cointegration.py  # Cointegration testing
│   │   ├── pair_selector.py  # Pair ranking/selection
│   │   └── validation.py     # Out-of-sample validation
│   │
│   ├── signal_generation/    # Signal generation module
│   │   ├── kalman_filter.py  # Kalman filter for hedge ratio
│   │   ├── spread_calculator.py
│   │   ├── zscore.py         # Z-score calculation
│   │   └── signals.py        # Signal logic
│   │
│   ├── execution/            # Execution engine
│   │   ├── position_manager.py
│   │   ├── order_router.py
│   │   └── risk_manager.py
│   │
│   ├── backtest/             # Backtesting framework
│   │   ├── engine.py         # Backtest engine
│   │   ├── metrics.py        # Performance metrics
│   │   └── visualization.py  # Charts and reports
│   │
│   ├── utils/                # Utilities
│   │   └── logging.py
│   │
│   ├── bot.py                # Main bot orchestration
│   ├── config.py             # Configuration management
│   └── main.py               # Entry point
│
├── tests/                    # Unit tests
├── examples/                 # Example scripts
├── config.yaml               # Configuration file
├── requirements.txt          # Dependencies
└── README.md
```

## Default ETF Universe

The bot includes a curated universe of liquid ETF pairs:

| Category | Pairs |
|----------|-------|
| Sector | XLF/KBE, XLE/XOP, XLK/IGV, XLV/IBB |
| Commodity | GLD/SLV, GLD/GDX, USO/XLE |
| Bond | TLT/IEF, LQD/HYG |
| International | EFA/EEM, FXE/FXB |
| Index | SPY/IWM, QQQ/IWM |

## Signal Logic

| Condition | Action |
|-----------|--------|
| Z-score < -2.0 | LONG spread (buy asset1, sell asset2) |
| Z-score > +2.0 | SHORT spread (sell asset1, buy asset2) |
| Z-score crosses 0 | EXIT position |
| Z-score exceeds ±3.5 | STOP LOSS |

## Risk Management

- Maximum 3 pairs traded simultaneously
- Maximum 40% exposure per pair
- Maximum 150% total gross exposure
- 5% daily loss limit (trading halted)
- 3% maximum loss per trade

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance Metrics

The backtesting framework calculates:

- Total Return, CAGR
- Sharpe Ratio, Sortino Ratio
- Max Drawdown, Calmar Ratio
- Win Rate, Profit Factor
- Average Holding Period
- Expectancy per trade

## Daily Schedule (Mon-Fri, Eastern Time)

| Time (ET) | Action |
|-----------|--------|
| 9:25 AM | Generate signals |
| 9:35 AM | Execute trades |
| 12:00 PM | Midday check |
| 3:50 PM | End of day processing |
| 6:00 PM | Daily pair scan |

## License

MIT License

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade first and never risk more than you can afford to lose.
