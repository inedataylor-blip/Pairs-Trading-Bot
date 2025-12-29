"""Configuration management for the pairs trading bot."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger


class Config:
    """Configuration manager that loads from YAML and environment variables."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file and environment."""
        load_dotenv()

        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._config = self._default_config()

    def _default_config(self) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "trading": {
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "stop_zscore": 3.5,
                "max_pairs": 3,
                "risk_per_trade": 0.02,
            },
            "kalman": {
                "delta": 0.0001,
                "lookback_zscore": 20,
            },
            "cointegration": {
                "lookback_days": 252,
                "min_half_life": 5,
                "max_half_life": 60,
                "p_value_threshold": 0.05,
                "retest_frequency_days": 7,
            },
            "risk": {
                "max_pair_exposure": 0.40,
                "max_gross_exposure": 1.50,
                "max_daily_loss": 0.05,
                "max_trade_loss": 0.03,
            },
            "schedule": {
                "signal_time": "09:25",
                "execution_time": "09:35",
                "midday_check": "12:00",
                "close_check": "15:50",
            },
            "alpaca": {
                "paper": True,
                "base_url": "https://paper-api.alpaca.markets",
            },
            "logging": {
                "level": "INFO",
                "file": "logs/trading_bot.log",
            },
            "database": {
                "path": "data/trading.db",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'trading.entry_zscore')."""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def alpaca_api_key(self) -> str:
        """Get Alpaca API key from environment."""
        return os.getenv("ALPACA_API_KEY", "")

    @property
    def alpaca_secret_key(self) -> str:
        """Get Alpaca secret key from environment."""
        return os.getenv("ALPACA_SECRET_KEY", "")

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        mode = os.getenv("TRADING_MODE", "paper")
        return mode.lower() == "paper" or self.get("alpaca.paper", True)

    @property
    def alpaca_base_url(self) -> str:
        """Get Alpaca base URL based on trading mode."""
        if self.is_paper_trading:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)


# Global configuration instance
config = Config()
