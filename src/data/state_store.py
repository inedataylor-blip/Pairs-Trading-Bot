"""State persistence for bot state (positions, signals, Kalman filter state)."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class StateStore:
    """Persistent state storage for bot state."""

    def __init__(self, state_dir: str = "data/state"):
        """
        Initialize state store.

        Args:
            state_dir: Directory for state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str, format: str = "json") -> Path:
        """Get file path for a state key."""
        return self.state_dir / f"{key}.{format}"

    def save_json(self, key: str, data: dict) -> None:
        """
        Save state as JSON.

        Args:
            key: State key (filename without extension)
            data: Data to save
        """
        path = self._get_path(key, "json")
        with open(path, "w") as f:
            json.dump(
                {
                    "data": data,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )
        logger.debug(f"Saved state to {path}")

    def load_json(self, key: str) -> Optional[dict]:
        """
        Load state from JSON.

        Args:
            key: State key

        Returns:
            Loaded data or None if not found
        """
        path = self._get_path(key, "json")
        if not path.exists():
            return None

        with open(path) as f:
            state = json.load(f)
            return state.get("data")

    def save_pickle(self, key: str, data: Any) -> None:
        """
        Save state as pickle (for complex objects like Kalman filter).

        Args:
            key: State key
            data: Data to save
        """
        path = self._get_path(key, "pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "data": data,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
            )
        logger.debug(f"Saved pickled state to {path}")

    def load_pickle(self, key: str) -> Optional[Any]:
        """
        Load state from pickle.

        Args:
            key: State key

        Returns:
            Loaded data or None if not found
        """
        path = self._get_path(key, "pkl")
        if not path.exists():
            return None

        with open(path, "rb") as f:
            state = pickle.load(f)
            return state.get("data")

    def save_kalman_state(self, pair: tuple[str, str], state: dict) -> None:
        """
        Save Kalman filter state for a pair.

        Args:
            pair: Asset pair tuple
            state: Kalman filter state dict
        """
        key = f"kalman_{pair[0]}_{pair[1]}"
        self.save_pickle(key, state)

    def load_kalman_state(self, pair: tuple[str, str]) -> Optional[dict]:
        """
        Load Kalman filter state for a pair.

        Args:
            pair: Asset pair tuple

        Returns:
            Kalman filter state or None
        """
        key = f"kalman_{pair[0]}_{pair[1]}"
        return self.load_pickle(key)

    def save_positions(self, positions: list[dict]) -> None:
        """Save current positions state."""
        self.save_json("positions", {"positions": positions})

    def load_positions(self) -> list[dict]:
        """Load positions state."""
        data = self.load_json("positions")
        return data.get("positions", []) if data else []

    def save_signals(self, signals: dict) -> None:
        """Save current signals state."""
        self.save_json("signals", signals)

    def load_signals(self) -> dict:
        """Load signals state."""
        return self.load_json("signals") or {}

    def save_daily_pnl(self, date: str, pnl: float, details: dict) -> None:
        """
        Save daily P&L record.

        Args:
            date: Date string (YYYY-MM-DD)
            pnl: Daily P&L value
            details: Additional details
        """
        key = f"pnl_{date}"
        self.save_json(key, {"pnl": pnl, "details": details})

    def clear_all(self) -> None:
        """Clear all state files."""
        for path in self.state_dir.iterdir():
            if path.is_file():
                path.unlink()
        logger.info("Cleared all state files")
