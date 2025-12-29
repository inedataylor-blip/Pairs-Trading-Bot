"""Price caching to reduce API calls."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger


class PriceCache:
    """In-memory cache for price data with expiration."""

    def __init__(self, default_ttl_minutes: int = 15):
        """
        Initialize price cache.

        Args:
            default_ttl_minutes: Default time-to-live for cached data
        """
        self._cache: dict[str, dict] = {}
        self.default_ttl = timedelta(minutes=default_ttl_minutes)

    def _cache_key(self, symbol: str, data_type: str = "bars") -> str:
        """Generate cache key for a symbol and data type."""
        return f"{symbol}:{data_type}"

    def get(
        self,
        symbol: str,
        data_type: str = "bars",
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data if not expired.

        Args:
            symbol: Stock symbol
            data_type: Type of data (bars, quote, etc.)

        Returns:
            Cached DataFrame or None if expired/not found
        """
        key = self._cache_key(symbol, data_type)

        if key not in self._cache:
            return None

        entry = self._cache[key]
        if datetime.now() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["data"]

    def set(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_type: str = "bars",
        ttl: Optional[timedelta] = None,
    ) -> None:
        """
        Cache data with expiration.

        Args:
            symbol: Stock symbol
            data: DataFrame to cache
            data_type: Type of data
            ttl: Time-to-live (uses default if not specified)
        """
        key = self._cache_key(symbol, data_type)
        ttl = ttl or self.default_ttl

        self._cache[key] = {
            "data": data,
            "cached_at": datetime.now(),
            "expires_at": datetime.now() + ttl,
        }

        logger.debug(f"Cached {data_type} data for {symbol}")

    def invalidate(self, symbol: str, data_type: Optional[str] = None) -> None:
        """
        Invalidate cached data.

        Args:
            symbol: Stock symbol
            data_type: Type of data (None to invalidate all types)
        """
        if data_type:
            key = self._cache_key(symbol, data_type)
            if key in self._cache:
                del self._cache[key]
        else:
            # Invalidate all data types for this symbol
            keys_to_remove = [k for k in self._cache if k.startswith(f"{symbol}:")]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Cleared price cache")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(
            1 for entry in self._cache.values() if entry["expires_at"] > now
        )

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
        }
