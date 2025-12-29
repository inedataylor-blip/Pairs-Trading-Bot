"""Define the universe of tradeable pair candidates."""

from typing import List, Tuple


class PairUniverse:
    """Manages the universe of candidate pairs for trading."""

    # Default ETF pair universe organized by category
    SECTOR_PAIRS = [
        ("XLF", "KBE"),  # Financials vs Regional Banks
        ("XLE", "XOP"),  # Energy vs Oil & Gas E&P
        ("XLK", "IGV"),  # Tech vs Software
        ("XLV", "IBB"),  # Healthcare vs Biotech
        ("XLI", "ITA"),  # Industrials vs Aerospace & Defense
        ("XLY", "XRT"),  # Consumer Discretionary vs Retail
        ("XLP", "PBJ"),  # Consumer Staples vs Food & Beverage
        ("XLU", "VPU"),  # Utilities vs Vanguard Utilities
    ]

    COMMODITY_PAIRS = [
        ("GLD", "SLV"),  # Gold vs Silver
        ("GLD", "GDX"),  # Gold vs Gold Miners
        ("USO", "XLE"),  # Oil vs Energy sector
        ("SLV", "SIL"),  # Silver vs Silver Miners
        ("DBA", "MOO"),  # Agriculture vs Agribusiness
    ]

    BOND_PAIRS = [
        ("TLT", "IEF"),  # Long-term vs Intermediate Treasuries
        ("LQD", "HYG"),  # Investment Grade vs High Yield
        ("TLT", "TIP"),  # Nominal vs Inflation-Protected Treasuries
        ("AGG", "BND"),  # iShares vs Vanguard Aggregate Bond
        ("MUB", "HYD"),  # Muni Bond vs High Yield Muni
    ]

    INTERNATIONAL_PAIRS = [
        ("EFA", "EEM"),  # Developed vs Emerging Markets
        ("FXE", "FXB"),  # Euro vs British Pound
        ("EWJ", "EWZ"),  # Japan vs Brazil
        ("VGK", "VWO"),  # Europe vs Emerging Markets
        ("EWG", "EWU"),  # Germany vs UK
    ]

    INDEX_PAIRS = [
        ("SPY", "IWM"),  # S&P 500 vs Russell 2000
        ("QQQ", "IWM"),  # Nasdaq vs Russell 2000
        ("DIA", "SPY"),  # Dow vs S&P 500
        ("MDY", "IWM"),  # Mid-Cap vs Small-Cap
    ]

    def __init__(
        self,
        include_sectors: bool = True,
        include_commodities: bool = True,
        include_bonds: bool = True,
        include_international: bool = True,
        include_indices: bool = True,
        custom_pairs: List[Tuple[str, str]] = None,
    ):
        """
        Initialize pair universe with selected categories.

        Args:
            include_sectors: Include sector ETF pairs
            include_commodities: Include commodity ETF pairs
            include_bonds: Include bond ETF pairs
            include_international: Include international ETF pairs
            include_indices: Include index ETF pairs
            custom_pairs: Additional custom pairs to include
        """
        self._pairs: List[Tuple[str, str]] = []

        if include_sectors:
            self._pairs.extend(self.SECTOR_PAIRS)
        if include_commodities:
            self._pairs.extend(self.COMMODITY_PAIRS)
        if include_bonds:
            self._pairs.extend(self.BOND_PAIRS)
        if include_international:
            self._pairs.extend(self.INTERNATIONAL_PAIRS)
        if include_indices:
            self._pairs.extend(self.INDEX_PAIRS)
        if custom_pairs:
            self._pairs.extend(custom_pairs)

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in self._pairs:
            # Normalize pair order
            normalized = tuple(sorted(pair))
            if normalized not in seen:
                seen.add(normalized)
                unique_pairs.append(pair)
        self._pairs = unique_pairs

    @property
    def pairs(self) -> List[Tuple[str, str]]:
        """Get list of all pairs in the universe."""
        return self._pairs.copy()

    @property
    def symbols(self) -> List[str]:
        """Get list of all unique symbols in the universe."""
        symbols = set()
        for pair in self._pairs:
            symbols.add(pair[0])
            symbols.add(pair[1])
        return sorted(list(symbols))

    def add_pair(self, asset1: str, asset2: str) -> None:
        """Add a pair to the universe."""
        pair = (asset1.upper(), asset2.upper())
        if pair not in self._pairs and (pair[1], pair[0]) not in self._pairs:
            self._pairs.append(pair)

    def remove_pair(self, asset1: str, asset2: str) -> bool:
        """Remove a pair from the universe. Returns True if removed."""
        pair = (asset1.upper(), asset2.upper())
        reverse_pair = (asset2.upper(), asset1.upper())

        if pair in self._pairs:
            self._pairs.remove(pair)
            return True
        elif reverse_pair in self._pairs:
            self._pairs.remove(reverse_pair)
            return True
        return False

    def get_pairs_by_symbol(self, symbol: str) -> List[Tuple[str, str]]:
        """Get all pairs containing a specific symbol."""
        symbol = symbol.upper()
        return [p for p in self._pairs if symbol in p]

    def __len__(self) -> int:
        """Return number of pairs in universe."""
        return len(self._pairs)

    def __iter__(self):
        """Iterate over pairs."""
        return iter(self._pairs)

    def __repr__(self) -> str:
        """String representation."""
        return f"PairUniverse({len(self._pairs)} pairs)"


def get_default_universe() -> PairUniverse:
    """Get the default pair universe with all categories."""
    return PairUniverse()


def get_liquid_etf_universe() -> PairUniverse:
    """Get a universe of highly liquid ETF pairs."""
    liquid_pairs = [
        ("SPY", "IWM"),
        ("QQQ", "IWM"),
        ("GLD", "SLV"),
        ("XLF", "KBE"),
        ("XLE", "XOP"),
        ("TLT", "IEF"),
        ("EFA", "EEM"),
    ]
    return PairUniverse(
        include_sectors=False,
        include_commodities=False,
        include_bonds=False,
        include_international=False,
        include_indices=False,
        custom_pairs=liquid_pairs,
    )
