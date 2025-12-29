"""Tests for signal generation."""

import pytest
from datetime import datetime

from src.signal_generation.signals import SignalGenerator, SignalType, TradingSignal


class TestSignalGenerator:
    """Tests for SignalGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a signal generator instance."""
        return SignalGenerator(
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_threshold=3.5,
        )

    @pytest.fixture
    def pair(self):
        """Sample pair."""
        return ("XLF", "KBE")

    def test_no_position_no_signal(self, generator, pair):
        """Test that no signal is generated when z-score is within thresholds."""
        signal = generator.generate_signal(
            pair=pair,
            zscore=1.0,  # Within entry threshold
            hedge_ratio=0.85,
            spread=0.5,
        )

        assert signal.signal_type == SignalType.HOLD

    def test_long_entry_signal(self, generator, pair):
        """Test long entry signal generation."""
        signal = generator.generate_signal(
            pair=pair,
            zscore=-2.5,  # Below negative entry threshold
            hedge_ratio=0.85,
            spread=-1.5,
        )

        assert signal.signal_type == SignalType.LONG_SPREAD
        assert signal.zscore == -2.5

    def test_short_entry_signal(self, generator, pair):
        """Test short entry signal generation."""
        signal = generator.generate_signal(
            pair=pair,
            zscore=2.5,  # Above positive entry threshold
            hedge_ratio=0.85,
            spread=1.5,
        )

        assert signal.signal_type == SignalType.SHORT_SPREAD
        assert signal.zscore == 2.5

    def test_long_exit_signal(self, generator, pair):
        """Test exit signal from long position."""
        # First enter a long position
        generator.set_position(pair, 1)

        signal = generator.generate_signal(
            pair=pair,
            zscore=-0.3,  # Above negative exit threshold
            hedge_ratio=0.85,
            spread=0.2,
        )

        assert signal.signal_type == SignalType.EXIT

    def test_short_exit_signal(self, generator, pair):
        """Test exit signal from short position."""
        # First enter a short position
        generator.set_position(pair, -1)

        signal = generator.generate_signal(
            pair=pair,
            zscore=0.3,  # Below positive exit threshold
            hedge_ratio=0.85,
            spread=0.2,
        )

        assert signal.signal_type == SignalType.EXIT

    def test_long_stop_loss(self, generator, pair):
        """Test stop loss from long position."""
        generator.set_position(pair, 1)

        signal = generator.generate_signal(
            pair=pair,
            zscore=-4.0,  # Below negative stop threshold
            hedge_ratio=0.85,
            spread=-2.5,
        )

        assert signal.signal_type == SignalType.STOP_LOSS

    def test_short_stop_loss(self, generator, pair):
        """Test stop loss from short position."""
        generator.set_position(pair, -1)

        signal = generator.generate_signal(
            pair=pair,
            zscore=4.0,  # Above positive stop threshold
            hedge_ratio=0.85,
            spread=2.5,
        )

        assert signal.signal_type == SignalType.STOP_LOSS

    def test_process_signal_updates_position(self, generator, pair):
        """Test that processing signal updates internal position."""
        # Generate and process long entry
        signal = generator.generate_signal(
            pair=pair,
            zscore=-2.5,
            hedge_ratio=0.85,
            spread=-1.5,
        )
        generator.process_signal(signal)

        assert generator.get_position(pair) == 1

        # Generate and process exit
        signal = generator.generate_signal(
            pair=pair,
            zscore=-0.3,
            hedge_ratio=0.85,
            spread=0.2,
        )
        generator.process_signal(signal)

        assert generator.get_position(pair) == 0

    def test_signal_to_dict(self, generator, pair):
        """Test signal serialization."""
        signal = generator.generate_signal(
            pair=pair,
            zscore=-2.5,
            hedge_ratio=0.85,
            spread=-1.5,
        )

        signal_dict = signal.to_dict()

        assert signal_dict["signal_type"] == "LONG_SPREAD"
        assert signal_dict["pair"] == pair
        assert signal_dict["zscore"] == -2.5
        assert "timestamp" in signal_dict

    def test_reset(self, generator, pair):
        """Test generator reset."""
        generator.set_position(pair, 1)
        assert generator.get_position(pair) == 1

        generator.reset()
        assert generator.get_position(pair) == 0
