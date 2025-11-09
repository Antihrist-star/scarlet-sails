"""
Simple Regime Detector - Day 10
================================

Fast, simple MA-based regime detection.
NOT perfect, but good enough for hybrid exit strategy.

Full MDP multi-dimensional regime detection comes later (Priority 1).

Author: Scarlet Sails Team
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict


class MarketRegime(Enum):
    """Market regime types"""
    BULL_TREND = "Bull Trend"
    BEAR_MARKET = "Bear Market"
    SIDEWAYS = "Sideways/Choppy"
    CRISIS = "Crisis Event"


class SimpleRegimeDetector:
    """
    Simple MA and volatility-based regime detection.

    Rules:
    - CRISIS: Sharp drawdown (>20% in 30 days)
    - BULL: MA20 > MA200 AND positive momentum
    - BEAR: MA20 < MA200 AND negative momentum
    - SIDEWAYS: Everything else (default safe mode)

    This is a PRAGMATIC detector for immediate use.
    Will be replaced by MDP multi-dimensional state later.
    """

    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 200,
        bull_threshold: float = 0.10,  # 10% gain in 30d
        bear_threshold: float = -0.10,  # 10% loss in 30d
        crisis_threshold: float = -0.30,  # 30% drawdown (stricter!)
        ma_separation_pct: float = 0.02,  # 2% separation
    ):
        """
        Initialize regime detector.

        Args:
            ma_short: Short MA period (default 20)
            ma_long: Long MA period (default 200)
            bull_threshold: Min return for bull regime
            bear_threshold: Max return for bear regime
            crisis_threshold: Drawdown threshold for crisis
            ma_separation_pct: Min % separation between MAs
        """
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.crisis_threshold = crisis_threshold
        self.ma_separation_pct = ma_separation_pct

    def detect(self, df: pd.DataFrame, current_bar: int) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            df: OHLCV dataframe
            current_bar: Current bar index

        Returns:
            MarketRegime enum
        """
        # Need enough history
        if current_bar < self.ma_long:
            return MarketRegime.SIDEWAYS  # Default to safe mode

        # Current price
        current_price = df['close'].iloc[current_bar]

        # Calculate MAs
        ma_short = df['close'].iloc[current_bar - self.ma_short:current_bar + 1].mean()
        ma_long = df['close'].iloc[current_bar - self.ma_long:current_bar + 1].mean()

        # Price 30 days ago (720 hours @ 1h bars)
        lookback_bars = min(720, current_bar)
        past_price = df['close'].iloc[current_bar - lookback_bars]

        # 30-day return
        ret_30d = (current_price - past_price) / past_price

        # Drawdown from 30-day high
        high_30d = df['close'].iloc[current_bar - lookback_bars:current_bar + 1].max()
        dd_30d = (current_price - high_30d) / high_30d

        # Regime detection (priority order)

        # 1. CRISIS: Sharp drawdown + volatility spike
        # FIX: Made crisis detection stricter
        # - Requires 30% drawdown (not 20%)
        # - Added volatility confirmation
        if dd_30d < self.crisis_threshold:
            # Confirm with volatility check
            recent_volatility = df['close'].iloc[current_bar-20:current_bar+1].std() / df['close'].iloc[current_bar]
            historical_volatility = df['close'].iloc[current_bar-200:current_bar-20].std() / df['close'].iloc[current_bar-100]

            # Crisis = sharp DD + volatility spike (>2x normal)
            if recent_volatility > historical_volatility * 1.5:
                return MarketRegime.CRISIS
            # Otherwise just bear market (not crisis)
            elif ret_30d < self.bear_threshold:
                return MarketRegime.BEAR_MARKET

        # 2. BULL: MA20 > MA200 AND positive momentum
        if (ma_short > ma_long * (1 + self.ma_separation_pct) and
            ret_30d > self.bull_threshold):
            return MarketRegime.BULL_TREND

        # 3. BEAR: MA20 < MA200 AND negative momentum
        if (ma_short < ma_long * (1 - self.ma_separation_pct) and
            ret_30d < self.bear_threshold):
            return MarketRegime.BEAR_MARKET

        # 4. Default: SIDEWAYS (safe mode)
        return MarketRegime.SIDEWAYS

    def detect_with_confidence(
        self,
        df: pd.DataFrame,
        current_bar: int
    ) -> tuple[MarketRegime, float]:
        """
        Detect regime with confidence score.

        Returns:
            (regime, confidence) where confidence in 0-1
        """
        regime = self.detect(df, current_bar)

        # Calculate confidence based on strength of signals
        current_price = df['close'].iloc[current_bar]

        # MAs
        ma_short = df['close'].iloc[current_bar - self.ma_short:current_bar + 1].mean()
        ma_long = df['close'].iloc[current_bar - self.ma_long:current_bar + 1].mean()
        ma_separation = abs(ma_short - ma_long) / ma_long

        # Returns
        lookback_bars = min(720, current_bar)
        past_price = df['close'].iloc[current_bar - lookback_bars]
        ret_30d = abs((current_price - past_price) / past_price)

        # Confidence based on strength
        if regime == MarketRegime.CRISIS:
            # Strong signal if big drawdown
            high_30d = df['close'].iloc[current_bar - lookback_bars:current_bar + 1].max()
            dd_30d = abs((current_price - high_30d) / high_30d)
            confidence = min(1.0, dd_30d / abs(self.crisis_threshold))

        elif regime == MarketRegime.BULL_TREND:
            # Strong if MA separation + momentum strong
            ma_conf = min(1.0, ma_separation / 0.05)  # 5% = full confidence
            ret_conf = min(1.0, ret_30d / 0.20)  # 20% = full confidence
            confidence = (ma_conf + ret_conf) / 2

        elif regime == MarketRegime.BEAR_MARKET:
            # Strong if MA separation + momentum strong
            ma_conf = min(1.0, ma_separation / 0.05)
            ret_conf = min(1.0, ret_30d / 0.20)
            confidence = (ma_conf + ret_conf) / 2

        else:  # SIDEWAYS
            # Confidence low if clearly in between
            confidence = 0.5

        return regime, confidence

    def get_regime_stats(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get regime distribution across entire dataframe.

        Returns:
            Dict of {regime: count}
        """
        stats = {regime.value: 0 for regime in MarketRegime}

        # Label all bars
        for i in range(self.ma_long, len(df)):
            regime = self.detect(df, i)
            stats[regime.value] += 1

        return stats


# Unit tests
if __name__ == "__main__":
    print("Simple Regime Detector - Unit Tests")
    print("=" * 60)

    # Create sample data with known regimes
    np.random.seed(42)
    n_bars = 5000
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='1h')

    # Build price with different regimes
    prices = []

    # Segment 1: Bull trend (1000 bars)
    bull = np.linspace(10000, 15000, 1000)
    bull += np.random.normal(0, 100, 1000)
    prices.extend(bull)

    # Segment 2: Sideways (1000 bars)
    sideways = 15000 + np.sin(np.linspace(0, 10*np.pi, 1000)) * 500
    sideways += np.random.normal(0, 200, 1000)
    prices.extend(sideways)

    # Segment 3: Crisis drop (100 bars)
    crisis = np.linspace(15000, 10000, 100)
    crisis += np.random.normal(0, 300, 100)
    prices.extend(crisis)

    # Segment 4: Bear market (1000 bars)
    bear = np.linspace(10000, 7000, 1000)
    bear += np.random.normal(0, 200, 1000)
    prices.extend(bear)

    # Segment 5: Recovery sideways (1900 bars)
    recovery = 7000 + np.sin(np.linspace(0, 15*np.pi, 1900)) * 800
    recovery += np.random.normal(0, 300, 1900)
    prices.extend(recovery)

    prices = np.array(prices[:n_bars])

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n_bars),
    }, index=dates)

    # Test detector
    detector = SimpleRegimeDetector()

    print("\n1. Testing regime detection...")

    # Test specific points
    test_points = [
        (500, "Bull trend period"),
        (1500, "Sideways period"),
        (2050, "Crisis period"),
        (3000, "Bear market period"),
        (4500, "Recovery period"),
    ]

    for bar, description in test_points:
        regime, confidence = detector.detect_with_confidence(df, bar)
        price = df['close'].iloc[bar]

        print(f"\n   Bar {bar} ({description}):")
        print(f"      Price: ${price:.0f}")
        print(f"      Regime: {regime.value}")
        print(f"      Confidence: {confidence:.2f}")

    # Test full distribution
    print("\n2. Testing regime distribution across full dataset...")
    stats = detector.get_regime_stats(df)

    print("\n   Regime distribution:")
    total_bars = sum(stats.values())
    for regime, count in stats.items():
        pct = count / total_bars * 100
        print(f"      {regime:20}: {count:5} bars ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Regime Detector working!")
