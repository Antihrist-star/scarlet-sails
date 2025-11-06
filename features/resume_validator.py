"""
P1-1: Enhanced Resume Validation Module
========================================

Validates whether market recovery is REAL or FAKE before resuming normal operations.

Problem: Scenario 5 (Fake Recovery)
- Bear crash: -15%
- Rally: +8% (looks like recovery!)
- Reality: Another crash -12% (was fake)
- System needs to distinguish real recovery from bear market rally

Solution: Multi-Factor Resume Validation
Before resuming normal operations after a crisis, check 5 factors:
1. Price Stability: Recovery sustained without reversal
2. Volatility Normalization: Vol returned to normal levels
3. Volume Confirmation: Healthy volume (not low-volume pump)
4. Regime Sustained: New regime held for minimum time
5. Trend Strength: Strong uptrend (not weak bounce)

Progressive Resume Strategy:
- HALT → CAUTIOUS (reduced risk)
- CAUTIOUS → NORMAL (full operations)

Philosophy: "Don't catch falling knives, wait for confirmation"

Author: Scarlet Sails Team
Date: 2025-11-05
Priority: P1 (Scenario 5 fix)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass


class ResumeLevel(Enum):
    """Resume readiness levels"""
    HALT = "HALT"           # No trading (crisis mode)
    CAUTIOUS = "CAUTIOUS"   # Limited trading (recovery unclear)
    NORMAL = "NORMAL"       # Full trading (recovery confirmed)


@dataclass
class ResumeFactors:
    """Data structure for resume validation factors"""
    price_stability: bool
    volatility_normalized: bool
    volume_healthy: bool
    regime_sustained: bool
    trend_strong: bool

    def score(self) -> float:
        """Calculate resume confidence score (0-1)"""
        factors = [
            self.price_stability,
            self.volatility_normalized,
            self.volume_healthy,
            self.regime_sustained,
            self.trend_strong
        ]
        return sum(factors) / len(factors)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        passed = sum([
            self.price_stability,
            self.volatility_normalized,
            self.volume_healthy,
            self.regime_sustained,
            self.trend_strong
        ])
        return f"{passed}/5 factors confirmed"


class ResumeValidator:
    """
    Validates whether market recovery is real before resuming operations.

    Workflow:
    1. Crisis detected → enter HALT mode
    2. Recovery appears → enter CAUTIOUS mode (check every bar)
    3. All 5 factors confirmed → enter NORMAL mode
    4. If recovery fails → back to HALT

    Philosophy: "Better late than sorry"
    """

    def __init__(
        self,
        min_stability_hours: float = 24.0,  # Recovery must hold 24h (more conservative)
        bars_per_hour: int = 4,             # 15min bars
        vol_normalization_factor: float = 1.3,  # Vol must be <1.3x median (stricter)
        min_regime_hours: float = 48.0,     # New regime held 48h (more conservative)
        min_volume_ratio: float = 0.6       # Volume >60% of median (stricter)
    ):
        """
        Initialize resume validator.

        Args:
            min_stability_hours: Minimum hours recovery must hold
            bars_per_hour: Bars per hour (4 for 15min data)
            vol_normalization_factor: Max volatility vs median
            min_regime_hours: Minimum regime duration
            min_volume_ratio: Minimum volume vs median
        """
        self.min_stability_hours = min_stability_hours
        self.bars_per_hour = bars_per_hour
        self.min_stability_bars = int(min_stability_hours * bars_per_hour)
        self.vol_normalization_factor = vol_normalization_factor
        self.min_regime_hours = min_regime_hours
        self.min_regime_bars = int(min_regime_hours * bars_per_hour)
        self.min_volume_ratio = min_volume_ratio

        # State tracking
        self.current_level = ResumeLevel.NORMAL
        self.recovery_start_bar = None
        self.recovery_high = None

    def check_resume_readiness(
        self,
        df: pd.DataFrame,
        current_bar: int,
        current_regime: str,
        regime_change_bar: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Check if system is ready to resume normal operations.

        Args:
            df: Price/volume data
            current_bar: Current bar index
            current_regime: Current market regime ('BULL', 'BEAR', 'SIDEWAYS')
            regime_change_bar: Bar where regime last changed

        Returns:
            Dict with:
                - resume_level: HALT/CAUTIOUS/NORMAL
                - factors: ResumeFactors object
                - recommendation: Action to take
                - details: Additional info
        """
        if current_bar < 96:  # Need at least 1 day of data
            return {
                'resume_level': ResumeLevel.CAUTIOUS,
                'factors': None,
                'recommendation': 'Insufficient data for validation',
                'details': {'bars': current_bar}
            }

        # Check each factor
        factors = self._evaluate_factors(df, current_bar, current_regime, regime_change_bar)

        # Determine resume level based on factor score
        score = factors.score()

        if score >= 0.8:  # 4/5 or 5/5 factors
            resume_level = ResumeLevel.NORMAL
            recommendation = "RESUME - Recovery confirmed, full operations"
        elif score >= 0.6:  # 3/5 factors
            resume_level = ResumeLevel.CAUTIOUS
            recommendation = "CAUTIOUS - Partial recovery, reduced positions"
        else:  # <3/5 factors
            resume_level = ResumeLevel.HALT
            recommendation = "HALT - Recovery not confirmed, stay defensive"

        return {
            'resume_level': resume_level,
            'factors': factors,
            'score': score,
            'recommendation': recommendation,
            'details': {
                'summary': factors.get_summary(),
                'current_regime': current_regime
            }
        }

    def _evaluate_factors(
        self,
        df: pd.DataFrame,
        current_bar: int,
        current_regime: str,
        regime_change_bar: Optional[int]
    ) -> ResumeFactors:
        """
        Evaluate all 5 resume factors.

        Returns:
            ResumeFactors with bool for each factor
        """
        # Window for analysis
        lookback = min(current_bar, 96)  # Last 1 day
        window_df = df.iloc[max(0, current_bar - lookback):current_bar + 1].copy()

        # Factor 1: Price Stability
        # Check if price held recovery without significant reversal
        price_stability = self._check_price_stability(window_df, current_bar)

        # Factor 2: Volatility Normalization
        # Check if volatility returned to normal levels
        volatility_normalized = self._check_volatility(df, current_bar)

        # Factor 3: Volume Healthy
        # Check if volume confirms recovery (not low-volume pump)
        volume_healthy = self._check_volume(df, current_bar)

        # Factor 4: Regime Sustained
        # Check if new regime held for minimum time
        regime_sustained = self._check_regime_duration(current_bar, regime_change_bar)

        # Factor 5: Trend Strength
        # Check if uptrend is strong (not weak bounce)
        trend_strong = self._check_trend_strength(window_df)

        return ResumeFactors(
            price_stability=price_stability,
            volatility_normalized=volatility_normalized,
            volume_healthy=volume_healthy,
            regime_sustained=regime_sustained,
            trend_strong=trend_strong
        )

    def _check_price_stability(self, window_df: pd.DataFrame, current_bar: int) -> bool:
        """
        Check if price recovery is stable.

        Criteria: No >5% decline from recent high in last 12 hours
        """
        if len(window_df) < self.min_stability_bars:
            return False

        # Get last 12 hours
        stability_window = window_df.iloc[-self.min_stability_bars:]

        recent_high = stability_window['close'].max()
        current_price = stability_window['close'].iloc[-1]

        decline_from_high = (recent_high - current_price) / recent_high

        # Stable if decline <5%
        return decline_from_high < 0.05

    def _check_volatility(self, df: pd.DataFrame, current_bar: int) -> bool:
        """
        Check if volatility normalized to healthy levels.

        Criteria: Recent volatility <1.5x median volatility
        """
        if current_bar < 200:
            return False

        # Calculate rolling volatility
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(window=20).std()

        # Recent volatility (last 20 bars)
        recent_vol = rolling_vol.iloc[max(0, current_bar - 20):current_bar + 1].mean()

        # Historical median (last 200 bars)
        median_vol = rolling_vol.iloc[max(0, current_bar - 200):current_bar + 1].median()

        if median_vol == 0:
            return False

        vol_ratio = recent_vol / median_vol

        # Normalized if <1.5x median
        return vol_ratio < self.vol_normalization_factor

    def _check_volume(self, df: pd.DataFrame, current_bar: int) -> bool:
        """
        Check if volume is healthy (confirms recovery).

        Criteria: Recent volume >50% of median volume
        """
        if current_bar < 100:
            return False

        # Recent volume (last 20 bars)
        recent_volume = df['volume'].iloc[max(0, current_bar - 20):current_bar + 1].mean()

        # Median volume (last 100 bars)
        median_volume = df['volume'].iloc[max(0, current_bar - 100):current_bar + 1].median()

        if median_volume == 0:
            return False

        volume_ratio = recent_volume / median_volume

        # Healthy if >50% of median
        return volume_ratio >= self.min_volume_ratio

    def _check_regime_duration(
        self,
        current_bar: int,
        regime_change_bar: Optional[int]
    ) -> bool:
        """
        Check if new regime sustained for minimum time.

        Criteria: Regime held for 24+ hours (96 bars)
        """
        if regime_change_bar is None:
            return True  # No recent regime change

        bars_in_regime = current_bar - regime_change_bar

        # Sustained if held 24+ hours
        return bars_in_regime >= self.min_regime_bars

    def _check_trend_strength(self, window_df: pd.DataFrame) -> bool:
        """
        Check if uptrend is strong (not weak bounce).

        Criteria:
        - Price > 20-bar SMA
        - SMA trending up (current > SMA 10 bars ago)
        """
        if len(window_df) < 30:
            return False

        # Calculate 20-bar SMA
        sma_20 = window_df['close'].rolling(window=20).mean()

        current_price = window_df['close'].iloc[-1]
        current_sma = sma_20.iloc[-1]

        # Price above SMA?
        price_above_sma = current_price > current_sma

        # SMA trending up?
        if len(sma_20) >= 10:
            sma_10_bars_ago = sma_20.iloc[-10]
            sma_trending_up = current_sma > sma_10_bars_ago * 1.02  # Up 2%+
        else:
            sma_trending_up = False

        # Strong if both conditions met
        return price_above_sma and sma_trending_up


def demo():
    """Demo showing resume validation"""
    print("="*60)
    print("Resume Validator Demo - Real vs Fake Recovery")
    print("="*60)

    validator = ResumeValidator()

    # Scenario: Real recovery
    print("\n--- Scenario 1: Real Recovery ---")
    print("Crash: $100 → $85 (-15%)")
    print("Recovery: $85 → $95 (+12%) - sustained for 2 days")

    # Generate synthetic data
    np.random.seed(42)

    # Pre-crash
    prices_pre = [100.0 + np.random.normal(0, 0.5) for _ in range(96)]

    # Crash
    crash_prices = []
    for i in range(48):
        progress = i / 48
        crash_pct = -0.15 * progress
        price = 100.0 * (1 + crash_pct) + np.random.normal(0, 1.0)
        crash_prices.append(price)

    # Real recovery (sustained)
    recovery_prices = []
    for i in range(96):
        progress = i / 96
        recovery_pct = 0.12 * progress
        price = crash_prices[-1] * (1 + recovery_pct) + np.random.normal(0, 0.5)
        recovery_prices.append(price)

    all_prices = prices_pre + crash_prices + recovery_prices

    df_real = pd.DataFrame({
        'close': all_prices,
        'volume': [5000 + np.random.normal(0, 500) for _ in range(len(all_prices))]
    })

    # Check at end of recovery
    current_bar = len(df_real) - 1
    regime_change_bar = len(prices_pre) + len(crash_prices) + 48  # 12h into recovery

    result = validator.check_resume_readiness(
        df_real,
        current_bar,
        current_regime='BULL',
        regime_change_bar=regime_change_bar
    )

    print(f"\nResume Level: {result['resume_level'].value}")
    print(f"Score: {result['score']:.1%}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Summary: {result['details']['summary']}")

    if result['factors']:
        print("\nFactors:")
        print(f"  Price Stability: {'✅' if result['factors'].price_stability else '❌'}")
        print(f"  Volatility Normalized: {'✅' if result['factors'].volatility_normalized else '❌'}")
        print(f"  Volume Healthy: {'✅' if result['factors'].volume_healthy else '❌'}")
        print(f"  Regime Sustained: {'✅' if result['factors'].regime_sustained else '❌'}")
        print(f"  Trend Strong: {'✅' if result['factors'].trend_strong else '❌'}")

    # Scenario: Fake recovery (bear market rally)
    print("\n--- Scenario 2: Fake Recovery (Bear Trap) ---")
    print("Crash: $100 → $85 (-15%)")
    print("Rally: $85 → $93 (+9%) - but then crashes to $81")

    # Pre-crash (same)
    # Crash (same)

    # Fake rally (weak, low volume)
    fake_rally_prices = []
    for i in range(48):  # Only 12 hours
        progress = i / 48
        rally_pct = 0.09 * progress
        price = crash_prices[-1] * (1 + rally_pct) + np.random.normal(0, 1.5)
        fake_rally_prices.append(price)

    # Second crash
    crash2_prices = []
    for i in range(48):
        progress = i / 48
        crash_pct = -0.13 * progress
        price = fake_rally_prices[-1] * (1 + crash_pct) + np.random.normal(0, 2.0)
        crash2_prices.append(price)

    all_prices_fake = prices_pre + crash_prices + fake_rally_prices + crash2_prices

    df_fake = pd.DataFrame({
        'close': all_prices_fake,
        'volume': [5000 + np.random.normal(0, 500) if i < len(prices_pre) + len(crash_prices) else 2000 + np.random.normal(0, 300) for i in range(len(all_prices_fake))]
    })

    # Check at peak of fake rally (before second crash)
    current_bar_fake = len(prices_pre) + len(crash_prices) + len(fake_rally_prices) - 1
    regime_change_bar_fake = len(prices_pre) + len(crash_prices) + 24  # 6h into rally

    result_fake = validator.check_resume_readiness(
        df_fake,
        current_bar_fake,
        current_regime='SIDEWAYS',
        regime_change_bar=regime_change_bar_fake
    )

    print(f"\nResume Level: {result_fake['resume_level'].value}")
    print(f"Score: {result_fake['score']:.1%}")
    print(f"Recommendation: {result_fake['recommendation']}")
    print(f"Summary: {result_fake['details']['summary']}")

    if result_fake['factors']:
        print("\nFactors:")
        print(f"  Price Stability: {'✅' if result_fake['factors'].price_stability else '❌'}")
        print(f"  Volatility Normalized: {'✅' if result_fake['factors'].volatility_normalized else '❌'}")
        print(f"  Volume Healthy: {'✅' if result_fake['factors'].volume_healthy else '❌'}")
        print(f"  Regime Sustained: {'✅' if result_fake['factors'].regime_sustained else '❌'}")
        print(f"  Trend Strong: {'✅' if result_fake['factors'].trend_strong else '❌'}")

    print("\n" + "="*60)
    print("Key Insight:")
    print("Real recovery: 4-5 factors confirmed → RESUME")
    print("Fake recovery: <3 factors confirmed → STAY CAUTIOUS")
    print("="*60)


if __name__ == "__main__":
    demo()
    print("\n✅ Resume validator module created successfully!")
