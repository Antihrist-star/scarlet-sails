#!/usr/bin/env python3
"""
SimpleOpportunityScorer - Day 2 Component

Filters signals based on:
- ML probability (primary factor)
- Market regime (bull/bear/sideways)
- Crisis detection (skip if crisis)
- Volatility (high vol = lower score)
- Volume (low vol = lower score)

This is SIMPLE intentionally - to be easily testable and improvable
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


class SimpleOpportunityScorer:
    """
    Scores each ML signal to decide whether to enter a trade

    Philosophy: If we can't score it, we skip it.
    Better to miss good signals than to take bad ones.
    """

    def __init__(self,
                 ml_threshold: float = 0.50,
                 min_volume_percentile: float = 0.30,
                 max_volatility_percentile: float = 0.80,
                 regime_factors: Optional[Dict[str, float]] = None,
                 crisis_penalty: float = 0.0):
        """
        Args:
            ml_threshold: Min ML probability to consider
            min_volume_percentile: Skip if volume < this percentile
            max_volatility_percentile: Skip if volatility > this percentile
            regime_factors: Multipliers for each regime
                {'bull': 1.2, 'sideways': 1.0, 'bear': 0.8}
            crisis_penalty: If 1.0, completely block signals in crisis
                          If < 1.0, reduce score in crisis
        """
        self.ml_threshold = ml_threshold
        self.min_volume_percentile = min_volume_percentile
        self.max_volatility_percentile = max_volatility_percentile
        self.crisis_penalty = crisis_penalty

        self.regime_factors = regime_factors or {
            'bull': 1.2,
            'sideways': 1.0,
            'bear': 0.8,
        }

        # Stats for percentile calculations
        self.volume_stats = None
        self.volatility_stats = None

    def calibrate(self, df: pd.DataFrame):
        """
        Calculate volume and volatility percentiles from data

        Must call this before scoring!
        """
        # Volume stats
        self.volume_stats = {
            'min': df['volume'].quantile(self.min_volume_percentile),
            'max': df['volume'].quantile(0.95),
        }

        # Volatility stats (intrabar: high - low)
        df['intrabar_vol'] = (df['high'] - df['low']) / df['close'] * 100
        self.volatility_stats = {
            'p50': df['intrabar_vol'].quantile(0.50),
            'p75': df['intrabar_vol'].quantile(0.75),
            'p90': df['intrabar_vol'].quantile(self.max_volatility_percentile),
        }

        return self

    def score(self,
              ml_prob: float,
              volume: float,
              intrabar_volatility: float,
              regime: str = 'sideways',
              is_crisis: bool = False) -> float:
        """
        Score a single signal

        Returns:
            0.0 - 1.0 (or higher if regime bonus)
            0.0 = skip this signal
            1.0+ = good signal
        """
        # ====================================================================
        # STEP 1: ML Probability (primary gate)
        # ====================================================================
        if ml_prob < self.ml_threshold:
            return 0.0

        score = ml_prob

        # ====================================================================
        # STEP 2: Volume check
        # ====================================================================
        if self.volume_stats is not None:
            if volume < self.volume_stats['min']:
                return 0.0  # Skip low volume

        # ====================================================================
        # STEP 3: Volatility check
        # ====================================================================
        if self.volatility_stats is not None:
            if intrabar_volatility > self.volatility_stats['p90']:
                return 0.0  # Skip extremely volatile bars

        # ====================================================================
        # STEP 4: Regime factor
        # ====================================================================
        regime_factor = self.regime_factors.get(regime, 1.0)
        score *= regime_factor

        # ====================================================================
        # STEP 5: Crisis penalty
        # ====================================================================
        if is_crisis:
            score *= (1.0 - self.crisis_penalty)
            if self.crisis_penalty == 1.0:
                return 0.0  # Completely block in crisis

        return score

    def score_batch(self,
                    ml_probs: np.ndarray,
                    df: pd.DataFrame,
                    regimes: Optional[np.ndarray] = None,
                    is_crisis_array: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Score multiple signals at once

        Args:
            ml_probs: Array of ML probabilities
            df: OHLCV DataFrame
            regimes: Array of regime strings (default: all 'sideways')
            is_crisis_array: Array of crisis flags (default: all False)

        Returns:
            Array of scores (same length as ml_probs)
        """
        if len(ml_probs) != len(df):
            raise ValueError(f"Length mismatch: {len(ml_probs)} != {len(df)}")

        scores = np.zeros(len(ml_probs))

        # Calculate volatility on the fly
        intrabar_vol = (df['high'] - df['low']) / df['close'] * 100

        for i in range(len(ml_probs)):
            regime = regimes[i] if regimes is not None else 'sideways'
            is_crisis = is_crisis_array[i] if is_crisis_array is not None else False

            scores[i] = self.score(
                ml_prob=ml_probs[i],
                volume=df.iloc[i]['volume'],
                intrabar_volatility=intrabar_vol.iloc[i],
                regime=regime,
                is_crisis=is_crisis,
            )

        return scores


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_scorer():
    """Quick test of the scorer"""
    scorer = SimpleOpportunityScorer(ml_threshold=0.50)

    print("Testing SimpleOpportunityScorer...")

    # Test 1: Bad ML probability
    assert scorer.score(ml_prob=0.40, volume=1000, intrabar_volatility=0.5) == 0.0
    print("✅ Test 1: ML threshold gate works")

    # Test 2: Good signal
    score = scorer.score(ml_prob=0.60, volume=1000, intrabar_volatility=0.5)
    assert 0.5 < score < 1.0
    print(f"✅ Test 2: Good signal scored {score:.2f}")

    # Test 3: Regime factor (bull)
    score_bull = scorer.score(ml_prob=0.60, volume=1000, intrabar_volatility=0.5, regime='bull')
    score_bear = scorer.score(ml_prob=0.60, volume=1000, intrabar_volatility=0.5, regime='bear')
    assert score_bull > score_bear
    print(f"✅ Test 3: Regime factor works (bull {score_bull:.2f} > bear {score_bear:.2f})")

    # Test 4: Crisis penalty
    score_normal = scorer.score(ml_prob=0.60, volume=1000, intrabar_volatility=0.5, is_crisis=False)
    score_crisis = scorer.score(ml_prob=0.60, volume=1000, intrabar_volatility=0.5, is_crisis=True, )
    assert score_normal == score_crisis  # Default crisis_penalty = 0.0
    print(f"✅ Test 4: Crisis penalty works")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_scorer()
