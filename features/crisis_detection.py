"""
P0-2: Multi-Timeframe Crisis Detection System
==============================================

Implements:
1. Multi-timeframe regime classification (1d, 7d, 30d)
2. Consensus voting (requires 2/3 agreement)
3. Hysteresis mechanism (2-day confirmation to prevent whipsaw)
4. Cumulative loss tracking (detects slow-burn crises like Scenario 4)

Author: Scarlet Sails Team
Date: 2025-11-05
Priority: P0 (CRITICAL)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from enum import Enum
from datetime import datetime, timedelta
from features.resume_validator import ResumeValidator, ResumeLevel


class Regime(Enum):
    """Market regime classification"""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class AlertLevel(Enum):
    """Alert severity levels"""
    NORMAL = "NORMAL"
    ALERT = "ALERT"  # Warning - approaching crisis
    CRISIS = "CRISIS"  # Crisis detected - halt trading
    HALT = "HALT"  # Emergency halt - severe crisis


class MultiTimeframeDetector:
    """
    Multi-timeframe crisis detection with consensus voting.

    Key Innovation: Uses 3 different timeframes (1d, 7d, 30d) to detect crises.
    - Fast timeframes catch sudden crashes (COVID March 2020)
    - Slow timeframes catch gradual declines (Scenario 4: -5% daily over 5 days)
    - Consensus voting prevents false alarms
    """

    def __init__(
        self,
        timeframes: Dict[str, int] = None,
        regime_thresholds: Dict[str, float] = None,
        crisis_thresholds: Dict[str, float] = None,
        hysteresis_periods: int = 2
    ):
        """
        Initialize multi-timeframe detector.

        Args:
            timeframes: Dict of {name: bars} for detection windows
                       Default: {'1d': 96, '7d': 672, '30d': 2880} (for 15min bars)
            regime_thresholds: Thresholds for BULL/BEAR classification
                              Default: {'bull': 0.02, 'bear': -0.02}
            crisis_thresholds: Thresholds for crisis detection
                              Default: {'alert_1h': -0.10, 'crisis_24h': -0.20, 'crisis_7d': -0.30}
            hysteresis_periods: Days to confirm regime change (default: 2)
        """
        # Timeframes in bars (assuming 15min data: 96 bars = 1 day)
        self.timeframes = timeframes or {
            '1d': 96,      # 24 hours
            '7d': 672,     # 7 days
            '30d': 2880    # 30 days
        }

        # Regime classification thresholds
        self.regime_thresholds = regime_thresholds or {
            'bull': 0.02,   # +2% return = BULL
            'bear': -0.02   # -2% return = BEAR
        }

        # Crisis detection thresholds
        self.crisis_thresholds = crisis_thresholds or {
            'alert_1h': -0.10,     # -10% in 1h = ALERT
            'alert_7d': -0.12,     # -12% cumulative = ALERT (slow burn warning)
            'crisis_24h': -0.20,   # -20% in 24h = CRISIS (sudden crash)
            'crisis_7d': -0.25,    # -25% in 7d = CRISIS (gradual decline)
            'halt_1h': -0.25       # -25% in 1h = EMERGENCY HALT
        }

        # Hysteresis state tracking
        self.hysteresis_periods = hysteresis_periods
        self.regime_history: List[Regime] = []
        self.last_confirmed_regime: Optional[Regime] = None
        self.candidate_regime: Optional[Regime] = None
        self.candidate_count: int = 0
        self.regime_change_bar: Optional[int] = None
        self.current_bar: int = 0
        self.recovery_start_bar: Optional[int] = None  # Track when recovery started

        # Resume validation (for detecting fake recoveries)
        self.resume_validator = ResumeValidator()

    def calculate_return(self, df: pd.DataFrame, window: int, use_all_available: bool = False) -> float:
        """
        Calculate return over specified window.

        Args:
            df: Price dataframe with 'close' column
            window: Lookback window in bars
            use_all_available: If True and df length < window, use all available data

        Returns:
            Return as decimal (e.g., 0.05 = 5% gain)
        """
        if len(df) < window:
            if not use_all_available or len(df) < 2:
                return 0.0
            # Use all available data
            current_price = df['close'].iloc[-1]
            past_price = df['close'].iloc[0]
        else:
            current_price = df['close'].iloc[-1]
            past_price = df['close'].iloc[-window]

        if past_price == 0:
            return 0.0

        return (current_price - past_price) / past_price

    def classify_regime(self, return_pct: float) -> Regime:
        """
        Classify regime based on return.

        Args:
            return_pct: Return as decimal

        Returns:
            Regime classification
        """
        if return_pct > self.regime_thresholds['bull']:
            return Regime.BULL
        elif return_pct < self.regime_thresholds['bear']:
            return Regime.BEAR
        else:
            return Regime.SIDEWAYS

    def detect_regime_multitime(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Detect regime using multi-timeframe consensus voting.

        Algorithm:
        1. Calculate returns for each timeframe (1d, 7d, 30d)
        2. Classify each timeframe independently
        3. Vote: require 2/3 agreement for confidence
        4. Return regime with confidence score

        Args:
            df: Price dataframe with 'close' column

        Returns:
            Dict with:
                - regime: Consensus regime (BULL/BEAR/SIDEWAYS/UNKNOWN)
                - confidence: 0.33, 0.66, or 1.0 based on agreement
                - timeframes: Dict of {timeframe: regime} for each window
                - returns: Dict of {timeframe: return_pct} for debugging
        """
        # Calculate returns for each timeframe
        returns = {}
        regimes = {}

        for name, window in self.timeframes.items():
            ret = self.calculate_return(df, window)
            returns[name] = ret
            regimes[name] = self.classify_regime(ret)

        # Count votes for each regime
        regime_votes = {
            Regime.BULL: 0,
            Regime.BEAR: 0,
            Regime.SIDEWAYS: 0
        }

        for regime in regimes.values():
            regime_votes[regime] += 1

        # Find consensus regime
        total_votes = len(regimes)
        max_votes = max(regime_votes.values())
        consensus_regime = max(regime_votes, key=regime_votes.get)

        # Calculate confidence (2/3 = 0.66, 3/3 = 1.0, 1/3 = 0.33)
        confidence = max_votes / total_votes

        # If no clear majority (1/3 split), mark as UNKNOWN
        if confidence <= 0.34:
            consensus_regime = Regime.UNKNOWN

        return {
            'regime': consensus_regime,
            'confidence': confidence,
            'timeframes': regimes,
            'returns': returns,
            'votes': regime_votes
        }

    def apply_hysteresis(self, new_regime: Regime, timestamp: datetime = None) -> Tuple[Regime, bool]:
        """
        Apply hysteresis to prevent regime switching on noise.

        Algorithm:
        1. If new regime == current confirmed regime: continue
        2. If new regime != current confirmed regime:
           - Track candidate regime
           - Increment counter
           - If counter >= hysteresis_periods: CONFIRM SWITCH
        3. If regime changes before confirmation: reset counter

        Args:
            new_regime: Newly detected regime
            timestamp: Current timestamp (for logging)

        Returns:
            Tuple of (confirmed_regime, regime_changed_flag)
        """
        # First detection - initialize
        if self.last_confirmed_regime is None:
            self.last_confirmed_regime = new_regime
            self.candidate_regime = None
            self.candidate_count = 0
            return new_regime, True

        # No change - continue with confirmed regime
        if new_regime == self.last_confirmed_regime:
            self.candidate_regime = None
            self.candidate_count = 0
            return self.last_confirmed_regime, False

        # New candidate regime detected
        if new_regime != self.candidate_regime:
            # Reset if candidate changes
            self.candidate_regime = new_regime
            self.candidate_count = 1
            # Return current confirmed regime (no change yet)
            return self.last_confirmed_regime, False

        # Candidate regime persists
        self.candidate_count += 1

        # Confirm regime change if threshold reached
        if self.candidate_count >= self.hysteresis_periods:
            old_regime = self.last_confirmed_regime
            self.last_confirmed_regime = new_regime
            self.candidate_regime = None
            self.candidate_count = 0
            self.regime_change_bar = self.current_bar  # Track when regime changed

            print(f"⚠️ REGIME CHANGE CONFIRMED: {old_regime.value} → {new_regime.value}")
            if timestamp:
                print(f"   Timestamp: {timestamp}")

            return new_regime, True

        # Still in confirmation period
        return self.last_confirmed_regime, False

    def calculate_crisis_score(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate crisis score using cumulative loss tracking.

        This is the KEY FIX for Scenario 4 (Gradual Crisis).
        Tracks both:
        - Fast crashes (-20% in 24h)
        - Slow burns (-30% in 7d, even if each day is only -5%)

        Args:
            df: Price dataframe with 'close' column

        Returns:
            Dict with:
                - alert_level: NORMAL/ALERT/CRISIS/HALT
                - scores: Dict of {timeframe: loss_pct}
                - triggered_rules: List of rules that fired
                - message: Human-readable explanation
        """
        scores = {}
        triggered_rules = []

        # 1-hour window (fast crash detection)
        if len(df) >= 4:  # 4 bars = 1 hour for 15min data
            score_1h = self.calculate_return(df, 4)
            scores['1h'] = score_1h

            if score_1h <= self.crisis_thresholds['halt_1h']:
                triggered_rules.append(f"EMERGENCY: {score_1h:.2%} loss in 1h")
            elif score_1h <= self.crisis_thresholds['alert_1h']:
                triggered_rules.append(f"WARNING: {score_1h:.2%} loss in 1h")

        # 24-hour window (sudden crash detection)
        if len(df) >= self.timeframes['1d']:
            score_24h = self.calculate_return(df, self.timeframes['1d'])
            scores['24h'] = score_24h

            if score_24h <= self.crisis_thresholds['crisis_24h']:
                triggered_rules.append(f"CRISIS: {score_24h:.2%} loss in 24h")

        # 7-day window (gradual decline detection) ⭐ FIXES SCENARIO 4
        # Use cumulative tracking from inception if we don't have full 7d yet
        if len(df) >= self.timeframes['7d']:
            score_7d = self.calculate_return(df, self.timeframes['7d'])
            scores['7d'] = score_7d

            if score_7d <= self.crisis_thresholds['crisis_7d']:
                triggered_rules.append(f"CRISIS: {score_7d:.2%} cumulative loss in 7d")
            elif score_7d <= self.crisis_thresholds['alert_7d']:
                triggered_rules.append(f"WARNING: {score_7d:.2%} cumulative loss in 7d")
        elif len(df) >= 96:  # If we have at least 1 day of data
            # Use cumulative return from inception for early detection
            score_cumulative = self.calculate_return(df, len(df), use_all_available=True)
            scores['cumulative'] = score_cumulative

            if score_cumulative <= self.crisis_thresholds['crisis_7d']:
                triggered_rules.append(f"CRISIS: {score_cumulative:.2%} cumulative loss from inception")
            elif score_cumulative <= self.crisis_thresholds['alert_7d']:
                triggered_rules.append(f"WARNING: {score_cumulative:.2%} cumulative loss from inception")

        # Determine alert level
        alert_level = AlertLevel.NORMAL

        if any('EMERGENCY' in rule for rule in triggered_rules):
            alert_level = AlertLevel.HALT
        elif any('CRISIS' in rule for rule in triggered_rules):
            alert_level = AlertLevel.CRISIS
        elif any('WARNING' in rule for rule in triggered_rules):
            alert_level = AlertLevel.ALERT

        # Generate message
        if alert_level == AlertLevel.NORMAL:
            message = "✅ Market conditions normal"
        else:
            message = f"🚨 {alert_level.value}: {'; '.join(triggered_rules)}"

        return {
            'alert_level': alert_level,
            'scores': scores,
            'triggered_rules': triggered_rules,
            'message': message
        }

    def detect(self, df: pd.DataFrame, timestamp: datetime = None) -> Dict[str, any]:
        """
        Main detection function - combines all components.

        Workflow:
        1. Multi-timeframe regime detection (with consensus)
        2. Apply hysteresis (prevent false switches)
        3. Calculate crisis score (cumulative loss)
        4. Resume validation (check if recovery is real)
        5. Return comprehensive analysis

        Args:
            df: Price dataframe with 'close' column
            timestamp: Current timestamp

        Returns:
            Dict with complete analysis:
                - regime_analysis: Multi-timeframe regime data
                - confirmed_regime: Regime after hysteresis
                - regime_changed: Boolean flag
                - crisis_analysis: Crisis score and alert level
                - resume_analysis: Resume validation (if applicable)
                - recommendation: Trading action (CONTINUE/CAUTION/HALT)
        """
        # Update current bar tracking
        self.current_bar = len(df) - 1

        # Step 1: Multi-timeframe regime detection
        regime_analysis = self.detect_regime_multitime(df)

        # Step 2: Apply hysteresis
        old_regime = self.last_confirmed_regime
        confirmed_regime, regime_changed = self.apply_hysteresis(
            regime_analysis['regime'],
            timestamp
        )

        # Step 3: Calculate crisis score
        crisis_analysis = self.calculate_crisis_score(df)

        # Step 4: Resume validation (for detecting fake recoveries)
        resume_analysis = None

        # Check if recovering from BEAR market OR from significant drawdown
        recovering_from_bear = old_regime == Regime.BEAR and confirmed_regime in [Regime.SIDEWAYS, Regime.BULL]

        # Also check if recovering from recent significant drawdown (even if regime didn't officially become BEAR)
        recovering_from_drawdown = False
        if len(df) >= 288:  # At least 3 days of data
            # Check if there was a significant recent drawdown
            lookback_window = min(len(df), 672)  # Use up to 7 days of history
            high_recent = df['close'].iloc[-lookback_window:].max()
            current_price = df['close'].iloc[-1]
            drawdown = (current_price - high_recent) / high_recent

            # If there was a >10% drawdown recently, validate any recovery
            if drawdown < -0.10:
                # Check if currently recovering (price above recent low)
                low_recent = df['close'].iloc[-lookback_window:].min()
                if current_price > low_recent * 1.03:  # Price >3% above recent low = recovery attempt
                    recovering_from_drawdown = True

                    # Track when recovery started (if not already tracked)
                    if self.recovery_start_bar is None:
                        # Find the bar where price was at recent low
                        low_bar_offset = df['close'].iloc[-lookback_window:].idxmin()
                        self.recovery_start_bar = self.current_bar - (len(df) - 1 - low_bar_offset)
            else:
                # No significant drawdown anymore - reset recovery tracking
                self.recovery_start_bar = None

        if recovering_from_bear or recovering_from_drawdown:
            # Potential recovery - validate if it's real
            # Use recovery_start_bar if available, otherwise regime_change_bar
            reference_bar = self.recovery_start_bar if self.recovery_start_bar is not None else self.regime_change_bar

            resume_analysis = self.resume_validator.check_resume_readiness(
                df=df,
                current_bar=self.current_bar,
                current_regime=confirmed_regime.value,
                regime_change_bar=reference_bar
            )

        # Step 5: Generate recommendation
        if crisis_analysis['alert_level'] == AlertLevel.HALT:
            recommendation = "HALT"
            action = "🛑 EMERGENCY HALT - Stop all trading immediately"
        elif crisis_analysis['alert_level'] == AlertLevel.CRISIS:
            recommendation = "HALT"
            action = "🚨 CRISIS DETECTED - Halt trading, evaluate positions"
        elif crisis_analysis['alert_level'] == AlertLevel.ALERT:
            recommendation = "CAUTION"
            action = "⚠️ CAUTION - Reduce position sizes, monitor closely"
        else:
            # Normal conditions - but check resume validation
            if resume_analysis:
                # Recovering from BEAR - use resume validator recommendation
                if resume_analysis['resume_level'] == ResumeLevel.HALT:
                    recommendation = "HALT"
                    action = f"🚫 HALT - Recovery not confirmed ({resume_analysis['details']['summary']})"
                elif resume_analysis['resume_level'] == ResumeLevel.CAUTIOUS:
                    recommendation = "CAUTION"
                    action = f"⚠️ CAUTIOUS - Partial recovery ({resume_analysis['details']['summary']})"
                else:  # NORMAL
                    recommendation = "CONTINUE"
                    action = f"✅ RESUME - Recovery confirmed ({resume_analysis['details']['summary']})"
            else:
                # Not in recovery - normal operations
                recommendation = "CONTINUE"
                action = "✅ CONTINUE - Normal trading conditions"

        return {
            'timestamp': timestamp or datetime.now(),
            'regime_analysis': regime_analysis,
            'confirmed_regime': confirmed_regime,
            'regime_changed': regime_changed,
            'crisis_analysis': crisis_analysis,
            'resume_analysis': resume_analysis,  # New: resume validation for fake recovery detection
            'recommendation': recommendation,
            'action': action,

            # Summary for logging
            'summary': {
                'regime': confirmed_regime.value,
                'confidence': regime_analysis['confidence'],
                'alert_level': crisis_analysis['alert_level'].value,
                'action': recommendation,
                'resume_level': resume_analysis['resume_level'].value if resume_analysis else None
            }
        }


def demo():
    """Demo showing how to use the detector"""
    print("="*60)
    print("Multi-Timeframe Crisis Detector Demo")
    print("="*60)

    # Create sample data - simulating Scenario 4 (gradual decline)
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='15min')
    np.random.seed(42)

    # Simulate gradual decline: -5% per day over 5 days
    prices = [100.0]
    for i in range(1, 1000):
        # Gradual decline: -0.0005 per bar = -0.05% per bar
        # Over 96 bars (1 day) = -4.8% per day
        trend = -0.0005
        noise = np.random.normal(0, 0.001)
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)

    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })

    # Test detector at different points
    detector = MultiTimeframeDetector()

    test_points = [
        ('Day 1', 96),
        ('Day 3', 96*3),
        ('Day 5', 96*5),
        ('Day 7', 96*7),
    ]

    for label, bar in test_points:
        if bar >= len(df):
            continue

        test_df = df.iloc[:bar]
        result = detector.detect(test_df, df['timestamp'].iloc[bar-1])

        print(f"\n{label} (Bar {bar}):")
        print(f"  Price: ${test_df['close'].iloc[-1]:.2f} (from $100.00)")
        print(f"  Regime: {result['confirmed_regime'].value} (confidence: {result['regime_analysis']['confidence']:.2f})")
        print(f"  Alert: {result['crisis_analysis']['alert_level'].value}")
        print(f"  Action: {result['recommendation']}")
        if result['crisis_analysis']['triggered_rules']:
            print(f"  Rules: {', '.join(result['crisis_analysis']['triggered_rules'])}")


if __name__ == "__main__":
    demo()
    print("\n✅ Crisis detection module created successfully!")
