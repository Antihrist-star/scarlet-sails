"""
Hybrid Entry System - Day 1 Implementation
==========================================

3-Layer unified entry system combining:
- Layer 1: Rule-based signals (RSI < 30)
- Layer 2: ML filter (XGBoost)
- Layer 3: Crisis gate (MultiTimeframeDetector)

Author: Scarlet Sails Team
Date: 2025-11-10
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from models.xgboost_model import XGBoostModel
from features.crisis_detection import MultiTimeframeDetector
from models.regime_detector import SimpleRegimeDetector
from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor


class HybridEntrySystem:
    """
    3-Layer Hybrid Entry System

    Architecture:
    ┌─────────────────────────────────┐
    │  Layer 1: Rule-Based Signal    │ ← RSI < 30
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  Layer 2: ML Filter             │ ← XGBoost probability
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │  Layer 3: Crisis Gate           │ ← Halt during crashes
    └────────────┬────────────────────┘
                 │
                 ▼
             [EXECUTE]

    Usage:
        >>> system = HybridEntrySystem(ml_threshold=0.6)
        >>> should_enter, reason = system.should_enter(df, bar_index)
        >>> if should_enter:
        ...     # Execute trade
    """

    def __init__(
        self,
        ml_threshold: float = 0.6,
        ml_model_path: str = 'models/xgboost_model.json',
        rsi_threshold: float = 30.0,
        min_bars_between_signals: int = 24,
        crisis_sensitivity: str = 'medium',
        enable_ml_filter: bool = True,
        enable_crisis_gate: bool = True,
        all_timeframes: Optional[Dict[str, pd.DataFrame]] = None,
        target_timeframe: Optional[str] = None
    ):
        """
        Initialize Hybrid Entry System

        Args:
            ml_threshold: ML probability threshold (0-1, default: 0.6)
                         Higher = fewer but better signals
            ml_model_path: Path to trained XGBoost model
            rsi_threshold: RSI oversold threshold (default: 30)
            min_bars_between_signals: Min bars between signals (prevent clustering)
            crisis_sensitivity: Crisis detector sensitivity ('low', 'medium', 'high')
            enable_ml_filter: Enable Layer 2 ML filtering
            enable_crisis_gate: Enable Layer 3 crisis protection
            all_timeframes: Dict of DataFrames for all 4 TFs (15m, 1h, 4h, 1d)
                           Required for ML feature extraction
            target_timeframe: Target timeframe being tested (e.g., '15m')
        """
        # Config
        self.ml_threshold = ml_threshold
        self.rsi_threshold = rsi_threshold
        self.min_bars_between_signals = min_bars_between_signals
        self.enable_ml_filter = enable_ml_filter
        self.enable_crisis_gate = enable_crisis_gate

        # Multi-timeframe data for feature extraction
        self.all_timeframes = all_timeframes
        self.target_timeframe = target_timeframe

        # Feature extractor for multi-TF features
        self.feature_extractor = MultiTimeframeFeatureExtractor()

        # Layer 2: ML Filter
        self.ml_model = None
        if enable_ml_filter:
            try:
                model_path = Path(ml_model_path)
                if model_path.exists():
                    self.ml_model = XGBoostModel()
                    self.ml_model.load(str(model_path))
                    print(f"✅ ML model loaded from {model_path}")
                else:
                    print(f"⚠️  ML model not found at {model_path}")
                    print(f"   Layer 2 (ML filter) will be disabled")
                    self.enable_ml_filter = False
            except Exception as e:
                print(f"❌ Failed to load ML model: {e}")
                print(f"   Layer 2 (ML filter) will be disabled")
                self.enable_ml_filter = False

        # Layer 3: Crisis Gate
        self.crisis_detector = None
        if enable_crisis_gate:
            try:
                self.crisis_detector = MultiTimeframeDetector()
                print(f"✅ Crisis detector initialized")
            except Exception as e:
                print(f"❌ Failed to initialize crisis detector: {e}")
                print(f"   Layer 3 (crisis gate) will be disabled")
                self.enable_crisis_gate = False

        # Regime detector (for context)
        self.regime_detector = SimpleRegimeDetector()

        # State tracking
        self.last_signal_bar: Optional[int] = None
        self.total_checks = 0
        self.layer1_passed = 0
        self.layer2_passed = 0
        self.layer3_passed = 0

    def should_enter(
        self,
        df: pd.DataFrame,
        bar_index: int,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, str]:
        """
        Check if we should enter at this bar

        Args:
            df: OHLCV DataFrame with indicators
            bar_index: Current bar index
            timestamp: Current timestamp (optional)

        Returns:
            Tuple of (should_enter: bool, reason: str)

        Example:
            >>> should_enter, reason = system.should_enter(df, 1000)
            >>> print(f"Enter: {should_enter}, Reason: {reason}")
            Enter: True, Reason: All layers passed (ML: 0.72)
        """
        self.total_checks += 1

        # ========================================
        # LAYER 1: Rule-based signal check
        # ========================================
        rule_passed, rule_reason = self._check_rule_signal(df, bar_index)
        if not rule_passed:
            return False, rule_reason

        self.layer1_passed += 1

        # ========================================
        # LAYER 2: ML Filter
        # ========================================
        if self.enable_ml_filter and self.ml_model is not None:
            ml_score, ml_passed = self._check_ml_filter(df, bar_index)

            if not ml_passed:
                return False, f"ML rejected (score: {ml_score:.2f} < {self.ml_threshold:.2f})"

            self.layer2_passed += 1
            ml_score_str = f"ML: {ml_score:.2f}"
        else:
            ml_score_str = "ML: disabled"

        # ========================================
        # LAYER 3: Crisis Gate
        # ========================================
        if self.enable_crisis_gate and self.crisis_detector is not None:
            crisis_halt, crisis_reason = self._check_crisis(df, bar_index, timestamp)

            if crisis_halt:
                return False, f"Crisis gate: {crisis_reason}"

            self.layer3_passed += 1

        # ========================================
        # ALL LAYERS PASSED → ENTER!
        # ========================================
        self.last_signal_bar = bar_index
        return True, f"✅ All layers passed ({ml_score_str})"

    def _check_rule_signal(
        self,
        df: pd.DataFrame,
        bar_index: int
    ) -> Tuple[bool, str]:
        """
        Layer 1: Check RSI < 30 rule

        Returns:
            (passed: bool, reason: str)
        """
        # Need enough history
        if bar_index < 200:
            return False, "Insufficient history (need 200+ bars)"

        # Check RSI (support multiple naming conventions)
        rsi_col = None

        # Try different RSI column names
        if 'rsi' in df.columns:
            rsi_col = 'rsi'
        elif 'RSI_14' in df.columns:
            rsi_col = 'RSI_14'
        elif '15m_RSI_14' in df.columns:
            rsi_col = '15m_RSI_14'
        else:
            # Search for any column with 'RSI' in name
            rsi_candidates = [col for col in df.columns if 'RSI' in col.upper()]
            if rsi_candidates:
                rsi_col = rsi_candidates[0]

        if rsi_col is None:
            return False, f"RSI indicator not found (checked: rsi, RSI_14, 15m_RSI_14, and others)"

        rsi = df[rsi_col].iloc[bar_index]

        if rsi >= self.rsi_threshold:
            return False, f"RSI {rsi:.1f} >= {self.rsi_threshold} (not oversold)"

        # Check min bars between signals (prevent clustering)
        if self.last_signal_bar is not None:
            bars_since_last = bar_index - self.last_signal_bar
            if bars_since_last < self.min_bars_between_signals:
                return False, f"Too soon after last signal ({bars_since_last} < {self.min_bars_between_signals} bars)"

        # Rule signal passed!
        return True, f"RSI {rsi:.1f} < {self.rsi_threshold}"

    def _check_ml_filter(
        self,
        df: pd.DataFrame,
        bar_index: int
    ) -> Tuple[float, bool]:
        """
        Layer 2: ML filter using XGBoost

        Returns:
            (ml_score: float, passed: bool)

        TODO (Day 2): Extract correct features in proper format!
        Currently using placeholder feature extraction.
        """
        try:
            # Extract features for ML model
            features = self._extract_features(df, bar_index)

            if features is None:
                return 0.0, False

            # Get ML prediction
            # Model output shape: (1, 2) where [:, 1] is P(UP)
            proba = self.ml_model.predict_proba(features.reshape(1, -1))
            ml_score = proba[0][1]  # Probability of UP class

            # Check threshold
            passed = ml_score >= self.ml_threshold

            return ml_score, passed

        except Exception as e:
            print(f"⚠️  ML filter error: {e}")
            return 0.0, False

    def _check_crisis(
        self,
        df: pd.DataFrame,
        bar_index: int,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, str]:
        """
        Layer 3: Crisis detection gate

        Returns:
            (should_halt: bool, reason: str)
        """
        try:
            # Slice dataframe up to current bar
            df_slice = df.iloc[:bar_index+1]

            # Detect crisis
            result = self.crisis_detector.detect(df_slice, timestamp)

            # Check recommendation
            if result['recommendation'] in ['HALT', 'CAUTION']:
                return True, result['action']

            return False, "No crisis detected"

        except Exception as e:
            print(f"⚠️  Crisis detector error: {e}")
            # On error, be conservative: don't halt
            return False, f"Crisis check failed: {e}"

    def _extract_features(
        self,
        df: pd.DataFrame,
        bar_index: int
    ) -> Optional[np.ndarray]:
        """
        Extract features for ML model using Multi-Timeframe data

        The trained XGBoost model expects 31 features from all 4 timeframes:
        - 13 features from primary timeframe (15m)
        - 18 features from higher timeframes (1h, 4h, 1d)

        Returns:
            np.array of shape (31,) or None if error
        """
        try:
            # Check if multi-TF data is available
            if self.all_timeframes is None or self.target_timeframe is None:
                # Fallback to old placeholder method (will give random predictions)
                print("⚠️  Multi-TF data not available, using placeholder features")
                return self._extract_features_fallback(df, bar_index)

            # Use proper multi-TF feature extraction
            features = self.feature_extractor.extract_features_at_bar(
                all_timeframes=self.all_timeframes,
                target_tf=self.target_timeframe,
                bar_index=bar_index
            )

            return features

        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return None

    def _extract_features_fallback(
        self,
        df: pd.DataFrame,
        bar_index: int
    ) -> Optional[np.ndarray]:
        """
        FALLBACK: Old placeholder feature extraction

        WARNING: This gives random predictions! Only use for testing without multi-TF data.

        Returns:
            np.array of shape (31,) or None if error
        """
        try:
            # OLD PLACEHOLDER CODE (commented out for brevity)

            features = []

            # Feature 1-5: RSI and related
            rsi_col = 'rsi' if 'rsi' in df.columns else 'RSI_14'
            features.append(df[rsi_col].iloc[bar_index])
            features.append(df[rsi_col].iloc[bar_index] / 100.0)  # Normalized RSI

            rsi_ma = df[rsi_col].iloc[bar_index-10:bar_index+1].mean()
            features.append(rsi_ma)
            features.append(df[rsi_col].iloc[bar_index] - rsi_ma)  # RSI deviation

            rsi_slope = df[rsi_col].iloc[bar_index] - df[rsi_col].iloc[bar_index-5]
            features.append(rsi_slope)

            # Feature 6-10: Price and volume
            close = df['close'].iloc[bar_index]
            features.append(close)

            volume_ratio = df['volume'].iloc[bar_index] / df['volume'].iloc[bar_index-20:bar_index].mean()
            features.append(volume_ratio)

            # ATR ratio
            atr_col = 'atr' if 'atr' in df.columns else 'ATR_14'
            if atr_col in df.columns:
                atr_ratio = df[atr_col].iloc[bar_index] / close
                features.append(atr_ratio)
            else:
                features.append(0.02)  # Default

            # Returns
            ret_5 = (close - df['close'].iloc[bar_index-5]) / df['close'].iloc[bar_index-5]
            features.append(ret_5)

            ret_20 = (close - df['close'].iloc[bar_index-20]) / df['close'].iloc[bar_index-20]
            features.append(ret_20)

            # Feature 11-15: Moving averages
            ma_20 = df['close'].iloc[bar_index-20:bar_index+1].mean()
            features.append(close / ma_20)

            ma_50 = df['close'].iloc[bar_index-50:bar_index+1].mean()
            features.append(close / ma_50)

            features.append(ma_20 / ma_50)

            ema_9 = df['close'].iloc[bar_index-9:bar_index+1].ewm(span=9, adjust=False).mean().iloc[-1]
            features.append(close / ema_9)

            ema_21 = df['close'].iloc[bar_index-21:bar_index+1].ewm(span=21, adjust=False).mean().iloc[-1]
            features.append(ema_9 / ema_21)

            # Feature 16-20: Volatility
            volatility_20 = df['close'].iloc[bar_index-20:bar_index+1].pct_change().std()
            features.append(volatility_20)

            volatility_50 = df['close'].iloc[bar_index-50:bar_index+1].pct_change().std()
            features.append(volatility_50)
            features.append(volatility_20 / (volatility_50 + 1e-8))

            # High/Low range
            high_20 = df['high'].iloc[bar_index-20:bar_index+1].max()
            low_20 = df['low'].iloc[bar_index-20:bar_index+1].min()
            features.append((close - low_20) / (high_20 - low_20 + 1e-8))
            features.append((high_20 - close) / (high_20 - low_20 + 1e-8))

            # Feature 21-25: Regime indicators
            regime = self.regime_detector.detect(df, bar_index)
            regime_encoding = {
                'Bull Trend': 1.0,
                'Sideways/Choppy': 0.0,
                'Bear Market': -1.0,
                'Crisis Event': -2.0
            }
            features.append(regime_encoding.get(regime.value, 0.0))

            # Drawdown
            high_50 = df['high'].iloc[bar_index-50:bar_index+1].max()
            drawdown = (close - high_50) / high_50
            features.append(drawdown)

            # Momentum
            momentum_10 = (close - df['close'].iloc[bar_index-10]) / df['close'].iloc[bar_index-10]
            features.append(momentum_10)

            momentum_30 = (close - df['close'].iloc[bar_index-30]) / df['close'].iloc[bar_index-30]
            features.append(momentum_30)

            # Acceleration
            features.append(momentum_10 - momentum_30)

            # Feature 26-31: Additional features
            # Candle body
            candle_body = abs(df['close'].iloc[bar_index] - df['open'].iloc[bar_index]) / df['open'].iloc[bar_index]
            features.append(candle_body)

            # Price vs high/low
            features.append((close - df['low'].iloc[bar_index]) / (df['high'].iloc[bar_index] - df['low'].iloc[bar_index] + 1e-8))

            # Volume momentum
            vol_ma_short = df['volume'].iloc[bar_index-5:bar_index+1].mean()
            vol_ma_long = df['volume'].iloc[bar_index-20:bar_index+1].mean()
            features.append(vol_ma_short / (vol_ma_long + 1e-8))

            # Distance from MA200
            if bar_index >= 200:
                ma_200 = df['close'].iloc[bar_index-200:bar_index+1].mean()
                features.append(close / ma_200)
            else:
                features.append(1.0)

            # Consecutive candles
            consecutive_down = 0
            for i in range(1, min(6, bar_index)):
                if df['close'].iloc[bar_index-i] < df['open'].iloc[bar_index-i]:
                    consecutive_down += 1
                else:
                    break
            features.append(float(consecutive_down))

            # Final feature: trend strength
            if bar_index >= 100:
                trend_100 = (close - df['close'].iloc[bar_index-100]) / df['close'].iloc[bar_index-100]
                features.append(trend_100)
            else:
                features.append(0.0)

            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)

            # Check shape
            if len(features_array) != 31:
                print(f"⚠️  Feature count mismatch: got {len(features_array)}, expected 31")
                # Pad or truncate if needed
                if len(features_array) < 31:
                    features_array = np.pad(features_array, (0, 31 - len(features_array)))
                else:
                    features_array = features_array[:31]

            # Replace NaN/Inf with 0
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)

            return features_array

        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return None

    def reset_state(self):
        """
        Reset internal state between different assets/tests

        CRITICAL: Must be called before testing each new asset!
        Otherwise last_signal_bar from previous asset will interfere.
        """
        self.last_signal_bar = None
        # Don't reset total_checks/layer counts - keep for statistics

    def get_statistics(self) -> dict:
        """
        Get statistics about signal filtering

        Returns:
            Dict with filtering statistics
        """
        if self.total_checks == 0:
            return {
                'total_checks': 0,
                'layer1_pass_rate': 0.0,
                'layer2_pass_rate': 0.0,
                'layer3_pass_rate': 0.0,
                'overall_pass_rate': 0.0
            }

        layer1_rate = self.layer1_passed / self.total_checks
        layer2_rate = self.layer2_passed / self.layer1_passed if self.layer1_passed > 0 else 0.0
        layer3_rate = self.layer3_passed / self.layer2_passed if self.layer2_passed > 0 else 0.0
        overall_rate = self.layer3_passed / self.total_checks

        return {
            'total_checks': self.total_checks,
            'layer1_passed': self.layer1_passed,
            'layer2_passed': self.layer2_passed,
            'layer3_passed': self.layer3_passed,
            'layer1_pass_rate': layer1_rate,
            'layer2_pass_rate': layer2_rate,
            'layer3_pass_rate': layer3_rate,
            'overall_pass_rate': overall_rate,
            'reduction_rate': 1.0 - overall_rate
        }

    def print_statistics(self):
        """Print filtering statistics"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("HYBRID ENTRY SYSTEM STATISTICS")
        print("="*60)
        print(f"Total checks:     {stats['total_checks']:,}")
        print(f"\nLayer 1 (Rules):  {stats['layer1_passed']:,} passed ({stats['layer1_pass_rate']:.1%})")
        print(f"Layer 2 (ML):     {stats['layer2_passed']:,} passed ({stats['layer2_pass_rate']:.1%} of Layer 1)")
        print(f"Layer 3 (Crisis): {stats['layer3_passed']:,} passed ({stats['layer3_pass_rate']:.1%} of Layer 2)")
        print(f"\nOverall:          {stats['layer3_passed']:,} / {stats['total_checks']:,} passed ({stats['overall_pass_rate']:.1%})")
        print(f"Signal reduction: {stats['reduction_rate']:.1%}")
        print("="*60)

    def __repr__(self):
        """String representation"""
        ml_status = "enabled" if self.enable_ml_filter and self.ml_model else "disabled"
        crisis_status = "enabled" if self.enable_crisis_gate and self.crisis_detector else "disabled"

        return (
            f"HybridEntrySystem(\n"
            f"  Layer 1: RSI < {self.rsi_threshold}\n"
            f"  Layer 2: ML filter {ml_status} (threshold: {self.ml_threshold})\n"
            f"  Layer 3: Crisis gate {crisis_status}\n"
            f"  Signals processed: {self.total_checks}\n"
            f"  Signals passed: {self.layer3_passed}\n"
            f")"
        )


# ============================================
# UNIT TESTS
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("HYBRID ENTRY SYSTEM - UNIT TESTS")
    print("="*60)

    # Test 1: Initialization
    print("\n1. Testing initialization...")
    system = HybridEntrySystem(
        ml_threshold=0.6,
        enable_ml_filter=True,
        enable_crisis_gate=True
    )
    print(f"   ✅ System initialized")
    print(system)

    # Test 2: Create sample data
    print("\n2. Creating sample data...")
    np.random.seed(42)
    n_bars = 1000

    sample_df = pd.DataFrame({
        'open': np.random.randn(n_bars).cumsum() + 10000,
        'high': np.random.randn(n_bars).cumsum() + 10050,
        'low': np.random.randn(n_bars).cumsum() + 9950,
        'close': np.random.randn(n_bars).cumsum() + 10000,
        'volume': np.random.uniform(1000, 10000, n_bars),
        'rsi': np.random.uniform(20, 80, n_bars),
        'atr': np.random.uniform(50, 200, n_bars),
    })

    # Add some oversold RSI bars
    oversold_bars = [300, 320, 500, 520, 750]
    sample_df.loc[oversold_bars, 'rsi'] = np.random.uniform(15, 29, len(oversold_bars))

    print(f"   ✅ Created {len(sample_df)} bars of sample data")
    print(f"   Oversold bars (RSI < 30): {len(oversold_bars)}")

    # Test 3: Check signals
    print("\n3. Testing signal generation...")

    signals_found = 0
    for bar_idx in oversold_bars:
        should_enter, reason = system.should_enter(sample_df, bar_idx)
        if should_enter:
            signals_found += 1
            print(f"   ✅ Bar {bar_idx}: {reason}")
        else:
            print(f"   ❌ Bar {bar_idx}: {reason}")

    print(f"\n   Signals generated: {signals_found} / {len(oversold_bars)}")

    # Test 4: Statistics
    print("\n4. Testing statistics...")
    system.print_statistics()

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\n⚠️  NOTE: Day 2 task - Implement proper feature extraction!")
    print("   Current feature extraction is placeholder only.")
    print("   Need to match features with trained XGBoost model.")
