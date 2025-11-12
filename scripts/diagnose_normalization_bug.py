#!/usr/bin/env python3
"""
NORMALIZATION BUG DIAGNOSTIC

Find exactly which features are not normalized properly.

HYPOTHESIS:
- Scaler trained on MIX of normalized and absolute features
- Some features are absolute prices (bad!)
- Some features are ratios (good!)

TEST:
1. Extract features from BTC 2018-02-19 (first test bar)
2. Compare with scaler statistics
3. Identify which features cause MILLION sigma values
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*100)
print("NORMALIZATION BUG DIAGNOSTIC")
print("="*100)

# Load scaler
scaler_path = MODELS_DIR / "xgboost_normalized_scaler.pkl"
scaler = joblib.load(scaler_path)

print(f"\n✅ Scaler loaded: {scaler_path}")
print(f"   Features: {len(scaler.mean_)} features")

# Load data
extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
all_tf, primary_df = extractor.prepare_multi_timeframe_data("BTC", "15m")

print(f"\n✅ BTC data loaded:")
print(f"   Bars: {len(primary_df)}")
print(f"   Date range: {primary_df.index[0]} to {primary_df.index[-1]}")

# Extract features from FIRST bar (2018-02-19)
# This is the bar that showed 2 MILLION sigma!
test_bar_idx = 10000  # A bar from 2018

features_raw = extractor.extract_features_at_bar(all_tf, "15m", test_bar_idx)

if features_raw is None:
    print("\n❌ Failed to extract features!")
    sys.exit(1)

print(f"\n✅ Features extracted from bar {test_bar_idx}")
print(f"   Timestamp: {primary_df.index[test_bar_idx]}")
print(f"   BTC price: ${primary_df['close'].iloc[test_bar_idx]:,.2f}")

# Feature names (must match training!)
feature_names = [
    'close', '15m_EMA_9', '15m_EMA_21', '15m_SMA_50', '15m_RSI_14',
    '15m_ATR_pct', '15m_MACD', '15m_MACD_signal', '15m_BB_upper', '15m_BB_lower',
    '15m_returns_5', '15m_returns_20', '15m_volatility_20', '15m_price_to_EMA9',
    '15m_volume_ratio', '1h_EMA_9', '1h_EMA_21', '1h_SMA_50', '1h_RSI_14',
    '1h_ATR_pct', '1h_returns_5', '1h_returns_20', '1h_price_to_EMA9',
    '4h_EMA_9', '4h_EMA_21', '4h_SMA_50', '4h_RSI_14', '4h_ATR_pct',
    '4h_returns_5', '4h_returns_20', '4h_price_to_EMA9'
]

# Compare with scaler
print(f"\n{'='*100}")
print("DETAILED FEATURE ANALYSIS")
print(f"{'='*100}")

print(f"\n{'Feature':<25} {'Raw Value':>15} {'Scaler Mean':>15} {'Scaler Std':>15} {'Sigma':>15} {'Status':>10}")
print("-"*100)

problems = []

for i, fname in enumerate(feature_names):
    raw_val = features_raw[i]
    scaler_mean = scaler.mean_[i]
    scaler_std = scaler.scale_[i]

    # Calculate sigma
    sigma = abs((raw_val - scaler_mean) / scaler_std)

    # Detect problems
    status = "✅ OK"

    # Problem 1: Scaler mean is absolute price (should be ~1.0 for ratios)
    if fname not in ['close', 'RSI'] and scaler_mean > 10:
        status = "💀 ABSOLUTE"
        problems.append({
            'feature': fname,
            'issue': 'Scaler mean is absolute price (not normalized)',
            'scaler_mean': scaler_mean,
            'expected': '~1.0 for ratios'
        })

    # Problem 2: Sigma > 1000 (catastrophic)
    if sigma > 1000:
        status = "💀 HUGE σ"
        problems.append({
            'feature': fname,
            'issue': f'Sigma = {sigma:,.0f} (catastrophic)',
            'raw_value': raw_val,
            'scaler_mean': scaler_mean
        })

    # Problem 3: Raw value is absolute price for ratio feature
    if fname not in ['close'] and '_EMA_' in fname and raw_val > 10:
        status = "⚠️ RAW ABS"
        problems.append({
            'feature': fname,
            'issue': 'Raw value is absolute (should be ratio)',
            'raw_value': raw_val,
            'expected': '~1.0'
        })

    print(f"{fname:<25} {raw_val:>15.4f} {scaler_mean:>15.4f} {scaler_std:>15.4f} {sigma:>15.2f} {status:>10}")

# Check for specific known issues
print(f"\n{'='*100}")
print("SPECIFIC CHECKS")
print(f"{'='*100}")

# Check 1: Close feature
close_idx = 0
close_val = features_raw[close_idx]
close_scaler_mean = scaler.mean_[close_idx]

print(f"\n1. CLOSE FEATURE:")
print(f"   Raw value: {close_val:.4f}")
print(f"   Scaler mean: {close_scaler_mean:.4f}")
print(f"   BTC price at bar: ${primary_df['close'].iloc[test_bar_idx]:,.2f}")

if abs(close_val - primary_df['close'].iloc[test_bar_idx]) < 1.0:
    print("   ❌ PROBLEM: 'close' is ABSOLUTE PRICE!")
    print("   Expected: close should be normalized (e.g., log or ratio)")
else:
    print("   ✅ OK: close appears normalized")

# Check 2: EMA features
print(f"\n2. EMA FEATURES:")
ema_9_idx = 1
ema_9_val = features_raw[ema_9_idx]
ema_9_scaler_mean = scaler.mean_[ema_9_idx]

print(f"   15m_EMA_9 raw: {ema_9_val:.4f}")
print(f"   15m_EMA_9 scaler mean: {ema_9_scaler_mean:.4f}")

if ema_9_scaler_mean > 10:
    print("   ❌ PROBLEM: Scaler thinks EMA is ABSOLUTE!")
    print("   This means training data had absolute EMA values")
elif ema_9_val > 10:
    print("   ❌ PROBLEM: Raw EMA is ABSOLUTE but scaler expects ratio!")
    print("   This means inference code doesn't normalize properly")
else:
    print("   ✅ OK: EMA appears normalized")

# Check 3: Returns features
print(f"\n3. RETURNS FEATURES:")
returns_5_idx = 10
returns_5_val = features_raw[returns_5_idx]
returns_5_scaler_mean = scaler.mean_[returns_5_idx]

print(f"   15m_returns_5 raw: {returns_5_val:.6f}")
print(f"   15m_returns_5 scaler mean: {returns_5_scaler_mean:.6f}")

if abs(returns_5_val) > 1.0:
    print("   ❌ PROBLEM: Returns should be small decimals (% change)")
    print("   Value > 1.0 suggests wrong calculation")
else:
    print("   ✅ OK: Returns look reasonable")

# Summary
print(f"\n{'='*100}")
print(f"PROBLEMS FOUND: {len(problems)}")
print(f"{'='*100}")

if problems:
    print("\nDETAILED PROBLEMS:")
    for i, prob in enumerate(problems[:10], 1):  # Show first 10
        print(f"\n{i}. {prob['feature']}")
        print(f"   Issue: {prob['issue']}")
        for key, val in prob.items():
            if key not in ['feature', 'issue']:
                if isinstance(val, float):
                    print(f"   {key}: {val:.4f}")
                else:
                    print(f"   {key}: {val}")

# Root cause analysis
print(f"\n{'='*100}")
print("ROOT CAUSE ANALYSIS")
print(f"{'='*100}")

# Count issues by type
absolute_scaler = sum(1 for p in problems if 'Scaler mean is absolute' in p['issue'])
absolute_raw = sum(1 for p in problems if 'Raw value is absolute' in p['issue'])
huge_sigma = sum(1 for p in problems if 'Sigma' in p['issue'])

print(f"\nIssue distribution:")
print(f"  - Scaler trained on absolute features: {absolute_scaler}")
print(f"  - Raw features are absolute: {absolute_raw}")
print(f"  - Huge sigma values: {huge_sigma}")

if absolute_scaler > 5:
    print(f"\n💀 ROOT CAUSE: TRAINING DATA BUG")
    print(f"   Location: retrain_xgboost_normalized.py")
    print(f"   Problem: Training used absolute features instead of normalized")
    print(f"   Fix: Review NormalizedMultiTFExtractor in training code")
elif absolute_raw > 5:
    print(f"\n💀 ROOT CAUSE: INFERENCE CODE BUG")
    print(f"   Location: MultiTimeframeFeatureExtractor.extract_features_at_bar()")
    print(f"   Problem: Inference returns absolute features")
    print(f"   Fix: Check that inference uses same normalization as training")
else:
    print(f"\n❓ ROOT CAUSE: UNCLEAR")
    print(f"   Need to check both training and inference code")

print(f"\n{'='*100}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*100}")
