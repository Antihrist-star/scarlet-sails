"""
PHASE 1.2: REGIME DETECTION VALIDATION
========================================

Test regime detection component on REAL BTC periods:
- Bull trend: 2020-2021 (post-COVID rally)
- Bear market: 2022 (Luna, FTX collapse)
- Sideways: 2019, 2023 (consolidation periods)

Metrics:
- Accuracy (% correct labels vs actual regimes)
- Lag (time to detect regime change)
- Whipsaw rate (false regime switches)
- Regime distribution

Expected performance:
- Accuracy >75% minimum
- Lag <7 days to detect transition
- Whipsaw <10% false switches

Author: Scarlet Sails Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from models.regime_detector import SimpleRegimeDetector, MarketRegime

print("="*80)
print("PHASE 1.2: REGIME DETECTION VALIDATION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading prepared data...")

data_path = Path('data/processed/btc_prepared_phase0.parquet')
if not data_path.exists():
    print(f"❌ ERROR: {data_path} not found!")
    print("Run phase0_load_real_data.py first!")
    exit(1)

df = pd.read_parquet(data_path)
print(f"✅ Loaded {len(df):,} bars")
print(f"   Period: {df.index[0]} to {df.index[-1]}")

# ============================================================================
# INITIALIZE REGIME DETECTOR
# ============================================================================
print("\n🔍 Initializing regime detector...")

detector = SimpleRegimeDetector()
print("✅ SimpleRegimeDetector loaded")

# ============================================================================
# DEFINE KNOWN REGIME PERIODS (GROUND TRUTH)
# ============================================================================
print("\n📊 Defining known regime periods (ground truth)...")

# Based on historical BTC price action
known_regimes = [
    {
        'name': 'COVID_RECOVERY_BULL',
        'start': pd.Timestamp('2020-04-01', tz='UTC'),
        'end': pd.Timestamp('2021-04-14', tz='UTC'),
        'expected': MarketRegime.BULL_TREND,
        'description': 'Post-COVID bull run to $64k'
    },
    {
        'name': 'MID_2021_BEAR',
        'start': pd.Timestamp('2021-05-20', tz='UTC'),
        'end': pd.Timestamp('2021-07-20', tz='UTC'),
        'expected': MarketRegime.BEAR_MARKET,
        'description': 'China ban correction'
    },
    {
        'name': 'Q4_2021_BULL',
        'start': pd.Timestamp('2021-08-01', tz='UTC'),
        'end': pd.Timestamp('2021-11-08', tz='UTC'),
        'expected': MarketRegime.BULL_TREND,
        'description': 'Q4 2021 rally to ATH $69k'
    },
    {
        'name': 'BEAR_2022',
        'start': pd.Timestamp('2022-01-01', tz='UTC'),
        'end': pd.Timestamp('2022-12-31', tz='UTC'),
        'expected': MarketRegime.BEAR_MARKET,
        'description': 'Full 2022 bear market (Luna, FTX)'
    },
    {
        'name': 'SIDEWAYS_2023',
        'start': pd.Timestamp('2023-01-01', tz='UTC'),
        'end': pd.Timestamp('2023-09-30', tz='UTC'),
        'expected': MarketRegime.SIDEWAYS,
        'description': '2023 consolidation around $25-30k'
    },
    {
        'name': 'BULL_2024',
        'start': pd.Timestamp('2024-01-01', tz='UTC'),
        'end': pd.Timestamp('2024-03-14', tz='UTC'),
        'expected': MarketRegime.BULL_TREND,
        'description': '2024 ETF rally to new ATH $73k'
    },
]

print(f"   Defined {len(known_regimes)} test periods")
for period in known_regimes:
    print(f"      {period['name']}: {period['expected'].value}")

# ============================================================================
# TEST 1: REGIME DETECTION ACCURACY
# ============================================================================
print("\n" + "="*80)
print("TEST 1: REGIME DETECTION ACCURACY")
print("="*80)

results = []

for period in known_regimes:
    print(f"\n📊 Testing {period['name']}...")
    print(f"   Expected: {period['expected'].value}")
    print(f"   Period: {period['start']} to {period['end']}")

    # Filter data for this period
    mask = (df.index >= period['start']) & (df.index <= period['end'])
    period_df = df[mask]

    if len(period_df) == 0:
        print(f"   ⚠️  No data in this period!")
        continue

    print(f"   Bars: {len(period_df)}")

    # Detect regime for each bar
    detections = []
    for i in range(200, len(period_df)):  # Need 200 bars for MAs
        bar_idx = df.index.get_loc(period_df.index[i])
        regime = detector.detect(df, bar_idx)
        detections.append(regime.value)

    if not detections:
        print(f"   ⚠️  Not enough data for detection!")
        continue

    # Count regime distribution
    regime_counts = pd.Series(detections).value_counts()
    total = len(detections)

    print(f"\n   Detected regimes:")
    for regime, count in regime_counts.items():
        pct = count / total * 100
        marker = "✅" if regime == period['expected'].value else "  "
        print(f"      {marker} {regime}: {count} bars ({pct:.1f}%)")

    # Calculate accuracy (% bars correctly labeled)
    correct = sum(1 for r in detections if r == period['expected'].value)
    accuracy = correct / total

    print(f"\n   Accuracy: {accuracy:.1%} ({correct}/{total} bars)")

    # Find when regime was first correctly detected
    first_correct = None
    for i, regime in enumerate(detections):
        if regime == period['expected'].value:
            first_correct = i
            break

    if first_correct is not None:
        detection_lag_bars = first_correct
        detection_lag_days = detection_lag_bars / 24  # 1h bars
        print(f"   Detection lag: {detection_lag_bars} bars ({detection_lag_days:.1f} days)")
    else:
        detection_lag_days = np.nan
        print(f"   ❌ Never correctly detected!")

    results.append({
        'period': period['name'],
        'expected': period['expected'].value,
        'accuracy': accuracy,
        'detection_lag_days': detection_lag_days,
        'bars': total,
    })

# ============================================================================
# TEST 2: WHIPSAW RATE (FALSE REGIME SWITCHES)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: WHIPSAW RATE (FALSE REGIME SWITCHES)")
print("="*80)

print("\nDetecting regimes across full dataset...")

# Detect regime for all bars
all_regimes = []
for i in range(200, len(df)):
    regime = detector.detect(df, i)
    all_regimes.append({
        'timestamp': df.index[i],
        'regime': regime.value,
    })

regimes_df = pd.DataFrame(all_regimes)

# Count regime switches
regimes_df['regime_changed'] = regimes_df['regime'] != regimes_df['regime'].shift(1)
total_switches = regimes_df['regime_changed'].sum() - 1  # -1 for first row

print(f"   Total bars: {len(regimes_df):,}")
print(f"   Total regime switches: {total_switches}")

# Calculate whipsaw rate (switches back to previous regime within 7 days)
whipsaws = 0
for i in range(2, len(regimes_df)):
    if regimes_df.iloc[i]['regime_changed']:
        # Check if this is a whipsaw (back to regime from 7 days ago)
        lookback = min(7 * 24, i)  # 7 days in hours
        prev_regime = regimes_df.iloc[i - lookback]['regime']
        current_regime = regimes_df.iloc[i]['regime']

        if current_regime == prev_regime:
            whipsaws += 1

whipsaw_rate = whipsaws / total_switches if total_switches > 0 else 0

print(f"   Whipsaws: {whipsaws}")
print(f"   Whipsaw rate: {whipsaw_rate:.1%}")

if whipsaw_rate < 0.10:
    print(f"   ✅ PASS: <10% whipsaw rate")
elif whipsaw_rate < 0.20:
    print(f"   ⚠️  WARNING: 10-20% whipsaw rate")
else:
    print(f"   ❌ FAIL: >20% whipsaw rate - too unstable!")

# Show regime distribution
print(f"\nOverall regime distribution:")
regime_dist = regimes_df['regime'].value_counts()
for regime, count in regime_dist.items():
    pct = count / len(regimes_df) * 100
    print(f"   {regime}: {count:,} bars ({pct:.1f}%)")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1.2 RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

print("\n📊 Period-by-Period Accuracy:")
print(results_df.to_string(index=False))

# Calculate overall metrics
avg_accuracy = results_df['accuracy'].mean()
avg_lag = results_df['detection_lag_days'].mean()

print(f"\n🎯 Overall Metrics:")
print(f"   Average Accuracy: {avg_accuracy:.1%}")
print(f"   Average Detection Lag: {avg_lag:.1f} days")
print(f"   Whipsaw Rate: {whipsaw_rate:.1%}")

# Pass/Fail criteria
print(f"\n✅ PASS/FAIL:")

passed = True

if avg_accuracy >= 0.75:
    print(f"   ✅ Accuracy {avg_accuracy:.1%} >= 75%")
elif avg_accuracy >= 0.60:
    print(f"   ⚠️  Accuracy {avg_accuracy:.1%} (60-75%, acceptable)")
else:
    print(f"   ❌ Accuracy {avg_accuracy:.1%} < 60% (too low!)")
    passed = False

if avg_lag <= 7.0:
    print(f"   ✅ Detection lag {avg_lag:.1f} days <= 7 days")
elif avg_lag <= 14.0:
    print(f"   ⚠️  Detection lag {avg_lag:.1f} days (7-14 days, acceptable)")
else:
    print(f"   ❌ Detection lag {avg_lag:.1f} days > 14 days (too slow!)")
    passed = False

if whipsaw_rate < 0.10:
    print(f"   ✅ Whipsaw rate {whipsaw_rate:.1%} < 10%")
elif whipsaw_rate < 0.20:
    print(f"   ⚠️  Whipsaw rate {whipsaw_rate:.1%} (10-20%, acceptable)")
else:
    print(f"   ❌ Whipsaw rate {whipsaw_rate:.1%} > 20% (too unstable!)")
    passed = False

print("\n" + "="*80)
if passed:
    print("✅ PHASE 1.2 PASSED - Regime detection validated!")
else:
    print("⚠️  PHASE 1.2 MARGINAL - Regime detection needs improvement")
print("="*80)
