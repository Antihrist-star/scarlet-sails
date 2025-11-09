"""
PHASE 1.3: ENTRY SIGNAL VALIDATION
====================================

Test entry signal quality on REAL BTC data (2017-2025):
- Signal count and frequency
- Entry accuracy (did price move favorably after entry?)
- Signal distribution across regimes
- Dangerous entries (signals before crashes)

Metrics:
- Signals per year (expect 100-200)
- Entry accuracy: % entries followed by +5% move within 7 days
- Regime distribution (are we entering at right times?)
- Pre-crash signals: <5% entries before known crashes

Expected performance:
- 100-200 signals per year
- >60% entry accuracy (+5% within 7 days)
- Well distributed across regimes
- <5% dangerous pre-crash entries

Author: Scarlet Sails Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("PHASE 1.3: ENTRY SIGNAL VALIDATION")
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

years = (df.index[-1] - df.index[0]).days / 365.25
print(f"   Years: {years:.1f}")

# ============================================================================
# GENERATE ENTRY SIGNALS (SIMPLE RSI STRATEGY)
# ============================================================================
print("\n📊 Generating entry signals...")
print("   Strategy: RSI < 30 (oversold)")

# Find entry signals
entry_signals = []

for i in range(200, len(df)):
    # RSI < 30 = oversold
    if df['rsi'].iloc[i] < 30:
        # Avoid signals too close together (min 24h spacing)
        if not entry_signals or (i - entry_signals[-1]['bar_index'] > 24):
            entry_signals.append({
                'bar_index': i,
                'timestamp': df.index[i],
                'price': df['close'].iloc[i],
                'rsi': df['rsi'].iloc[i],
            })

print(f"   ✅ Generated {len(entry_signals)} entry signals")
signals_per_year = len(entry_signals) / years
print(f"   Signals per year: {signals_per_year:.1f}")

if 100 <= signals_per_year <= 200:
    print(f"   ✅ PASS: 100-200 signals/year target")
elif 50 <= signals_per_year <= 300:
    print(f"   ⚠️  WARNING: Outside optimal range (50-300)")
else:
    print(f"   ❌ FAIL: Too few/many signals!")

# ============================================================================
# TEST 1: ENTRY ACCURACY
# ============================================================================
print("\n" + "="*80)
print("TEST 1: ENTRY ACCURACY (+5% MOVE WITHIN 7 DAYS)")
print("="*80)

print("\nAnalyzing price movement after each entry...")

entry_outcomes = []

for signal in entry_signals:
    entry_bar = signal['bar_index']
    entry_price = signal['price']

    # Look ahead 7 days (168 hours)
    lookback_bars = min(168, len(df) - entry_bar - 1)

    if lookback_bars < 24:  # Skip if too close to end
        continue

    # Find max price in next 7 days
    future_prices = df['close'].iloc[entry_bar+1:entry_bar+1+lookback_bars]
    max_price = future_prices.max()
    max_gain = (max_price - entry_price) / entry_price

    # Did we hit +5% within 7 days?
    success = max_gain >= 0.05

    # How long to hit target (if hit)?
    if success:
        target_price = entry_price * 1.05
        hit_bars = 0
        for j in range(len(future_prices)):
            if future_prices.iloc[j] >= target_price:
                hit_bars = j + 1
                break
        hit_hours = hit_bars
    else:
        hit_hours = np.nan

    entry_outcomes.append({
        'timestamp': signal['timestamp'],
        'entry_price': entry_price,
        'max_gain': max_gain,
        'success': success,
        'hit_hours': hit_hours,
    })

outcomes_df = pd.DataFrame(entry_outcomes)

# Calculate metrics
success_rate = outcomes_df['success'].mean()
avg_max_gain = outcomes_df['max_gain'].mean()
avg_hit_time = outcomes_df[outcomes_df['success']]['hit_hours'].mean()

print(f"\n📊 Entry Accuracy Results:")
print(f"   Total entries analyzed: {len(outcomes_df)}")
print(f"   Success rate (+5% within 7d): {success_rate:.1%}")
print(f"   Average max gain in 7d: {avg_max_gain:+.1%}")

if not pd.isna(avg_hit_time):
    print(f"   Average time to +5%: {avg_hit_time:.0f}h ({avg_hit_time/24:.1f} days)")

if success_rate >= 0.60:
    print(f"   ✅ PASS: Success rate >= 60%")
elif success_rate >= 0.50:
    print(f"   ⚠️  WARNING: Success rate 50-60% (marginal)")
else:
    print(f"   ❌ FAIL: Success rate < 50% (worse than random!)")

# ============================================================================
# TEST 2: SIGNAL DISTRIBUTION ACROSS TIME
# ============================================================================
print("\n" + "="*80)
print("TEST 2: SIGNAL DISTRIBUTION ACROSS TIME")
print("="*80)

# Group signals by year
signals_df = pd.DataFrame(entry_signals)
signals_df['year'] = signals_df['timestamp'].dt.year

signals_by_year = signals_df.groupby('year').size()

print(f"\n📊 Signals per year:")
for year, count in signals_by_year.items():
    print(f"   {year}: {count} signals")

# Check if distribution is reasonable
std_signals = signals_by_year.std()
mean_signals = signals_by_year.mean()
cv = std_signals / mean_signals if mean_signals > 0 else 0

print(f"\n   Mean: {mean_signals:.0f} signals/year")
print(f"   Std Dev: {std_signals:.0f}")
print(f"   Coefficient of Variation: {cv:.2f}")

if cv < 0.5:
    print(f"   ✅ PASS: Consistent signal generation")
else:
    print(f"   ⚠️  WARNING: High variability in signal frequency")

# ============================================================================
# TEST 3: PRE-CRASH ENTRIES (DANGEROUS SIGNALS)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: PRE-CRASH ENTRIES (DANGEROUS SIGNALS)")
print("="*80)

print("\nChecking entries before known crashes...")

# Check if any entries happened within 7 days BEFORE crash events
crash_periods = df[df['is_crash'] == True]

dangerous_entries = 0
for signal in entry_signals:
    signal_time = signal['timestamp']

    # Check if crash started within 7 days after entry
    seven_days_after = signal_time + pd.Timedelta(days=7)

    crash_in_window = crash_periods[
        (crash_periods.index >= signal_time) &
        (crash_periods.index <= seven_days_after)
    ]

    if len(crash_in_window) > 0:
        dangerous_entries += 1
        crash_event = crash_in_window.iloc[0]['crash_event']
        print(f"   ⚠️  Entry at {signal_time} → {crash_event} started within 7d")

dangerous_rate = dangerous_entries / len(entry_signals) if entry_signals else 0

print(f"\n📊 Pre-Crash Entry Analysis:")
print(f"   Total entries: {len(entry_signals)}")
print(f"   Dangerous entries (crash within 7d): {dangerous_entries}")
print(f"   Dangerous entry rate: {dangerous_rate:.1%}")

if dangerous_rate < 0.05:
    print(f"   ✅ PASS: <5% dangerous entries")
elif dangerous_rate < 0.10:
    print(f"   ⚠️  WARNING: 5-10% dangerous entries")
else:
    print(f"   ❌ FAIL: >10% dangerous entries - poor timing!")

# ============================================================================
# TEST 4: RSI DISTRIBUTION AT ENTRY
# ============================================================================
print("\n" + "="*80)
print("TEST 4: RSI DISTRIBUTION AT ENTRY")
print("="*80)

rsi_values = [s['rsi'] for s in entry_signals]

print(f"\n📊 RSI at entry points:")
print(f"   Mean: {np.mean(rsi_values):.1f}")
print(f"   Median: {np.median(rsi_values):.1f}")
print(f"   Min: {np.min(rsi_values):.1f}")
print(f"   Max: {np.max(rsi_values):.1f}")

# Should be concentrated around 20-30 (oversold)
in_range = sum(1 for rsi in rsi_values if 20 <= rsi <= 30)
in_range_pct = in_range / len(rsi_values)

print(f"   Entries with RSI 20-30: {in_range_pct:.1%}")

if in_range_pct >= 0.80:
    print(f"   ✅ PASS: Most entries in oversold zone")
else:
    print(f"   ⚠️  WARNING: Entries spread outside oversold zone")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1.3 RESULTS SUMMARY")
print("="*80)

print(f"\n📊 Overall Metrics:")
print(f"   Signals per year: {signals_per_year:.1f}")
print(f"   Entry accuracy (+5% in 7d): {success_rate:.1%}")
print(f"   Avg max gain in 7d: {avg_max_gain:+.1%}")
print(f"   Dangerous entry rate: {dangerous_rate:.1%}")
print(f"   RSI concentration (20-30): {in_range_pct:.1%}")

# Pass/Fail criteria
print(f"\n✅ PASS/FAIL:")

passed = True

if 100 <= signals_per_year <= 200:
    print(f"   ✅ Signal frequency: {signals_per_year:.0f}/year (optimal)")
elif 50 <= signals_per_year <= 300:
    print(f"   ⚠️  Signal frequency: {signals_per_year:.0f}/year (acceptable)")
else:
    print(f"   ❌ Signal frequency: {signals_per_year:.0f}/year (too extreme!)")
    passed = False

if success_rate >= 0.60:
    print(f"   ✅ Entry accuracy: {success_rate:.1%} >= 60%")
elif success_rate >= 0.50:
    print(f"   ⚠️  Entry accuracy: {success_rate:.1%} (50-60%, marginal)")
else:
    print(f"   ❌ Entry accuracy: {success_rate:.1%} < 50%")
    passed = False

if dangerous_rate < 0.05:
    print(f"   ✅ Dangerous entries: {dangerous_rate:.1%} < 5%")
elif dangerous_rate < 0.10:
    print(f"   ⚠️  Dangerous entries: {dangerous_rate:.1%} (5-10%, acceptable)")
else:
    print(f"   ❌ Dangerous entries: {dangerous_rate:.1%} > 10%")
    passed = False

print("\n" + "="*80)
if passed:
    print("✅ PHASE 1.3 PASSED - Entry signals validated!")
else:
    print("⚠️  PHASE 1.3 MARGINAL - Entry signals need improvement")
print("="*80)
