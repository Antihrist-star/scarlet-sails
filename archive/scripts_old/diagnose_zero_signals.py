"""
DIAGNOSTIC SCRIPT - Find why 0 signals generated
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.hybrid_entry_system import HybridEntrySystem

print("=" * 100)
print("DIAGNOSTIC: Why 0 signals?")
print("=" * 100)

# Test on BTC 15m
asset = 'BTC'
target_tf = '15m'

print(f"\n1. Loading multi-TF data for {asset} {target_tf}...")
extractor = MultiTimeframeFeatureExtractor(data_dir='data/raw')

try:
    all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, target_tf)
    print(f"   ✅ Loaded {len(primary_df)} bars")
    print(f"   ✅ Prepared {len(primary_df.columns)} columns")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check columns
print(f"\n2. Checking columns in primary_df...")
print(f"   Total columns: {len(primary_df.columns)}")
print(f"   Column names (first 20): {list(primary_df.columns[:20])}")

# Check for RSI columns
rsi_cols = [col for col in primary_df.columns if 'rsi' in col.lower() or 'RSI' in col]
print(f"\n   RSI columns found: {rsi_cols}")

atr_cols = [col for col in primary_df.columns if 'atr' in col.lower() or 'ATR' in col]
print(f"   ATR columns found: {atr_cols}")

# Check RSI < 30
if '15m_RSI_14' in primary_df.columns:
    rsi_col = '15m_RSI_14'
elif 'rsi' in primary_df.columns:
    rsi_col = 'rsi'
elif 'RSI_14' in primary_df.columns:
    rsi_col = 'RSI_14'
else:
    print(f"\n   ❌ NO RSI COLUMN FOUND!")
    print(f"   Available columns: {list(primary_df.columns)}")
    sys.exit(1)

print(f"\n3. Checking RSI values (using column: {rsi_col})...")
rsi_below_30 = (primary_df[rsi_col] < 30).sum()
rsi_below_30_pct = rsi_below_30 / len(primary_df) * 100
print(f"   RSI < 30: {rsi_below_30:,} bars ({rsi_below_30_pct:.2f}%)")

if rsi_below_30 == 0:
    print(f"   ❌ PROBLEM: No RSI < 30 found!")
    print(f"   RSI min: {primary_df[rsi_col].min():.2f}")
    print(f"   RSI max: {primary_df[rsi_col].max():.2f}")
    print(f"   RSI mean: {primary_df[rsi_col].mean():.2f}")
    print(f"   Sample RSI values: {primary_df[rsi_col].iloc[100:110].values}")
else:
    print(f"   ✅ RSI < 30 exists, should generate signals")

# Test HybridEntrySystem
print(f"\n4. Testing HybridEntrySystem...")
entry_system = HybridEntrySystem(
    ml_threshold=0.6,
    enable_ml_filter=True,
    enable_crisis_gate=False,
    all_timeframes=all_tf,
    target_timeframe=target_tf
)

# Test Layer 1 (Rules) directly
print(f"\n5. Testing Layer 1 (Rules) - checking rule signals...")
rule_signals = []
for i in range(100, min(1000, len(primary_df))):
    # Check if RSI < 30
    if primary_df[rsi_col].iloc[i] < 30:
        rule_signals.append(i)

print(f"   Layer 1 would generate: {len(rule_signals)} signals (first 900 bars)")

if len(rule_signals) == 0:
    print(f"   ❌ Layer 1 generates 0 signals!")
else:
    # Test full entry system on first few signals
    print(f"\n6. Testing full 3-layer system on first 5 signals...")
    for idx, bar_idx in enumerate(rule_signals[:5]):
        timestamp = primary_df.index[bar_idx]
        should_enter, reason = entry_system.should_enter(primary_df, bar_idx, timestamp)

        print(f"\n   Signal {idx+1} at bar {bar_idx}:")
        print(f"      RSI: {primary_df[rsi_col].iloc[bar_idx]:.2f}")
        print(f"      Should enter: {should_enter}")
        print(f"      Reason: {reason}")

# Check statistics
print(f"\n7. Entry system statistics...")
stats = entry_system.get_statistics()
print(f"   Total checks: {stats.get('total_checks', 0)}")
print(f"   Layer 1 passed: {stats.get('layer1_passed', 0)}")
print(f"   Layer 2 passed: {stats.get('layer2_passed', 0)}")
print(f"   Layer 3 passed: {stats.get('layer3_passed', 0)}")
print(f"   Final signals: {stats.get('final_signals', 0)}")

print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
