"""
Run Backtest on Enriched Model (54 features)
Compare with baseline (31 features)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
import joblib
from backtesting.honest_backtest_v2 import HonestBacktestV2
from models.logistic_baseline import LogisticBaseline

print("="*80)
print("BACKTEST - ENRICHED MODEL (54 FEATURES)")
print("="*80)

project_root = os.path.dirname(os.path.dirname(__file__))

# Load enriched model
print("\n[1/5] Loading enriched model...")
model = LogisticBaseline(input_dim=54)
model.load_state_dict(torch.load(os.path.join(project_root, "models", "logistic_enriched_v2.pth")))
model.eval()
print("✅ Model loaded")

# Load test data
print("\n[2/5] Loading test data...")
X_test = torch.load(os.path.join(project_root, "models", "X_test_enriched_v2.pt"))
print(f"X_test: {X_test.shape}")

# Load OHLCV
print("\n[3/5] Loading OHLCV...")
ohlcv_full = pd.read_parquet(os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet"))
test_start_idx = len(ohlcv_full) - len(X_test)
ohlcv_subset = ohlcv_full.iloc[test_start_idx:].reset_index(drop=True)
print(f"OHLCV subset: {len(ohlcv_subset)} bars")

# Generate predictions
print("\n[4/5] Generating predictions...")
with torch.no_grad():
    y_pred_proba = model.predict_proba(X_test)

# Use optimal threshold from training
OPTIMAL_THRESHOLD = 0.80  # From training results
y_pred_signals = (y_pred_proba[:, 1] > OPTIMAL_THRESHOLD).astype(int)

print(f"Threshold: {OPTIMAL_THRESHOLD}")
print(f"Total signals: {len(y_pred_signals)}")
print(f"BUY signals: {(y_pred_signals == 1).sum()} ({(y_pred_signals == 1).sum()/len(y_pred_signals)*100:.1f}%)")
print(f"Probability stats: min={y_pred_proba[:, 1].min():.4f}, max={y_pred_proba[:, 1].max():.4f}, mean={y_pred_proba[:, 1].mean():.4f}")

# Run backtest
print("\n[5/5] Running backtest...")
print("Config:")
print("  - Initial capital: $100,000")
print("  - Take Profit: 2%")
print("  - Stop Loss: 1%")
print("  - Max Hold: 288 bars (3 days)")
print("  - Cooldown: 10 bars")
print("  - Commission: 0.1% per side")
print("  - Slippage: 0.05%")

backtest = HonestBacktestV2(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    position_size_pct=0.95,
    take_profit=0.02,
    stop_loss=0.01,
    max_hold_bars=288,
    cooldown_bars=10
)

metrics = backtest.run(ohlcv_subset, y_pred_signals)
backtest.print_report()

# Comparison with baseline
print("\n" + "="*80)
print("COMPARISON: BASELINE (31) vs ENRICHED (54)")
print("="*80)

baseline_results = {
    'features': 31,
    'threshold': 0.82,
    'signals': 1434,
    'trades': 104,
    'win_rate': 0.3846,
    'pf': 1.4199,
    'return': 13.94,
    'mdd': 9.33
}

enriched_results = {
    'features': 54,
    'threshold': OPTIMAL_THRESHOLD,
    'signals': (y_pred_signals == 1).sum(),
    'trades': metrics['n_trades'],
    'win_rate': metrics['win_rate'],
    'pf': metrics['profit_factor'],
    'return': metrics['total_return_pct'],
    'mdd': metrics['max_drawdown_pct']
}

print(f"\n{'Metric':<20} {'Baseline (31)':<20} {'Enriched (54)':<20} {'Change':<15}")
print("-" * 75)

for key in ['features', 'threshold', 'signals', 'trades']:
    base_val = baseline_results[key]
    enrich_val = enriched_results[key]
    if key in ['threshold']:
        print(f"{key.capitalize():<20} {base_val:<20.2f} {enrich_val:<20.2f} {'':<15}")
    else:
        print(f"{key.capitalize():<20} {base_val:<20} {enrich_val:<20} {'':<15}")

print("-" * 75)

for key in ['win_rate', 'pf', 'return', 'mdd']:
    base_val = baseline_results[key]
    enrich_val = enriched_results[key]
    
    if base_val != 0:
        change = ((enrich_val - base_val) / abs(base_val)) * 100
        change_str = f"{change:+.1f}%"
    else:
        change_str = "N/A"
    
    symbol = "✅" if enrich_val > base_val else "❌"
    if key == 'mdd':
        symbol = "✅" if enrich_val < base_val else "❌"
    
    print(f"{key.upper():<20} {base_val:<20.4f} {enrich_val:<20.4f} {change_str:<10} {symbol}")

print("="*80)

# Goal check
print("\n🎯 WEEK 3 GOALS:")
pf_status = "✅" if enriched_results['pf'] >= 2.0 else "❌"
mdd_status = "✅" if enriched_results['mdd'] <= 15.0 else "❌"

print(f"{pf_status} Profit Factor ≥ 2.0: {enriched_results['pf']:.4f}")
print(f"{mdd_status} Max Drawdown ≤ 15%: {enriched_results['mdd']:.2f}%")

if enriched_results['pf'] >= 2.0 and enriched_results['mdd'] <= 15.0:
    print("\n🎉 GOALS ACHIEVED WITH ENRICHED FEATURES!")
elif enriched_results['pf'] > baseline_results['pf']:
    print(f"\n⚠️  Improvement over baseline (+{((enriched_results['pf']/baseline_results['pf'])-1)*100:.1f}% PF), but still below Week 3 goals")
else:
    print(f"\n❌ Enriched model performed worse than baseline")

# Save plot
try:
    backtest.plot_results(save_path='reports/backtest_enriched_v2_results.png')
except Exception as e:
    print(f"Could not save plot: {e}")

print("\n✅ Backtest completed!")