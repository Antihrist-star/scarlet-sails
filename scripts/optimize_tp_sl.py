"""
TP/SL Optimization for Baseline Model (31 features) - FIXED
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
import joblib
from itertools import product
from backtesting.honest_backtest_v2 import HonestBacktestV2
from models.logistic_baseline import LogisticBaseline
import json
from datetime import datetime

print("="*80)
print("TP/SL OPTIMIZATION FOR BASELINE MODEL (31 FEATURES)")
print("="*80)

project_root = os.path.dirname(os.path.dirname(__file__))

# Load baseline model
print("\n[1/5] Loading baseline model...")
model = LogisticBaseline(input_dim=31)
model.load_state_dict(torch.load(
    os.path.join(project_root, "models", "logistic_baseline_clean_2d.pth"),
    map_location='cpu'
))
model.eval()
print("✅ Model loaded")

# Load scaler
print("\n[2/5] Loading scaler...")
scaler = joblib.load(os.path.join(project_root, "models", "scaler_clean_2d.pkl"))
print("✅ Scaler loaded")

# Load test data (NOT scaled yet!)
print("\n[3/5] Loading test data...")
X_test = torch.load(os.path.join(project_root, "models", "X_test_clean.pt"))
print(f"X_test shape: {X_test.shape}")

# Apply scaler
print("Scaling test data...")
X_test_np = X_test.numpy()
X_test_scaled = scaler.transform(X_test_np)
X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
print("✅ Data scaled")

# Load OHLCV
print("\n[4/5] Loading OHLCV data...")
ohlcv_full = pd.read_parquet(os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet"))
test_start_idx = len(ohlcv_full) - len(X_test)
ohlcv_subset = ohlcv_full.iloc[test_start_idx:].reset_index(drop=True)
print(f"OHLCV subset: {len(ohlcv_subset)} bars")

# Generate predictions with optimal threshold
print("\n[5/5] Generating predictions...")
BASELINE_THRESHOLD = 0.82
with torch.no_grad():
    y_pred_proba = model.predict_proba(X_test_scaled_tensor)

print(f"Probability stats:")
print(f"  Min:  {y_pred_proba[:, 1].min():.4f}")
print(f"  Max:  {y_pred_proba[:, 1].max():.4f}")
print(f"  Mean: {y_pred_proba[:, 1].mean():.4f}")

y_pred_signals = (y_pred_proba[:, 1] > BASELINE_THRESHOLD).astype(int)
print(f"Threshold: {BASELINE_THRESHOLD}")
print(f"Total BUY signals: {y_pred_signals.sum()}")

if y_pred_signals.sum() == 0:
    print("\n❌ ERROR: No signals generated!")
    print("This means the model is not working correctly.")
    print("Check:")
    print("  1. Model file is correct")
    print("  2. Scaler is correct")
    print("  3. Threshold is reasonable")
    sys.exit(1)

# Define parameter grid
print("\n" + "="*80)
print("STARTING TP/SL GRID SEARCH")
print("="*80)

tp_values = [0.015, 0.020, 0.025, 0.030, 0.035]
sl_values = [0.007, 0.008, 0.010, 0.012, 0.015]

print(f"\nTP range: {tp_values[0]*100:.1f}% to {tp_values[-1]*100:.1f}%")
print(f"SL range: {sl_values[0]*100:.1f}% to {sl_values[-1]*100:.1f}%")
print(f"Total combinations: {len(tp_values) * len(sl_values)}")
print("-" * 80)

results = []
best_pf = 0
best_config = None
combo_num = 0
total_combos = len(tp_values) * len(sl_values)

# Test all combinations
for tp, sl in product(tp_values, sl_values):
    combo_num += 1
    ratio = tp / sl
    
    if ratio < 1.5:
        continue
    
    print(f"\n[{combo_num}/{total_combos}] Testing TP={tp*100:.1f}%, SL={sl*100:.1f}%, Ratio={ratio:.2f}")
    
    backtest = HonestBacktestV2(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        position_size_pct=0.95,
        take_profit=tp,
        stop_loss=sl,
        max_hold_bars=288,
        cooldown_bars=10
    )
    
    metrics = backtest.run(ohlcv_subset, y_pred_signals)
    
    # Calculate actual ratio - FIXED KEYS!
    if metrics['n_trades'] > 0:
        actual_ratio = abs(metrics['avg_win']) / abs(metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
    else:
        actual_ratio = 0
    
    result = {
        'tp': tp * 100,
        'sl': sl * 100,
        'expected_ratio': ratio,
        'actual_ratio': actual_ratio,
        'trades': metrics['n_trades'],
        'wins': metrics['n_wins'],
        'losses': metrics['n_losses'],
        'win_rate': metrics['win_rate'],
        'pf': metrics['profit_factor'],
        'return': metrics['total_return_pct'],
        'mdd': metrics['max_drawdown_pct'],
        'sharpe': metrics['sharpe_ratio'],
        'calmar': metrics['calmar_ratio']
    }
    
    results.append(result)
    
    if metrics['profit_factor'] > best_pf:
        best_pf = metrics['profit_factor']
        best_config = result
    
    status = "✅" if metrics['profit_factor'] >= 2.0 and metrics['max_drawdown_pct'] <= 15 else ""
    print(f"  → PF={metrics['profit_factor']:.3f}, MDD={metrics['max_drawdown_pct']:.1f}%, "
          f"Return={metrics['total_return_pct']:.1f}%, Trades={metrics['n_trades']}, "
          f"WR={metrics['win_rate']:.1%} {status}")

# Analysis
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

results_sorted = sorted(results, key=lambda x: x['pf'], reverse=True)

print("\n🏆 TOP 10 CONFIGURATIONS:")
print("-" * 100)
print(f"{'Rank':<5} {'TP%':<6} {'SL%':<6} {'Ratio':<7} {'PF':<8} {'MDD%':<7} {'Return%':<9} {'Trades':<7} {'WR%':<7} {'Goal':<5}")
print("-" * 100)

for i, r in enumerate(results_sorted[:10], 1):
    goal_met = "✅" if r['pf'] >= 2.0 and r['mdd'] <= 15 else ""
    print(f"{i:<5} {r['tp']:<6.1f} {r['sl']:<6.1f} {r['expected_ratio']:<7.2f} "
          f"{r['pf']:<8.3f} {r['mdd']:<7.2f} {r['return']:<9.2f} "
          f"{r['trades']:<7} {r['win_rate']*100:<7.1f} {goal_met:<5}")

# Find goal configs
goal_configs = [r for r in results if r['pf'] >= 2.0 and r['mdd'] <= 15]

if goal_configs:
    print("\n" + "="*80)
    print("✅ CONFIGURATIONS MEETING WEEK 3 GOALS")
    print("="*80)
    for i, r in enumerate(sorted(goal_configs, key=lambda x: x['pf'], reverse=True), 1):
        print(f"\n{i}. TP={r['tp']:.1f}%, SL={r['sl']:.1f}%")
        print(f"   PF:      {r['pf']:.3f} ✅")
        print(f"   MDD:     {r['mdd']:.2f}% ✅")
        print(f"   Return:  {r['return']:.2f}%")
        print(f"   Sharpe:  {r['sharpe']:.4f}")
        print(f"   Trades:  {r['trades']}")
        print(f"   WR:      {r['win_rate']:.2%}")
else:
    print("\n" + "="*80)
    print("⚠️  NO CONFIG MEETS BOTH GOALS")
    print("="*80)

# Ratio analysis
print("\n📈 RATIO EFFICIENCY (TOP 3):")
print("-" * 80)
for i, r in enumerate(results_sorted[:3], 1):
    if r['actual_ratio'] > 0:
        efficiency = (r['actual_ratio'] / r['expected_ratio']) * 100
        print(f"{i}. TP={r['tp']:.1f}%, SL={r['sl']:.1f}%")
        print(f"   Expected: {r['expected_ratio']:.2f}, Actual: {r['actual_ratio']:.2f}")
        print(f"   Efficiency: {efficiency:.1f}% (lost {100-efficiency:.1f}% to costs)")

# Save
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
csv_path = os.path.join(project_root, 'reports', 'tp_sl_optimization_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"✅ CSV saved: {csv_path}")

best_config_data = {
    'model': 'baseline_31_features',
    'threshold': BASELINE_THRESHOLD,
    'tp': best_config['tp'] / 100,
    'sl': best_config['sl'] / 100,
    'metrics': {
        'pf': best_config['pf'],
        'mdd': best_config['mdd'],
        'return': best_config['return'],
        'trades': best_config['trades'],
        'win_rate': best_config['win_rate']
    },
    'timestamp': datetime.now().isoformat()
}

json_path = os.path.join(project_root, 'models', 'best_tp_sl_config.json')
with open(json_path, 'w') as f:
    json.dump(best_config_data, f, indent=2)
print(f"✅ JSON saved: {json_path}")

# Final
print("\n" + "="*80)
print("🎯 RECOMMENDATION")
print("="*80)

if goal_configs:
    best = goal_configs[0]
    print(f"\n✅ USE: TP={best['tp']:.1f}%, SL={best['sl']:.1f}%")
    print(f"   PF={best['pf']:.3f}, MDD={best['mdd']:.2f}%")
    print("\n🎉 WEEK 3 GOALS ACHIEVED!")
else:
    print(f"\n⚠️  BEST: TP={best_config['tp']:.1f}%, SL={best_config['sl']:.1f}%")
    print(f"   PF={best_config['pf']:.3f}, MDD={best_config['mdd']:.2f}%")
    gap_pf = 2.0 - best_config['pf']
    gap_mdd = best_config['mdd'] - 15.0
    print(f"\nGaps:")
    if best_config['pf'] < 2.0:
        print(f"  PF: -{gap_pf:.2f} points")
    if best_config['mdd'] > 15.0:
        print(f"  MDD: +{gap_mdd:.2f}%")
    print("\nNext: Try XGBoost")

print("\n✅ COMPLETE!")