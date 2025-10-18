"""
Threshold Optimization V2 - Multi-Criteria Analysis
Finds optimal threshold based on PF, WR, Sharpe, and Trade Count
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
import matplotlib.pyplot as plt

# Paths
project_root = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(project_root, "models", "logistic_baseline_clean_2d.pth")
scaler_path = os.path.join(project_root, "models", "scaler_clean_2d.pkl")
X_test_path = os.path.join(project_root, "models", "X_test_clean.pt")
ohlcv_path = os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet")

print("="*80)
print("THRESHOLD OPTIMIZATION - MULTI-CRITERIA ANALYSIS")
print("="*80)

# Load model and data
print("\n[1/5] Loading model and data...")
model = LogisticBaseline(input_dim=31)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

scaler = joblib.load(scaler_path)
X_test = torch.load(X_test_path)
X_test_np = X_test.numpy()
X_test_scaled = scaler.transform(X_test_np)

# Get probabilities
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_pred_proba = model.predict_proba(X_test_tensor)

print(f"Probabilities: min={y_pred_proba[:, 1].min():.4f}, max={y_pred_proba[:, 1].max():.4f}, mean={y_pred_proba[:, 1].mean():.4f}")

# Load OHLCV
ohlcv_full = pd.read_parquet(ohlcv_path)
test_start_idx = len(ohlcv_full) - len(X_test)
ohlcv_subset = ohlcv_full.iloc[test_start_idx:].reset_index(drop=True)
print(f"OHLCV subset: {len(ohlcv_subset)} bars")

# Backtest params
BACKTEST_PARAMS = {
    'initial_capital': 100000,
    'commission': 0.001,
    'slippage': 0.0005,
    'position_size_pct': 0.95,
    'take_profit': 0.02,
    'stop_loss': 0.01,
    'max_hold_bars': 288,
    'cooldown_bars': 10
}

# Threshold range
thresholds = np.arange(0.50, 0.91, 0.02)
MIN_TRADES = 20  # Minimum trades for valid result

print(f"\n[2/5] Testing {len(thresholds)} thresholds from {thresholds[0]:.2f} to {thresholds[-1]:.2f}...")
print(f"Min trades filter: {MIN_TRADES}")

results = []

for i, thr in enumerate(thresholds):
    y_pred_signals = (y_pred_proba[:, 1] > thr).astype(int)
    n_signals = y_pred_signals.sum()
    
    if n_signals == 0:
        results.append({
            'threshold': thr,
            'n_signals': 0,
            'n_trades': 0,
            'pf': 0.0,
            'wr': 0.0,
            'return_pct': 0.0,
            'mdd_pct': 0.0,
            'sharpe': 0.0,
            'calmar': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'valid': False
        })
        continue
    
    # Run backtest
    backtest = HonestBacktestV2(**BACKTEST_PARAMS)
    metrics = backtest.run(ohlcv_subset, y_pred_signals)
    
    # Valid if enough trades
    is_valid = metrics['n_trades'] >= MIN_TRADES
    
    results.append({
        'threshold': thr,
        'n_signals': n_signals,
        'n_trades': metrics['n_trades'],
        'pf': metrics['profit_factor'],
        'wr': metrics['win_rate'],
        'return_pct': metrics['total_return_pct'],
        'mdd_pct': metrics['max_drawdown_pct'],
        'sharpe': metrics['sharpe_ratio'],
        'calmar': metrics['calmar_ratio'],
        'avg_win': metrics['avg_win'],
        'avg_loss': metrics['avg_loss'],
        'valid': is_valid
    })
    
    if (i + 1) % 5 == 0:
        print(f"  Progress: {i+1}/{len(thresholds)} ({(i+1)/len(thresholds)*100:.0f}%)")

print("\n[3/5] Analyzing results...")

results_df = pd.DataFrame(results)
valid_df = results_df[results_df['valid']].copy()

print(f"\nValid results: {len(valid_df)}/{len(results_df)} (>= {MIN_TRADES} trades)")

if len(valid_df) == 0:
    print("❌ No valid thresholds found! All have < {MIN_TRADES} trades.")
    sys.exit(1)

# Find best thresholds by different criteria
best_indices = {}

# 1. Best PF
best_indices['pf'] = valid_df['pf'].idxmax()

# 2. Best Sharpe (if positive)
positive_sharpe = valid_df[valid_df['sharpe'] > 0]
if len(positive_sharpe) > 0:
    best_indices['sharpe'] = positive_sharpe['sharpe'].idxmax()
else:
    best_indices['sharpe'] = None

# 3. Best Return
best_indices['return'] = valid_df['return_pct'].idxmax()

# 4. Best WR (among profitable: PF > 1.0)
profitable = valid_df[valid_df['pf'] > 1.0]
if len(profitable) > 0:
    best_indices['wr'] = profitable['wr'].idxmax()
else:
    best_indices['wr'] = None

# 5. Best Calmar (if positive)
positive_calmar = valid_df[valid_df['calmar'] > 0]
if len(positive_calmar) > 0:
    best_indices['calmar'] = positive_calmar['calmar'].idxmax()
else:
    best_indices['calmar'] = None

# Print summary
print("\n" + "="*80)
print("TOP CANDIDATES BY DIFFERENT CRITERIA")
print("="*80)

for criterion, idx in best_indices.items():
    if idx is None:
        print(f"\n{criterion.upper()}: No valid candidate")
        continue
    
    row = results_df.loc[idx]
    print(f"\n{criterion.upper()}:")
    print(f"  Threshold:    {row['threshold']:.2f}")
    print(f"  PF:           {row['pf']:.4f}")
    print(f"  Win Rate:     {row['wr']:.2%}")
    print(f"  Return:       {row['return_pct']:.2f}%")
    print(f"  Max DD:       {row['mdd_pct']:.2f}%")
    print(f"  Sharpe:       {row['sharpe']:.4f}")
    print(f"  Trades:       {row['n_trades']}")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Choose best by PF as primary
recommended_idx = best_indices['pf']
rec_row = results_df.loc[recommended_idx]

print(f"\n🎯 RECOMMENDED THRESHOLD: {rec_row['threshold']:.2f}")
print(f"\nExpected Performance:")
print(f"  Profit Factor:    {rec_row['pf']:.4f}")
print(f"  Win Rate:         {rec_row['wr']:.2%}")
print(f"  Total Return:     {rec_row['return_pct']:.2f}%")
print(f"  Max Drawdown:     {rec_row['mdd_pct']:.2f}%")
print(f"  Sharpe Ratio:     {rec_row['sharpe']:.4f}")
print(f"  Number of Trades: {rec_row['n_trades']}")

# Status check
if rec_row['pf'] >= 2.0 and rec_row['mdd_pct'] <= 15.0:
    print("\n✅ GOALS ACHIEVED!")
elif rec_row['pf'] >= 1.0:
    print("\n⚠️  Profitable but below Week 3 goals")
else:
    print("\n❌ Still not profitable")

# Save results
print("\n[4/5] Saving results...")
results_path = os.path.join(project_root, 'reports', 'threshold_optimization_v2.csv')
results_df.to_csv(results_path, index=False)
print(f"💾 CSV saved: {results_path}")

# Plot
print("\n[5/5] Creating plots...")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Plot 1: PF vs Threshold
ax = axes[0, 0]
ax.plot(results_df['threshold'], results_df['pf'], marker='o', label='All')
if len(valid_df) > 0:
    ax.scatter(valid_df['threshold'], valid_df['pf'], color='green', s=100, label=f'Valid (≥{MIN_TRADES} trades)', zorder=3)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='PF=1.0 (Break-even)')
ax.axhline(2.0, color='green', linestyle='--', alpha=0.5, label='PF=2.0 (Goal)')
ax.scatter([rec_row['threshold']], [rec_row['pf']], color='gold', s=200, marker='*', zorder=4, label='Recommended')
ax.set_xlabel('Threshold')
ax.set_ylabel('Profit Factor')
ax.set_title('Profit Factor vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: WR vs Threshold
ax = axes[0, 1]
ax.plot(results_df['threshold'], results_df['wr'] * 100, marker='s', label='Win Rate')
ax.axhline(42.7, color='red', linestyle='--', alpha=0.5, label='BE WR ~42.7%')
ax.axhline(50, color='green', linestyle='--', alpha=0.5, label='Goal WR 50%')
ax.scatter([rec_row['threshold']], [rec_row['wr'] * 100], color='gold', s=200, marker='*', zorder=4, label='Recommended')
ax.set_xlabel('Threshold')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Return vs Threshold
ax = axes[1, 0]
ax.plot(results_df['threshold'], results_df['return_pct'], marker='d', color='purple')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.scatter([rec_row['threshold']], [rec_row['return_pct']], color='gold', s=200, marker='*', zorder=4)
ax.set_xlabel('Threshold')
ax.set_ylabel('Total Return (%)')
ax.set_title('Total Return vs Threshold')
ax.grid(True, alpha=0.3)

# Plot 4: Trades vs Threshold
ax = axes[1, 1]
ax.plot(results_df['threshold'], results_df['n_trades'], marker='v', color='red')
ax.axhline(MIN_TRADES, color='orange', linestyle='--', alpha=0.5, label=f'Min trades ({MIN_TRADES})')
ax.scatter([rec_row['threshold']], [rec_row['n_trades']], color='gold', s=200, marker='*', zorder=4)
ax.set_xlabel('Threshold')
ax.set_ylabel('Number of Trades')
ax.set_title('Trade Frequency vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Sharpe vs Threshold
ax = axes[2, 0]
ax.plot(results_df['threshold'], results_df['sharpe'], marker='^', color='teal')
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Sharpe=1.0')
ax.scatter([rec_row['threshold']], [rec_row['sharpe']], color='gold', s=200, marker='*', zorder=4)
ax.set_xlabel('Threshold')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: MDD vs Threshold
ax = axes[2, 1]
ax.plot(results_df['threshold'], results_df['mdd_pct'], marker='x', color='darkred')
ax.axhline(15, color='green', linestyle='--', alpha=0.5, label='Goal MDD 15%')
ax.scatter([rec_row['threshold']], [rec_row['mdd_pct']], color='gold', s=200, marker='*', zorder=4)
ax.set_xlabel('Threshold')
ax.set_ylabel('Max Drawdown (%)')
ax.set_title('Max Drawdown vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(project_root, 'reports', 'threshold_optimization_v2.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📊 Plot saved: {plot_path}")
plt.close()

print("\n" + "="*80)
print("✅ THRESHOLD OPTIMIZATION COMPLETED!")
print("="*80)