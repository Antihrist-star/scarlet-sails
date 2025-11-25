"""
Analyze individual trades from Day 11 forensics
"""
import pandas as pd

print("=" * 80)
print("📋 ANALYZING TRADES (Day 11 Forensics)")
print("=" * 80)
print()

# Load trades
df_all = pd.read_csv('reports/day11_forensics/all_trades_detailed.csv')
df_all['pnl_pct'] = (df_all['exit_price'] / df_all['entry_price'] - 1) * 100
df_all['hold_hours'] = df_all['holding_bars']

print(f"📊 Total trades: {len(df_all)}")
print()

# Best trades
print("🏆 BEST 10 TRADES:")
print("-" * 80)
best = pd.read_csv('reports/day11_forensics/best_10_trades.csv')
best['pnl_pct'] = (best['exit_price'] / best['entry_price'] - 1) * 100
for i, row in best.iterrows():
    print(f"{i+1:2d}. PnL: +{row['pnl_pct']:5.1f}% | "
          f"Hold: {row['holding_bars']:3.0f}h | "
          f"Entry RSI: {row['rsi']:.1f} | "
          f"Regime: {row['regime'].split('.')[-1]}")

print()

# Worst trades  
print("💀 WORST 10 TRADES:")
print("-" * 80)
worst = pd.read_csv('reports/day11_forensics/worst_10_trades.csv')
worst['pnl_pct'] = (worst['exit_price'] / worst['entry_price'] - 1) * 100
for i, row in worst.iterrows():
    print(f"{i+1:2d}. PnL: {row['pnl_pct']:6.1f}% | "
          f"Hold: {row['holding_bars']:3.0f}h | "
          f"Entry RSI: {row['rsi']:.1f} | "
          f"Exit: {row['exit_reason']}")

print()

# Statistics
print("📊 TRADE STATISTICS:")
print("-" * 80)

winning = df_all[df_all['pnl_pct'] > 0]
losing = df_all[df_all['pnl_pct'] <= 0]

print(f"Trades: {len(df_all)}")
print(f"Win rate: {len(winning)/len(df_all)*100:.1f}%")
print(f"Average win: +{winning['pnl_pct'].mean():.2f}%")
print(f"Average loss: {losing['pnl_pct'].mean():.2f}%")
print(f"Best trade: +{df_all['pnl_pct'].max():.2f}%")
print(f"Worst trade: {df_all['pnl_pct'].min():.2f}%")

if len(losing) > 0:
    pf = winning['pnl_pct'].sum() / abs(losing['pnl_pct'].sum())
    print(f"Profit factor: {pf:.2f}")

print(f"Average holding: {df_all['hold_hours'].mean():.1f} hours")
print()

# By regime
print("🌍 BY REGIME:")
print("-" * 80)
df_all['regime_simple'] = df_all['regime'].str.split('.').str[-1]
by_regime = df_all.groupby('regime_simple').agg({
    'pnl_pct': ['mean', 'count']
})
by_regime.columns = ['Avg PnL %', 'Count']
by_regime = by_regime.sort_values('Avg PnL %', ascending=False)
print(by_regime)
print()

# By exit reason
print("🚪 BY EXIT REASON:")
print("-" * 80)
exit_stats = df_all.groupby('exit_reason').agg({
    'pnl_pct': ['mean', 'count']
})
exit_stats.columns = ['Avg PnL %', 'Count']
exit_stats['Pct'] = exit_stats['Count'] / len(df_all) * 100
exit_stats = exit_stats.sort_values('Count', ascending=False)
print(exit_stats)

print()
print("=" * 80)
print("✅ TRADE ANALYSIS COMPLETE!")
print("=" * 80)
