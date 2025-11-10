"""
Analyze individual trades from Day 11 forensics
"""
import pandas as pd

print("=" * 80)
print("📋 ANALYZING TRADES (Day 11 Forensics)")
print("=" * 80)
print()

# Load trades
df = pd.read_csv('reports/day11_forensics/all_trades_detailed.csv')

print(f"📊 Total trades: {len(df)}")
print()

# Best trades
print("🏆 BEST 10 TRADES:")
print("-" * 80)
best = pd.read_csv('reports/day11_forensics/best_10_trades.csv')
for i, row in enumerate(best.itertuples(), 1):
    print(f"{i:2d}. PnL: {row.pnl_pct:6.2f}% | "
          f"Hold: {row.hold_time:3.0f}h | "
          f"Entry: RSI {row.entry_rsi:.1f} | "
          f"Regime: {row.regime}")

print()

# Worst trades  
print("💀 WORST 10 TRADES:")
print("-" * 80)
worst = pd.read_csv('reports/day11_forensics/worst_10_trades.csv')
for i, row in enumerate(worst.itertuples(), 1):
    print(f"{i:2d}. PnL: {row.pnl_pct:6.2f}% | "
          f"Hold: {row.hold_time:3.0f}h | "
          f"Entry: RSI {row.entry_rsi:.1f} | "
          f"Regime: {row.regime} | "
          f"Exit: {row.exit_reason}")

print()

# Statistics
print("📊 TRADE STATISTICS:")
print("-" * 80)

winning = df[df['pnl_pct'] > 0]
losing = df[df['pnl_pct'] <= 0]

print(f"Win rate: {len(winning)/len(df)*100:.1f}%")
print(f"Average win: {winning['pnl_pct'].mean():.2f}%")
print(f"Average loss: {losing['pnl_pct'].mean():.2f}%")
print(f"Best trade: {df['pnl_pct'].max():.2f}%")
print(f"Worst trade: {df['pnl_pct'].min():.2f}%")
print(f"Profit factor: {winning['pnl_pct'].sum() / abs(losing['pnl_pct'].sum()):.2f}")
print()

# By regime
print("🌍 BY REGIME:")
print("-" * 80)
by_regime = df.groupby('regime').agg({
    'pnl_pct': ['mean', 'count'],
    'exit_reason': lambda x: (x == 'tp').sum()
})
print(by_regime)
print()

# By exit reason
print("🚪 BY EXIT REASON:")
print("-" * 80)
exit_counts = df['exit_reason'].value_counts()
for reason, count in exit_counts.items():
    pct = count / len(df) * 100
    avg_pnl = df[df['exit_reason'] == reason]['pnl_pct'].mean()
    print(f"  {reason:15s}: {count:4d} trades ({pct:5.1f}%) | Avg PnL: {avg_pnl:6.2f}%")

print()
print("=" * 80)
print("✅ TRADE ANALYSIS COMPLETE!")
print("=" * 80)
