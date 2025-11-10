"""
Analyze EXISTING results from master audit and Day 11 forensics
"""
import json
import pandas as pd
from pathlib import Path

print("=" * 80)
print("🔍 ANALYZING EXISTING RESULTS")
print("=" * 80)
print()

# Load master audit results
print("📊 MASTER AUDIT RESULTS (56 combinations tested)")
print("-" * 80)

with open('reports/master_audit/raw_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Show summary
print(f"✅ Total combinations: {len(df)}")
print(f"✅ Successful: {len(df[df['status'] == 'SUCCESS'])}")
print(f"❌ Failed: {len(df[df['status'] != 'SUCCESS'])}")
print()

# Top performers
print("🏆 TOP 10 PERFORMERS (by annual return):")
print("-" * 80)

df_success = df[df['status'] == 'SUCCESS'].copy()
df_success['annual_return'] = df_success['metrics'].apply(lambda x: x['annual_return'])
df_success['win_rate'] = df_success['metrics'].apply(lambda x: x['win_rate'])
df_success['sharpe'] = df_success['metrics'].apply(lambda x: x['sharpe'])
df_success['period_years'] = df_success['metrics'].apply(lambda x: x['period_years'])

top10 = df_success.nlargest(10, 'annual_return')

for i, row in enumerate(top10.itertuples(), 1):
    print(f"{i:2d}. {row.asset:4s} {row.timeframe:3s}: {row.annual_return:6.1f}% annual | "
          f"WR: {row.win_rate*100:5.1f}% | Sharpe: {row.sharpe:.2f} | "
          f"Period: {row.period_years:.1f}y")

print()
print("📈 STATISTICS:")
print("-" * 80)
print(f"Average annual return: {df_success['annual_return'].mean():.1f}%")
print(f"Median annual return:  {df_success['annual_return'].median():.1f}%")
print(f"Best: {df_success['annual_return'].max():.1f}%")
print(f"Worst: {df_success['annual_return'].min():.1f}%")
print()

# By asset
print("💰 BY ASSET (average annual return):")
print("-" * 80)
by_asset = df_success.groupby('asset')['annual_return'].mean().sort_values(ascending=False)
for asset, ret in by_asset.items():
    print(f"  {asset:6s}: {ret:6.1f}%")

print()

# By timeframe
print("⏰ BY TIMEFRAME (average annual return):")
print("-" * 80)
by_tf = df_success.groupby('timeframe')['annual_return'].mean().sort_values(ascending=False)
for tf, ret in by_tf.items():
    print(f"  {tf:4s}: {ret:6.1f}%")

print()

# Day 11 forensics
print("🔬 DAY 11 FORENSICS TESTS:")
print("-" * 80)

with open('reports/day11_forensics/test_results.json', 'r') as f:
    forensics = json.load(f)

for test, result in forensics.items():
    emoji = "✅" if "PASS" in result else "⚠️" if "MARGINAL" in result else "❌"
    print(f"  {emoji} {test}: {result}")

print()
print("=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
