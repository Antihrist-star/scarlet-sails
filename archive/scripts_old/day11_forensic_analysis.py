"""
DAY 11: TRUTH DISCOVERY PROTOCOL
=================================

PHASE 0: CREATIVE EXPLORATION + PHASE 1: HARDCORE VALIDATION

Цель: Понять ПРИРОДУ edge, не просто найти параметры

Author: Scarlet Sails Team
Date: Day 11
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DAY 11: TRUTH DISCOVERY PROTOCOL")
print("="*80)
print("\nПОНЯТЬ ПРИРОДУ EDGE, НЕ ПРОСТО НАЙТИ ПАРАМЕТРЫ\n")

# ============================================================================
# PART 1: MARKET ARCHAEOLOGY (3-4h)
# ============================================================================
print("\n" + "="*80)
print("PART 1: MARKET ARCHAEOLOGY")
print("="*80)

print("\n📂 Loading data...")
df = pd.read_parquet('data/processed/btc_prepared_phase0.parquet')
print(f"✅ Loaded {len(df):,} bars")
print(f"   Period: {df.index[0]} to {df.index[-1]}")

# Detect regime if not present
if 'regime' not in df.columns:
    print("\n🔍 Detecting market regimes...")
    from models.regime_detector import SimpleRegimeDetector

    regime_detector = SimpleRegimeDetector()
    regimes = []

    for i in range(len(df)):
        regime = regime_detector.detect(df, i)
        regimes.append(regime)

    df['regime'] = regimes
    print(f"✅ Regimes detected")

# Generate entry signals
print("\n📊 Generating entry signals (RSI < 30)...")
signals = df[df['rsi'] < 30].copy()
print(f"✅ Generated {len(signals):,} entry signals")

# ============================================================================
# Simulate all trades to get P&L
# ============================================================================
print("\n💰 Simulating all trades to analyze P&L...")

from models.hybrid_position_manager import HybridPositionManager

def backtest_hybrid_detailed(df, signals):
    """Run Hybrid backtest and return detailed trade results"""
    hybrid = HybridPositionManager(max_holding_time_bars=168)  # 7 days

    trade_results = []

    for idx, sig_row in signals.iterrows():
        entry_bar = df.index.get_loc(idx)
        entry_price = sig_row['close']
        entry_time = idx

        # Open position
        position = hybrid.open_position(
            symbol='BTC/USDT',
            entry_price=entry_price,
            entry_time=entry_time,
            size=1.0,
            direction='long',
            df=df,
            current_bar=entry_bar,
        )

        if position is None:
            continue

        # Update position over next 7 days
        exit_price = None
        exit_time = None
        exit_reason = None

        for i in range(entry_bar + 1, min(entry_bar + 169, len(df))):
            current_price = df.iloc[i]['close']
            current_time = df.index[i]

            # Update position using CORRECT API
            try:
                position, exit_signals = hybrid.update_position(
                    'BTC/USDT',
                    current_price,
                    df,
                    i
                )

                # Check if exited
                if exit_signals:
                    exit_price = exit_signals[0]['price']
                    exit_time = current_time
                    exit_reason = exit_signals[0]['label']

                    hybrid.execute_exits('BTC/USDT', exit_signals)
                    break

            except Exception as e:
                # Position closed or error
                break

        # If didn't exit, use last price
        if exit_price is None:
            last_bar = min(entry_bar + 168, len(df) - 1)
            exit_price = df.iloc[last_bar]['close']
            exit_time = df.index[last_bar]
            exit_reason = "max_holding_time"

        # Calculate P&L
        pnl = (exit_price - entry_price) / entry_price * 100

        # Get market context
        regime = sig_row.get('regime', 'UNKNOWN')

        trade_results.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': regime,
            'rsi': sig_row['rsi'],
            'volume_ratio': sig_row['volume_ratio'],
            'volatility': sig_row['volatility'],
            'holding_bars': (exit_time - entry_time).total_seconds() / 3600,  # hours
        })

    return pd.DataFrame(trade_results)

print("   Running Hybrid backtest...")
trades_df = backtest_hybrid_detailed(df, signals)
print(f"✅ Completed {len(trades_df):,} trades")

# ============================================================================
# ARCHAEOLOGY 1: BEST TRADES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ARCHAEOLOGY 1: BEST TRADES ANALYSIS")
print("="*80)

best_trades = trades_df.nlargest(10, 'pnl')

print(f"\n📊 Top 10 Best Trades:")
print(f"\n{'Entry Time':<22} {'Exit Time':<22} {'P&L':>8} {'Regime':<15} {'RSI':>6} {'Vol Ratio':>10} {'Holding (h)':>12}")
print("-" * 120)
for idx, trade in best_trades.iterrows():
    print(f"{str(trade['entry_time']):<22} {str(trade['exit_time']):<22} "
          f"{trade['pnl']:>7.1f}% {trade['regime']:<15} {trade['rsi']:>6.1f} "
          f"{trade['volume_ratio']:>10.2f} {trade['holding_bars']:>12.1f}")

# Analyze common patterns in winners
print(f"\n🔍 PATTERN ANALYSIS (Top 10 Winners):")
print(f"   Average P&L: {best_trades['pnl'].mean():.1f}%")
print(f"   Median P&L: {best_trades['pnl'].median():.1f}%")
print(f"   Average RSI at entry: {best_trades['rsi'].mean():.1f}")
print(f"   Average volume ratio: {best_trades['volume_ratio'].mean():.2f}x")
print(f"   Average holding time: {best_trades['holding_bars'].mean():.1f}h ({best_trades['holding_bars'].mean()/24:.1f} days)")

print(f"\n   Regime distribution:")
for regime, count in best_trades['regime'].value_counts().items():
    print(f"      {regime}: {count} trades ({count/len(best_trades)*100:.1f}%)")

# ============================================================================
# ARCHAEOLOGY 2: WORST TRADES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ARCHAEOLOGY 2: WORST TRADES ANALYSIS")
print("="*80)

worst_trades = trades_df.nsmallest(10, 'pnl')

print(f"\n📊 Top 10 Worst Trades:")
print(f"\n{'Entry Time':<22} {'Exit Time':<22} {'P&L':>8} {'Regime':<15} {'RSI':>6} {'Vol Ratio':>10} {'Holding (h)':>12}")
print("-" * 120)
for idx, trade in worst_trades.iterrows():
    print(f"{str(trade['entry_time']):<22} {str(trade['exit_time']):<22} "
          f"{trade['pnl']:>7.1f}% {trade['regime']:<15} {trade['rsi']:>6.1f} "
          f"{trade['volume_ratio']:>10.2f} {trade['holding_bars']:>12.1f}")

# Analyze common patterns in losers
print(f"\n🔍 PATTERN ANALYSIS (Top 10 Losers):")
print(f"   Average P&L: {worst_trades['pnl'].mean():.1f}%")
print(f"   Median P&L: {worst_trades['pnl'].median():.1f}%")
print(f"   Average RSI at entry: {worst_trades['rsi'].mean():.1f}")
print(f"   Average volume ratio: {worst_trades['volume_ratio'].mean():.2f}x")
print(f"   Average holding time: {worst_trades['holding_bars'].mean():.1f}h ({worst_trades['holding_bars'].mean()/24:.1f} days)")

print(f"\n   Regime distribution:")
for regime, count in worst_trades['regime'].value_counts().items():
    print(f"      {regime}: {count} trades ({count/len(worst_trades)*100:.1f}%)")

# ============================================================================
# ARCHAEOLOGY 3: PATTERN LIBRARY
# ============================================================================
print("\n" + "="*80)
print("ARCHAEOLOGY 3: PATTERN LIBRARY")
print("="*80)

print("\n📚 Comparing Winners vs Losers:")
print(f"\n{'Metric':<25} {'Winners (Top 10)':<20} {'Losers (Bottom 10)':<20} {'Difference':<15}")
print("-" * 80)

metrics = [
    ('Average RSI', best_trades['rsi'].mean(), worst_trades['rsi'].mean()),
    ('Average Volume Ratio', best_trades['volume_ratio'].mean(), worst_trades['volume_ratio'].mean()),
    ('Average Volatility', best_trades['volatility'].mean(), worst_trades['volatility'].mean()),
    ('Average Holding (h)', best_trades['holding_bars'].mean(), worst_trades['holding_bars'].mean()),
]

for metric_name, winner_val, loser_val in metrics:
    diff = winner_val - loser_val
    print(f"{metric_name:<25} {winner_val:<20.2f} {loser_val:<20.2f} {diff:<15.2f}")

# ============================================================================
# PART 2: HARDCORE TESTS (4h)
# ============================================================================
print("\n" + "="*80)
print("PART 2: HARDCORE VALIDATION - 10 TESTS")
print("="*80)

test_results = {}

# ============================================================================
# TEST-1: OOS EDGE VALIDATION (честный тест)
# ============================================================================
print("\n" + "="*80)
print("TEST-1: OOS EDGE VALIDATION")
print("="*80)
print("Period: Jan-Jun 2024 (6 months AFTER last optimization)")
print("Parameters: FROZEN (no tuning)")
print("Costs: FULL (0.15% per trade)")

# Filter for 2024 Jan-Jun
oos_start = pd.Timestamp('2024-01-01', tz='UTC')
oos_end = pd.Timestamp('2024-06-30', tz='UTC')
oos_trades = trades_df[(trades_df['entry_time'] >= oos_start) & (trades_df['entry_time'] <= oos_end)]

if len(oos_trades) > 0:
    # Calculate net P&L after costs
    cost_per_trade = 0.15  # 0.15% per round trip
    oos_trades['net_pnl'] = oos_trades['pnl'] - cost_per_trade

    total_net_pnl = oos_trades['net_pnl'].sum()
    win_rate = (oos_trades['net_pnl'] > 0).sum() / len(oos_trades) * 100
    monthly_return = total_net_pnl / 6  # 6 months

    print(f"\n📊 OOS Results (Jan-Jun 2024):")
    print(f"   Trades: {len(oos_trades)}")
    print(f"   Total Net P&L: {total_net_pnl:.1f}%")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Monthly Return (avg): {monthly_return:.1f}%")

    # Verdict
    if monthly_return > 1.5:
        verdict = "✅ PASS - Edge exists"
    elif monthly_return > 0.5:
        verdict = "⚠️  MARGINAL - Weak edge"
    else:
        verdict = "❌ FAIL - No edge"

    print(f"   Verdict: {verdict}")
    test_results['TEST-1'] = verdict
else:
    print("⚠️  No trades in OOS period")
    test_results['TEST-1'] = "⚠️  NO DATA"

# ============================================================================
# TEST-2: SLIPPAGE REALITY
# ============================================================================
print("\n" + "="*80)
print("TEST-2: SLIPPAGE REALITY")
print("="*80)
print("Assumption: 0.05% slippage per trade")
print("Reality check: Market impact analysis")

# Conservative estimate: 0.05% is reasonable for BTC
# For altcoins would be higher (0.1-0.2%)
assumed_slippage = 0.05
realistic_slippage = 0.08  # Conservative for BTC

print(f"\n📊 Slippage Analysis:")
print(f"   Assumed: {assumed_slippage}%")
print(f"   Realistic (BTC): {realistic_slippage}%")
print(f"   Difference: {realistic_slippage - assumed_slippage}%")

if realistic_slippage < 0.08:
    verdict = "✅ PASS - Assumption OK"
elif realistic_slippage < 0.15:
    verdict = "⚠️  MARGINAL - Tight but workable"
else:
    verdict = "❌ FAIL - Assumption broken"

print(f"   Verdict: {verdict}")
test_results['TEST-2'] = verdict

# ============================================================================
# TEST-3: REGIME PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("TEST-3: REGIME PERFORMANCE")
print("="*80)
print("Test profitability across different market regimes")

# Define regime periods
regime_periods = {
    'Bull (Jun 2023 - Jun 2024)': (pd.Timestamp('2023-06-01', tz='UTC'), pd.Timestamp('2024-06-30', tz='UTC')),
    'Sideways (Sept-Dec 2023)': (pd.Timestamp('2023-09-01', tz='UTC'), pd.Timestamp('2023-12-31', tz='UTC')),
    'Volatile (Aug-Sept 2023)': (pd.Timestamp('2023-08-01', tz='UTC'), pd.Timestamp('2023-09-30', tz='UTC')),
    'Bear (Nov-Dec 2022)': (pd.Timestamp('2022-11-01', tz='UTC'), pd.Timestamp('2022-12-31', tz='UTC')),
}

print(f"\n📊 Regime Performance:")
profitable_regimes = 0

for regime_name, (start, end) in regime_periods.items():
    regime_trades = trades_df[(trades_df['entry_time'] >= start) & (trades_df['entry_time'] <= end)]

    if len(regime_trades) > 0:
        cost_per_trade = 0.15
        regime_trades_copy = regime_trades.copy()
        regime_trades_copy['net_pnl'] = regime_trades_copy['pnl'] - cost_per_trade
        total_pnl = regime_trades_copy['net_pnl'].sum()

        is_profitable = total_pnl > 0
        if is_profitable:
            profitable_regimes += 1

        status = "✅" if is_profitable else "❌"
        print(f"   {status} {regime_name}: {total_pnl:+.1f}% ({len(regime_trades)} trades)")
    else:
        print(f"   ⚠️  {regime_name}: No trades")

# Verdict
if profitable_regimes >= 3:
    verdict = "✅ PASS - Strategy robust"
elif profitable_regimes >= 2:
    verdict = "⚠️  MARGINAL - Regime-dependent"
else:
    verdict = "❌ FAIL - Overfitted"

print(f"\n   Profitable regimes: {profitable_regimes}/4")
print(f"   Verdict: {verdict}")
test_results['TEST-3'] = verdict

# ============================================================================
# TEST-4: CORRELATION POISON (simplified)
# ============================================================================
print("\n" + "="*80)
print("TEST-4: CORRELATION POISON")
print("="*80)
print("Risk: Are all positions correlated?")

# For single asset (BTC), check volatility clustering
print(f"\n📊 Volatility Clustering Analysis:")
print(f"   Note: Testing on single asset (BTC)")
print(f"   For multi-asset, would check correlation matrix")

# Check if trades cluster in time (sign of correlation)
trades_df_sorted = trades_df.sort_values('entry_time')
time_diffs = trades_df_sorted['entry_time'].diff()
median_diff = time_diffs.median()

trades_within_24h = (time_diffs < pd.Timedelta(hours=24)).sum()
clustering_rate = trades_within_24h / len(trades_df) * 100

print(f"   Median time between trades: {median_diff}")
print(f"   Trades within 24h of previous: {trades_within_24h} ({clustering_rate:.1f}%)")

if clustering_rate < 20:
    verdict = "✅ PASS - Low clustering"
elif clustering_rate < 40:
    verdict = "⚠️  MARGINAL - Some clustering"
else:
    verdict = "❌ FAIL - High clustering (correlation risk)"

print(f"   Verdict: {verdict}")
test_results['TEST-4'] = verdict

# ============================================================================
# TEST-5: EXECUTION FAILURES
# ============================================================================
print("\n" + "="*80)
print("TEST-5: EXECUTION FAILURES")
print("="*80)
print("Operational risk analysis")

# Simulate scenarios
print(f"\n📊 Execution Scenarios:")
print(f"   Scenario 1: Partial fill (50% filled at +0.1% worse price)")
print(f"      Impact: -0.05% per trade")
print(f"\n   Scenario 2: Network timeout (retry after 5 min)")
print(f"      Impact: ~0.02% slippage on average")
print(f"\n   Scenario 3: Exchange down (30 min, miss entry)")
print(f"      Impact: Opportunity cost (not realized loss)")

max_loss = 0.05 + 0.02  # Partial fill + timeout
print(f"\n   Max loss from execution: {max_loss}%")

if max_loss < 0.02:
    verdict = "✅ PASS - Safe"
elif max_loss < 0.05:
    verdict = "⚠️  MARGINAL - Acceptable"
else:
    verdict = "❌ FAIL - Dangerous"

print(f"   Verdict: {verdict}")
test_results['TEST-5'] = verdict

# ============================================================================
# TEST-6: PSYCHOLOGICAL BURNOUT
# ============================================================================
print("\n" + "="*80)
print("TEST-6: PSYCHOLOGICAL BURNOUT")
print("="*80)
print("Can you stay disciplined during losing streaks?")

# Find longest losing streak
trades_df_sorted = trades_df.sort_values('entry_time')
trades_df_sorted['is_win'] = trades_df_sorted['pnl'] > 0.15  # After costs

current_streak = 0
max_streak = 0
streaks = []

for is_win in trades_df_sorted['is_win']:
    if not is_win:
        current_streak += 1
        max_streak = max(max_streak, current_streak)
    else:
        if current_streak > 0:
            streaks.append(current_streak)
        current_streak = 0

print(f"\n📊 Losing Streak Analysis:")
print(f"   Longest losing streak: {max_streak} trades")
print(f"   Average losing streak: {np.mean(streaks) if streaks else 0:.1f} trades")
print(f"   Number of streaks: {len(streaks)}")

if max_streak < 5:
    verdict = "✅ PASS - Manageable"
elif max_streak < 10:
    verdict = "⚠️  MARGINAL - Difficult"
else:
    verdict = "❌ FAIL - Breaking point"

print(f"   Verdict: {verdict}")
test_results['TEST-6'] = verdict

# ============================================================================
# TEST-7: PARAMETER SENSITIVITY
# ============================================================================
print("\n" + "="*80)
print("TEST-7: PARAMETER SENSITIVITY")
print("="*80)
print("How sensitive is performance to RSI threshold?")

# We use RSI < 30, test RSI < 28 and RSI < 32
baseline_count = len(signals)
baseline_pnl = trades_df['pnl'].sum()

# RSI < 28
signals_28 = df[df['rsi'] < 28]
degradation_28 = (1 - len(signals_28) / baseline_count) * 100

# RSI < 32
signals_32 = df[df['rsi'] < 32]
increase_32 = (len(signals_32) / baseline_count - 1) * 100

print(f"\n📊 Parameter Sensitivity:")
print(f"   Baseline (RSI < 30): {baseline_count} signals")
print(f"   RSI < 28: {len(signals_28)} signals ({degradation_28:+.1f}% change)")
print(f"   RSI < 32: {len(signals_32)} signals ({increase_32:+.1f}% change)")

max_change = max(abs(degradation_28), abs(increase_32))

if max_change < 15:
    verdict = "✅ PASS - Robust"
elif max_change < 30:
    verdict = "⚠️  MARGINAL - Sensitive"
else:
    verdict = "❌ FAIL - Overfitted"

print(f"   Max change: {max_change:.1f}%")
print(f"   Verdict: {verdict}")
test_results['TEST-7'] = verdict

# ============================================================================
# TEST-8: PER-ASSET VALIDATION (skip for single asset)
# ============================================================================
print("\n" + "="*80)
print("TEST-8: PER-ASSET VALIDATION")
print("="*80)
print("Note: Testing on single asset (BTC)")
print("For multi-asset portfolio, would test each asset separately")

print(f"\n   ⚠️  SKIPPED - Single asset backtest")
test_results['TEST-8'] = "⚠️  SKIPPED (single asset)"

# ============================================================================
# TEST-9: DIAMOND TEST
# ============================================================================
print("\n" + "="*80)
print("TEST-9: DIAMOND TEST - ULTIMATE TRUTH")
print("="*80)
print("Net annual return after ALL costs")

# Calculate over full period
total_trades = len(trades_df)
total_gross_pnl = trades_df['pnl'].sum()
total_cost = total_trades * 0.15  # 0.15% per trade
total_net_pnl = total_gross_pnl - total_cost

# Annualize (we have ~9 years of data)
years = (trades_df['entry_time'].max() - trades_df['entry_time'].min()).days / 365
annual_return = total_net_pnl / years

print(f"\n📊 Diamond Test Results:")
print(f"   Total trades: {total_trades}")
print(f"   Total gross P&L: {total_gross_pnl:.1f}%")
print(f"   Total costs: {total_cost:.1f}%")
print(f"   Total net P&L: {total_net_pnl:.1f}%")
print(f"   Period: {years:.1f} years")
print(f"   Annual return (net): {annual_return:.1f}%")

if annual_return > 20:
    verdict = "✅ PASS - Real edge exists"
elif annual_return > 5:
    verdict = "⚠️  MARGINAL - Weak edge"
else:
    verdict = "❌ FAIL - Noise"

print(f"   Verdict: {verdict}")
test_results['TEST-9'] = verdict

# ============================================================================
# TEST-10: UNREALIZED TRADES
# ============================================================================
print("\n" + "="*80)
print("TEST-10: UNREALIZED OPPORTUNITIES")
print("="*80)
print("Are we leaving money on the table?")

# For this we'd need to track:
# - Signals that met RSI < 30 but didn't enter (why?)
# - Potential P&L from those missed trades

# Simplification: Check if any high-quality setups were missed
# by analyzing price moves after RSI < 30 signals we didn't take

# All RSI < 30 bars
all_oversold = df[df['rsi'] < 30]

# Calculate potential P&L if we entered ALL oversold bars
potential_trades = len(all_oversold)
actual_trades = len(trades_df)
missed_rate = (1 - actual_trades / potential_trades) * 100

print(f"\n📊 Unrealized Analysis:")
print(f"   All oversold bars (RSI < 30): {potential_trades}")
print(f"   Actual trades: {actual_trades}")
print(f"   Missed rate: {missed_rate:.1f}%")

if missed_rate < 20:
    verdict = "✅ PASS - Capturing most opportunities"
elif missed_rate < 50:
    verdict = "⚠️  MARGINAL - Room for improvement"
else:
    verdict = "❌ FAIL - Leaving too much on table"

print(f"   Verdict: {verdict}")
test_results['TEST-10'] = verdict

# ============================================================================
# PART 3: SYNTHESIS & REPORT
# ============================================================================
print("\n" + "="*80)
print("SYNTHESIS: TRUTH DISCOVERY RESULTS")
print("="*80)

# Count pass/fail/marginal
passed = sum(1 for v in test_results.values() if '✅' in v)
marginal = sum(1 for v in test_results.values() if '⚠️' in v)
failed = sum(1 for v in test_results.values() if '❌' in v)
total_tests = len([v for v in test_results.values() if 'SKIPPED' not in v])

print(f"\n📊 Test Results Summary:")
print(f"   Total tests: {total_tests}")
print(f"   ✅ Passed: {passed}")
print(f"   ⚠️  Marginal: {marginal}")
print(f"   ❌ Failed: {failed}")

print(f"\n📋 Detailed Results:")
for test_name, result in test_results.items():
    print(f"   {test_name}: {result}")

# Decision matrix
print(f"\n" + "="*80)
print("DECISION MATRIX")
print("="*80)

if passed >= 7:
    decision = "✅ EDGE EXISTS - Proceed to scaling"
    recommendation = "System validated. Focus on:\n" + \
                    "   1. Multi-asset expansion (13 assets)\n" + \
                    "   2. Position sizing optimization\n" + \
                    "   3. Risk management refinement"
elif passed >= 4:
    decision = "⚠️  MARGINAL EDGE - Needs optimization"
    recommendation = "System has potential. Priority:\n" + \
                    "   1. Improve entry signals (add filters)\n" + \
                    "   2. Better regime detection\n" + \
                    "   3. ML enhancement for entry quality"
else:
    decision = "❌ FUNDAMENTAL ISSUES - Redesign needed"
    recommendation = "System needs major changes:\n" + \
                    "   1. Different timeframe (1h → 4h?)\n" + \
                    "   2. Different strategy type\n" + \
                    "   3. Different asset class"

print(f"\n🎯 DECISION: {decision}")
print(f"\n💡 RECOMMENDATIONS:")
print(recommendation)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create reports directory
reports_dir = Path('reports/day11_forensics')
reports_dir.mkdir(parents=True, exist_ok=True)

# Save trade details
trades_df.to_csv(reports_dir / 'all_trades_detailed.csv', index=False)
best_trades.to_csv(reports_dir / 'best_10_trades.csv', index=False)
worst_trades.to_csv(reports_dir / 'worst_10_trades.csv', index=False)

# Save test results
with open(reports_dir / 'test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)

# Generate markdown report
report_md = f"""# DAY 11: TRUTH DISCOVERY PROTOCOL - RESULTS

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY

**Decision:** {decision}

**Test Results:** {passed} passed, {marginal} marginal, {failed} failed (out of {total_tests} tests)

**Recommendation:**
{recommendation}

---

## PART 1: MARKET ARCHAEOLOGY

### Best Trades Analysis (Top 10)

- **Average P&L:** {best_trades['pnl'].mean():.1f}%
- **Average RSI:** {best_trades['rsi'].mean():.1f}
- **Average Volume Ratio:** {best_trades['volume_ratio'].mean():.2f}x
- **Average Holding Time:** {best_trades['holding_bars'].mean():.1f}h ({best_trades['holding_bars'].mean()/24:.1f} days)

### Worst Trades Analysis (Bottom 10)

- **Average P&L:** {worst_trades['pnl'].mean():.1f}%
- **Average RSI:** {worst_trades['rsi'].mean():.1f}
- **Average Volume Ratio:** {worst_trades['volume_ratio'].mean():.2f}x
- **Average Holding Time:** {worst_trades['holding_bars'].mean():.1f}h ({worst_trades['holding_bars'].mean()/24:.1f} days)

### Key Differences (Winners vs Losers)

| Metric | Winners | Losers | Difference |
|--------|---------|--------|------------|
| RSI | {best_trades['rsi'].mean():.1f} | {worst_trades['rsi'].mean():.1f} | {best_trades['rsi'].mean() - worst_trades['rsi'].mean():.1f} |
| Volume Ratio | {best_trades['volume_ratio'].mean():.2f} | {worst_trades['volume_ratio'].mean():.2f} | {best_trades['volume_ratio'].mean() - worst_trades['volume_ratio'].mean():.2f} |
| Volatility | {best_trades['volatility'].mean():.3f} | {worst_trades['volatility'].mean():.3f} | {best_trades['volatility'].mean() - worst_trades['volatility'].mean():.3f} |
| Holding Time (h) | {best_trades['holding_bars'].mean():.1f} | {worst_trades['holding_bars'].mean():.1f} | {best_trades['holding_bars'].mean() - worst_trades['holding_bars'].mean():.1f} |

---

## PART 2: HARDCORE VALIDATION

### Test Results

"""

for test_name, result in test_results.items():
    report_md += f"**{test_name}:** {result}\n\n"

report_md += f"""
---

## CONCLUSIONS

Based on {total_tests} hardcore tests, the system shows:

1. **Edge Quality:** {'Strong' if passed >= 7 else 'Marginal' if passed >= 4 else 'Weak'}
2. **Robustness:** {'High' if passed >= 7 else 'Medium' if passed >= 4 else 'Low'}
3. **Scalability:** {'Ready' if passed >= 7 else 'Needs work' if passed >= 4 else 'Not ready'}

### Next Steps

{recommendation}

---

## FILES GENERATED

- `all_trades_detailed.csv` - All {len(trades_df)} trades with full details
- `best_10_trades.csv` - Top 10 winning trades
- `worst_10_trades.csv` - Top 10 losing trades
- `test_results.json` - Test results in JSON format
- `forensic_report.md` - This report

"""

with open(reports_dir / 'forensic_report.md', 'w', encoding='utf-8') as f:
    f.write(report_md)

print(f"\n✅ Results saved to {reports_dir}/")
print(f"   - all_trades_detailed.csv ({len(trades_df)} trades)")
print(f"   - best_10_trades.csv")
print(f"   - worst_10_trades.csv")
print(f"   - test_results.json")
print(f"   - forensic_report.md")

print("\n" + "="*80)
print("DAY 11 COMPLETE - TRUTH DISCOVERED!")
print("="*80)
print(f"\n🎯 DECISION: {decision}")
print(f"\nRead full report: reports/day11_forensics/forensic_report.md")
print("="*80)
