"""
PHASE 2: WALK-FORWARD VALIDATION
==================================

Test full system performance year-by-year on real data.
No lookahead bias, no parameter fitting to test period.

Methodology:
- Use SAME strategy across all years (no re-optimization)
- Test Naive, PM, and Hybrid on each year
- Calculate metrics: P&L, Sharpe, DD, Win Rate, Trades
- Identify which years worked, which failed

Expected insights:
- Bull years (2020-2021, 2024): Should profit
- Bear years (2022): Should survive or minimal loss
- Sideways years (2019, 2023): Modest gains
- Overall: Realistic annual returns

Author: Scarlet Sails Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from models.position_manager import PositionManager
from models.hybrid_position_manager import HybridPositionManager

print("="*80)
print("PHASE 2: WALK-FORWARD VALIDATION")
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
# GENERATE ENTRY SIGNALS (SAME AS PHASE 1.3)
# ============================================================================
print("\n📊 Generating entry signals (RSI < 30)...")

entry_signals = []
for i in range(200, len(df)):
    if df['rsi'].iloc[i] < 30:
        if not entry_signals or (i - entry_signals[-1]['bar_index'] > 24):
            entry_signals.append({
                'bar_index': i,
                'timestamp': df.index[i],
                'price': df['close'].iloc[i],
                'year': df.index[i].year,
            })

print(f"   ✅ Generated {len(entry_signals)} entry signals")

# Group by year
signals_df = pd.DataFrame(entry_signals)
signals_by_year = signals_df.groupby('year').size()

print("\n   Signals per year:")
for year, count in signals_by_year.items():
    print(f"      {year}: {count}")

# ============================================================================
# BACKTEST FUNCTIONS (SAME AS comprehensive_exit_test_REAL.py)
# ============================================================================

def backtest_naive_yearly(df, signals, year):
    """Naive strategy for specific year"""
    year_signals = [s for s in signals if s['year'] == year]

    trades = []
    for sig in year_signals:
        entry_bar = sig['bar_index']
        entry_price = sig['price']

        tp_price = entry_price * 1.15
        sl_price = entry_price * 0.95

        for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            if high >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price
                trades.append({'pnl_pct': pnl_pct, 'exit_reason': 'TP'})
                break
            elif low <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                trades.append({'pnl_pct': pnl_pct, 'exit_reason': 'SL'})
                break

    return trades

def backtest_pm_yearly(df, signals, year):
    """Position Manager for specific year"""
    year_signals = [s for s in signals if s['year'] == year]

    pm = PositionManager(
        max_holding_time_bars=168,
        enable_trailing=True,
        enable_partial_exits=True,
    )

    trades = []
    for sig in year_signals:
        entry_bar = sig['bar_index']
        entry_price = sig['price']
        entry_time = sig['timestamp']

        position = pm.open_position(
            symbol='BTC/USDT',
            entry_price=entry_price,
            entry_time=entry_time,
            size=1.0,
            direction='long',
            df=df,
            current_bar=entry_bar,
            regime='BULL',
        )

        for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
            current_price = df['close'].iloc[i]

            try:
                position, exit_signals = pm.update_position('BTC/USDT', current_price, df, i)

                if exit_signals:
                    exit_price = exit_signals[0]['price']
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append({'pnl_pct': pnl_pct, 'exit_reason': exit_signals[0]['label']})
                    pm.execute_exits('BTC/USDT', exit_signals)
                    break
            except:
                break

    return trades

def backtest_hybrid_yearly(df, signals, year):
    """Hybrid for specific year"""
    year_signals = [s for s in signals if s['year'] == year]

    hybrid = HybridPositionManager(max_holding_time_bars=168)

    trades = []
    for sig in year_signals:
        entry_bar = sig['bar_index']
        entry_price = sig['price']
        entry_time = sig['timestamp']

        position = hybrid.open_position(
            symbol='BTC/USDT',
            entry_price=entry_price,
            entry_time=entry_time,
            size=1.0,
            direction='long',
            df=df,
            current_bar=entry_bar,
        )

        for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
            current_price = df['close'].iloc[i]

            try:
                position, exit_signals = hybrid.update_position('BTC/USDT', current_price, df, i)

                if exit_signals:
                    exit_price = exit_signals[0]['price']
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trades.append({'pnl_pct': pnl_pct, 'exit_reason': exit_signals[0]['label']})
                    hybrid.execute_exits('BTC/USDT', exit_signals)
                    break
            except:
                break

    return trades

def calc_metrics_yearly(trades):
    """Calculate yearly metrics"""
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'pf': 0}

    pnls = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / len(trades) if trades else 0
    total_pnl = sum(pnls) * 100

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 1e-6
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'pf': pf
    }

# ============================================================================
# RUN YEAR-BY-YEAR TESTS
# ============================================================================
print("\n" + "="*80)
print("RUNNING YEAR-BY-YEAR BACKTESTS")
print("="*80)

years = sorted(signals_df['year'].unique())
results = []

for year in years:
    print(f"\n📊 Testing {year}...")

    year_signals_count = signals_by_year[year]
    print(f"   Signals: {year_signals_count}")

    # Test all three strategies
    naive_trades = backtest_naive_yearly(df, entry_signals, year)
    pm_trades = backtest_pm_yearly(df, entry_signals, year)
    hybrid_trades = backtest_hybrid_yearly(df, entry_signals, year)

    naive_m = calc_metrics_yearly(naive_trades)
    pm_m = calc_metrics_yearly(pm_trades)
    hybrid_m = calc_metrics_yearly(hybrid_trades)

    print(f"   Naive:  {naive_m['trades']} trades, {naive_m['total_pnl']:+.1f}% P&L, WR {naive_m['win_rate']:.1%}")
    print(f"   PM:     {pm_m['trades']} trades, {pm_m['total_pnl']:+.1f}% P&L, WR {pm_m['win_rate']:.1%}")
    print(f"   Hybrid: {hybrid_m['trades']} trades, {hybrid_m['total_pnl']:+.1f}% P&L, WR {hybrid_m['win_rate']:.1%}")

    results.append({
        'year': year,
        'signals': year_signals_count,
        'naive_pnl': naive_m['total_pnl'],
        'pm_pnl': pm_m['total_pnl'],
        'hybrid_pnl': hybrid_m['total_pnl'],
        'naive_wr': naive_m['win_rate'],
        'pm_wr': pm_m['win_rate'],
        'hybrid_wr': hybrid_m['win_rate'],
        'naive_pf': naive_m['pf'],
        'pm_pf': pm_m['pf'],
        'hybrid_pf': hybrid_m['pf'],
    })

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

print("\n📊 Year-by-Year Performance (Total P&L %):")
print(f"\n{'Year':<6} {'Signals':<10} {'Naive':>10} {'PM':>10} {'Hybrid':>10}")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"{row['year']:<6} {row['signals']:<10} {row['naive_pnl']:>9.1f}% {row['pm_pnl']:>9.1f}% {row['hybrid_pnl']:>9.1f}%")

print("\n📊 Year-by-Year Win Rate:")
print(f"\n{'Year':<6} {'Naive':>10} {'PM':>10} {'Hybrid':>10}")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"{row['year']:<6} {row['naive_wr']:>9.1%} {row['pm_wr']:>9.1%} {row['hybrid_wr']:>9.1%}")

# Overall totals
print("\n📊 OVERALL TOTALS (all years combined):")
print(f"   Naive:  Total P&L: {results_df['naive_pnl'].sum():+.1f}%")
print(f"   PM:     Total P&L: {results_df['pm_pnl'].sum():+.1f}%")
print(f"   Hybrid: Total P&L: {results_df['hybrid_pnl'].sum():+.1f}%")

years_count = len(years)
print(f"\n📊 ANNUALIZED RETURNS ({years_count} years):")
print(f"   Naive:  {results_df['naive_pnl'].sum() / years_count:.1f}% per year")
print(f"   PM:     {results_df['pm_pnl'].sum() / years_count:.1f}% per year")
print(f"   Hybrid: {results_df['hybrid_pnl'].sum() / years_count:.1f}% per year")

# Identify best/worst years
best_hybrid_year = results_df.loc[results_df['hybrid_pnl'].idxmax()]
worst_hybrid_year = results_df.loc[results_df['hybrid_pnl'].idxmin()]

print(f"\n📊 BEST/WORST YEARS (Hybrid):")
print(f"   Best:  {best_hybrid_year['year']} ({best_hybrid_year['hybrid_pnl']:+.1f}%)")
print(f"   Worst: {worst_hybrid_year['year']} ({worst_hybrid_year['hybrid_pnl']:+.1f}%)")

# Count profitable years
profitable_years = (results_df['hybrid_pnl'] > 0).sum()
print(f"\n📊 CONSISTENCY:")
print(f"   Profitable years: {profitable_years}/{years_count} ({profitable_years/years_count:.1%})")

print("\n" + "="*80)
print("✅ PHASE 2 COMPLETE - Walk-forward validation done!")
print("="*80)

print("\n💡 KEY INSIGHTS:")
print(f"   - System tested on {years_count} years of real data")
print(f"   - Hybrid best: {best_hybrid_year['year']} ({best_hybrid_year['hybrid_pnl']:+.1f}%)")
print(f"   - Hybrid worst: {worst_hybrid_year['year']} ({worst_hybrid_year['hybrid_pnl']:+.1f}%)")
print(f"   - Consistency: {profitable_years}/{years_count} profitable years")
print(f"   - Realistic annual return: {results_df['hybrid_pnl'].sum() / years_count:.1f}%/year")
