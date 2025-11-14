#!/usr/bin/env python3
"""
TP/SL GRID SEARCH OPTIMIZATION
===============================

Finds optimal Take-Profit and Stop-Loss levels

Test matrix:
- TP: 0.5% to 5% in 0.5% increments
- SL: 0.25% to 3% in 0.25% increments

Metrics tracked:
- Win Rate
- Profit Factor
- Total P&L
- Sharpe ratio (approximated)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from backtesting.backtest_pjs_framework import (
    PjSBacktestEngine, BacktestConfig, RuleBasedStrategy
)
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def run_backtest(tp_pct, sl_pct):
    """Run backtest with specific TP/SL parameters"""
    project_root = Path(__file__).parent.parent
    btc_file = project_root / "data" / "raw" / "BTC_USDT_15m.parquet"
    ohlcv = pd.read_parquet(btc_file)

    # Generate signals
    strategy = RuleBasedStrategy(rsi_threshold=30, period=14)
    signals = strategy.generate_signals(ohlcv)
    ml_scores = signals.astype(float) * 0.7

    # Configure backtest with this TP/SL
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.95,
        take_profit=tp_pct / 100,  # Convert % to decimal
        stop_loss=sl_pct / 100,
        max_hold_bars=288,
        commission=0.001,
        slippage=0.0005,
        cooldown_bars=10,
        ml_enabled=False,
        filters_enabled=True,
        opportunity_enabled=True,
        cost_enabled=True,
        risk_penalty_enabled=True,
    )

    # Run backtest
    backtest = PjSBacktestEngine(config)
    results = backtest.run(ohlcv=ohlcv, raw_signals=signals, ml_scores=ml_scores)

    return results


def main():
    print("\n" + "="*100)
    print("TP/SL GRID SEARCH OPTIMIZATION")
    print("="*100)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define grid
    tp_values = np.arange(0.5, 5.1, 0.5)  # 0.5% to 5%
    sl_values = np.arange(0.25, 3.1, 0.25)  # 0.25% to 3%

    print(f"\nGrid: TP {tp_values.tolist()} / SL {sl_values.tolist()}")
    print(f"Total combinations: {len(tp_values) * len(sl_values)}")

    results_list = []
    total_runs = len(tp_values) * len(sl_values)
    run_count = 0

    print(f"\n{'TP %':>6} {'SL %':>6} {'Trades':>8} {'WR %':>8} {'PF':>8} {'P&L $':>12} {'Return %':>10}")
    print("-" * 80)

    for tp in tp_values:
        for sl in sl_values:
            run_count += 1
            try:
                results = run_backtest(tp, sl)

                results_list.append({
                    'TP': tp,
                    'SL': sl,
                    'Trades': results['trades'],
                    'WR': results['win_rate'],
                    'PF': results['profit_factor'],
                    'PnL': results['total_pnl'],
                    'Return': results['total_pnl_pct'],
                    'Capital': results['final_capital']
                })

                print(f"{tp:6.2f} {sl:6.2f} {results['trades']:8d} {results['win_rate']:8.1f} "
                      f"{results['profit_factor']:8.2f} ${results['total_pnl']:11,.0f} "
                      f"{results['total_pnl_pct']:9.2f}%")

            except Exception as e:
                print(f"{tp:6.2f} {sl:6.2f} ERROR: {str(e)[:40]}")

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)

    if not results_list:
        print("❌ No results generated")
        return

    results_df = pd.DataFrame(results_list)

    # Find best by different metrics
    print("\nTop Results by Win Rate:")
    top_wr = results_df.nlargest(5, 'WR')[['TP', 'SL', 'Trades', 'WR', 'PF', 'PnL', 'Return']]
    print(top_wr.to_string(index=False))

    print("\nTop Results by Profit Factor:")
    top_pf = results_df.nlargest(5, 'PF')[['TP', 'SL', 'Trades', 'WR', 'PF', 'PnL', 'Return']]
    print(top_pf.to_string(index=False))

    print("\nTop Results by Total P&L:")
    top_pnl = results_df.nlargest(5, 'PnL')[['TP', 'SL', 'Trades', 'WR', 'PF', 'PnL', 'Return']]
    print(top_pnl.to_string(index=False))

    # Find best balanced (WR > 40%, PF > 1.0)
    print("\nBalanced Results (WR > 40%, PF > 1.0):")
    balanced = results_df[(results_df['WR'] > 40) & (results_df['PF'] > 1.0)].nlargest(5, 'Return')
    if len(balanced) > 0:
        print(balanced[['TP', 'SL', 'Trades', 'WR', 'PF', 'PnL', 'Return']].to_string(index=False))
    else:
        print("No results meeting criteria")

    # Find best overall recommendation
    best_overall = results_df.loc[results_df['PnL'].idxmax()]
    print(f"\n🏆 BEST OVERALL RECOMMENDATION:")
    print(f"   TP: {best_overall['TP']:.2f}%")
    print(f"   SL: {best_overall['SL']:.2f}%")
    print(f"   Trades: {best_overall['Trades']}")
    print(f"   Win Rate: {best_overall['WR']:.1f}%")
    print(f"   Profit Factor: {best_overall['PF']:.2f}")
    print(f"   Total P&L: ${best_overall['PnL']:,.0f}")
    print(f"   Return: {best_overall['Return']:.2f}%")

    # Save results
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_file = reports_dir / f"tp_sl_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(report_file, index=False)
    print(f"\n✅ Results saved to: {report_file.name}")

    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
