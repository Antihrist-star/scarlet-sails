#!/usr/bin/env python3
"""
TEST: P_j(S) Framework - VERSION 1
===================================

Тестирует базовую интеграцию с Rule-Based стратегией

Stage 1 (V1):
  - Rule-Based strategy
  - ТОЛЬКО costs (остальные компоненты disabled)
  - Baseline для сравнения

Следующие этапы:
  - V2: + OpportunityScorer
  - V3: + Filters (crisis detection, regime)
  - V4: + RiskPenalty
  - V5: Full integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting.backtest_pjs_framework import (
    PjSBacktestEngine, BacktestConfig, RuleBasedStrategy
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*80)
    print("P_j(S) FRAMEWORK TEST - VERSION 1 (Rule-Based + Costs)")
    print("="*80 + "\n")

    # Setup
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: Load data (простой BTC для начала)
    # =========================================================================
    print("[1/4] Loading BTC data...")
    btc_file = data_dir / "BTC_USDT_15m.parquet"
    if not btc_file.exists():
        logger.error(f"Data not found: {btc_file}")
        return

    ohlcv = pd.read_parquet(btc_file)
    logger.info(f"Loaded {len(ohlcv)} bars")

    # =========================================================================
    # STEP 2: Generate signals (Rule-Based: RSI < 30)
    # =========================================================================
    print("[2/4] Generating Rule-Based signals (RSI < 30)...")
    strategy = RuleBasedStrategy(rsi_threshold=30, period=14)
    signals = strategy.generate_signals(ohlcv)
    signal_count = np.sum(signals)
    logger.info(f"Generated {signal_count} signals")

    # =========================================================================
    # STEP 3: Configure P_j(S) backtest
    # =========================================================================
    print("[3/4] Configuring P_j(S) backtest...")

    # VERSION 1: Only costs enabled
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.95,
        take_profit=0.02,      # 2%
        stop_loss=0.01,        # 1%
        max_hold_bars=288,     # 3 days for 15m
        commission=0.001,      # 0.1% per side
        slippage=0.0005,       # 0.05%
        cooldown_bars=10,

        # Component status
        ml_enabled=False,           # No ML yet (using signals from strategy)
        filters_enabled=False,       # NO filters
        opportunity_enabled=False,   # NO opportunity scorer
        cost_enabled=True,          # YES costs
        risk_penalty_enabled=False,  # NO risk penalty
        ml_threshold=0.5            # Not used (ml_enabled=False)
    )

    logger.info(f"Config: TP={config.take_profit*100:.1f}%, "
               f"SL={config.stop_loss*100:.1f}%, "
               f"Costs enabled={config.cost_enabled}")

    # =========================================================================
    # STEP 4: Run backtest
    # =========================================================================
    print("[4/4] Running P_j(S) backtest (V1)...")

    # For V1, use simple base scores (0 for no signal, 0.7 for signal)
    # In V2+, these will be replaced with real ML scores
    ml_scores = signals.astype(float) * 0.7  # 0 for no signal, 0.7 for rule-based signal

    backtest = PjSBacktestEngine(config)
    results = backtest.run(
        ohlcv=ohlcv,
        raw_signals=signals,
        ml_scores=ml_scores  # Explicit scores
    )

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS - P_j(S) FRAMEWORK V1")
    print("="*80)

    print(f"\nTrading Summary:")
    print(f"  Initial Capital: ${config.initial_capital:,.0f}")
    print(f"  Final Capital:   ${results['final_capital']:,.0f}")
    print(f"  Total P&L:       ${results['total_pnl']:,.0f} ({results['total_pnl_pct']:+.1f}%)")
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {results['trades']}")
    print(f"  Wins:            {results['wins']}")
    print(f"  Losses:          {results['losses']}")
    print(f"  Win Rate:        {results['win_rate']:.1f}%")
    print(f"  Profit Factor:   {results['profit_factor']:.2f}")

    # Breakdown of trades
    if results['trades'] > 0:
        print(f"\nTrade Breakdown:")
        print(f"  TP exits:   {sum(1 for t in results['trades_detail'] if t.exit_reason == 'tp')}")
        print(f"  SL exits:   {sum(1 for t in results['trades_detail'] if t.exit_reason == 'sl')}")
        print(f"  TIME exits: {sum(1 for t in results['trades_detail'] if t.exit_reason == 'time')}")

        # Sample trades (first 5)
        print(f"\nFirst 5 trades:")
        for i, trade in enumerate(results['trades_detail'][:5]):
            print(f"  {i+1}. {trade}")

    # =========================================================================
    # ANALYSIS: What components did what?
    # =========================================================================
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS")
    print("="*80)

    if results['trades'] > 0:
        avg_ml = np.mean([t.ml_score for t in results['trades_detail']])
        avg_filter = np.mean([t.filter_product for t in results['trades_detail']])
        avg_opportunity = np.mean([t.opportunity_score for t in results['trades_detail']])
        avg_costs = np.mean([t.costs for t in results['trades_detail']])
        avg_risk_penalty = np.mean([t.risk_penalty for t in results['trades_detail']])
        avg_pjs = np.mean([t.total_pjs for t in results['trades_detail']])

        print(f"\nAverage P_j(S) Components (across {results['trades']} trades):")
        print(f"  ML Score:           {avg_ml:.4f} (disabled - N/A)")
        print(f"  Filter Product:     {avg_filter:.4f} (disabled)")
        print(f"  Opportunity Score:  {avg_opportunity:.4f} (disabled)")
        print(f"  Costs:              {avg_costs:.4f}")
        print(f"  Risk Penalty:       {avg_risk_penalty:.4f} (disabled)")
        print(f"  ─────────────────────────────")
        print(f"  Final P_j(S):       {avg_pjs:.4f}")

    # =========================================================================
    # NEXT STEPS
    # =========================================================================
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
✅ V1 DONE: Rule-Based with costs
🟡 V2 TODO: Add OpportunityScorer (currently disabled)
🟡 V3 TODO: Add Filters (crisis, regime)
🟡 V4 TODO: Add RiskPenalty
🟡 V5 TODO: Full integration + ML

Change config flags to test different components:
  - opportunity_enabled=True for V2
  - filters_enabled=True for V3
  - risk_penalty_enabled=True for V4
    """)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    report_file = reports_dir / "pjs_framework_v1_test.txt"
    with open(report_file, 'w') as f:
        f.write("P_j(S) FRAMEWORK TEST - V1\n")
        f.write("="*80 + "\n\n")
        f.write(f"Trades: {results['trades']}\n")
        f.write(f"Win Rate: {results['win_rate']:.1f}%\n")
        f.write(f"Profit Factor: {results['profit_factor']:.2f}\n")
        f.write(f"Total P&L: ${results['total_pnl']:,.0f} ({results['total_pnl_pct']:+.1f}%)\n")

    logger.info(f"Results saved to {report_file}")

    print(f"\n✅ Test complete! Results saved to {report_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
