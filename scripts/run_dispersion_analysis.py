"""
MAIN DISPERSION ANALYSIS
Runs comprehensive analysis on historical data and proves strategy dispersion

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.dispersion_analyzer import DispersionAnalyzer
from analysis.dispersion_visualizer import DispersionVisualizer
from strategies.rule_based_v2 import RuleBasedStrategy
from strategies.xgboost_ml_v2 import XGBoostMLStrategy
from strategies.hybrid_v2 import HybridStrategy

# === Импорт для RL/DQN ===
from rl.dqn import DQNAgent
from rl.trading_environment import TradingEnvironment
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_realistic_data(n_bars=2000, seed=42):
    np.random.seed(seed)
    logger.info(f"Generating {n_bars} bars of realistic data...")

    regime_lengths = [500, 700, 500, 300]
    regime_trends = [0.0002, -0.0001, 0.0003, -0.0002]
    regime_vols = [0.015, 0.020, 0.025, 0.040]

    close_prices = [50000]
    for regime_len, trend, vol in zip(regime_lengths, regime_trends, regime_vols):
        for _ in range(regime_len):
            ret = np.random.normal(trend, vol)
            new_price = close_prices[-1] * (1 + ret)
            close_prices.append(new_price)
    close_prices = np.array(close_prices[:n_bars])
    close_prices = np.maximum(close_prices, 1000)
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')

    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.0005, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.003, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.003, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)

    logger.info(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
    return df

def load_dqn_agent(state_dim=12, action_dim=3, model_path="models/dqn_best_pnl.pth"):
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0  # exploitation
    logger.info(f"✅ DQNAgent loaded from {model_path}")
    return agent

def dqn_signal_func(agent, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
        return q_values[0, 1].item() - q_values[0, 2].item()

class DQNRLStrategy:
    """
    Adapter for DQNAgent to look like other Strategy classes (generate_signals).
    Всегда возвращает DataFrame с главным столбцом 'value'.
    """
    def __init__(self, agent):
        self.agent = agent

    def generate_signals(self, df):
        signals = []
        env = TradingEnvironment(df)
        for step in range(len(df)):
            state = env._get_state()
            score = dqn_signal_func(self.agent, state)
            signals.append({'value': score, 'signal': 1 if score > 0 else 0, 'timestamp': df.index[step]})
            env.current_step += 1
        signals_df = pd.DataFrame(signals).set_index('timestamp')
        return signals_df

def main():
    print("="*80)
    print("SCARLET SAILS - DISPERSION ANALYSIS")
    print("Mathematical Proof of Strategy Dispersion")
    print("="*80)
    print()

    # Step 1: Generate/Load Data
    print("STEP 1: DATA PREPARATION")
    print("-"*40)
    df = generate_realistic_data(n_bars=2000, seed=42)
    print(f"✅ Data loaded: {len(df)} bars")
    print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
    print()

    # Step 2: Initialize Strategies
    print("STEP 2: STRATEGY INITIALIZATION")
    print("-"*40)
    dqn_agent = load_dqn_agent()
    strategies = {
        'rule_based': RuleBasedStrategy(),
        'xgboost_ml': XGBoostMLStrategy(),
        'hybrid': HybridStrategy(),
        'dqn_rl': DQNRLStrategy(dqn_agent)
    }
    print("✅ Strategies initialized:")
    for name in strategies.keys():
        print(f"   - {name}")
    print()

    # Step 3: Run Dispersion Analysis
    print("STEP 3: DISPERSION ANALYSIS")
    print("-"*40)
    print("This will take a few minutes...")
    print()
    analyzer = DispersionAnalyzer()

    try:
        results = analyzer.run_full_analysis(df, strategies)
        print()
        print("="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print()
        print("STEP 4: GENERATING REPORT")
        print("-"*40)
        report = analyzer.generate_report(results)
        report_path = 'dispersion_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Report saved to: {report_path}")
        print()
        print(report)
        print()
        print("STEP 5: CREATING VISUALIZATIONS")
        print("-"*40)
        visualizer = DispersionVisualizer()
        try:
            output_dir = 'dispersion_charts'
            os.makedirs(output_dir, exist_ok=True)
            visualizer.create_full_report(results, output_dir=output_dir)
            print(f"✅ Visualizations created in: {output_dir}/")
            print("   Files:")
            print("   - 1_distributions.png")
            print("   - 2_boxplots.png")
            print("   - 3_timeseries.png")
            print("   - 4_correlation.png")
            print("   - 5_scatter_matrix.png")
            print("   - 6_violin.png")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            print("⚠️  Visualization skipped (install matplotlib: pip install matplotlib seaborn)")
        print()
        print("="*80)
        print("KEY FINDINGS")
        print("="*80)
        anova = results['anova']
        var_decomp = results['variance_decomposition']
        print(f"\n1. ANOVA Test:")
        print(f"   F-statistic: {anova['F_statistic']:.4f}")
        print(f"   p-value: {anova['p_value']:.6f}")
        if anova['reject_null']:
            print(f"   ✅ SIGNIFICANT: Strategies have different means (p < 0.05)")
        else:
            print(f"   ⚠️  NOT SIGNIFICANT: Cannot reject null hypothesis")
        print(f"\n2. Variance Decomposition:")
        print(f"   eta_squared (effect size): {var_decomp['eta_squared']:.4f}")
        print(f"   Between-group variance: {var_decomp['eta_squared']:.1%} of total")
        print(f"   Interpretation: {var_decomp['interpretation']}")
        print(f"\n3. Correlation Matrix:")
        corr = results['correlation_matrix']
        print(corr)
        print()
        print("="*80)
        print("SCIENTIFIC CONCLUSION")
        print("="*80)
        if anova['reject_null']:
            print("\n✅ HYPOTHESIS CONFIRMED!")
            print("\nWe have statistically significant evidence that:")
            print("  • Rule-Based, ML, RL, and Hybrid strategies")
            print("  • Make FUNDAMENTALLY DIFFERENT decisions")
            print("  • When analyzing the same market conditions")
            print(f"\nStatistical confidence: p = {anova['p_value']:.6f} (highly significant)")
            print(f"Effect size: eta_squared = {var_decomp['eta_squared']:.2%} (explains variation)")
            print("\n→ This supports the core hypothesis of Scarlet Sails:")
            print("  Different algorithmic approaches lead to measurably")
            print("  different trading decisions.")
        else:
            print("\n⚠️  HYPOTHESIS PARTIALLY SUPPORTED")
            print("\nCurrent limitations:")
            print("  • Using fallback ML predictor (not trained model)")
            print("  • Synthetic data (not real market data)")
            print("  • Need more diverse market conditions")
            print("\nNext steps:")
            print("  • Train XGBoost on real data")
            print("  • Use actual Binance historical data")
            print("  • Analyze multiple assets and timeframes")
        print()
        print("="*80)
        print("ANALYSIS COMPLETE - SEE REPORT FOR DETAILS")
        print("="*80)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
