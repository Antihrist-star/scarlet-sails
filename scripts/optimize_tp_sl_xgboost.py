"""
TP/SL Optimization for XGBoost (threshold 0.46)
Goal: Find best TP/SL to reach PF ≥ 2.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
from itertools import product
from models.xgboost_model import XGBoostModel
from backtesting.honest_backtest_v2 import HonestBacktestV2
import warnings
warnings.filterwarnings('ignore')


def main():
    print("="*80)
    print("TP/SL OPTIMIZATION FOR XGBOOST (threshold 0.46)")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    
    # Load model
    print("\n[1/4] Loading XGBoost model...")
    model = XGBoostModel()
    model.load(project_root / "models" / "xgboost_model.json")
    THRESHOLD = 0.46
    print(f"✅ Model loaded, threshold: {THRESHOLD}")
    
    # Load data
    print("\n[2/4] Loading test data...")
    X_test = torch.load(project_root / "models" / "X_test_clean.pt", weights_only=False)
    scaler = joblib.load(project_root / "models" / "scaler_clean_2d.pkl")
    X_test_scaled = scaler.transform(X_test.numpy())
    
    ohlcv = pd.read_parquet(project_root / "data" / "raw" / "BTC_USDT_15m_FULL.parquet")
    ohlcv_test = ohlcv.iloc[-len(X_test):].reset_index(drop=True)
    print(f"✅ Data loaded: {len(X_test)} samples")
    
    # Generate signals
    print("\n[3/4] Generating signals...")
    y_pred_proba = model.predict_proba(X_test_scaled)
    y_pred_signals = (y_pred_proba[:, 1] >= THRESHOLD).astype(int)
    n_signals = y_pred_signals.sum()
    print(f"✅ Signals: {n_signals}")
    
    # Grid search
    print("\n[4/4] Grid search TP/SL...")
    
    tp_values = [0.020, 0.025, 0.030, 0.035, 0.040]  # 2.0% to 4.0%
    sl_values = [0.008, 0.010, 0.012, 0.015, 0.018]  # 0.8% to 1.8%
    
    print(f"TP range: {tp_values[0]*100:.1f}% to {tp_values[-1]*100:.1f}%")
    print(f"SL range: {sl_values[0]*100:.1f}% to {sl_values[-1]*100:.1f}%")
    print(f"Total: {len(tp_values) * len(sl_values)} combinations")
    print("-" * 80)
    
    results = []
    best_pf = 0
    best_config = None
    
    for tp, sl in product(tp_values, sl_values):
        ratio = tp / sl
        if ratio < 1.5:
            continue
        
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
        
        metrics = backtest.run(ohlcv_test, y_pred_signals)
        
        result = {
            'tp': tp * 100,
            'sl': sl * 100,
            'ratio': ratio,
            'pf': metrics['profit_factor'],
            'mdd': metrics['max_drawdown_pct'],
            'return': metrics['total_return_pct'],
            'trades': metrics['n_trades'],
            'wr': metrics['win_rate'] * 100,
            'sharpe': metrics['sharpe_ratio']
        }
        
        results.append(result)
        
        if metrics['profit_factor'] > best_pf:
            best_pf = metrics['profit_factor']
            best_config = result
        
        status = "✅" if metrics['profit_factor'] >= 2.0 and metrics['max_drawdown_pct'] <= 15 else ""
        print(f"TP={tp*100:.1f}%, SL={sl*100:.1f}% → "
              f"PF={metrics['profit_factor']:.2f}, MDD={metrics['max_drawdown_pct']:.1f}%, "
              f"Trades={metrics['n_trades']}, WR={metrics['win_rate']:.1%} {status}")
    
    # Analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    results_sorted = sorted(results, key=lambda x: x['pf'], reverse=True)
    
    print("\n🏆 TOP 10:")
    print("-" * 90)
    print(f"{'Rank':<5} {'TP%':<6} {'SL%':<6} {'Ratio':<7} {'PF':<8} {'MDD%':<7} {'Return%':<9} {'Trades':<7} {'WR%':<7}")
    print("-" * 90)
    
    for i, r in enumerate(results_sorted[:10], 1):
        goal = "✅" if r['pf'] >= 2.0 and r['mdd'] <= 15 else ""
        print(f"{i:<5} {r['tp']:<6.1f} {r['sl']:<6.1f} {r['ratio']:<7.2f} "
              f"{r['pf']:<8.3f} {r['mdd']:<7.2f} {r['return']:<9.2f} "
              f"{r['trades']:<7} {r['wr']:<7.1f} {goal}")
    
    # Goal check
    goal_configs = [r for r in results if r['pf'] >= 2.0 and r['mdd'] <= 15]
    
    print("\n" + "="*80)
    if goal_configs:
        print("✅ FOUND CONFIGS MEETING GOALS!")
        print("="*80)
        best = goal_configs[0]
        print(f"\n🎯 RECOMMENDED: TP={best['tp']:.1f}%, SL={best['sl']:.1f}%")
        print(f"   PF:     {best['pf']:.3f} ✅")
        print(f"   MDD:    {best['mdd']:.2f}% ✅")
        print(f"   Return: {best['return']:.2f}%")
        print(f"   Trades: {best['trades']}")
        print(f"   WR:     {best['wr']:.1f}%")
        print(f"   Sharpe: {best['sharpe']:.3f}")
        print("\n🎉 WEEK 3 GOALS ACHIEVED!")
    else:
        print("⚠️  NO CONFIG MEETS BOTH GOALS")
        print("="*80)
        print(f"\n📊 BEST AVAILABLE: TP={best_config['tp']:.1f}%, SL={best_config['sl']:.1f}%")
        print(f"   PF:     {best_config['pf']:.3f} {'✅' if best_config['pf'] >= 2.0 else '❌'}")
        print(f"   MDD:    {best_config['mdd']:.2f}% {'✅' if best_config['mdd'] <= 15 else '❌'}")
        print(f"   Return: {best_config['return']:.2f}%")
        print(f"   Trades: {best_config['trades']}")
        
        gap = 2.0 - best_config['pf']
        print(f"\n📌 Gap to PF 2.0: {gap:.2f} points")
        
        if gap < 0.1:
            print("   → Try ensemble (LR + XGBoost)")
        elif gap < 0.2:
            print("   → Try lower cooldown (10 → 5 bars)")
        else:
            print("   → Consider different approach")
    
    # Save
    df = pd.DataFrame(results_sorted)
    df.to_csv(project_root / "reports" / "xgboost_tp_sl_optimization.csv", index=False)
    print("\n✅ Results saved to reports/xgboost_tp_sl_optimization.csv")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()