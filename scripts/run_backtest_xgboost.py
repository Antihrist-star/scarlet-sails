"""
Backtest XGBoost Model - FIXED VERSION with hardcoded threshold 0.46
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from models.xgboost_model import XGBoostModel
from backtesting.honest_backtest_v2 import HonestBacktestV2


def print_header(title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def print_step(step_num, total_steps, description):
    print(f"[{step_num}/{total_steps}] {description}")


def main():
    print_header("XGBOOST BACKTEST - THRESHOLD 0.46")
    
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    data_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ========================================================================
    # [1/5] LOAD MODEL
    # ========================================================================
    print_step(1, 5, "Loading model...")
    
    model = XGBoostModel()
    model_path = models_dir / "xgboost_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load(model_path)
    
    # HARDCODED THRESHOLD
    THRESHOLD = 0.46  # 🔥 FIXED!
    
    print(f"✅ Model loaded")
    print(f"✅ Threshold (hardcoded): {THRESHOLD:.2f}")
    
    # ========================================================================
    # [2/5] LOAD TEST DATA
    # ========================================================================
    print_step(2, 5, "Loading test data...")
    
    X_test = torch.load(models_dir / "X_test_clean.pt", weights_only=False)
    y_test = torch.load(models_dir / "y_test_clean.pt", weights_only=False)
    scaler = joblib.load(models_dir / "scaler_clean_2d.pkl")
    
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"✅ Data loaded")
    
    # ========================================================================
    # [3/5] LOAD OHLCV
    # ========================================================================
    print_step(3, 5, "Loading OHLCV...")
    
    ohlcv_full = pd.read_parquet(data_dir / "BTC_USDT_15m_FULL.parquet")
    test_start_idx = len(ohlcv_full) - len(X_test)
    ohlcv_test = ohlcv_full.iloc[test_start_idx:].reset_index(drop=True)
    
    print(f"OHLCV test: {ohlcv_test.shape}")
    print(f"✅ OHLCV aligned")
    
    # ========================================================================
    # [4/5] GENERATE PREDICTIONS
    # ========================================================================
    print_step(4, 5, "Generating predictions...")
    
    X_test_np = X_test.numpy()
    X_test_scaled = scaler.transform(X_test_np)
    
    y_pred_proba = model.predict_proba(X_test_scaled)
    y_pred_signals = (y_pred_proba[:, 1] >= THRESHOLD).astype(int)
    
    n_signals = np.sum(y_pred_signals)
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"BUY signals: {n_signals} ({n_signals/len(y_pred_signals)*100:.2f}%)")
    
    if n_signals == 0:
        print("❌ No signals! Cannot backtest.")
        return
    
    # Win rate verification
    y_test_np = y_test.numpy()
    signal_mask = y_pred_signals == 1
    actual_wr = np.sum(y_test_np[signal_mask] == 1) / n_signals * 100
    print(f"Actual WR on signals: {actual_wr:.1f}%")
    print(f"✅ Predictions generated")
    
    # ========================================================================
    # [5/5] RUN BACKTEST
    # ========================================================================
    print_step(5, 5, "Running backtest...")
    
    backtest = HonestBacktestV2(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        position_size_pct=0.95,
        take_profit=0.03,
        stop_loss=0.012,
        max_hold_bars=288,
        cooldown_bars=10
    )
    
    print(f"Config: TP={backtest.take_profit*100:.1f}%, SL={backtest.stop_loss*100:.1f}%")
    print()
    
    metrics = backtest.run(ohlcv_test, y_pred_signals)
    backtest.print_report()
    
    # Save results
    if hasattr(backtest, 'trades') and backtest.trades:
        trades_df = pd.DataFrame(backtest.trades)
        trades_df.to_csv(reports_dir / "xgboost_046_trades.csv", index=False)
        print(f"\n💾 Trades saved")
    
    backtest.plot_results(save_path=str(reports_dir / "xgboost_046_equity.png"))
    print(f"📊 Chart saved")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print_header("COMPARISON WITH BASELINE")
    
    baseline_pf = 1.64
    baseline_mdd = 12.33
    baseline_trades = 45
    
    xgb_pf = metrics['profit_factor']
    xgb_mdd = metrics['max_drawdown_pct']
    xgb_trades = metrics['n_trades']
    
    print(f"{'Metric':<20} {'Baseline (LR)':<15} {'XGBoost':<15} {'Change':<10}")
    print("-" * 60)
    print(f"{'Profit Factor':<20} {baseline_pf:<15.2f} {xgb_pf:<15.2f} {(xgb_pf-baseline_pf)/baseline_pf*100:>+6.1f}%")
    print(f"{'Max Drawdown':<20} {baseline_mdd:<15.2f} {xgb_mdd:<15.2f} {xgb_mdd-baseline_mdd:>+6.1f}pp")
    print(f"{'Trades':<20} {baseline_trades:<15} {xgb_trades:<15} {xgb_trades-baseline_trades:>+6}")
    print("-" * 60)
    
    # Goal check
    print()
    if xgb_pf >= 2.0 and xgb_mdd <= 15:
        print("🎉 GOALS ACHIEVED! PF ≥ 2.0, MDD ≤ 15%")
    elif xgb_pf >= 2.0:
        print("⚠️  PF goal achieved, but MDD too high")
    elif xgb_mdd <= 15:
        print("⚠️  MDD goal achieved, but PF too low")
        print(f"   Gap to PF 2.0: {2.0 - xgb_pf:.2f} points")
    else:
        print("❌ Goals not achieved yet")
        print(f"   PF gap: {2.0 - xgb_pf:.2f} points")
        print(f"   MDD gap: {xgb_mdd - 15:.2f}pp")
    
    print_header("BACKTEST COMPLETE")


if __name__ == "__main__":
    main()