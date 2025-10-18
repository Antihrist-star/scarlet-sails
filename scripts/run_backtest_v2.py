"""
Run Backtest V2 - Triple Barrier + Fixed Hold
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
import joblib
from backtesting.honest_backtest_v2 import HonestBacktestV2
from models.logistic_baseline import LogisticBaseline

# Paths
project_root = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(project_root, "models", "logistic_baseline_clean_2d.pth")
scaler_path = os.path.join(project_root, "models", "scaler_clean_2d.pkl")
X_test_path = os.path.join(project_root, "models", "X_test_clean.pt")
ohlcv_path = os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet")

print("="*80)
print("HONEST BACKTEST V2 - TRIPLE BARRIER + FIXED HOLD")
print("="*80)

# Load model and scaler
print("\nLoading model and scaler...")
model = LogisticBaseline(input_dim=31)
model.load_state_dict(torch.load(model_path))
model.eval()

scaler = joblib.load(scaler_path)

# Load test features
print("Loading test features...")
X_test = torch.load(X_test_path)
X_test_np = X_test.numpy()
X_test_scaled = scaler.transform(X_test_np)

# Load OHLCV data
print(f"Loading OHLCV data...")
ohlcv_full = pd.read_parquet(ohlcv_path)

# Get subset matching test period
test_start_idx = len(ohlcv_full) - len(X_test)
ohlcv_subset = ohlcv_full.iloc[test_start_idx:].reset_index(drop=True)

print(f"OHLCV subset: {len(ohlcv_subset)} bars")

# Generate predictions
print("\nGenerating predictions...")
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_pred_proba = model.predict_proba(X_test_tensor)
    
    OPTIMAL_THRESHOLD = 0.69
    y_pred_signals = (y_pred_proba[:, 1] > OPTIMAL_THRESHOLD).astype(int)

print(f"Signals: {len(y_pred_signals)}")
print(f"BUY signals: {(y_pred_signals == 1).sum()}")

# Run backtest V2
print("\nRunning HONEST BACKTEST V2...")
backtest = HonestBacktestV2(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    take_profit=0.02,
    stop_loss=0.01,
    max_hold_bars=288,
    cooldown_bars=10
)

metrics = backtest.run(ohlcv_subset, y_pred_signals)
backtest.print_report()

try:
    backtest.plot_results()
except Exception as e:
    print(f"Could not save plot: {e}")

print("\n✅ Backtest V2 completed!")