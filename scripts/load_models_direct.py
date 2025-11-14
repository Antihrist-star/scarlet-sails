#!/usr/bin/env python3
"""
Load and test trained models directly without torch dependency
Tests XGBoost model JSON directly using xgboost library
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 100)
print("LOADING TRAINED MODELS DIRECTLY")
print("=" * 100)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: Load data
# ============================================================================

print("[1/3] Loading data...")
data_file = PROJECT_ROOT / "data" / "raw" / "BTC_USDT_15m.parquet"

if not data_file.exists():
    print(f"❌ ERROR: Data file not found: {data_file}")
    sys.exit(1)

ohlcv = pd.read_parquet(data_file)
print(f"✅ Data loaded: {len(ohlcv)} bars")
print(f"   Columns: {list(ohlcv.columns)}")

# ============================================================================
# STEP 2: Load XGBoost model
# ============================================================================

print("\n[2/3] Loading XGBoost model directly...")
xgb_model_file = PROJECT_ROOT / "models" / "xgboost_model.json"

if not xgb_model_file.exists():
    print(f"❌ XGBoost model not found: {xgb_model_file}")
    sys.exit(1)

try:
    # Load XGBoost model directly
    booster = xgb.Booster()
    booster.load_model(str(xgb_model_file))

    # Get model info
    with open(xgb_model_file, 'r') as f:
        model_json = json.load(f)

    num_trees = len(model_json.get('learner', {}).get('gradient_booster', {}).get('model', {}).get('trees', []))
    num_features = model_json.get('learner', {}).get('gradient_booster', {}).get('model', {}).get('tree_param', {}).get('num_feature', 'N/A')

    print(f"✅ XGBoost model loaded successfully")
    print(f"   Trees: {num_trees}")
    print(f"   Features: {num_features if isinstance(num_features, str) else num_features}")
    print(f"   File size: {xgb_model_file.stat().st_size / 1024:.1f} KB")

except Exception as e:
    print(f"❌ Failed to load XGBoost: {str(e)[:100]}")
    sys.exit(1)

# ============================================================================
# STEP 3: Test predictions on random data
# ============================================================================

print("\n[3/3] Testing model predictions...")

try:
    # Create test data: 100 samples with 31 features
    print("   Creating test data (100 samples, 31 features)...")
    test_data = np.random.randn(100, 31).astype(np.float32)

    # Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(test_data)

    # Get predictions
    predictions = booster.predict(dmatrix)

    print(f"✅ Model predictions generated successfully")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Mean prediction: {predictions.mean():.4f}")
    print(f"   Std dev: {predictions.std():.4f}")

    # Show sample predictions
    print(f"\n   Sample predictions (first 10):")
    for i in range(min(10, len(predictions))):
        print(f"      {i+1:2d}. {predictions[i]:.6f}")

except Exception as e:
    print(f"❌ Prediction failed: {str(e)[:100]}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Check best config
# ============================================================================

print("\n[4/4] Best configuration from optimization:")
config_file = PROJECT_ROOT / "models" / "best_tp_sl_config.json"

if config_file.exists():
    with open(config_file) as f:
        config = json.load(f)

    metrics = config.get('metrics', {})
    print(f"✅ Configuration loaded")
    print(f"   Model: {config.get('model', 'N/A')}")
    print(f"   TP: {config.get('tp', 'N/A')} ({config.get('tp', 0)*100:.1f}%)")
    print(f"   SL: {config.get('sl', 'N/A')} ({config.get('sl', 0)*100:.2f}%)")
    print(f"   Threshold: {config.get('threshold', 'N/A')}")
    print(f"\n   Performance Metrics:")
    print(f"      Win Rate: {metrics.get('win_rate', 'N/A')*100:.2f}%")
    print(f"      Profit Factor: {metrics.get('pf', 'N/A'):.2f}")
    print(f"      Total Return: {metrics.get('return', 'N/A'):.2f}%")
    print(f"      Trades: {metrics.get('trades', 'N/A')}")
else:
    print(f"⚠️  Config file not found: {config_file}")

print(f"\n" + "=" * 100)
print(f"✅ SUCCESS: XGBoost model is ready for backtesting")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)
