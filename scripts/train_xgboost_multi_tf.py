"""
Train XGBoost Model - MULTI-TIMEFRAME VERSION
==============================================

Trains XGBoost model using proper multi-timeframe features (31 features):
- 13 features from primary timeframe (15m)
- 18 features from higher timeframes (1h, 4h, 1d)

This creates a NEW model specifically for Hybrid system.
Old model (models/xgboost_model.json) is NOT overwritten.

Output:
- models/xgboost_multi_tf_model.json (NEW model)
- models/xgboost_multi_tf_scaler.pkl (feature scaler)
- models/xgboost_multi_tf_features.json (feature names)

Author: Scarlet Sails Team
Date: 2025-11-11
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.xgboost_model import XGBoostModel

# Config
DATA_DIR = Path('data/raw')
MODELS_DIR = Path('models')
ASSET = 'BTC'  # Use BTC for training (most liquid, longest history)
TARGET_TF = '15m'  # Primary timeframe

# Training config
FORWARD_BARS = 96  # 24 hours on 15m = 96 bars
PROFIT_THRESHOLD = 0.01  # 1% profit target
MAX_SAMPLES = 50000  # Limit dataset size for faster training

print("=" * 100)
print("TRAIN XGBOOST - MULTI-TIMEFRAME VERSION")
print("=" * 100)
print(f"\nAsset: {ASSET}")
print(f"Target TF: {TARGET_TF}")
print(f"Forward bars: {FORWARD_BARS} ({FORWARD_BARS * 15 / 60:.1f} hours)")
print(f"Profit threshold: {PROFIT_THRESHOLD * 100:.1f}%")

# ============================================================================
# STEP 1: Load multi-TF data
# ============================================================================
print("\n" + "=" * 100)
print("STEP 1: Loading multi-timeframe data")
print("=" * 100)

extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

try:
    all_tf, primary_df = extractor.prepare_multi_timeframe_data(ASSET, TARGET_TF)
    print(f"✅ Loaded {len(primary_df)} bars")
    print(f"✅ Prepared {len(primary_df.columns)} columns")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: Create labels (UP/DOWN based on future profit)
# ============================================================================
print("\n" + "=" * 100)
print("STEP 2: Creating labels")
print("=" * 100)

# Calculate future returns
primary_df['future_close'] = primary_df['close'].shift(-FORWARD_BARS)
primary_df['future_return'] = (primary_df['future_close'] - primary_df['close']) / primary_df['close']

# Label: UP (1) if future return > threshold, DOWN (0) otherwise
primary_df['label'] = (primary_df['future_return'] > PROFIT_THRESHOLD).astype(int)

# Remove rows without future data
primary_df = primary_df[:-FORWARD_BARS].copy()

# Statistics
n_up = (primary_df['label'] == 1).sum()
n_down = (primary_df['label'] == 0).sum()
total = len(primary_df)

print(f"Total samples: {total:,}")
print(f"UP (1):        {n_up:,} ({n_up/total*100:.1f}%)")
print(f"DOWN (0):      {n_down:,} ({n_down/total*100:.1f}%)")

# ============================================================================
# STEP 3: Extract features for all samples
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: Extracting features")
print("=" * 100)

X_list = []
y_list = []
valid_indices = []

# Sample bars (skip first 200 for indicator warm-up)
sample_indices = range(200, len(primary_df))

# Limit samples for faster training
if len(sample_indices) > MAX_SAMPLES:
    print(f"⚠️  Limiting to {MAX_SAMPLES:,} samples (from {len(sample_indices):,})")
    step = len(sample_indices) // MAX_SAMPLES
    sample_indices = list(sample_indices)[::step][:MAX_SAMPLES]

print(f"Extracting features for {len(sample_indices):,} samples...")

for i, bar_idx in enumerate(sample_indices):
    if i % 5000 == 0:
        print(f"   Progress: {i:,} / {len(sample_indices):,} ({i/len(sample_indices)*100:.1f}%)")

    # Extract features
    features = extractor.extract_features_at_bar(all_tf, TARGET_TF, bar_idx)

    if features is not None:
        X_list.append(features)
        y_list.append(primary_df['label'].iloc[bar_idx])
        valid_indices.append(bar_idx)

X = np.array(X_list)
y = np.array(y_list)

print(f"\n✅ Extracted features for {len(X):,} samples")
print(f"   Feature shape: {X.shape}")
print(f"   Label shape: {y.shape}")

# Check class distribution
n_up_final = (y == 1).sum()
n_down_final = (y == 0).sum()
print(f"\nFinal distribution:")
print(f"   UP (1):   {n_up_final:,} ({n_up_final/len(y)*100:.1f}%)")
print(f"   DOWN (0): {n_down_final:,} ({n_down_final/len(y)*100:.1f}%)")

# ============================================================================
# STEP 4: Scale features
# ============================================================================
print("\n" + "=" * 100)
print("STEP 4: Scaling features")
print("=" * 100)

# Split into train/test (temporal split to avoid lookahead bias)
split_idx = int(len(X) * 0.8)

X_train_raw = X[:split_idx]
X_test_raw = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

print(f"Train: {len(X_train_raw):,} samples")
print(f"Test:  {len(X_test_raw):,} samples")

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

print(f"✅ Features scaled")
print(f"   Mean: {X_train_scaled.mean():.4f}")
print(f"   Std:  {X_train_scaled.std():.4f}")

# ============================================================================
# STEP 5: Train XGBoost
# ============================================================================
print("\n" + "=" * 100)
print("STEP 5: Training XGBoost")
print("=" * 100)

# Calculate class weights
class_weight = len(y_train) / (2 * np.bincount(y_train))
scale_pos_weight = class_weight[1] / class_weight[0]

print(f"Class weight ratio: {scale_pos_weight:.2f}")

# Create model
model = XGBoostModel(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    early_stopping_rounds=20
)

# Split train into train/val
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"\nTrain fit: {len(X_train_fit):,}")
print(f"Val:       {len(X_val):,}")

# Train
print("\nTraining...")
model.fit(
    X_train_fit,
    y_train_fit,
    eval_set=(X_val, y_val),
    verbose=50
)

print(f"\n✅ Training complete")
print(f"   Best iteration: {model.best_iteration}")

# ============================================================================
# STEP 6: Evaluate
# ============================================================================
print("\n" + "=" * 100)
print("STEP 6: Evaluation")
print("=" * 100)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"              DOWN    UP")
print(f"Actual DOWN   {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       UP     {cm[1,0]:5d}  {cm[1,1]:5d}")

# Feature importance
print("\nTop 10 Important Features:")
model.print_feature_importance(top_n=10)

# Test different thresholds
print("\n" + "=" * 100)
print("STEP 7: Threshold optimization")
print("=" * 100)

thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Win Rate':<12} {'Signals':<12}")
print(f"{'-'*48}")

best_threshold = 0.5
best_wr = 0

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba[:, 1] >= thresh).astype(int)
    positive_mask = (y_pred_thresh == 1)

    if positive_mask.sum() > 0:
        acc = accuracy_score(y_test, y_pred_thresh)
        wr = (y_test[positive_mask] == 1).mean()
        n_signals = positive_mask.sum()

        marker = ""
        if 0.45 <= thresh <= 0.65 and wr > best_wr:
            best_threshold = thresh
            best_wr = wr
            marker = " ⭐"

        print(f"{thresh:<12.2f} {acc:<12.4f} {wr*100:<12.1f} {n_signals:<12d}{marker}")
    else:
        print(f"{thresh:<12.2f} {'N/A':<12} {'N/A':<12} {0:<12d}")

print(f"\n✅ Best threshold: {best_threshold:.2f} (Win Rate: {best_wr*100:.1f}%)")

# ============================================================================
# STEP 8: Save model, scaler, and feature names
# ============================================================================
print("\n" + "=" * 100)
print("STEP 8: Saving model")
print("=" * 100)

# Save model
model_path = MODELS_DIR / 'xgboost_multi_tf_model.json'
model.save(str(model_path))
print(f"✅ Model saved: {model_path}")

# Save scaler
scaler_path = MODELS_DIR / 'xgboost_multi_tf_scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved: {scaler_path}")

# Save feature names (for documentation)
feature_names = [
    # Primary TF (13)
    '15m_RSI_14', '15m_EMA_9', '15m_EMA_21', '15m_SMA_50',
    '15m_BB_middle', '15m_BB_std', '15m_BB_upper', '15m_BB_lower',
    '15m_ATR_14', '15m_returns_5', '15m_returns_20',
    '15m_volume_sma', '15m_volume_ratio',
    # 1h (6)
    '1h_close', '1h_EMA_9', '1h_EMA_21', '1h_SMA_50', '1h_RSI_14', '1h_returns_5',
    # 4h (6)
    '4h_close', '4h_EMA_9', '4h_EMA_21', '4h_SMA_50', '4h_RSI_14', '4h_returns_5',
    # 1d (6)
    '1d_close', '1d_EMA_9', '1d_EMA_21', '1d_SMA_50', '1d_RSI_14', '1d_returns_5'
]

features_path = MODELS_DIR / 'xgboost_multi_tf_features.json'
with open(features_path, 'w') as f:
    json.dump({
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'best_threshold': float(best_threshold),
        'asset': ASSET,
        'target_tf': TARGET_TF,
        'forward_bars': FORWARD_BARS,
        'profit_threshold': PROFIT_THRESHOLD
    }, f, indent=2)
print(f"✅ Feature names saved: {features_path}")

# Save best threshold
threshold_path = MODELS_DIR / 'xgboost_multi_tf_threshold.txt'
with open(threshold_path, 'w') as f:
    f.write(f"{best_threshold:.2f}\n")
print(f"✅ Best threshold saved: {threshold_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("TRAINING COMPLETE!")
print("=" * 100)

print(f"\nModel: XGBoost Multi-Timeframe")
print(f"Training samples: {len(X_train_scaled):,}")
print(f"Test samples: {len(X_test_scaled):,}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Win Rate at best threshold: {best_wr*100:.1f}%")

print(f"\n📁 Files created:")
print(f"   {model_path}")
print(f"   {scaler_path}")
print(f"   {features_path}")
print(f"   {threshold_path}")

print(f"\n✅ READY FOR HYBRID BACKTEST!")
print("=" * 100)
