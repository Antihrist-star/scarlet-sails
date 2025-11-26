#!/usr/bin/env python3
"""
DEEP ML DIAGNOSTIC - Why ML generates 266K trades and OOD 98.3%?

CRITICAL QUESTIONS:
1. Why exactly 266,491 trades on BTC/ETH/LTC 15m?
2. Why OOD 98.3% on 15m but 0% on 1h/4h/1d?
3. Is there data leakage?
4. Is normalization broken?
5. Which features cause OOD?
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.xgboost_model import XGBoostModel

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Load normalized model
model_path = MODELS_DIR / "xgboost_normalized_model.json"
scaler_path = MODELS_DIR / "xgboost_normalized_scaler.pkl"

print("="*100)
print("DEEP ML DIAGNOSTIC - INVESTIGATING 266K TRADES AND 98.3% OOD")
print("="*100)

# Load model
model = XGBoostModel()
model.load(str(model_path))
scaler = joblib.load(scaler_path)

print("\n✅ Model loaded")
print(f"   Model: {model_path}")
print(f"   Scaler: {scaler_path}")

# Test on BTC 15m first
asset = "BTC"
timeframe = "15m"
threshold = 0.65
date_cutoff = "2025-10-01"

print(f"\n{'='*100}")
print(f"TEST 1: Analyzing {asset}_{timeframe}")
print(f"{'='*100}")

# Load data
extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, timeframe)

# Apply date cutoff
cutoff_ts = pd.Timestamp(date_cutoff, tz='UTC')
primary_df = primary_df[primary_df.index < cutoff_ts]
for tf_key in all_tf:
    all_tf[tf_key] = all_tf[tf_key][all_tf[tf_key].index < cutoff_ts]

print(f"\n📊 Data loaded:")
print(f"   Total bars: {len(primary_df)}")
print(f"   Date range: {primary_df.index[0]} to {primary_df.index[-1]}")

# Analyze predictions
print(f"\n{'─'*100}")
print("ANALYZING PREDICTIONS...")
print(f"{'─'*100}")

predictions = []
ood_bars = []
feature_stats = []

FORWARD_BARS = 96
for i in range(len(primary_df) - FORWARD_BARS):
    features = extractor.extract_features_at_bar(all_tf, timeframe, i)

    if features is None:
        continue

    # Scale
    features_scaled = scaler.transform(features.reshape(1, -1))[0]

    # Check OOD
    is_ood = np.any(np.abs(features_scaled) > 3.0)
    if is_ood:
        ood_bars.append({
            'bar': i,
            'timestamp': str(primary_df.index[i]),
            'max_sigma': float(np.max(np.abs(features_scaled))),
            'features_scaled': features_scaled.tolist()
        })

    # Predict
    prob = model.predict_proba(features.reshape(1, -1))[0]
    prob_up = prob[1]

    predictions.append({
        'bar': i,
        'timestamp': str(primary_df.index[i]),
        'prob_up': float(prob_up),
        'is_ood': is_ood,
        'signal': prob_up >= threshold
    })

    # Store feature stats
    feature_stats.append(features_scaled)

predictions_df = pd.DataFrame(predictions)
feature_stats = np.array(feature_stats)

print(f"\n✅ Predictions analyzed:")
print(f"   Total predictions: {len(predictions_df)}")
print(f"   Signals generated: {predictions_df['signal'].sum()}")
print(f"   OOD bars: {predictions_df['is_ood'].sum()} ({predictions_df['is_ood'].mean()*100:.1f}%)")

# Analyze probability distribution
print(f"\n{'─'*100}")
print("PROBABILITY DISTRIBUTION:")
print(f"{'─'*100}")

prob_stats = predictions_df['prob_up'].describe()
print(f"   Min:    {prob_stats['min']:.4f}")
print(f"   25%:    {prob_stats['25%']:.4f}")
print(f"   Median: {prob_stats['50%']:.4f}")
print(f"   75%:    {prob_stats['75%']:.4f}")
print(f"   Max:    {prob_stats['max']:.4f}")
print(f"   Mean:   {prob_stats['mean']:.4f}")
print(f"   Std:    {prob_stats['std']:.4f}")

print(f"\n   Signals (prob >= {threshold}):")
signals_df = predictions_df[predictions_df['signal']]
if len(signals_df) > 0:
    print(f"   - Count: {len(signals_df)}")
    print(f"   - Prob mean: {signals_df['prob_up'].mean():.4f}")
    print(f"   - Prob min: {signals_df['prob_up'].min():.4f}")
    print(f"   - Prob max: {signals_df['prob_up'].max():.4f}")
else:
    print(f"   - Count: 0 (NO SIGNALS!)")

# Check if all predictions are similar
unique_probs = predictions_df['prob_up'].nunique()
print(f"\n   Unique probability values: {unique_probs}")
if unique_probs < 10:
    print("   ⚠️  WARNING: Very few unique probabilities - model may be broken!")
    print(f"   Top 10 most common probabilities:")
    print(predictions_df['prob_up'].value_counts().head(10))

# Analyze feature scaling
print(f"\n{'─'*100}")
print("FEATURE SCALING ANALYSIS:")
print(f"{'─'*100}")

feature_names = [
    'close', '15m_EMA_9', '15m_EMA_21', '15m_SMA_50', '15m_RSI_14',
    '15m_ATR_pct', '15m_MACD', '15m_MACD_signal', '15m_BB_upper', '15m_BB_lower',
    '15m_returns_5', '15m_returns_20', '15m_volatility_20', '15m_price_to_EMA9',
    '15m_volume_ratio', '1h_EMA_9', '1h_EMA_21', '1h_SMA_50', '1h_RSI_14',
    '1h_ATR_pct', '1h_returns_5', '1h_returns_20', '1h_price_to_EMA9',
    '4h_EMA_9', '4h_EMA_21', '4h_SMA_50', '4h_RSI_14', '4h_ATR_pct',
    '4h_returns_5', '4h_returns_20', '4h_price_to_EMA9'
]

print(f"\n   Feature stats (in sigma units):")
print(f"   {'Feature':<30} {'Mean σ':>10} {'Max σ':>10} {'% > 3σ':>10}")
print(f"   {'-'*60}")

ood_features = []
for i, fname in enumerate(feature_names):
    mean_sigma = np.mean(np.abs(feature_stats[:, i]))
    max_sigma = np.max(np.abs(feature_stats[:, i]))
    pct_ood = np.mean(np.abs(feature_stats[:, i]) > 3.0) * 100

    marker = "💀" if pct_ood > 50 else "⚠️" if pct_ood > 10 else ""
    print(f"   {fname:<30} {mean_sigma:>10.2f} {max_sigma:>10.2f} {pct_ood:>9.1f}% {marker}")

    if pct_ood > 10:
        ood_features.append({
            'feature': fname,
            'mean_sigma': float(mean_sigma),
            'max_sigma': float(max_sigma),
            'pct_ood': float(pct_ood)
        })

# Check scaler statistics
print(f"\n{'─'*100}")
print("SCALER TRAINING STATISTICS:")
print(f"{'─'*100}")

print(f"\n   Training data inferred from scaler:")
print(f"   {'Feature':<30} {'Train Mean':>15} {'Train Std':>15}")
print(f"   {'-'*60}")

for i, fname in enumerate(feature_names):
    train_mean = scaler.mean_[i]
    train_std = scaler.scale_[i]
    print(f"   {fname:<30} {train_mean:>15.6f} {train_std:>15.6f}")

# Sample some OOD bars
print(f"\n{'─'*100}")
print("SAMPLE OOD BARS (first 5):")
print(f"{'─'*100}")

for i, ood in enumerate(ood_bars[:5]):
    print(f"\n   OOD Bar #{i+1}:")
    print(f"   - Timestamp: {ood['timestamp']}")
    print(f"   - Max sigma: {ood['max_sigma']:.2f}σ")
    print(f"   - Features causing OOD:")

    features_scaled = np.array(ood['features_scaled'])
    ood_idx = np.where(np.abs(features_scaled) > 3.0)[0]

    for idx in ood_idx[:5]:  # Show top 5
        print(f"     - {feature_names[idx]}: {features_scaled[idx]:.2f}σ")

# Save diagnostic report
report = {
    'asset': asset,
    'timeframe': timeframe,
    'threshold': threshold,
    'total_bars': len(primary_df),
    'total_predictions': len(predictions_df),
    'signals_generated': int(predictions_df['signal'].sum()),
    'ood_ratio': float(predictions_df['is_ood'].mean()),
    'probability_stats': {
        'min': float(prob_stats['min']),
        'median': float(prob_stats['50%']),
        'max': float(prob_stats['max']),
        'mean': float(prob_stats['mean']),
        'std': float(prob_stats['std']),
        'unique_values': int(unique_probs)
    },
    'ood_features': ood_features,
    'sample_predictions': predictions_df.head(20).to_dict('records'),
    'sample_ood_bars': ood_bars[:10]
}

output_file = REPORTS_DIR / "ml_deep_diagnostic.json"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*100}")
print(f"DIAGNOSTIC REPORT SAVED: {output_file}")
print(f"{'='*100}")

# CRITICAL FINDINGS
print(f"\n{'='*100}")
print("🚨 CRITICAL FINDINGS:")
print(f"{'='*100}")

if predictions_df['signal'].sum() > 100000:
    print("\n❌ PROBLEM 1: TOO MANY SIGNALS")
    print(f"   - Signals: {predictions_df['signal'].sum():,}")
    print(f"   - Expected: ~10,000-20,000")
    print(f"   - Ratio: {predictions_df['signal'].sum() / len(predictions_df) * 100:.1f}% of all bars")
    print(f"   - LIKELY CAUSE: Threshold too low OR model overfits to 'UP' class")

if predictions_df['is_ood'].mean() > 0.5:
    print("\n❌ PROBLEM 2: HIGH OOD RATIO")
    print(f"   - OOD ratio: {predictions_df['is_ood'].mean()*100:.1f}%")
    print(f"   - Expected: <10%")
    print(f"   - LIKELY CAUSE: Model trained on different data distribution")

if len(ood_features) > 10:
    print("\n❌ PROBLEM 3: MANY OOD FEATURES")
    print(f"   - OOD features: {len(ood_features)}/31")
    print(f"   - LIKELY CAUSE: Normalization didn't work properly")
    print(f"   - Top 5 worst features:")
    for feat in sorted(ood_features, key=lambda x: x['pct_ood'], reverse=True)[:5]:
        print(f"     - {feat['feature']}: {feat['pct_ood']:.1f}% OOD")

if unique_probs < 100:
    print("\n❌ PROBLEM 4: FEW UNIQUE PROBABILITIES")
    print(f"   - Unique probs: {unique_probs}")
    print(f"   - Expected: 10,000+")
    print(f"   - LIKELY CAUSE: Model collapsed or data leakage")

print(f"\n{'='*100}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*100}")
