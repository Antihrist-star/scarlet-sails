#!/usr/bin/env python3
"""
ML BREAKDOWN DIAGNOSTICS
Глубокий анализ почему ML модель не работает на свежих данных

Проверяет:
1. Distribution shift (train vs test features)
2. Feature scaling issues
3. Prediction confidence breakdown
4. Out-of-distribution detection
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.xgboost_model import XGBoostModel

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Constants
TARGET_TF = "15m"
FORWARD_BARS = 96


def analyze_feature_distributions(
    scaler: any,
    all_timeframes: Dict,
    primary_df: pd.DataFrame,
    n_samples: int = 1000
) -> Dict:
    """
    Анализ distribution shift между training и testing данными
    """
    print("\n" + "="*100)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*100)

    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    # Training statistics (from scaler)
    train_mean = scaler.mean_
    train_std = scaler.scale_

    # Extract features from test data
    test_features = []

    # Sample from recent data (skip first 10K bars)
    start_idx = max(10000, len(primary_df) - n_samples - FORWARD_BARS)
    end_idx = len(primary_df) - FORWARD_BARS

    print(f"\n📊 Extracting {end_idx - start_idx} test samples from recent data...")
    print(f"   Range: bar {start_idx} to {end_idx}")

    for bar_idx in range(start_idx, end_idx):
        if bar_idx % 200 == 0:
            print(f"   Progress: {bar_idx - start_idx}/{end_idx - start_idx}")

        features = extractor.extract_features_at_bar(all_timeframes, TARGET_TF, bar_idx)
        if features is not None:
            test_features.append(features)

    if len(test_features) == 0:
        print("❌ No valid test features extracted!")
        return None

    test_features = np.array(test_features)
    print(f"✅ Extracted {len(test_features)} valid samples")

    # Calculate test statistics
    test_mean = np.mean(test_features, axis=0)
    test_std = np.std(test_features, axis=0)

    # Load feature names
    with open(MODELS_DIR / "xgboost_multi_tf_features.json", 'r') as f:
        meta = json.load(f)
        feature_names = meta['feature_names']

    # Calculate distribution shift metrics
    print("\n" + "─"*100)
    print("DISTRIBUTION SHIFT METRICS")
    print("─"*100)

    results = {
        'feature_names': feature_names,
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'test_mean': test_mean.tolist(),
        'test_std': test_std.tolist(),
        'shift_metrics': []
    }

    print(f"\n{'Feature':<20} {'Train Mean':<15} {'Test Mean':<15} {'Shift (σ)':<12} {'Status':<10}")
    print("─"*100)

    for i, name in enumerate(feature_names):
        # Mean shift in standard deviations
        mean_shift_sigma = (test_mean[i] - train_mean[i]) / train_std[i]

        # Classify severity
        if abs(mean_shift_sigma) < 1.0:
            status = "✅ OK"
        elif abs(mean_shift_sigma) < 2.0:
            status = "⚠️ MILD"
        elif abs(mean_shift_sigma) < 3.0:
            status = "⚠️ MODERATE"
        elif abs(mean_shift_sigma) < 5.0:
            status = "🔴 SEVERE"
        else:
            status = "💀 CRITICAL"

        results['shift_metrics'].append({
            'feature': name,
            'train_mean': float(train_mean[i]),
            'test_mean': float(test_mean[i]),
            'shift_sigma': float(mean_shift_sigma),
            'status': status
        })

        print(f"{name:<20} {train_mean[i]:>12,.2f}   {test_mean[i]:>12,.2f}   {mean_shift_sigma:>+8.2f}σ   {status:<10}")

    # Summary
    print("\n" + "─"*100)
    print("SUMMARY")
    print("─"*100)

    critical_features = [m for m in results['shift_metrics'] if abs(m['shift_sigma']) >= 5.0]
    severe_features = [m for m in results['shift_metrics'] if 3.0 <= abs(m['shift_sigma']) < 5.0]
    moderate_features = [m for m in results['shift_metrics'] if 2.0 <= abs(m['shift_sigma']) < 3.0]

    print(f"\n💀 CRITICAL shift (>5σ): {len(critical_features)} features")
    for feat in critical_features:
        print(f"   - {feat['feature']}: {feat['shift_sigma']:+.2f}σ")

    print(f"\n🔴 SEVERE shift (3-5σ): {len(severe_features)} features")
    for feat in severe_features:
        print(f"   - {feat['feature']}: {feat['shift_sigma']:+.2f}σ")

    print(f"\n⚠️ MODERATE shift (2-3σ): {len(moderate_features)} features")

    return results


def analyze_predictions(
    model: XGBoostModel,
    scaler: any,
    all_timeframes: Dict,
    primary_df: pd.DataFrame,
    n_samples: int = 1000
) -> Dict:
    """
    Анализ ML predictions на test данных
    """
    print("\n" + "="*100)
    print("ML PREDICTION ANALYSIS")
    print("="*100)

    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    predictions = []

    # Sample from recent data
    start_idx = max(10000, len(primary_df) - n_samples - FORWARD_BARS)
    end_idx = len(primary_df) - FORWARD_BARS

    print(f"\n🔮 Generating predictions for {end_idx - start_idx} samples...")

    for bar_idx in range(start_idx, end_idx):
        if bar_idx % 200 == 0:
            print(f"   Progress: {bar_idx - start_idx}/{end_idx - start_idx}")

        features = extractor.extract_features_at_bar(all_timeframes, TARGET_TF, bar_idx)
        if features is None:
            continue

        # Scale
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        prob = model.predict_proba(features_scaled)[0]
        prob_up = prob[1]

        # Check if RSI < 30 (would Rule-Based enter?)
        rsi_col = None
        for col in ['rsi', 'RSI_14', '15m_RSI_14']:
            if col in primary_df.columns:
                rsi_col = col
                break

        rsi_value = primary_df[rsi_col].iloc[bar_idx] if rsi_col else None
        rule_would_enter = rsi_value is not None and rsi_value < 30

        predictions.append({
            'bar_idx': bar_idx,
            'prob_up': prob_up,
            'rsi': rsi_value,
            'rule_would_enter': rule_would_enter
        })

    print(f"✅ Generated {len(predictions)} predictions")

    # Analysis
    print("\n" + "─"*100)
    print("PREDICTION STATISTICS")
    print("─"*100)

    probs = [p['prob_up'] for p in predictions]

    print(f"\nProbability P(UP) distribution:")
    print(f"  Min:      {np.min(probs):.4f}")
    print(f"  Q1 (25%): {np.percentile(probs, 25):.4f}")
    print(f"  Median:   {np.median(probs):.4f}")
    print(f"  Q3 (75%): {np.percentile(probs, 75):.4f}")
    print(f"  Max:      {np.max(probs):.4f}")
    print(f"  Mean:     {np.mean(probs):.4f}")
    print(f"  Std:      {np.std(probs):.4f}")

    # Count by threshold
    print(f"\nThreshold analysis:")
    print(f"  P(UP) ≥ 0.65 (would enter): {sum(1 for p in probs if p >= 0.65)} ({sum(1 for p in probs if p >= 0.65)/len(probs)*100:.1f}%)")
    print(f"  P(UP) ≥ 0.50 (neutral):     {sum(1 for p in probs if p >= 0.50)} ({sum(1 for p in probs if p >= 0.50)/len(probs)*100:.1f}%)")
    print(f"  P(UP) < 0.35 (confident DN): {sum(1 for p in probs if p < 0.35)} ({sum(1 for p in probs if p < 0.35)/len(probs)*100:.1f}%)")

    # Rule-based comparison
    rule_signals = [p for p in predictions if p['rule_would_enter']]
    print(f"\nRule-Based vs ML:")
    print(f"  Rule-Based would enter: {len(rule_signals)} times")
    if len(rule_signals) > 0:
        rule_probs = [p['prob_up'] for p in rule_signals]
        print(f"  ML P(UP) on Rule signals:")
        print(f"    Mean:   {np.mean(rule_probs):.4f}")
        print(f"    Median: {np.median(rule_probs):.4f}")
        print(f"    ≥ 0.65: {sum(1 for p in rule_probs if p >= 0.65)} ({sum(1 for p in rule_probs if p >= 0.65)/len(rule_probs)*100:.1f}%)")

    # Histogram
    print(f"\nProbability histogram:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(probs, bins=bins)

    for i in range(len(bins)-1):
        bar = '█' * int(hist[i] / max(hist) * 50)
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:>5} {bar}")

    return {
        'predictions': predictions,
        'stats': {
            'min': float(np.min(probs)),
            'q1': float(np.percentile(probs, 25)),
            'median': float(np.median(probs)),
            'q3': float(np.percentile(probs, 75)),
            'max': float(np.max(probs)),
            'mean': float(np.mean(probs)),
            'std': float(np.std(probs))
        }
    }


def diagnose_why_ml_broken(
    model: XGBoostModel,
    scaler: any,
    all_timeframes: Dict,
    primary_df: pd.DataFrame
) -> None:
    """
    Главная диагностика: ПОЧЕМУ ML сломана?
    """
    print("\n" + "="*100)
    print("🔬 DIAGNOSING WHY ML IS BROKEN")
    print("="*100)

    # Get one sample from recent data
    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    # Find a bar with RSI < 30 (where Rule-Based would enter)
    rsi_col = None
    for col in ['rsi', 'RSI_14', '15m_RSI_14']:
        if col in primary_df.columns:
            rsi_col = col
            break

    oversold_bars = primary_df[primary_df[rsi_col] < 30].index

    # Take from recent data
    recent_oversold = [
        primary_df.index.get_loc(ts) for ts in oversold_bars[-100:]
        if primary_df.index.get_loc(ts) >= 10000 and primary_df.index.get_loc(ts) < len(primary_df) - FORWARD_BARS
    ]

    if len(recent_oversold) == 0:
        print("❌ No recent oversold bars found!")
        return

    bar_idx = recent_oversold[0]
    timestamp = primary_df.index[bar_idx]
    price = primary_df['close'].iloc[bar_idx]
    rsi = primary_df[rsi_col].iloc[bar_idx]

    print(f"\n📍 Analyzing bar {bar_idx}: {timestamp}")
    print(f"   Price: ${price:,.2f}")
    print(f"   RSI: {rsi:.2f} (< 30 → Rule-Based would ENTER)")

    # Extract features
    features_raw = extractor.extract_features_at_bar(all_timeframes, TARGET_TF, bar_idx)
    if features_raw is None:
        print("❌ Could not extract features!")
        return

    # Load feature names
    with open(MODELS_DIR / "xgboost_multi_tf_features.json", 'r') as f:
        meta = json.load(f)
        feature_names = meta['feature_names']

    # Scale
    features_scaled = scaler.transform(features_raw.reshape(1, -1))[0]

    # Predict
    prob = model.predict_proba(features_raw.reshape(1, -1))[0]
    prob_up = prob[1]

    print(f"\n🔮 ML Prediction:")
    print(f"   P(DOWN): {prob[0]:.4f}")
    print(f"   P(UP):   {prob_up:.4f}")
    print(f"   Threshold: 0.65")
    print(f"   Decision: {'ENTER ✅' if prob_up >= 0.65 else 'REJECT ❌'}")

    # Show features
    print(f"\n📊 Feature Analysis:")
    print(f"\n{'Feature':<20} {'Raw Value':<15} {'Scaled':<12} {'Train μ':<15} {'Σ from μ':<12} {'Status'}")
    print("─"*100)

    train_mean = scaler.mean_
    train_std = scaler.scale_

    for i, name in enumerate(feature_names):
        raw_val = features_raw[i]
        scaled_val = features_scaled[i]
        train_mu = train_mean[i]
        sigma_from_mean = scaled_val

        if abs(sigma_from_mean) < 2.0:
            status = "✅ Normal"
        elif abs(sigma_from_mean) < 3.0:
            status = "⚠️ Unusual"
        elif abs(sigma_from_mean) < 5.0:
            status = "🔴 Outlier"
        else:
            status = "💀 EXTREME"

        print(f"{name:<20} {raw_val:>12,.2f}   {scaled_val:>+8.3f}σ   {train_mu:>12,.2f}   {sigma_from_mean:>+8.3f}σ   {status}")

    # Top contributors
    print(f"\n🔝 Top 10 Most Extreme Features (by |scaled value|):")
    abs_scaled = np.abs(features_scaled)
    top_indices = np.argsort(abs_scaled)[-10:][::-1]

    for i in top_indices:
        print(f"   {feature_names[i]:<20}: {features_scaled[i]:>+7.3f}σ (raw: {features_raw[i]:>12,.2f}, train_μ: {train_mean[i]:>12,.2f})")

    # Diagnosis
    print(f"\n" + "─"*100)
    print("💡 DIAGNOSIS")
    print("─"*100)

    extreme_features = sum(1 for x in abs_scaled if abs(x) > 5.0)
    outlier_features = sum(1 for x in abs_scaled if abs(x) > 3.0)

    print(f"\n📊 Feature Distribution:")
    print(f"   Total features: 31")
    print(f"   EXTREME (>5σ): {extreme_features} features ({extreme_features/31*100:.1f}%)")
    print(f"   Outliers (>3σ): {outlier_features} features ({outlier_features/31*100:.1f}%)")

    if extreme_features >= 5:
        print(f"\n💀 CRITICAL: {extreme_features} features are >5σ from training distribution!")
        print(f"   → Model has NEVER seen data like this during training")
        print(f"   → Predictions are unreliable (out-of-distribution)")
        print(f"\n   Root cause: Price-based features (SMA, EMA, close) use ABSOLUTE prices")
        print(f"   → Training: BTC was $13K-$70K")
        print(f"   → Testing:  BTC is $107K-$111K")
        print(f"   → Scaled features are >5σ outliers!")
    elif outlier_features >= 10:
        print(f"\n🔴 SEVERE: {outlier_features} features are >3σ from training distribution!")
        print(f"   → Model is extrapolating beyond training range")
        print(f"   → Predictions may be unreliable")
    else:
        print(f"\n✅ Feature distribution looks reasonable")

    print(f"\n🔧 SOLUTIONS:")
    if extreme_features >= 5:
        print(f"   1. RETRAIN on recent data (2020-2025) including $70K-$111K prices")
        print(f"   2. USE NORMALIZED FEATURES:")
        print(f"      - Instead of: close_1d = $114,194 (absolute)")
        print(f"      - Use:        returns_1d = (close - close_prev) / close_prev")
        print(f"      - Or:         price_ratio = close / SMA_200")
        print(f"   3. ADD REGIME DETECTION:")
        print(f"      - Detect when features are out-of-distribution")
        print(f"      - Switch to fallback strategy (e.g., Rule-Based only)")


def main():
    """Main diagnostics"""

    print("="*100)
    print("ML BREAKDOWN DIAGNOSTICS")
    print("="*100)
    print("\nThis tool diagnoses WHY ML model is not working on recent data")

    # Load model
    model_path = MODELS_DIR / "xgboost_multi_tf_model.json"
    scaler_path = MODELS_DIR / "xgboost_multi_tf_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print("❌ Model files not found!")
        return

    print("\n📦 Loading model...")
    model = XGBoostModel()
    model.load(str(model_path))
    scaler = joblib.load(scaler_path)
    print("✅ Model loaded")

    # Load data
    asset = "BTC"
    print(f"\n📊 Loading data for {asset}...")
    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
    all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, TARGET_TF)
    print(f"✅ Loaded {len(primary_df)} bars")

    # Run diagnostics
    print("\n" + "="*100)
    print("RUNNING DIAGNOSTICS...")
    print("="*100)

    # 1. Why is ML broken on this specific example?
    diagnose_why_ml_broken(model, scaler, all_tf, primary_df)

    # 2. Feature distribution shift
    dist_results = analyze_feature_distributions(scaler, all_tf, primary_df, n_samples=1000)

    # 3. Prediction analysis
    pred_results = analyze_predictions(model, scaler, all_tf, primary_df, n_samples=1000)

    # Save results
    output_file = PROJECT_ROOT / "reports" / "ml_breakdown_diagnostics.json"
    output_file.parent.mkdir(exist_ok=True)

    results = {
        'distribution_shift': dist_results,
        'prediction_analysis': {
            'stats': pred_results['stats']
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "="*100)
    print("DIAGNOSTICS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
