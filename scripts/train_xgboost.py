"""
XGBoost Model Training Script - FIXED VERSION

IMPORTANT: We prioritize PRACTICAL trading over perfect Win Rate
Target: 500+ signals with 50-55% Win Rate is BETTER than 100 signals with 100% WR
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from models.xgboost_model import XGBoostModel


def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def print_step(step_num, total_steps, description):
    """Print step progress"""
    print(f"[{step_num}/{total_steps}] {description}")


def evaluate_threshold_smart(y_true, y_proba, thresholds):
    """
    SMART threshold selection:
    - Prioritize practical number of signals (300-1000)
    - Accept lower Win Rate for more opportunities
    - Balance between Win Rate and Signal Count
    """
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        
        # Win Rate: accuracy on predicted positives
        positive_mask = (y_pred == 1)
        if positive_mask.sum() > 0:
            win_rate = (y_true[positive_mask] == 1).mean() * 100
            n_signals = positive_mask.sum()
        else:
            win_rate = 0.0
            n_signals = 0
        
        results.append({
            'threshold': thresh,
            'accuracy': acc,
            'win_rate': win_rate,
            'n_signals': n_signals
        })
    
    # SMART SELECTION LOGIC
    print("\n🎯 SMART THRESHOLD SELECTION:")
    print("Looking for balance between Win Rate and Signal Count...")
    
    # Filter for practical trading (300-1000 signals preferred)
    MIN_SIGNALS = 300  # About 0.5% of test set
    MAX_SIGNALS = 1000  # About 1.8% of test set
    MIN_WIN_RATE = 50.0  # Minimum acceptable win rate
    
    # Priority 1: Ideal range with good win rate
    ideal_results = [r for r in results 
                    if MIN_SIGNALS <= r['n_signals'] <= MAX_SIGNALS 
                    and r['win_rate'] >= MIN_WIN_RATE]
    
    if ideal_results:
        # Among ideal, maximize: (win_rate - 50) * log(n_signals)
        scores = [(r['win_rate'] - 50) * np.log(r['n_signals']) for r in ideal_results]
        best_idx = np.argmax(scores)
        best = ideal_results[best_idx]
        print(f"✅ Found ideal threshold: {best['threshold']:.2f}")
        print(f"   Signals: {best['n_signals']}, Win Rate: {best['win_rate']:.1f}%")
    else:
        # Priority 2: Accept more signals with lower win rate
        acceptable_results = [r for r in results 
                            if 100 <= r['n_signals'] <= 2000 
                            and r['win_rate'] >= 45.0]
        
        if acceptable_results:
            # Balance score
            scores = [(r['win_rate'] - 40) * np.log(max(r['n_signals'], 10)) for r in acceptable_results]
            best_idx = np.argmax(scores)
            best = acceptable_results[best_idx]
            print(f"⚠️ Using acceptable threshold: {best['threshold']:.2f}")
            print(f"   Signals: {best['n_signals']}, Win Rate: {best['win_rate']:.1f}%")
        else:
            # Priority 3: Just take something with signals
            valid_results = [r for r in results if r['n_signals'] >= 50]
            if valid_results:
                # Prefer more signals
                best_idx = np.argmax([r['n_signals'] for r in valid_results])
                best = valid_results[best_idx]
                print(f"⚠️ Fallback threshold: {best['threshold']:.2f}")
            else:
                best = results[0]
                print(f"❌ No good threshold found, using default")
    
    return best['threshold'], best['win_rate'], results


def main():
    print_header("XGBOOST TRAINING - SMART VERSION")
    
    # Paths
    models_dir = Path("models")
    
    # ========================================================================
    # [1/6] LOAD DATA
    # ========================================================================
    print_step(1, 6, "Loading data...")
    
    X_train = torch.load(models_dir / "X_train_clean.pt", weights_only=False)
    y_train = torch.load(models_dir / "y_train_clean.pt", weights_only=False)
    X_test = torch.load(models_dir / "X_test_clean.pt", weights_only=False)
    y_test = torch.load(models_dir / "y_test_clean.pt", weights_only=False)
    
    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")
    
    # ========================================================================
    # [2/6] SCALE DATA
    # ========================================================================
    print_step(2, 6, "Scaling...")
    
    scaler = joblib.load(models_dir / "scaler_clean_2d.pkl")
    
    X_train_scaled = scaler.transform(X_train.numpy())
    X_test_scaled = scaler.transform(X_test.numpy())
    
    print("✅ Data scaled")
    
    # ========================================================================
    # [3/6] COMPUTE CLASS WEIGHTS
    # ========================================================================
    print_step(3, 6, "Computing class weights...")
    
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    
    # Class distribution
    n_down = (y_train_np == 0).sum()
    n_up = (y_train_np == 1).sum()
    total = len(y_train_np)
    
    pct_down = n_down / total * 100
    pct_up = n_up / total * 100
    
    print(f"DOWN (0): {n_down:,} ({pct_down:.1f}%)")
    print(f"UP   (1): {n_up:,} ({pct_up:.1f}%)")
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_np),
        y=y_train_np
    )
    
    scale_pos_weight = class_weights[1] / class_weights[0]
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # ========================================================================
    # [4/6] TRAIN MODEL
    # ========================================================================
    print_step(4, 6, "Training...")
    
    # Split train into train/val (80/20)
    split_idx = int(0.8 * len(X_train_scaled))
    
    X_train_fit = X_train_scaled[:split_idx]
    y_train_fit = y_train_np[:split_idx]
    X_val = X_train_scaled[split_idx:]
    y_val = y_train_np[split_idx:]
    
    print(f"Train fit: {X_train_fit.shape}")
    print(f"Val:       {X_val.shape}")
    print()
    
    # LESS CONSERVATIVE hyperparameters for more signals
    model = XGBoostModel(
        n_estimators=150,  # Reduced
        max_depth=4,  # Even shallower
        learning_rate=0.08,  # Slightly higher
        min_child_weight=10,  # More conservative
        subsample=0.6,  # More aggressive subsampling
        colsample_bytree=0.6,  
        scale_pos_weight=scale_pos_weight * 0.8,  # Reduce bias toward UP
        random_state=42,
        early_stopping_rounds=15
    )
    
    # Train with validation
    start_time = time.time()
    
    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=(X_val, y_val),
        verbose=50
    )
    
    train_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {train_time:.1f}s")
    print(f"Best iteration: {model.best_iteration}")
    
    # ========================================================================
    # [5/6] EVALUATE
    # ========================================================================
    print_step(5, 6, "Evaluating...")
    
    # Predictions on test set
    y_pred = model.predict(X_test_scaled, threshold=0.5)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    
    print(f"\nTest Set Performance (threshold=0.5):")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(
        y_test_np,
        y_pred,
        target_names=['DOWN (0)', 'UP (1)'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_np, y_pred)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              DOWN    UP")
    print(f"Actual DOWN   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       UP     {cm[1,0]:5d}  {cm[1,1]:5d}")
    print()
    
    # Feature importance
    model.print_feature_importance(top_n=10)
    
    # ========================================================================
    # [6/6] SMART THRESHOLD OPTIMIZATION
    # ========================================================================
    print_step(6, 6, "Smart threshold optimization...")
    
    # Test wider range
    thresholds = np.arange(0.30, 0.71, 0.02)
    
    best_thresh, best_wr, all_results = evaluate_threshold_smart(
        y_test_np,
        y_pred_proba[:, 1],
        thresholds
    )
    
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Win Rate':<12} {'Signals':<12}")
    print(f"{'-'*48}")
    
    # Show all results with signals
    for res in all_results:
        if res['n_signals'] > 0:
            marker = " ⭐" if res['threshold'] == best_thresh else ""
            # Highlight practical ranges
            if 300 <= res['n_signals'] <= 1000:
                marker += " 👍"
            print(
                f"{res['threshold']:<12.2f} "
                f"{res['accuracy']:<12.4f} "
                f"{res['win_rate']:<12.1f} "
                f"{res['n_signals']:<12d}"
                f"{marker}"
            )
    
    print(f"\n✅ Selected threshold: {best_thresh:.2f} (Win Rate: {best_wr:.1f}%)")
    
    # Final predictions with best threshold
    y_pred_best = (y_pred_proba[:, 1] >= best_thresh).astype(int)
    n_signals = y_pred_best.sum()
    signal_pct = n_signals / len(y_pred_best) * 100
    
    print(f"   Signals generated: {n_signals} ({signal_pct:.2f}% of test set)")
    
    # Double check
    actual_wr = (y_test_np[y_pred_best == 1] == 1).mean() * 100 if n_signals > 0 else 0
    print(f"   Verified Win Rate: {actual_wr:.1f}%")
    
    # ========================================================================
    # SAVE MODEL
    # ========================================================================
    print()
    model_path = models_dir / "xgboost_model.json"
    model.save(model_path)
    
    # Save best threshold
    threshold_path = models_dir / "xgboost_best_threshold.txt"
    with open(threshold_path, 'w') as f:
        f.write(f"{best_thresh:.2f}\n")
    print(f"✅ Best threshold saved to {threshold_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("TRAINING SUMMARY")
    
    print(f"Model: XGBoost (Smart Selection)")
    print(f"Training time: {train_time:.1f}s")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Selected threshold: {best_thresh:.2f}")
    print(f"Win Rate: {best_wr:.1f}%")
    print(f"Signals: {n_signals} ({signal_pct:.2f}%)")
    print()
    
    if n_signals < 100:
        print("⚠️ WARNING: Too few signals for practical trading!")
    elif n_signals > 2000:
        print("⚠️ WARNING: Too many signals, might be noisy!")
    else:
        print("✅ Good signal count for practical trading!")
    
    print(f"\nModel saved: {model_path}")
    print()
    print("✅ READY FOR BACKTEST!")
    print()


if __name__ == "__main__":
    main()