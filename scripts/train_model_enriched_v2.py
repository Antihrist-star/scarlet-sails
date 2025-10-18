"""
Train Logistic Baseline on Enriched Features (54 features)
FIXED: Correct class_weights handling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from models.logistic_baseline import LogisticBaseline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report

print("="*80)
print("TRAINING LOGISTIC MODEL - ENRICHED FEATURES (54)")
print("="*80)

project_root = os.path.dirname(os.path.dirname(__file__))

# Load enriched data
print("\n[1/5] Loading enriched data...")
X_train = torch.load(os.path.join(project_root, "models", "X_train_enriched_v2.pt"))
y_train = torch.load(os.path.join(project_root, "models", "y_train_enriched_v2.pt"))
X_test = torch.load(os.path.join(project_root, "models", "X_test_enriched_v2.pt"))
y_test = torch.load(os.path.join(project_root, "models", "y_test_enriched_v2.pt"))

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

# Convert to numpy
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

# Class distribution
print("\n[2/5] Analyzing class distribution...")
n_down = (y_train_np == 0).sum()
n_up = (y_train_np == 1).sum()
total = len(y_train_np)

print(f"Class distribution:")
print(f"  DOWN (0): {n_down:,} ({n_down/total*100:.1f}%)")
print(f"  UP (1):   {n_up:,} ({n_up/total*100:.1f}%)")

# Compute class weights
print("\n[3/5] Computing class weights...")
class_weights_np = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train_np), 
    y=y_train_np
)
class_weights_dict = {0: class_weights_np[0], 1: class_weights_np[1]}
class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

print(f"Class weights: DOWN={class_weights_dict[0]:.4f}, UP={class_weights_dict[1]:.4f}")

# Initialize model WITH class weights
print("\n[4/5] Training model...")
input_dim = X_train.shape[1]  # 54
print(f"Input dimension: {input_dim}")

model = LogisticBaseline(
    input_dim=input_dim, 
    class_weights=class_weights_tensor  # ← CORRECT: Pass here!
)

# Train (NO class_weights in fit!)
model.fit(
    X_train, 
    y_train_np,
    epochs=100,
    lr=0.0001  # ← NO class_weights parameter!
)

print("\n✅ Training complete!")

# Evaluate
print("\n[5/5] Evaluating on test set...")

# Get predictions
with torch.no_grad():
    y_pred_proba = model.predict_proba(X_test)
    y_pred_default = model.predict(X_test)  # Default threshold 0.5

# Metrics at default threshold
acc_default = accuracy_score(y_test_np, y_pred_default)

print("\n" + "="*80)
print("EVALUATION RESULTS - DEFAULT THRESHOLD (0.5)")
print("="*80)
print(f"Accuracy: {acc_default:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred_default, target_names=['DOWN', 'UP']))

# Threshold optimization for Win Rate
print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)

results = []
for thr in np.arange(0.50, 0.91, 0.02):
    y_pred_thr = (y_pred_proba[:, 1] > thr).astype(int)
    n_signals = y_pred_thr.sum()
    
    if n_signals == 0:
        continue
    
    # Win rate (precision for class 1)
    tp = ((y_pred_thr == 1) & (y_test_np == 1)).sum()
    wr = tp / n_signals if n_signals > 0 else 0
    
    # Accuracy
    acc = accuracy_score(y_test_np, y_pred_thr)
    
    results.append({
        'threshold': thr,
        'win_rate': wr,
        'accuracy': acc,
        'n_signals': n_signals
    })

# Find best by WR
best_by_wr = max(results, key=lambda x: x['win_rate'])

print(f"\nBest threshold (by Win Rate): {best_by_wr['threshold']:.2f}")
print(f"  Win Rate:   {best_by_wr['win_rate']:.2%}")
print(f"  Accuracy:   {best_by_wr['accuracy']:.4f}")
print(f"  Signals:    {best_by_wr['n_signals']}")

# Show top 5 thresholds
print("\nTop 5 thresholds by Win Rate:")
top5 = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:5]
for i, r in enumerate(top5, 1):
    print(f"  {i}. Thr={r['threshold']:.2f}: WR={r['win_rate']:.2%}, Acc={r['accuracy']:.4f}, Signals={r['n_signals']}")

# Save model
print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model_path = os.path.join(project_root, "models", "logistic_enriched_v2.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved: {model_path}")

# Save metadata
metadata = {
    'input_dim': input_dim,
    'n_features': input_dim,
    'best_threshold': best_by_wr['threshold'],
    'best_win_rate': best_by_wr['win_rate'],
    'accuracy_default': acc_default,
    'class_weights': class_weights_dict
}

import json
metadata_path = os.path.join(project_root, "models", "logistic_enriched_v2_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved: {metadata_path}")

# Summary
print("\n" + "="*80)
print("✅ TRAINING COMPLETED!")
print("="*80)
print(f"\nModel summary:")
print(f"  Input features:     {input_dim}")
print(f"  Training samples:   {len(X_train):,}")
print(f"  Test samples:       {len(X_test):,}")
print(f"  Default accuracy:   {acc_default:.4f}")
print(f"  Optimal threshold:  {best_by_wr['threshold']:.2f}")
print(f"  Best Win Rate:      {best_by_wr['win_rate']:.2%}")

print("\n🎯 COMPARISON WITH BASELINE (31 features):")
print("  Old (31 feat): WR ~50%, Threshold 0.82")
print(f"  New (54 feat): WR {best_by_wr['win_rate']:.1%}, Threshold {best_by_wr['threshold']:.2f}")

if best_by_wr['win_rate'] > 0.50:
    improvement = (best_by_wr['win_rate'] - 0.50) * 100
    print(f"  ✅ IMPROVEMENT: +{improvement:.1f}% Win Rate!")
else:
    print(f"  ⚠️  No improvement in Win Rate")

print("\n🚀 Next step: Run backtest!")
print("   python scripts/run_backtest_enriched_v2.py")