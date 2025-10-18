"""
Model Comparison: Logistic Baseline
Uses cleaned 2D data (output of clean_nan_data.py)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from models.logistic_baseline import LogisticBaseline

# Paths to CLEANED 2D data
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)

X_train_path = os.path.join(project_root, "models", "X_train_clean.pt")
y_train_path = os.path.join(project_root, "models", "y_train_clean.pt")
X_test_path = os.path.join(project_root, "models", "X_test_clean.pt")
y_test_path = os.path.join(project_root, "models", "y_test_clean.pt")
# We will create a new scaler for the clean 2D data
# scaler_path = os.path.join(project_root, "models", "scaler_clean_2d.pkl") # If you saved it from clean_nan_data.py

print("="*80)
print("LOGISTIC BASELINE MODEL - TRAINING & EVALUATION (Clean 2D Data)")
print("="*80)

# Load data
print("\n[1/4] Loading preprocessed CLEAN 2D data...")
try:
    X_train = torch.load(X_train_path)
    y_train = torch.load(y_train_path)
    X_test = torch.load(X_test_path)
    y_test = torch.load(y_test_path)
except FileNotFoundError as e:
    print(f"Error loading  {e}")
    print("Make sure clean data files exist (run clean_nan_data.py first).")
    sys.exit(1)

print(f"Loaded shapes:")
print(f"  X_train: {X_train.shape}")  # Should be (cleaned_N, 31)
print(f"  y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")   # Should be (cleaned_M, 31)
print(f"  y_test: {y_test.shape}")

# Convert to numpy - DO THIS BEFORE using *_np variables
X_train_np = X_train.numpy()
y_train_np = y_train.numpy()
X_test_np = X_test.numpy()
y_test_np = y_test.numpy()

print(f"  X_train_np shape: {X_train_np.shape}")
print(f"  X_test_np shape: {X_test_np.shape}")

# No 3D to 2D conversion needed here, data is already 2D
print("\n[2/4] Data is already 2D, applying StandardScaler...")

# Use StandardScaler fitted on the training data to avoid issues with old scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np) # Fit on clean training data
X_test_scaled = scaler.transform(X_test_np)       # Transform test data with the same scaler

print("✅ Data normalized with new StandardScaler")

# Class weights
classes = np.unique(y_train_np)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_np)
class_weights_dict = dict(zip(classes, class_weights))
print(f"\nClass distribution in training data:")
print(f"  Class 0 (DOWN): {(y_train_np == 0).sum()} ({(y_train_np == 0).sum()/len(y_train_np)*100:.1f}%)")
print(f"  Class 1 (UP):   {(y_train_np == 1).sum()} ({(y_train_np == 1).sum()/len(y_train_np)*100:.1f}%)")
print(f"Class weights: {class_weights_dict}")

cw_tensor = torch.tensor([class_weights_dict[0], class_weights_dict[1]], dtype=torch.float)

# CREATE MODEL with CORRECT input_dim (from 2D data)
print("\n[3/4] Training Logistic Baseline model...")
input_dim = X_train_scaled.shape[1]  # Should be 31
print(f"Input dimension: {input_dim}")

model = LogisticBaseline(input_dim=input_dim, class_weights=cw_tensor)
model.fit(X_train_scaled, y_train_np, epochs=100, lr=0.0001)

# EVALUATION
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_pred_proba = model.predict_proba(X_test_tensor)
    y_pred_default = model.predict(X_test_tensor, threshold=0.5)

# Default threshold
print("\n--- Default Threshold (0.5) ---")
print(f"Accuracy: {accuracy_score(y_test_np, y_pred_default):.4f}")
print(classification_report(y_test_np, y_pred_default, target_names=['DOWN', 'UP']))

# Optimize threshold
print("\n--- Threshold Optimization ---")
thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = 0.5
best_wr = 0

for threshold in thresholds:
    y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
    n_up = (y_pred == 1).sum()
    if n_up > 0:
        wr = ((y_pred == 1) & (y_test_np == 1)).sum() / n_up
        if wr > best_wr:
            best_wr = wr
            best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best Win Rate: {best_wr:.4f} ({best_wr*100:.2f}%)")

# Final predictions with optimal threshold
y_pred_opt = (y_pred_proba[:, 1] > best_threshold).astype(int)
acc_opt = accuracy_score(y_test_np, y_pred_opt)
n_trades = (y_pred_opt == 1).sum()
n_wins = ((y_pred_opt == 1) & (y_test_np == 1)).sum()

print(f"\n--- Optimal Threshold Results ---")
print(f"Accuracy: {acc_opt:.4f} ({acc_opt*100:.2f}%)")
print(f"Win Rate: {best_wr:.4f} ({best_wr*100:.2f}%)")
print(f"Trades: {n_trades}/{len(y_test_np)} ({n_trades/len(y_test_np)*100:.1f}%)")
print(f"Wins: {n_wins}/{n_trades}")

# Expected Value (simplified)
if n_trades > 0:
    profit_per_win = 0.02  # 2%
    loss_per_loss = 0.015  # 1.5%
    n_losses = n_trades - n_wins
    ev = (n_wins * profit_per_win - n_losses * loss_per_loss) / n_trades
    print(f"\n--- Expected Value ---")
    print(f"EV per trade: {ev:.6f} ({ev*100:.4f}%)")
    print(f"Total EV: {ev * n_trades:.6f}")

print("\n" + "="*80)
print("✅ LOGISTIC BASELINE COMPLETE!")
print("="*80)

# Save model
model_path = os.path.join(project_root, "models", "logistic_baseline_clean_2d.pth")
torch.save(model.state_dict(), model_path)
print(f"\n💾 Model saved: {model_path}")

# Save the scaler used for this model
scaler_path_saved = os.path.join(project_root, "models", "scaler_clean_2d.pkl")
joblib.dump(scaler, scaler_path_saved)
print(f"💾 Scaler saved: {scaler_path_saved}")
