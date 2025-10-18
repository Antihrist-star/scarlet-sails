"""
Clean NaN from training data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import joblib

project_root = os.path.dirname(os.path.dirname(__file__))

print("="*60)
print("CLEANING NaN FROM TRAINING DATA")
print("="*60)

# Load data
print("\n[1/4] Loading data...")
X_train = torch.load(os.path.join(project_root, "models", "X_train_swing_3d.pt"))
y_train = torch.load(os.path.join(project_root, "models", "y_train_swing_3d.pt"))
X_test = torch.load(os.path.join(project_root, "models", "X_test_swing_3d.pt"))
y_test = torch.load(os.path.join(project_root, "models", "y_test_swing_3d.pt"))

X_train_np = X_train.numpy()
y_train_np = y_train.numpy()
X_test_np = X_test.numpy()
y_test_np = y_test.numpy()

print(f"Original shapes:")
print(f"  X_train: {X_train_np.shape}")
print(f"  X_test: {X_test_np.shape}")

# Convert 3D to 2D
print("\n[2/4] Converting 3D to 2D...")
X_train_2d = X_train_np[:, -1, :]  # (227640, 31)
X_test_2d = X_test_np[:, -1, :]

print(f"After conversion:")
print(f"  X_train_2d: {X_train_2d.shape}")
print(f"  X_test_2d: {X_test_2d.shape}")

# Check for NaN
print("\n[3/4] Checking for NaN...")
train_nan_mask = np.isnan(X_train_2d).any(axis=1)
test_nan_mask = np.isnan(X_test_2d).any(axis=1)

print(f"  Train rows with NaN: {train_nan_mask.sum()}/{len(train_nan_mask)} ({train_nan_mask.sum()/len(train_nan_mask)*100:.2f}%)")
print(f"  Test rows with NaN: {test_nan_mask.sum()}/{len(test_nan_mask)} ({test_nan_mask.sum()/len(test_nan_mask)*100:.2f}%)")

# Remove NaN rows
print("\n[4/4] Removing NaN rows...")
X_train_clean = X_train_2d[~train_nan_mask]
y_train_clean = y_train_np[~train_nan_mask]
X_test_clean = X_test_2d[~test_nan_mask]
y_test_clean = y_test_np[~test_nan_mask]

print(f"After cleaning:")
print(f"  X_train_clean: {X_train_clean.shape}")
print(f"  y_train_clean: {y_train_clean.shape}")
print(f"  X_test_clean: {X_test_clean.shape}")
print(f"  y_test_clean: {y_test_clean.shape}")

# Verify no NaN
print("\n[5/5] Verifying...")
print(f"  Train NaN: {np.isnan(X_train_clean).sum()}")
print(f"  Test NaN: {np.isnan(X_test_clean).sum()}")

# Save cleaned data
print("\n[6/6] Saving cleaned data...")
torch.save(torch.tensor(X_train_clean), os.path.join(project_root, "models", "X_train_clean.pt"))
torch.save(torch.tensor(y_train_clean), os.path.join(project_root, "models", "y_train_clean.pt"))
torch.save(torch.tensor(X_test_clean), os.path.join(project_root, "models", "X_test_clean.pt"))
torch.save(torch.tensor(y_test_clean), os.path.join(project_root, "models", "y_test_clean.pt"))

print("✅ Cleaned data saved:")
print(f"   models/X_train_clean.pt")
print(f"   models/y_train_clean.pt")
print(f"   models/X_test_clean.pt")
print(f"   models/y_test_clean.pt")

print("\n" + "="*60)
print("✅ DATA CLEANING COMPLETE!")
print("="*60)