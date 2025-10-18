"""
Debug: Check for NaN/Inf in data
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import joblib

project_root = os.path.dirname(os.path.dirname(__file__))

# Load data
X_train = torch.load(os.path.join(project_root, "models", "X_train_swing_3d.pt"))
X_test = torch.load(os.path.join(project_root, "models", "X_test_swing_3d.pt"))

X_train_np = X_train.numpy()
X_test_np = X_test.numpy()

# Convert 3D to 2D
X_train_2d = X_train_np[:, -1, :]
X_test_2d = X_test_np[:, -1, :]

# Apply scaler
scaler = joblib.load(os.path.join(project_root, "models", "scaler_swing_3d.pkl"))
X_train_scaled = scaler.transform(X_train_2d)
X_test_scaled = scaler.transform(X_test_2d)

print("="*60)
print("DATA DEBUG")
print("="*60)

print(f"\n1. Original data (before scaling):")
print(f"   Train - NaN: {np.isnan(X_train_2d).sum()}, Inf: {np.isinf(X_train_2d).sum()}")
print(f"   Test  - NaN: {np.isnan(X_test_2d).sum()}, Inf: {np.isinf(X_test_2d).sum()}")
print(f"   Train - min: {X_train_2d.min():.6f}, max: {X_train_2d.max():.6f}")
print(f"   Test  - min: {X_test_2d.min():.6f}, max: {X_test_2d.max():.6f}")

print(f"\n2. After scaling:")
print(f"   Train - NaN: {np.isnan(X_train_scaled).sum()}, Inf: {np.isinf(X_train_scaled).sum()}")
print(f"   Test  - NaN: {np.isnan(X_test_scaled).sum()}, Inf: {np.isinf(X_test_scaled).sum()}")
print(f"   Train - min: {X_train_scaled.min():.6f}, max: {X_train_scaled.max():.6f}")
print(f"   Test  - min: {X_test_scaled.min():.6f}, max: {X_test_scaled.max():.6f}")

print(f"\n3. Distribution check:")
print(f"   Train - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"   Test  - mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")

print("\n" + "="*60)

if np.isnan(X_train_scaled).sum() > 0 or np.isinf(X_train_scaled).sum() > 0:
    print("❌ DATA HAS NaN/Inf! Need cleaning!")
else:
    print("✅ Data looks clean!")