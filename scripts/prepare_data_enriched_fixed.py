"""
Prepare Enriched Data - FIXED VERSION
Handles inf/NaN and missing timestamps
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib
from features.advanced_features import compute_all_advanced_features

print("="*80)
print("ENRICHED DATA PREPARATION - FIXED")
print("="*80)

project_root = os.path.dirname(os.path.dirname(__file__))

# [1/7] Load old clean data
print("\n[1/7] Loading old clean 2D data...")
X_train_old = torch.load(os.path.join(project_root, "models", "X_train_clean.pt"))
y_train_old = torch.load(os.path.join(project_root, "models", "y_train_clean.pt"))
X_test_old = torch.load(os.path.join(project_root, "models", "X_test_clean.pt"))
y_test_old = torch.load(os.path.join(project_root, "models", "y_test_clean.pt"))

print(f"Old data:")
print(f"  X_train: {X_train_old.shape}")  # (223038, 31)
print(f"  X_test: {X_test_old.shape}")    # (56910, 31)

# [2/7] Load raw OHLCV
print("\n[2/7] Loading raw OHLCV...")
df = pd.read_parquet(os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet"))
print(f"Raw data: {len(df)} bars")
print(f"Columns: {list(df.columns)}")

# [3/7] Compute basic indicators
print("\n[3/7] Computing basic indicators...")
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

df['RSI_14'] = RSIIndicator(df['close'], 14).rsi()
df['EMA_9'] = EMAIndicator(df['close'], 9).ema_indicator()
df['EMA_21'] = EMAIndicator(df['close'], 21).ema_indicator()
df['SMA_50'] = SMAIndicator(df['close'], 50).sma_indicator()

bb = BollingerBands(df['close'], 20, 2)
df['BB_middle'] = bb.bollinger_mavg()
df['BB_upper'] = bb.bollinger_hband()
df['BB_lower'] = bb.bollinger_lband()

df['ATR_14'] = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
df['returns_5'] = df['close'].pct_change(5)
df['returns_20'] = df['close'].pct_change(20)

print("✅ Basic indicators computed")

# [4/7] Compute advanced features
print("\n[4/7] Computing advanced features...")
df_enriched, advanced_features = compute_all_advanced_features(df)
print(f"✅ Added {len(advanced_features)} advanced features")

# [5/7] CRITICAL: Replace inf and extreme values
print("\n[5/7] Cleaning inf/extreme values...")
df_advanced = df_enriched[advanced_features].copy()

# Replace inf with NaN
df_advanced = df_advanced.replace([np.inf, -np.inf], np.nan)

# Clip extreme values (beyond 10 std)
for col in advanced_features:
    mean = df_advanced[col].mean()
    std = df_advanced[col].std()
    if std > 0:
        lower_bound = mean - 10 * std
        upper_bound = mean + 10 * std
        df_advanced[col] = df_advanced[col].clip(lower_bound, upper_bound)

print(f"Inf replaced with NaN")
print(f"Extreme values clipped")

# Check NaN count
nan_count = df_advanced.isna().sum().sum()
print(f"NaN count after cleaning: {nan_count}")

# [6/7] Align with old data
print("\n[6/7] Aligning with old data...")

total_samples = len(X_train_old) + len(X_test_old)
print(f"Need {total_samples} samples total")

# Take last N samples from enriched data
df_aligned = df_advanced.iloc[-total_samples:].reset_index(drop=True)

# Split into train/test
split_idx = len(X_train_old)
new_train_features = df_aligned.iloc[:split_idx].values
new_test_features = df_aligned.iloc[split_idx:].values

print(f"New train features: {new_train_features.shape}")
print(f"New test features: {new_test_features.shape}")

# [7/7] Combine old + new features
print("\n[7/7] Combining old + new features...")

X_train_old_np = X_train_old.numpy()
X_test_old_np = X_test_old.numpy()

# Combine horizontally
X_train_combined = np.hstack([X_train_old_np, new_train_features])
X_test_combined = np.hstack([X_test_old_np, new_test_features])

print(f"Combined train: {X_train_combined.shape}")  # (223038, 54)
print(f"Combined test: {X_test_combined.shape}")    # (56910, 54)

# Remove rows with NaN
print("\nRemoving NaN rows...")
train_mask = ~np.isnan(X_train_combined).any(axis=1)
test_mask = ~np.isnan(X_test_combined).any(axis=1)

X_train_clean = X_train_combined[train_mask]
y_train_clean = y_train_old.numpy()[train_mask]
X_test_clean = X_test_combined[test_mask]
y_test_clean = y_test_old.numpy()[test_mask]

print(f"After NaN removal:")
print(f"  X_train: {X_train_clean.shape}")
print(f"  X_test: {X_test_clean.shape}")

# Check for inf again
if np.isinf(X_train_clean).any() or np.isinf(X_test_clean).any():
    print("⚠️  Still have inf! Replacing with 0...")
    X_train_clean = np.nan_to_num(X_train_clean, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_clean = np.nan_to_num(X_test_clean, nan=0.0, posinf=0.0, neginf=0.0)

# Scale
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test_clean)

print("✅ Scaling complete")

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_clean, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_clean, dtype=torch.long)

# Save
print("\n[SAVE] Saving enriched data...")
output_dir = os.path.join(project_root, "models")

torch.save(X_train_tensor, os.path.join(output_dir, "X_train_enriched_v2.pt"))
torch.save(y_train_tensor, os.path.join(output_dir, "y_train_enriched_v2.pt"))
torch.save(X_test_tensor, os.path.join(output_dir, "X_test_enriched_v2.pt"))
torch.save(y_test_tensor, os.path.join(output_dir, "y_test_enriched_v2.pt"))

joblib.dump(scaler, os.path.join(output_dir, "scaler_enriched_v2.pkl"))

# Save feature names (31 old + 23 new)
old_features = [f"old_feat_{i}" for i in range(31)]
all_features = old_features + advanced_features
joblib.dump(all_features, os.path.join(output_dir, "features_enriched_v2.pkl"))

print("\n✅ FILES SAVED:")
print("   - X_train_enriched_v2.pt")
print("   - y_train_enriched_v2.pt")
print("   - X_test_enriched_v2.pt")
print("   - y_test_enriched_v2.pt")
print("   - scaler_enriched_v2.pkl")
print("   - features_enriched_v2.pkl")

print("\n" + "="*80)
print("✅ ENRICHED DATA READY!")
print("="*80)
print(f"\nSummary:")
print(f"  Total features: {len(all_features)} (31 old + {len(advanced_features)} new)")
print(f"  Train samples: {len(X_train_clean)}")
print(f"  Test samples: {len(X_test_clean)}")
print(f"  Class balance: {(y_train_clean==1).sum()/len(y_train_clean)*100:.1f}% UP")

print("\n🎯 Next: Train model on enriched features!")