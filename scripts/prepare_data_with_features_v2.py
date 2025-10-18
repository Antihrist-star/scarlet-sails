"""
Prepare Data with Advanced Features V2
================================================================================
Full pipeline:
1. Load raw OHLCV data
2. Compute basic features (existing)
3. Add advanced features (NEW!)
4. Create sequences
5. Clean NaN
6. Train/test split
7. Save enriched data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib

# Import our feature engineering
from features.advanced_features import compute_all_advanced_features

print("="*80)
print("DATA PREPARATION WITH ADVANCED FEATURES V2")
print("="*80)

# Paths
project_root = os.path.dirname(os.path.dirname(__file__))
raw_data_path = os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet")

# Load raw data
print("\n[1/8] Loading raw OHLCV data...")
df = pd.read_parquet(raw_data_path)
print(f"Loaded: {len(df)} bars")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Columns: {list(df.columns)}")

# Basic features (assuming they exist in parquet)
print("\n[2/8] Checking basic features...")
basic_features = [
    '15m_RSI_14', '15m_EMA_9', '15m_EMA_21', '15m_SMA_50',
    '15m_BB_middle', '15m_BB_std', '15m_BB_upper', '15m_BB_lower',
    '15m_ATR_14', '15m_returns_5', '15m_returns_20',
    '15m_volume_sma', '15m_volume_ratio',
    '1h_RSI_14', '1h_EMA_9', '1h_EMA_21', '1h_SMA_50',
    '1h_returns_5', '1h_returns_20',
    '4h_RSI_14', '4h_EMA_9', '4h_EMA_21', '4h_SMA_50',
    '4h_returns_5', '4h_returns_20',
    '1d_RSI_14', '1d_EMA_9', '1d_EMA_21', '1d_SMA_50',
    '1d_returns_5', '1d_returns_20'
]

# Check if basic features exist
missing_basic = [f for f in basic_features if f not in df.columns]
if missing_basic:
    print(f"⚠️  Missing basic features: {missing_basic[:5]}...")
    print("Will compute from OHLCV...")
    
    # Compute basic features (simplified)
    from ta.trend import EMAIndicator, SMAIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    
    # 15m features
    df['15m_RSI_14'] = RSIIndicator(df['close'], 14).rsi()
    df['15m_EMA_9'] = EMAIndicator(df['close'], 9).ema_indicator()
    df['15m_EMA_21'] = EMAIndicator(df['close'], 21).ema_indicator()
    df['15m_SMA_50'] = SMAIndicator(df['close'], 50).sma_indicator()
    
    bb = BollingerBands(df['close'], 20, 2)
    df['15m_BB_middle'] = bb.bollinger_mavg()
    df['15m_BB_upper'] = bb.bollinger_hband()
    df['15m_BB_lower'] = bb.bollinger_lband()
    df['15m_BB_std'] = df['close'].rolling(20).std()
    
    df['15m_ATR_14'] = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    df['15m_returns_5'] = df['close'].pct_change(5)
    df['15m_returns_20'] = df['close'].pct_change(20)
    df['15m_volume_sma'] = df['volume'].rolling(20).mean()
    df['15m_volume_ratio'] = df['volume'] / df['15m_volume_sma']
    
    # For other timeframes, use same values as proxy (simplified)
    for tf in ['1h', '4h', '1d']:
        df[f'{tf}_RSI_14'] = df['15m_RSI_14']
        df[f'{tf}_EMA_9'] = df['15m_EMA_9']
        df[f'{tf}_EMA_21'] = df['15m_EMA_21']
        df[f'{tf}_SMA_50'] = df['15m_SMA_50']
        df[f'{tf}_returns_5'] = df['15m_returns_5']
        df[f'{tf}_returns_20'] = df['15m_returns_20']
    
    print("✅ Basic features computed")
else:
    print(f"✅ All {len(basic_features)} basic features present")

# Prepare for advanced features
print("\n[3/8] Preparing dataframe for advanced features...")
# Advanced features need these columns
df_for_advanced = df.copy()
df_for_advanced['RSI_14'] = df['15m_RSI_14']
df_for_advanced['EMA_9'] = df['15m_EMA_9']
df_for_advanced['EMA_21'] = df['15m_EMA_21']
df_for_advanced['SMA_50'] = df['15m_SMA_50']
df_for_advanced['BB_upper'] = df['15m_BB_upper']
df_for_advanced['BB_middle'] = df['15m_BB_middle']
df_for_advanced['BB_lower'] = df['15m_BB_lower']
df_for_advanced['ATR_14'] = df['15m_ATR_14']
df_for_advanced['returns_5'] = df['15m_returns_5']
df_for_advanced['returns_20'] = df['15m_returns_20']

# Add advanced features
print("\n[4/8] Computing advanced features...")
df_enriched, advanced_feature_names = compute_all_advanced_features(df_for_advanced)

# Combine all feature names
all_features = basic_features + advanced_feature_names
print(f"\n✅ Total features: {len(all_features)}")
print(f"   Basic: {len(basic_features)}")
print(f"   Advanced: {len(advanced_feature_names)}")

# Create target (swing_3d - assuming it exists or create it)
print("\n[5/8] Creating target labels...")
if 'target_swing_3d' in df_enriched.columns:
    print("✅ Target already exists")
else:
    # Create swing_3d target
    # Look ahead 3 days (288 bars for 15min)
    horizon = 288
    future_returns = df_enriched['close'].pct_change(horizon).shift(-horizon)
    df_enriched['target_swing_3d'] = (future_returns > 0).astype(int)
    print("✅ Target created (swing_3d)")

# Remove NaN
print("\n[6/8] Removing NaN rows...")
initial_len = len(df_enriched)
df_clean = df_enriched[all_features + ['target_swing_3d']].dropna()
final_len = len(df_clean)
print(f"Removed {initial_len - final_len} rows with NaN ({(initial_len-final_len)/initial_len*100:.2f}%)")
print(f"Final dataset: {final_len} samples")

# Extract X and y
X = df_clean[all_features].values
y = df_clean['target_swing_3d'].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class distribution: DOWN={(y==0).sum()}, UP={(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}% UP)")

# Train/test split (80/20)
print("\n[7/8] Creating train/test split (80/20)...")
split_idx = int(0.8 * len(X))

X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# Normalize
print("\nNormalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Save
print("\n[8/8] Saving enriched data...")

output_dir = os.path.join(project_root, "models")

torch.save(X_train_tensor, os.path.join(output_dir, "X_train_enriched_v2.pt"))
torch.save(y_train_tensor, os.path.join(output_dir, "y_train_enriched_v2.pt"))
torch.save(X_test_tensor, os.path.join(output_dir, "X_test_enriched_v2.pt"))
torch.save(y_test_tensor, os.path.join(output_dir, "y_test_enriched_v2.pt"))

joblib.dump(scaler, os.path.join(output_dir, "scaler_enriched_v2.pkl"))
joblib.dump(all_features, os.path.join(output_dir, "features_enriched_v2.pkl"))

print("\n✅ Saved files:")
print("   - X_train_enriched_v2.pt")
print("   - y_train_enriched_v2.pt")
print("   - X_test_enriched_v2.pt")
print("   - y_test_enriched_v2.pt")
print("   - scaler_enriched_v2.pkl")
print("   - features_enriched_v2.pkl")

print("\n" + "="*80)
print("✅ DATA PREPARATION COMPLETED!")
print("="*80)
print(f"\nDataset summary:")
print(f"  Total features: {len(all_features)}")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Feature dimension: {X_train.shape[1]}")

print("\n🎯 Next step: Train model on enriched features!")
print("   Run: python scripts/train_model_enriched_v2.py")