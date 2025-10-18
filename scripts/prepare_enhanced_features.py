"""
Prepare Enhanced Features V2
- Load raw OHLCV (BTC_USDT_15m_FULL.parquet)
- Compute advanced features using features/advanced_features.py on the raw data
- Load old 3D data (X_train_swing_3d.pt, y_train_swing_3d.pt, X_test_swing_3d.pt, y_test_swing_3d.pt)
- The old 3D data was created from a slice of the raw data.
- Apply 3D->2D conversion (last timestep) to old data to get 2D old features.
- Align the index range of the old 2D features with the corresponding slice from the raw data + new features.
- Combine the old 31 features with the new features computed on the aligned slice.
- Clean NaN from the combined features.
- Scale the combined features using a new StandardScaler.
- Save the new combined features as X_train_enhanced.pt, y_train_enhanced.pt, X_test_enhanced.pt, y_test_enhanced.pt
- Save the new scaler as scaler_enhanced.pkl
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from features.advanced_features import compute_all_advanced_features

# --- Пути ---
project_root = os.path.dirname(os.path.dirname(__file__))
raw_data_path = os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet")

# Старые 3D данные (из которых был получен PF 1.42)
old_X_train_path = os.path.join(project_root, "models", "X_train_swing_3d.pt")
old_y_train_path = os.path.join(project_root, "models", "y_train_swing_3d.pt")
old_X_test_path = os.path.join(project_root, "models", "X_test_swing_3d.pt")
old_y_test_path = os.path.join(project_root, "models", "y_test_swing_3d.pt")

# Новые файлы для сохранения
new_X_train_path = os.path.join(project_root, "models", "X_train_enhanced.pt")
new_y_train_path = os.path.join(project_root, "models", "y_train_enhanced.pt")
new_X_test_path = os.path.join(project_root, "models", "X_test_enhanced.pt")
new_y_test_path = os.path.join(project_root, "models", "y_test_enhanced.pt")
new_scaler_path = os.path.join(project_root, "models", "scaler_enhanced.pkl")

print("="*80)
print("PREPARING ENHANCED FEATURES (PF 1.42 -> PF 2.0 Pipeline)")
print("="*80)

# 1. Загрузка старых 3D данных для получения длины и сопоставления
print("\n[1/6] Loading old 3D data to get shapes and alignment info...")
try:
    X_train_3d = torch.load(old_X_train_path) # Shape: (N_train, 60, 31)
    y_train_old = torch.load(old_y_train_path)   # Shape: (N_train,)
    X_test_3d = torch.load(old_X_test_path)  # Shape: (N_test, 60, 31)
    y_test_old = torch.load(old_y_test_path)     # Shape: (N_test,)
except FileNotFoundError as e:
    print(f"❌ Error loading old data: {e}")
    print("Make sure the files X_train_swing_3d.pt, y_train_swing_3d.pt, etc. exist in the models/ directory.")
    sys.exit(1)

N_train_old = X_train_3d.shape[0]
N_test_old = X_test_3d.shape[0]

print(f"Old X_train_3d shape: {X_train_3d.shape}")
print(f"Old y_train shape: {y_train_old.shape}")
print(f"Old X_test_3d shape: {X_test_3d.shape}")
print(f"Old y_test shape: {y_test_old.shape}")

# 2. Загрузка raw OHLCV данных
print("\n[2/6] Loading raw OHLCV data...")
try:
    ohlcv_raw = pd.read_parquet(raw_data_path)
except FileNotFoundError:
    print(f"❌ Error: Raw data file not found at {raw_data_path}")
    sys.exit(1)

print(f"Raw data shape: {ohlcv_raw.shape}")

# 3. Вычисление новых признаков на raw OHLCV
# Для этого нам нужны базовые индикаторы, которые использовались в compute_all_advanced_features
# Предположим, что в compute_all_advanced_features они вычисляются на лету или уже есть.
# Если они не вычислены, их нужно добавить в raw_data_path или вычислить здесь.
# Проверим, какие колонки есть:
print(f"Raw data columns: {list(ohlcv_raw.columns)}")

# Вычисляем базовые индикаторы, необходимые для advanced_features
print("  Computing basic indicators (RSI, EMA, BB, ATR, returns)...")
def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(window=window).mean()

def compute_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    bb_upper = rolling_mean + (rolling_std * num_std)
    bb_lower = rolling_mean - (rolling_std * num_std)
    return bb_upper, rolling_mean, bb_lower, rolling_std

# Вычисляем
ohlcv_raw['RSI_14'] = compute_rsi(ohlcv_raw['close'], window=14)
ohlcv_raw['EMA_9'] = ohlcv_raw['close'].ewm(span=9).mean()
ohlcv_raw['EMA_21'] = ohlcv_raw['close'].ewm(span=21).mean()
ohlcv_raw['SMA_50'] = ohlcv_raw['close'].rolling(window=50).mean()
bb_upper, bb_middle, bb_lower, bb_std = compute_bollinger_bands(ohlcv_raw['close'], window=20, num_std=2)
ohlcv_raw['BB_upper'] = bb_upper
ohlcv_raw['BB_middle'] = bb_middle
ohlcv_raw['BB_lower'] = bb_lower
ohlcv_raw['BB_std'] = bb_std
ohlcv_raw['ATR_14'] = compute_atr(ohlcv_raw, window=14)
ohlcv_raw['returns_5'] = ohlcv_raw['close'].pct_change(5)
ohlcv_raw['returns_20'] = ohlcv_raw['close'].pct_change(20)

# Убедимся, что нет NaN от вычисления индикаторов (обычно первые N значений)
# Удалим строки с NaN, которые могли появиться при вычислении индикаторов
print(f"  Raw data shape before dropna: {ohlcv_raw.shape}")
ohlcv_raw.dropna(inplace=True)
print(f"  Raw data shape after dropna: {ohlcv_raw.shape}")

# 4. Применение advanced_features
print("\n[3/6] Computing advanced features on raw data...")
# Проверим, что все необходимые колонки есть перед вызовом
required_cols = ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'EMA_9', 'EMA_21', 'SMA_50', 'ATR_14', 'returns_5', 'returns_20', 'BB_upper', 'BB_middle', 'BB_lower', 'BB_std']
missing_cols = [col for col in required_cols if col not in ohlcv_raw.columns]
if missing_cols:
    print(f"❌ Error: Missing required columns for advanced features: {missing_cols}")
    sys.exit(1)

ohlcv_with_features, new_feature_names = compute_all_advanced_features(ohlcv_raw)
print(f"  Shape after advanced features: {ohlcv_with_features.shape}")
print(f"  New feature names ({len(new_feature_names)}): {new_feature_names[:10]}...") # Печать первых 10

# Убедимся, что нет NaN в новых признаках (могли появиться от rolling quantile, деления на 0 и т.п.)
print(f"  NaN in new features before final dropna: {ohlcv_with_features[new_feature_names].isna().sum().sum()}")
# Удаляем строки с NaN из объединённого датафрейма (включая новые признаки)
print(f"  Data shape before final dropna: {ohlcv_with_features.shape}")
ohlcv_with_features_clean = ohlcv_with_features.dropna()
print(f"  Data shape after final dropna: {ohlcv_with_features_clean.shape}")

# 5. Сопоставление индексов и извлечение соответствующих срезов
print("\n[4/6] Aligning indices and extracting slices...")
# Старые 3D данные (X_train_3d, X_test_3d) были созданы из *конца* исходного датасета.
# X_test_3d.shape[0] = N_test_old
# X_train_3d.shape[0] = N_train_old
# Общее количество срезов (последних шагов) = N_train_old + N_test_old
# Эти срезы соответствуют последним N_train_old + N_test_old строкам из исходного датасета *после* создания 3D.
# Но мы хотим получить признаки для *последнего временного шага* в каждом срезе 3D.
# Т.е., срез [i, -1, :] из X_3d соответствует строке i из датасета *после* 3D->2D преобразования.
# Этот индекс i указывает на строку в *оригинальном* датасете (после очистки и добавления признаков).
# Итак, X_test_3d.shape[0] срезов (с последними шагами) соответствует строкам [-(N_train_old + N_test_old) : -N_train_old] в 2D-представлении исходного датасета.
# X_train_3d.shape[0] срезов соответствует строкам [-N_train_old : ] в 2D-представлении исходного датасета.
# Мы должны выбрать строки из ohlcv_with_features_clean, соответствующие этим индексам.

total_old_2d_len = N_train_old + N_test_old
if len(ohlcv_with_features_clean) < total_old_2d_len:
    print(f"❌ Error: Not enough rows in cleaned features ({len(ohlcv_with_features_clean)}) for old data length ({total_old_2d_len}).")
    print("This might happen if too many NaN values were dropped.")
    sys.exit(1)

# Сбросим индекс, чтобы использовать позиционную индексацию
ohlcv_with_features_clean = ohlcv_with_features_clean.reset_index(drop=True)

# Выбираем срезы
X_test_start_idx = -(N_train_old + N_test_old)
X_test_end_idx = -N_train_old
X_train_start_idx = -N_train_old
X_train_end_idx = len(ohlcv_with_features_clean) # Should be the end

X_test_features_df = ohlcv_with_features_clean.iloc[X_test_start_idx:X_test_end_idx]
X_train_features_df = ohlcv_with_features_clean.iloc[X_train_start_idx:X_train_end_idx]

print(f"  Selected X_train features slice shape: {X_train_features_df.shape}")
print(f"  Selected X_test features slice shape: {X_test_features_df.shape}")

# 6. Преобразование старых 3D в 2D (последний шаг) для получения старых признаков
print("\n[5/6] Converting old 3D features to 2D and combining...")
X_train_2d_old = X_train_3d.numpy()[:, -1, :] # Shape: (N_train_old, 31)
X_test_2d_old = X_test_3d.numpy()[:, -1, :]   # Shape: (N_test_old, 31)

print(f"  Old X_train_2d shape: {X_train_2d_old.shape}")
print(f"  Old X_test_2d shape: {X_test_2d_old.shape}")

# Извлечение новых признаков из датафреймов
X_train_new_features_np = X_train_features_df[new_feature_names].values
X_test_new_features_np = X_test_features_df[new_feature_names].values

print(f"  New X_train features shape: {X_train_new_features_np.shape}")
print(f"  New X_test features shape: {X_test_new_features_np.shape}")

# Проверка совпадения количества строк
if X_train_2d_old.shape[0] != X_train_new_features_np.shape[0]:
    print(f"❌ Error: Row count mismatch for train: old {X_train_2d_old.shape[0]} vs new {X_train_new_features_np.shape[0]}")
    sys.exit(1)
if X_test_2d_old.shape[0] != X_test_new_features_np.shape[0]:
    print(f"❌ Error: Row count mismatch for test: old {X_test_2d_old.shape[0]} vs new {X_test_new_features_np.shape[0]}")
    sys.exit(1)

# Объединение старых и новых признаков
X_train_combined = np.concatenate([X_train_2d_old, X_train_new_features_np], axis=1)
X_test_combined = np.concatenate([X_test_2d_old, X_test_new_features_np], axis=1)

print(f"  Combined X_train shape: {X_train_combined.shape} (old {X_train_2d_old.shape[1]} + new {X_train_new_features_np.shape[1]})")
print(f"  Combined X_test shape: {X_test_combined.shape} (old {X_test_2d_old.shape[1]} + new {X_test_new_features_np.shape[1]})")

# 7. Очистка NaN из объединённых признаков (как в clean_nan_data.py)
print("\n[6/6] Cleaning NaN from combined features...")
train_nan_mask = np.isnan(X_train_combined).any(axis=1)
test_nan_mask = np.isnan(X_test_combined).any(axis=1)

print(f"  Train rows with NaN: {train_nan_mask.sum()}/{len(train_nan_mask)}")
print(f"  Test rows with NaN: {test_nan_mask.sum()}/{len(test_nan_mask)}")

# Удаление строк с NaN
X_train_clean = X_train_combined[~train_nan_mask]
y_train_clean = y_train_old.numpy()[~train_nan_mask] # Применяем тот же фильтр к метке
X_test_clean = X_test_combined[~test_nan_mask]
y_test_clean = y_test_old.numpy()[~test_nan_mask]

print(f"  After cleaning - X_train: {X_train_clean.shape}, y_train: {y_train_clean.shape}")
print(f"  After cleaning - X_test: {X_test_clean.shape}, y_test: {y_test_clean.shape}")

# 8. Нормализация объединённых признаков
print("\n[7/7] Scaling combined features...")
scaler_enhanced = StandardScaler()
X_train_scaled = scaler_enhanced.fit_transform(X_train_clean) # Обучаем на очищенном трейне
X_test_scaled = scaler_enhanced.transform(X_test_clean)       # Трансформим очищенный тест

# 9. Сохранение новых файлов
print("\n[8/8] Saving enhanced data files...")
torch.save(torch.tensor(X_train_scaled), new_X_train_path)
torch.save(torch.tensor(y_train_clean), new_y_train_path)
torch.save(torch.tensor(X_test_scaled), new_X_test_path)
torch.save(torch.tensor(y_test_clean), new_y_test_path)
joblib.dump(scaler_enhanced, new_scaler_path)

print(f"✅ Saved: {new_X_train_path}")
print(f"✅ Saved: {new_y_train_path}")
print(f"✅ Saved: {new_X_test_path}")
print(f"✅ Saved: {new_y_test_path}")
print(f"✅ Saved: {new_scaler_path}")

print("\n" + "="*80)
print("✅ ENHANCED FEATURES PREPARATION COMPLETE!")
print(f"Final shapes - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Total features: {X_train_scaled.shape[1]} (31 old + {len(new_feature_names)} new)")
print("="*80)
