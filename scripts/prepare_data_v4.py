import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def load_all_timeframes(data_dir="data/raw", symbol="BTC_USDT"):
    """
    Загрузить все timeframes для одного символа
    """
    print(f"\n=== Loading {symbol} All Timeframes ===")
    
    timeframes = {}
    for tf in ['15m', '1h', '4h', '1d']:
        filename = f"{data_dir}/{symbol}_{tf}_FULL.parquet"
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
            timeframes[tf] = df
            print(f"Loaded {tf}: {df.shape[0]} bars, {df.index[0]} to {df.index[-1]}")
        else:
            print(f"WARNING: {filename} not found!")
    
    return timeframes

def calculate_technical_indicators(df, prefix=""):
    """
    Добавить технические индикаторы
    """
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df[f'{prefix}RSI_14'] = rsi(df['close'], 14)
    df[f'{prefix}EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df[f'{prefix}EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df[f'{prefix}SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df[f'{prefix}BB_middle'] = df['close'].rolling(window=20).mean()
    df[f'{prefix}BB_std'] = df['close'].rolling(window=20).std()
    df[f'{prefix}BB_upper'] = df[f'{prefix}BB_middle'] + (df[f'{prefix}BB_std'] * 2)
    df[f'{prefix}BB_lower'] = df[f'{prefix}BB_middle'] - (df[f'{prefix}BB_std'] * 2)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    df[f'{prefix}ATR_14'] = true_range.rolling(window=14).mean()
    
    # Returns
    df[f'{prefix}returns_5'] = df['close'].pct_change(5)
    df[f'{prefix}returns_20'] = df['close'].pct_change(20)
    
    # Volume
    if 'volume' in df.columns:
        df[f'{prefix}volume_sma'] = df['volume'].rolling(20).mean()
        df[f'{prefix}volume_ratio'] = df['volume'] / df[f'{prefix}volume_sma']
    
    return df

def create_multitimeframe_features(timeframes):
    """
    Объединить все timeframes в единый датафрейм на 15m resolution
    """
    print("\n=== Creating Multi-Timeframe Features ===")
    
    # Базовый датафрейм - 15m
    df = timeframes['15m'].copy()
    
    # Добавить индикаторы для 15m
    df = calculate_technical_indicators(df, prefix='15m_')
    
    # Resample и merge higher timeframes
    for tf in ['1h', '4h', '1d']:
        df_tf = timeframes[tf].copy()
        df_tf = calculate_technical_indicators(df_tf, prefix=f'{tf}_')
        
        # Resample к 15m (forward fill)
        df_tf_resampled = df_tf.resample('15min').ffill()
        
        # Merge только ключевые колонки (избежать слишком много features)
        key_cols = [col for col in df_tf_resampled.columns if any(x in col for x in ['close', 'EMA', 'SMA', 'RSI', 'returns'])]
        
        for col in key_cols:
            df[col] = df_tf_resampled[col]
    
    print(f"Total features before cleaning: {len(df.columns)}")
    
    return df

def create_daily_direction_target(df):
    """
    Target: Направление цены на следующий день
    Более простой и предсказуемый чем profitable trades
    """
    print("\n=== Creating Daily Direction Target ===")
    
    # Future close через 96 bars (24 часа для 15m)
    FUTURE_BARS = 96
    THRESHOLD = 0.005  # 0.5% минимальное движение (выше noise)
    
    future_close = df['close'].shift(-FUTURE_BARS)
    price_change = (future_close - df['close']) / df['close']
    
    # Target = 1 если цена вырастет >0.5%
    df['target_daily_up'] = (price_change > THRESHOLD).astype(int)
    
    # Remove rows без future data
    df = df[:-FUTURE_BARS].copy()
    
    # Statistics
    target_dist = df['target_daily_up'].value_counts()
    print(f"Target distribution:")
    print(f"  DOWN (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0) / len(df) * 100:.1f}%)")
    print(f"  UP (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0) / len(df) * 100:.1f}%)")
    
    return df

def prepare_features_and_target(df):
    """
    Финальная подготовка features и target
    """
    print("\n=== Preparing Features ===")
    
    # Remove NaN
    df = df.dropna()
    print(f"After dropna: {df.shape}")
    
    # Exclude target и raw price columns
    exclude_cols = ['target_daily_up', 'close', 'open', 'high', 'low', 'volume']
    exclude_cols += [col for col in df.columns if 'close' in col.lower() and col != 'target_daily_up']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Selected features: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")
    
    X_data = df[feature_cols].values
    y_data = df['target_daily_up'].values
    
    return X_data, y_data, feature_cols

def load_and_preprocess_data(data_dir="data/raw", symbol="BTC_USDT", sequence_length=60):
    print(f"\n{'='*60}")
    print(f"MULTI-TIMEFRAME DATA PREPARATION V4")
    print(f"Symbol: {symbol}")
    print(f"{'='*60}")
    
    # Load all timeframes
    timeframes = load_all_timeframes(data_dir, symbol)
    
    if len(timeframes) < 4:
        raise ValueError(f"Missing timeframes for {symbol}. Need 15m, 1h, 4h, 1d")
    
    # Create multi-TF features
    df = create_multitimeframe_features(timeframes)
    
    # Create target
    df = create_daily_direction_target(df)
    
    # Prepare features
    X_data, y_data, feature_names = prepare_features_and_target(df)
    
    print(f"\nX shape: {X_data.shape}")
    print(f"y shape: {y_data.shape}")
    
    # Create sequences
    print("\n=== Creating Sequences ===")
    X_sequences = []
    y_targets = []
    
    for i in range(len(X_data) - sequence_length):
        X_sequences.append(X_data[i:i+sequence_length])
        y_targets.append(y_data[i+sequence_length])
    
    X = np.array(X_sequences)
    y = np.array(y_targets)
    
    print(f"Sequences created: {X.shape}")
    print(f"Target distribution in sequences: {np.bincount(y.astype(int))}")
    
    # TEMPORAL SPLIT
    split_idx = int(len(X) * 0.8)
    X_train_raw = X[:split_idx]
    X_test_raw = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\nTrain: {X_train_raw.shape}, Test: {X_test_raw.shape}")
    print(f"Train target: {np.bincount(y_train.astype(int))}")
    print(f"Test target: {np.bincount(y_test.astype(int))}")
    
    # SCALE ONLY ON TRAIN
    print("\n=== Scaling ===")
    scaler = MinMaxScaler()
    X_train_reshaped = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train_raw.shape)
    X_test_reshaped = X_test_raw.reshape(-1, X_test_raw.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test_raw.shape)
    
    # To tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, f"models/scaler_{symbol}_v4.pkl")
    joblib.dump(feature_names, f"models/features_{symbol}_v4.pkl")
    
    print("\n=== COMPLETE ===")
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test with BTC
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(symbol="BTC_USDT")
    
    print(f"\n{'='*60}")
    print("FINAL SHAPES:")
    print(f"Train X: {X_train.shape}, Train y: {y_train.shape}")
    print(f"Test X: {X_test.shape}, Test y: {y_test.shape}")
    print(f"Features: {X_train.shape[2]}")
    print(f"{'='*60}")