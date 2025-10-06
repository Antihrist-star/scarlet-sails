import pandas as pd
import numpy as np
import os
import talib
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def add_technical_indicators(df):
    """Добавление технических индикаторов"""
    
    high = df['high'].values
    low = df['low'].values  
    close = df['close'].values
    volume = df['volume'].values
    
    # Price-based features (относительные, не абсолютные)
    df['high_low_ratio'] = high / low
    df['close_open_ratio'] = close / df['open'].values
    df['high_close_ratio'] = high / close
    df['low_close_ratio'] = low / close
    
    # Trend indicators
    df['ema_9'] = talib.EMA(close, timeperiod=9) / close  # Нормализация
    df['ema_21'] = talib.EMA(close, timeperiod=21) / close
    
    # Momentum indicators  
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    
    # MACD (нормализованный)
    macd, macd_signal, macd_hist = talib.MACD(close)
    df['macd_norm'] = macd / close
    df['macd_histogram_norm'] = macd_hist / close
    
    # Bollinger Bands (относительные позиции)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # Volatility
    df['atr_norm'] = talib.ATR(high, low, close, timeperiod=14) / close
    
    # Volume indicators
    volume_sma = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / volume_sma
    
    # Returns
    df['return_1'] = df['close'].pct_change(1)
    df['return_4'] = df['close'].pct_change(4)
    
    return df

def create_balanced_target(df, future_bars=4):
    """Создание сбалансированного target"""
    
    # Будущая доходность
    future_return = (df['close'].shift(-future_bars) - df['close']) / df['close']
    
    # Используем процентили для баланса классов
    threshold_upper = future_return.quantile(0.6)  # Топ 40% = класс 1
    threshold_lower = future_return.quantile(0.4)  # Низ 40% = класс 0
    
    # Трехклассовая классификация -> бинарная
    df['target'] = 0
    df.loc[future_return > threshold_upper, 'target'] = 1
    df.loc[future_return < threshold_lower, 'target'] = 0
    
    # Удаляем средние 20% (нейтральные движения)
    df = df[(future_return > threshold_upper) | (future_return < threshold_lower)].copy()
    
    # Удаляем строки без будущих данных
    df = df[:-future_bars].copy()
    
    return df

def check_multicollinearity_fixed(df, feature_columns, threshold=0.85):
    """Исправленная проверка мультиколлинеарности"""
    
    # Проверяем корреляцию на ИСХОДНЫХ данных, не нормализованных
    corr_matrix = df[feature_columns].corr().abs()
    
    features_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                # Оставляем более простой индикатор
                if 'ratio' in col2 or 'norm' in col2:
                    features_to_drop.add(col2)
                else:
                    features_to_drop.add(col1)
                print(f"High correlation: {col1} vs {col2} ({corr_matrix.iloc[i, j]:.3f})")
    
    final_features = [col for col in feature_columns if col not in features_to_drop]
    print(f"Dropped {len(features_to_drop)} features: {list(features_to_drop)}")
    
    return final_features

def load_and_preprocess_data_v3(data_dir="data/raw", sequence_length=100, symbol="BTC/USDT"):
    """Исправленная версия препроцессинга"""
    
    # Загрузка данных
    filename = f"{symbol.replace('/', '_')}_15m.parquet"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise ValueError(f"File {filepath} not found")
    
    df = pd.read_parquet(filepath)
    print(f"Processing {symbol}: {len(df)} rows")
    
    # Добавление индикаторов
    df = add_technical_indicators(df)
    
    # Создание сбалансированного target
    df = create_balanced_target(df, future_bars=4)
    
    # Удаление NaN
    df = df.dropna()
    print(f"After processing: {len(df)} rows")
    
    # Feature columns (исключаем абсолютные цены)
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'open', 'high', 'low', 'close', 'volume'] 
                      and not col.startswith('future')]
    
    # Проверка корреляции
    feature_columns = check_multicollinearity_fixed(df, feature_columns, threshold=0.85)
    
    print(f"Final features ({len(feature_columns)}): {feature_columns}")
    
    # Проверка баланса классов
    target_dist = df['target'].value_counts()
    print(f"Target distribution: {target_dist.to_dict()}")
    
    # Подготовка данных
    X_data = df[feature_columns].values
    y_data = df['target'].values
    
    # Нормализация ТОЛЬКО features
    scaler_X = MinMaxScaler()
    scaled_X_data = scaler_X.fit_transform(X_data)
    
    # Создание sequences
    X, y = [], []
    for i in range(len(scaled_X_data) - sequence_length + 1):
        X.append(scaled_X_data[i:i+sequence_length])
        y.append(y_data[i+sequence_length-1])
    
    X = np.array(X)
    y = np.array(y)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Сохранение
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_X, "models/scaler_X_v3.pkl")
    
    return X, y, scaler_X, feature_columns

if __name__ == "__main__":
    X, y, scaler_X, features = load_and_preprocess_data_v3(symbol="BTC/USDT")
    
    print(f"\nРезультаты v3:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(features)}")
    print(f"Target balance: {torch.bincount(y.long())}")