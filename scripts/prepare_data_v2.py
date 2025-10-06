import pandas as pd
import numpy as np
import os
import talib
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def add_technical_indicators(df):
    """Добавление технических индикаторов с помощью TA-Lib"""
    
    # Преобразование в numpy arrays для TA-Lib
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # Trend indicators
    df['ema_9'] = talib.EMA(close, timeperiod=9)
    df['ema_21'] = talib.EMA(close, timeperiod=21)
    df['ema_50'] = talib.EMA(close, timeperiod=50)
    
    # Momentum indicators
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
    
    # Volatility
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
    
    # Volume indicators
    df['obv'] = talib.OBV(close, volume)
    df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = volume / df['volume_sma_20']
    
    return df

def create_classification_target(df, future_bars=4, threshold=0.002):
    """Создание бинарной классификации target"""
    
    # Будущая доходность через future_bars
    df['future_return'] = (df['close'].shift(-future_bars) - df['close']) / df['close']
    
    # Бинарная цель: 1 если рост > threshold, 0 иначе
    df['target'] = (df['future_return'] > threshold).astype(int)
    
    # Удаляем строки без будущих данных
    df = df[:-future_bars].copy()
    
    return df

def check_multicollinearity(df, feature_columns, threshold=0.90):
    """Проверка и удаление сильно коррелирующих features"""
    
    corr_matrix = df[feature_columns].corr().abs()
    
    # Находим пары с высокой корреляцией
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    # Удаляем второй feature из каждой пары
    features_to_drop = []
    for pair in high_corr_pairs:
        if pair[1] not in features_to_drop:
            features_to_drop.append(pair[1])
            print(f"Dropping {pair[1]} (corr with {pair[0]}: {pair[2]:.3f})")
    
    return [col for col in feature_columns if col not in features_to_drop]

def load_and_preprocess_data(data_dir="data/raw", sequence_length=100, symbol=None):
    """Загрузка и обработка данных с техническими индикаторами"""
    
    if symbol:
        # Обработка одного символа
        filename = f"{symbol.replace('/', '_')}_15m.parquet"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} not found")
        
        df = pd.read_parquet(filepath)
        print(f"Processing {symbol}: {len(df)} rows")
        
    else:
        # Объединение всех символов
        all_data = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".parquet") and "15m" in filename:
                filepath = os.path.join(data_dir, filename)
                df_temp = pd.read_parquet(filepath)
                all_data.append(df_temp)
        
        if not all_data:
            raise ValueError("No 15m parquet files found")
        
        df = pd.concat(all_data).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        print(f"Combined data: {len(df)} rows")
    
    # Добавление технических индикаторов
    df = add_technical_indicators(df)
    
    # Создание classification target
    df = create_classification_target(df, future_bars=4, threshold=0.002)
    
    # Удаление NaN
    df = df.dropna()
    print(f"After dropna: {len(df)} rows")
    
    # Определение feature columns
    feature_columns = [col for col in df.columns if col not in ['target', 'future_return']]
    
    # Проверка мультиколлинеарности
    feature_columns = check_multicollinearity(df, feature_columns, threshold=0.90)
    
    print(f"Final features ({len(feature_columns)}): {feature_columns}")
    
    # Проверка распределения target
    target_dist = df['target'].value_counts()
    print(f"Target distribution: {target_dist.to_dict()}")
    
    # Подготовка данных для модели
    X_data = df[feature_columns].values
    y_data = df['target'].values
    
    # Нормализация features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_X_data = scaler_X.fit_transform(X_data)
    
    # Создание sequences
    X, y = [], []
    for i in range(len(scaled_X_data) - sequence_length + 1):
        X.append(scaled_X_data[i:i+sequence_length])
        y.append(y_data[i+sequence_length-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Конвертация в PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Сохранение
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    
    return X, y, scaler_X, feature_columns

if __name__ == "__main__":
    # Тест с BTC
    X, y, scaler_X, features = load_and_preprocess_data(symbol="BTC/USDT")
    
    print(f"\nРезультаты:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features: {len(features)}")
    print(f"Target balance: {torch.bincount(y.long())}")