import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def load_and_preprocess_data(data_dir="data/raw", sequence_length=60, target_column="target_binary"):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and "15m" in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No 15m parquet files found in data/raw.")
    
    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    
    # RSI (14 periods)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    combined_df['RSI_14'] = calculate_rsi(combined_df['close'], 14)
    
    # EMA (9, 21)
    combined_df['EMA_9'] = combined_df['close'].ewm(span=9, adjust=False).mean()
    combined_df['EMA_21'] = combined_df['close'].ewm(span=21, adjust=False).mean()
    
    # Bollinger Bands (20, 2std)
    combined_df['BB_middle'] = combined_df['close'].rolling(window=20).mean()
    combined_df['BB_std'] = combined_df['close'].rolling(window=20).std()
    combined_df['BB_upper'] = combined_df['BB_middle'] + (combined_df['BB_std'] * 2)
    combined_df['BB_lower'] = combined_df['BB_middle'] - (combined_df['BB_std'] * 2)
    
    # ATR (14 periods)
    combined_df['HL'] = combined_df['high'] - combined_df['low']
    combined_df['HC'] = abs(combined_df['high'] - combined_df['close'].shift())
    combined_df['LC'] = abs(combined_df['low'] - combined_df['close'].shift())
    combined_df['TR'] = combined_df[['HL', 'HC', 'LC']].max(axis=1)
    combined_df['ATR_14'] = combined_df['TR'].rolling(window=14).mean()
    combined_df.drop(['HL', 'HC', 'LC', 'TR'], axis=1, inplace=True)
    
    # OBV
    combined_df['OBV'] = (np.sign(combined_df['close'].diff()) * combined_df['volume']).fillna(0).cumsum()
    
    # Удалить NaN после индикаторов
    combined_df = combined_df.dropna()
    
    # Binary classification target
    combined_df['target_binary'] = (combined_df['close'].pct_change(4).shift(-4) > 0.002).astype(int)
    combined_df = combined_df[:-4]
    
    # Features (все кроме target)
    feature_cols = [col for col in combined_df.columns if col != target_column]
    
    X_data = combined_df[feature_cols].values
    y_data = combined_df[target_column].values
    
    # Scalers
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_X_data = scaler_X.fit_transform(X_data)
    
    # Binary target не нормализуется
    scaled_y_data = y_data
    
    X, y = [], []
    for i in range(len(scaled_X_data) - sequence_length):
        X.append(scaled_X_data[i:i+sequence_length])
        y.append(scaled_y_data[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    
    return X, y, scaler_X, None

if __name__ == "__main__":
    X, y, scaler_X, _ = load_and_preprocess_data()
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Number of features: {X.shape[2]}")
    print("Data preparation complete.")