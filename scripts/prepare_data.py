import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch

def load_and_preprocess_data(data_dir="data/raw", sequence_length=60, target_column='close'):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and '15m' in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)

    if not all_data:
        raise ValueError("No 15m parquet files found in data/raw.")

    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

    # Use only relevant columns for scaling and model input
    features = ['open', 'high', 'low', 'close', 'volume']
    data = combined_df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, features.index(target_column)]) # Predict the next 'close' price

    X = np.array(X)
    y = np.array(y)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Save scaler for later use (e.g., inverse transform for predictions)
    os.makedirs("models", exist_ok=True)
    np.save('models/scaler_min.npy', scaler.data_min_)
    np.save('models/scaler_scale.npy', scaler.scale_)

    return X, y, scaler

if __name__ == '__main__':
    X, y, scaler = load_and_preprocess_data()
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print("Data preparation complete.")

