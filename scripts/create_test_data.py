#!/usr/bin/env python3
"""Create synthetic OHLCV data for testing"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_synthetic_ohlcv(length=10000, seed=42):
    """Create synthetic OHLCV data for testing"""
    np.random.seed(seed)

    # Generate price movements
    returns = np.random.normal(0.0001, 0.005, length)
    prices = 50000 * np.exp(np.cumsum(returns))

    # Create OHLCV
    data = []
    for i, price in enumerate(prices):
        volatility = np.random.uniform(0.001, 0.01)
        open_price = price * (1 + np.random.normal(0, volatility))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, volatility)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, volatility)))
        close_price = price
        volume = np.random.uniform(1000, 50000)

        data.append({
            'timestamp': pd.Timestamp('2020-01-01') + pd.Timedelta(minutes=15*i),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'time': str(pd.Timestamp('2020-01-01') + pd.Timedelta(minutes=15*i))
        })

    df = pd.DataFrame(data)
    return df

# Create data
print("Creating synthetic OHLCV data...")
df = create_synthetic_ohlcv(length=10000)

# Save
data_dir = Path(__file__).parent.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

parquet_file = data_dir / "BTC_USDT_15m.parquet"
df.to_parquet(parquet_file)
print(f"✓ Saved to {parquet_file}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
