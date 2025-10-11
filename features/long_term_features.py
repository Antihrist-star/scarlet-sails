def add_long_term_memory_features(df):
    df = df.copy()
    import pandas as pd
    import numpy as np
    
    # 1. Distance from 200MA
    ma_200 = df['close'].rolling(window=200, min_periods=200).mean()
    distance = (df['close'] - ma_200) / df['close']
    df['distance_from_200MA'] = distance.clip(-0.5, 0.5) * 2
    df['distance_from_200MA'].fillna(0, inplace=True)

    # 2-4. Volatility
    returns = df['close'].pct_change()
    df['volatility_3d'] = returns.rolling(window=288, min_periods=100).std()
    df['volatility_7d'] = returns.rolling(window=672, min_periods=200).std()
    df['volatility_14d'] = returns.rolling(window=1344, min_periods=400).std()
    df['volatility_3d'].fillna(df['volatility_3d'].median(), inplace=True)
    df['volatility_7d'].fillna(df['volatility_7d'].median(), inplace=True)
    df['volatility_14d'].fillna(df['volatility_14d'].median(), inplace=True)

    # 5. Vol regime
    vol_7d_median = df['volatility_7d'].median()
    df['vol_regime'] = 0
    df.loc[df['volatility_7d'] > vol_7d_median * 1.5, 'vol_regime'] = 1
    df.loc[df['volatility_7d'] < vol_7d_median * 0.7, 'vol_regime'] = -1

    # 6. Bars since high
    df['bars_since_7d_high'] = 0
    for i in range(len(df)):
        if i < 672:
            df.loc[df.index[i], 'bars_since_7d_high'] = i
        else:
            window_high = df['high'].iloc[i-672:i+1]
            last_high_idx = window_high.idxmax()
            bars = i - df.index.get_loc(last_high_idx)
            df.loc[df.index[i], 'bars_since_7d_high'] = min(bars, 672)

    # 7. Price position
    rolling_min = df['low'].rolling(window=2880, min_periods=100).min()
    rolling_max = df['high'].rolling(window=2880, min_periods=100).max()
    df['price_position_30d'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
    df['price_position_30d'] = df['price_position_30d'].clip(0, 1).fillna(0.5)

    return df

if __name__ == "__main__":
    print("✅ Features module created")