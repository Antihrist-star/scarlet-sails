"""
Advanced Feature Engineering
Computes features that improve edge:
- Volatility regime
- Market regime (trend)
- Volume dynamics
- Momentum features
- Multi-timeframe strength
"""

import pandas as pd
import numpy as np


def add_volatility_features(df, window=20):
    """
    Volatility regime indicators
    High vol = risky, Low vol = opportunity
    """
    # ATR-based volatility
    df['vol_atr_ratio'] = df['ATR_14'] / df['close']
    
    # Rolling volatility
    df['vol_std'] = df['returns_5'].rolling(window).std()
    
    # Volatility regime (0=low, 1=medium, 2=high)
    vol_quantiles = df['vol_std'].quantile([0.33, 0.67])
    df['vol_regime'] = pd.cut(
        df['vol_std'], 
        bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.67], np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    
    return df


def add_market_regime_features(df, short=20, long=50):
    """
    Market regime: bull, bear, choppy
    """
    # Trend strength
    df['trend_strength'] = (df['EMA_9'] - df['EMA_21']) / df['close']
    
    # Price position vs MA
    df['price_vs_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50']
    
    # Trend consistency (% of time EMA9 > EMA21)
    df['ema_cross'] = (df['EMA_9'] > df['EMA_21']).astype(int)
    df['trend_consistency'] = df['ema_cross'].rolling(short).mean()
    
    # Market regime classification
    # Bull: EMA9 > EMA21, price > SMA50
    # Bear: EMA9 < EMA21, price < SMA50
    # Choppy: mixed
    conditions = [
        (df['EMA_9'] > df['EMA_21']) & (df['close'] > df['SMA_50']),
        (df['EMA_9'] < df['EMA_21']) & (df['close'] < df['SMA_50'])
    ]
    df['market_regime'] = np.select(conditions, [1, -1], default=0)
    
    return df


def add_volume_features(df, window=20):
    """
    Volume dynamics and imbalance
    """
    # Volume momentum
    df['volume_momentum'] = df['volume'].pct_change(5)
    
    # Volume imbalance (short vs long)
    df['volume_sma_short'] = df['volume'].rolling(5).mean()
    df['volume_sma_long'] = df['volume'].rolling(20).mean()
    df['volume_imbalance'] = (df['volume_sma_short'] - df['volume_sma_long']) / df['volume_sma_long']
    
    # High volume days
    vol_threshold = df['volume'].rolling(window).quantile(0.8)
    df['high_volume_day'] = (df['volume'] > vol_threshold).astype(int)
    
    return df


def add_momentum_features(df):
    """
    Multi-period momentum
    """
    # RSI divergence (price vs RSI)
    df['rsi_divergence'] = df['RSI_14'].diff(5) - (df['close'].pct_change(5) * 100)
    
    # Momentum strength
    df['momentum_5'] = df['returns_5']
    df['momentum_20'] = df['returns_20']
    df['momentum_acceleration'] = df['momentum_5'] - df['momentum_20']
    
    # Price distance from extremes
    df['rolling_high'] = df['high'].rolling(50).max()
    df['rolling_low'] = df['low'].rolling(50).min()
    df['dist_from_high'] = (df['rolling_high'] - df['close']) / df['rolling_high']
    df['dist_from_low'] = (df['close'] - df['rolling_low']) / (df['rolling_high'] - df['rolling_low'])
    
    return df


def add_bollinger_features(df):
    """
    Bollinger Band-based features
    """
    # BB position (where is price in the band?)
    df['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # BB width (volatility proxy)
    df['bb_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # BB squeeze (low volatility)
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(int)
    
    return df


def add_price_action_features(df):
    """
    Candlestick and price action
    """
    # Candle body size
    df['candle_body'] = abs(df['close'] - df['open']) / df['open']
    
    # Candle direction
    df['candle_direction'] = np.sign(df['close'] - df['open'])
    
    # Upper/lower shadows
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    
    # Consecutive candles in same direction
    df['consecutive_up'] = (df['candle_direction'] == 1).astype(int).groupby(
        (df['candle_direction'] != df['candle_direction'].shift()).cumsum()
    ).cumsum()
    
    df['consecutive_down'] = (df['candle_direction'] == -1).astype(int).groupby(
        (df['candle_direction'] != df['candle_direction'].shift()).cumsum()
    ).cumsum()
    
    return df


def compute_all_advanced_features(df):
    """
    Master function to compute all advanced features
    
    Args:
        df: DataFrame with OHLCV and basic features
        
    Returns:
        DataFrame with all advanced features added
    """
    print("Computing advanced features...")
    
    df = df.copy()
    
    # Add each feature group
    df = add_volatility_features(df)
    df = add_market_regime_features(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)
    df = add_bollinger_features(df)
    df = add_price_action_features(df)
    
    # Count features
    advanced_features = [
        'vol_atr_ratio', 'vol_std', 'vol_regime',
        'trend_strength', 'price_vs_sma50', 'trend_consistency', 'market_regime',
        'volume_momentum', 'volume_imbalance', 'high_volume_day',
        'rsi_divergence', 'momentum_acceleration', 'dist_from_high', 'dist_from_low',
        'bb_position', 'bb_width', 'bb_squeeze',
        'candle_body', 'candle_direction', 'upper_shadow', 'lower_shadow',
        'consecutive_up', 'consecutive_down'
    ]
    
    print(f"✅ Added {len(advanced_features)} advanced features")
    
    # Check for NaN
    nan_count = df[advanced_features].isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️  {nan_count} NaN values in advanced features (will be cleaned later)")
    
    return df, advanced_features


if __name__ == "__main__":
    # Test
    print("Testing advanced features...")
    
    # Create dummy data
    test_df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'RSI_14': np.random.uniform(20, 80, 1000),
        'EMA_9': np.random.randn(1000).cumsum() + 100,
        'EMA_21': np.random.randn(1000).cumsum() + 100,
        'SMA_50': np.random.randn(1000).cumsum() + 100,
        'BB_upper': np.random.randn(1000).cumsum() + 105,
        'BB_middle': np.random.randn(1000).cumsum() + 100,
        'BB_lower': np.random.randn(1000).cumsum() + 95,
        'ATR_14': np.random.uniform(1, 5, 1000),
        'returns_5': np.random.randn(1000) * 0.01,
        'returns_20': np.random.randn(1000) * 0.01
    })
    
    result_df, features = compute_all_advanced_features(test_df)
    
    print(f"\nOriginal features: {len(test_df.columns)}")
    print(f"New features: {len(result_df.columns)}")
    print(f"Advanced features added: {len(features)}")
    print("\nSample features:")
    print(result_df[features].head())
    
    print("\n✅ Test passed!")