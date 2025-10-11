import sys
import os
# Добавляем корень проекта в Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from features.long_term_features import add_long_term_memory_features

def test_no_nan():
    print("\n🧪 Test 1: No NaN values...")
    
    # ПУТЬ ДЛЯ MSI - ПРОВЕРЬ ТОЧНОЕ ИМЯ!
    try:
        df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet')
    except FileNotFoundError:
        try:
            df = pd.read_parquet('data/raw/BTC_USDT_15m_FULL.parquet')
        except FileNotFoundError:
            print("❌ No data file found. Creating synthetic data...")
            # Создаём синтетические данные как fallback
            import numpy as np
            n = 10000
            df = pd.DataFrame({
                'open': np.random.randn(n) * 1000 + 30000,
                'high': np.random.randn(n) * 1000 + 30000,
                'low': np.random.randn(n) * 1000 + 30000,
                'close': np.random.randn(n) * 1000 + 30000,
                'volume': np.abs(np.random.randn(n)) * 1000
            })
            df['high'] = np.maximum(df['high'], df['open'])
            df['low'] = np.minimum(df['low'], df['open'])
    
    print(f"   Loaded: {len(df)} rows")
    
    df = add_long_term_memory_features(df)
    
    features = [
        'distance_from_200MA',
        'volatility_3d',
        'volatility_7d',
        'volatility_14d',
        'vol_regime',
        'bars_since_7d_high',
        'price_position_30d'
    ]
    
    total_nans = sum(df[col].isna().sum() for col in features)
    
    for col in features:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            status = "✅" if nan_count == 0 else "❌"
            print(f"   {status} {col}: {nan_count} NaN")
        else:
            print(f"   ❌ {col}: COLUMN MISSING")
            total_nans += 1
    
    assert total_nans == 0, f"Found {total_nans} NaN or missing columns"
    print("✅ Test 1 PASSED\n")

def test_feature_ranges():
    print("🧪 Test 2: Feature ranges...")
    
    try:
        df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet')
    except FileNotFoundError:
        try:
            df = pd.read_parquet('data/raw/BTC_USDT_15m_FULL.parquet')
        except FileNotFoundError:
            # Synthetic data
            import numpy as np
            n = 10000
            df = pd.DataFrame({
                'open': np.random.randn(n) * 1000 + 30000,
                'high': np.random.randn(n) * 1000 + 30000,
                'low': np.random.randn(n) * 1000 + 30000,
                'close': np.random.randn(n) * 1000 + 30000,
                'volume': np.abs(np.random.randn(n)) * 1000
            })
            df['high'] = np.maximum(df['high'], df['open'])
            df['low'] = np.minimum(df['low'], df['open'])
    
    df = add_long_term_memory_features(df)
    
    # Check distance_from_200MA
    if 'distance_from_200MA' in df.columns:
        assert -2 <= df['distance_from_200MA'].min() <= 2
        assert -2 <= df['distance_from_200MA'].max() <= 2
        print("   ✅ distance_from_200MA in range")
    
    # Check vol_regime
    if 'vol_regime' in df.columns:
        assert set(df['vol_regime'].dropna().unique()).issubset({-1, 0, 1})
        print("   ✅ vol_regime values correct")
    
    # Check price_position
    if 'price_position_30d' in df.columns:
        assert -0.1 <= df['price_position_30d'].min() <= 1.1
        assert -0.1 <= df['price_position_30d'].max() <= 1.1
        print("   ✅ price_position in [0,1]")
    
    print("✅ Test 2 PASSED\n")

def test_all_features():
    print("🧪 Test 3: All features present...")
    
    try:
        df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet')
    except FileNotFoundError:
        try:
            df = pd.read_parquet('data/raw/BTC_USDT_15m_FULL.parquet')
        except FileNotFoundError:
            # Synthetic data
            import numpy as np
            n = 10000
            df = pd.DataFrame({
                'open': np.random.randn(n) * 1000 + 30000,
                'high': np.random.randn(n) * 1000 + 30000,
                'low': np.random.randn(n) * 1000 + 30000,
                'close': np.random.randn(n) * 1000 + 30000,
                'volume': np.abs(np.random.randn(n)) * 1000
            })
            df['high'] = np.maximum(df['high'], df['open'])
            df['low'] = np.minimum(df['low'], df['open'])
    
    original = set(df.columns)
    df = add_long_term_memory_features(df)
    new_cols = set(df.columns) - original
    
    required = {
        'distance_from_200MA',
        'volatility_3d',
        'volatility_7d',
        'volatility_14d',
        'vol_regime',
        'bars_since_7d_high',
        'price_position_30d'
    }
    
    missing = required - new_cols
    assert len(missing) == 0, f"Missing: {missing}"
    
    print(f"   ✅ All {len(required)} features present")
    print("✅ Test 3 PASSED\n")

if __name__ == "__main__":
    print("="*60)
    print("FEATURE TESTS (MSI)")
    print("="*60)
    
    test_no_nan()
    test_feature_ranges()
    test_all_features()
    
    print("="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)