"""
CHECK DATA COLUMNS
Find out REAL column names in our data

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import pandas as pd
from pathlib import Path

def main():
    """Check actual column names"""
    print("="*80)
    print("CHECKING DATA COLUMNS")
    print("="*80)
    print()
    
    # Load data
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    
    if not Path(data_file).exists():
        print(f"❌ File not found: {data_file}")
        return
    
    df = pd.read_parquet(data_file)
    
    print(f"Loaded: {len(df)} bars")
    print(f"Total columns: {len(df.columns)}")
    print()
    
    # Group columns by type
    print("COLUMNS BY TYPE:")
    print("-"*80)
    
    # Basic OHLCV
    print("\n1. BASIC (OHLCV):")
    basic = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    for col in df.columns:
        if any(b in col.lower() for b in ['open', 'high', 'low', 'close', 'volume', 'time']):
            print(f"   {col}")
    
    # RSI
    print("\n2. RSI:")
    for col in df.columns:
        if 'rsi' in col.lower():
            print(f"   {col}")
    
    # MACD
    print("\n3. MACD:")
    for col in df.columns:
        if 'macd' in col.lower():
            print(f"   {col}")
    
    # Bollinger Bands
    print("\n4. BOLLINGER BANDS:")
    for col in df.columns:
        if 'bb' in col.lower() or 'boll' in col.lower():
            print(f"   {col}")
    
    # ATR
    print("\n5. ATR:")
    for col in df.columns:
        if 'atr' in col.lower():
            print(f"   {col}")
    
    # SMA/EMA
    print("\n6. MOVING AVERAGES:")
    for col in df.columns:
        if 'sma' in col.lower() or 'ema' in col.lower():
            print(f"   {col}")
    
    # Normalized
    print("\n7. NORMALIZED:")
    for col in df.columns:
        if 'norm' in col.lower():
            print(f"   {col}")
    
    # Derivative
    print("\n8. DERIVATIVE:")
    for col in df.columns:
        if 'deriv' in col.lower():
            print(f"   {col}")
    
    # Other
    print("\n9. OTHER:")
    known_prefixes = ['open', 'high', 'low', 'close', 'volume', 'time', 
                     'rsi', 'macd', 'bb', 'boll', 'atr', 'sma', 'ema', 
                     'norm', 'deriv']
    for col in df.columns:
        if not any(p in col.lower() for p in known_prefixes):
            print(f"   {col}")
    
    print()
    print("="*80)
    print("ALL COLUMNS:")
    print("-"*80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")
    
    print()
    print("="*80)
    print("SAVE THIS OUTPUT!")
    print("We need exact column names for strategy tests")
    print("="*80)


if __name__ == "__main__":
    main()