"""
ANALYZE RULE-BASED SIGNALS
Understand what 'signal' column contains and how to convert to actions

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')

from strategies.rule_based_v2 import RuleBasedStrategy

def prepare_data_for_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Map advanced features to classic indicators"""
    df = df.copy()
    
    if 'norm_rsi_pctile' in df.columns:
        df['RSI_14'] = df['norm_rsi_pctile'] * 100
    
    if 'norm_macd_zscore' in df.columns:
        df['MACD_12_26_9'] = df['norm_macd_zscore']
    
    if 'deriv_macd_diff1' in df.columns:
        df['MACDs_12_26_9'] = df['deriv_macd_diff1']
    
    if 'norm_bb_width_pctile' in df.columns and 'close' in df.columns:
        bb_width = df['norm_bb_width_pctile'] * df['close'] * 0.02
        df['BBM_20_2.0'] = df['close']
        df['BBL_20_2.0'] = df['close'] - bb_width
        df['BBU_20_2.0'] = df['close'] + bb_width
    
    if 'norm_atr_pctile' in df.columns and 'close' in df.columns:
        df['ATRr_14'] = df['norm_atr_pctile'] * df['close'] * 0.02
    
    return df


def main():
    """Analyze signal values"""
    print("="*80)
    print("ANALYZING RULE-BASED SIGNALS")
    print("="*80)
    print()
    
    # Load data
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    df = pd.read_parquet(data_file)
    test_df = df[df.index >= '2024-01-01']
    test_df = prepare_data_for_strategy(test_df)
    
    # Initialize strategy
    strategy = RuleBasedStrategy()
    
    # Generate signals on sample
    print("GENERATING SIGNALS:")
    print("-"*80)
    sample = test_df.iloc[-1000:]
    result = strategy.generate_signals(sample)
    
    print(f"Result shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print()
    
    # Analyze 'signal' column
    print("ANALYZING 'signal' COLUMN:")
    print("-"*80)
    
    if 'signal' in result.columns:
        signal_values = result['signal'].value_counts()
        print(f"Unique values in 'signal':")
        print(signal_values)
        print()
        
        print(f"Signal statistics:")
        print(f"  Min: {result['signal'].min()}")
        print(f"  Max: {result['signal'].max()}")
        print(f"  Mean: {result['signal'].mean():.4f}")
        print(f"  Non-zero: {(result['signal'] != 0).sum()}")
        print()
    
    # Analyze P_rb column
    print("ANALYZING 'P_rb' COLUMN:")
    print("-"*80)
    
    if 'P_rb' in result.columns:
        p_rb_stats = result['P_rb'].describe()
        print(p_rb_stats)
        print()
        
        print(f"Non-NaN P_rb values: {result['P_rb'].notna().sum()}")
        print()
    
    # Look for rows with actual signals
    print("ROWS WITH SIGNALS (signal != 0):")
    print("-"*80)
    
    if 'signal' in result.columns:
        signal_rows = result[result['signal'] != 0]
        if len(signal_rows) > 0:
            print(f"Found {len(signal_rows)} rows with signals")
            print(signal_rows[['P_rb', 'signal', 'W_opportunity', 'filters_product']].head(10))
        else:
            print("No rows with signal != 0")
    
    print()
    
    # Check if there's a conversion method
    print("CHECKING STRATEGY METHODS:")
    print("-"*80)
    
    methods = [m for m in dir(strategy) if not m.startswith('_')]
    
    signal_methods = []
    for method in methods:
        if any(keyword in method.lower() for keyword in ['action', 'trade', 'execute', 'decision']):
            signal_methods.append(method)
    
    if signal_methods:
        print("Found potential action methods:")
        for method in signal_methods:
            print(f"   ✅ {method}()")
    else:
        print("⚠️ No obvious action conversion methods")
    
    print()
    
    # Suggest conversion logic
    print("SUGGESTED CONVERSION LOGIC:")
    print("-"*80)
    print("Based on typical trading logic:")
    print("  signal == 1  → action = 'buy'")
    print("  signal == -1 → action = 'sell'")
    print("  signal == 0  → action = 'hold'")
    print()
    print("OR based on P_rb (probability):")
    print("  P_rb > 0.6  → action = 'buy'")
    print("  P_rb < 0.4  → action = 'sell'")
    print("  else        → action = 'hold'")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()