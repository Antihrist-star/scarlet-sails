"""
TEST RULE-BASED STRATEGY
Isolated test to verify Rule-Based strategy works

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from strategies.rule_based_v2 import RuleBasedStrategy

def main():
    """Test Rule-Based strategy in isolation"""
    print("="*80)
    print("TESTING RULE-BASED STRATEGY")
    print("="*80)
    print()
    
    # Load data
    logger.info("Loading data...")
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded: {len(df)} bars")
    logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Use 2024 data
    test_df = df[df.index >= '2024-01-01']
    logger.info(f"Test period: {len(test_df)} bars")
    
    # Check required columns
    print()
    print("CHECKING REQUIRED COLUMNS:")
    print("-"*80)
    
    required_base = ['open', 'high', 'low', 'close', 'volume']
    required_indicators = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14']
    
    for col in required_base + required_indicators:
        if col in test_df.columns:
            print(f"✅ {col}")
        else:
            print(f"❌ {col} MISSING!")
    
    # Initialize strategy
    print()
    print("INITIALIZING STRATEGY:")
    print("-"*80)
    
    try:
        strategy = RuleBasedStrategy()
        logger.info("✅ Rule-Based strategy initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return
    
    # Test signal generation
    print()
    print("TESTING SIGNAL GENERATION:")
    print("-"*80)
    
    try:
        # Test on last 1000 bars
        test_sample = test_df.iloc[-1000:]
        logger.info(f"Testing on {len(test_sample)} bars...")
        
        result = strategy.generate_signal(test_sample)
        
        if result:
            print(f"✅ Signal generated!")
            print(f"   Action: {result.get('action', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Keys: {list(result.keys())}")
        else:
            print(f"⚠️ No signal generated (returned None or empty)")
        
    except Exception as e:
        logger.error(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test on multiple windows
    print()
    print("TESTING MULTIPLE WINDOWS:")
    print("-"*80)
    
    signals_count = 0
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    
    # Test every 100 bars
    for i in range(100, len(test_df), 100):
        try:
            window = test_df.iloc[:i]
            result = strategy.generate_signal(window)
            
            if result:
                action = result.get('action', 'hold')
                signals_count += 1
                
                if action == 'buy':
                    buy_signals += 1
                elif action == 'sell':
                    sell_signals += 1
                else:
                    hold_signals += 1
        except Exception as e:
            logger.error(f"Error at bar {i}: {e}")
    
    print(f"Total signals: {signals_count}")
    print(f"  Buy: {buy_signals}")
    print(f"  Sell: {sell_signals}")
    print(f"  Hold: {hold_signals}")
    
    if signals_count == 0:
        print()
        print("⚠️ WARNING: No signals generated!")
        print("This means the strategy is not working correctly.")
    else:
        print()
        print("✅ Strategy is generating signals!")
    
    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()