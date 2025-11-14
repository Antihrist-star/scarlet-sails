"""
Download new data: Oct 24 - Nov 12, 2025
For all 14 assets × 4 timeframes
"""

import ccxt
import pandas as pd
from datetime import datetime, timezone
import time
import os

# Configuration
ASSETS = ['BTC', 'ETH', 'SOL', 'LINK', 'LDO', 'SUI', 'HBAR', 'ENA', 
          'ALGO', 'AVAX', 'DOT', 'LTC', 'ONDO', 'UNI']
TIMEFRAMES = ['15m', '1h', '4h', '1d']
START_DATE = '2025-10-24 14:15:00'  # After last data
END_DATE = '2025-11-12 23:59:59'     # Today

# Output directory
OUTPUT_DIR = 'data/raw'

def download_data(exchange, symbol, timeframe, since, limit=1000):
    """Download OHLCV data from Binance"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv:
        return None
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df

def main():
    print("="*80)
    print("DOWNLOADING NEW DATA: Oct 24 - Nov 12, 2025")
    print("="*80)
    
    # Initialize Binance
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(END_DATE, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    total = len(ASSETS) * len(TIMEFRAMES)
    current = 0
    
    for asset in ASSETS:
        print(f"\n{asset}:")
        symbol = f"{asset}/USDT"
        
        for tf in TIMEFRAMES:
            current += 1
            print(f"  [{current}/{total}] {tf}...", end=' ', flush=True)
            
            try:
                # Load existing data
                existing_file = f'{OUTPUT_DIR}/{asset}_USDT_{tf}.parquet'
                if not os.path.exists(existing_file):
                    print(f"SKIP (no existing file)")
                    continue
                
                existing_df = pd.read_parquet(existing_file)
                
                # Download new data
                all_data = []
                current_ts = start_ts
                
                while current_ts < end_ts:
                    df = download_data(exchange, symbol, tf, current_ts)
                    if df is None or len(df) == 0:
                        break
                    
                    all_data.append(df)
                    current_ts = int(df.index[-1].timestamp() * 1000) + 1
                    
                    time.sleep(0.1)  # Rate limit
                
                if not all_data:
                    print("NO NEW DATA")
                    continue
                
                # Combine with existing
                new_df = pd.concat(all_data)
                combined_df = pd.concat([existing_df, new_df])
                
                # Remove duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.sort_index(inplace=True)
                
                # Save
                combined_df.to_parquet(existing_file)
                
                new_bars = len(combined_df) - len(existing_df)
                print(f"✅ +{new_bars} bars (now {len(combined_df):,} total)")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                continue
    
    print("\n" + "="*80)
    print("✅ DOWNLOAD COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()