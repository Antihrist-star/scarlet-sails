import ccxt
import pandas as pd
from datetime import datetime
import time
import os

def download_full_history():
    """
    Download maximum available history for multiple symbols and timeframes
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,  # Respect API limits
        'rateLimit': 1200,  # milliseconds between requests
    })
    
    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    timeframes = ['15m', '1h', '4h', '1d']
    
    # Start dates for each symbol
    start_dates = {
        'BTC/USDT': '2017-08-17T00:00:00Z',  # Binance launch
        'ETH/USDT': '2017-08-17T00:00:00Z',
        'SOL/USDT': '2020-08-11T00:00:00Z',  # SOL listing
    }
    
    os.makedirs('data/raw', exist_ok=True)
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*50}")
            print(f"Downloading {symbol} {timeframe}...")
            print(f"{'='*50}")
            
            since = exchange.parse8601(start_dates[symbol])
            all_ohlcv = []
            
            while True:
                try:
                    # Fetch data
                    ohlcv = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=since, 
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    
                    # Update progress
                    last_date = pd.to_datetime(ohlcv[-1][0], unit='ms')
                    print(f"  Downloaded up to {last_date}: {len(all_ohlcv)} bars")
                    
                    # Check if we reached present
                    if ohlcv[-1][0] >= exchange.milliseconds() - 86400000:
                        print("  Reached current time!")
                        break
                    
                    # Move to next batch
                    since = ohlcv[-1][0] + 1
                    
                    # Rate limit
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    time.sleep(5)
                    continue
            
            if all_ohlcv:
                # Convert to DataFrame
                df = pd.DataFrame(
                    all_ohlcv, 
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Save
                filename = f"data/raw/{symbol.replace('/', '_')}_{timeframe}_FULL.parquet"
                df.to_parquet(filename)
                
                # Statistics
                days_of_data = (df.index[-1] - df.index[0]).days
                print(f"\n✅ SAVED: {filename}")
                print(f"  Shape: {df.shape}")
                print(f"  Period: {df.index[0]} to {df.index[-1]}")
                print(f"  Total days: {days_of_data}")
                print(f"  Years of data: {days_of_data/365:.1f}")
            else:
                print(f"❌ No data downloaded for {symbol} {timeframe}")
    
    print("\n" + "="*50)
    print("DOWNLOAD COMPLETE!")
    print("="*50)
    
    # Summary
    print("\nSUMMARY OF DOWNLOADED DATA:")
    for file in os.listdir('data/raw'):
        if file.endswith('_FULL.parquet'):
            df = pd.read_parquet(f'data/raw/{file}')
            print(f"{file}: {len(df):,} bars, {(df.index[-1]-df.index[0]).days/365:.1f} years")

if __name__ == "__main__":
    download_full_history()