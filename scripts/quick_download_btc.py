"""
Quick download BTC 1h data for testing (last 3 months)
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

print("🚀 QUICK DOWNLOAD: BTC 1h (last 3 months)")
print("=" * 60)

# Create directory
output_dir = Path('data/raw')
output_dir.mkdir(exist_ok=True, parents=True)

# Initialize exchange
exchange = ccxt.binance()

# Calculate time range (last 3 months)
end_time = datetime.now()
start_time = end_time - timedelta(days=90)

print(f"📅 Period: {start_time.date()} to {end_time.date()}")
print(f"📊 Downloading BTC/USDT 1h...")

# Download
since = int(start_time.timestamp() * 1000)
all_ohlcv = []

while since < int(end_time.timestamp() * 1000):
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
        if not ohlcv:
            break

        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1

        print(f"  Downloaded {len(all_ohlcv)} bars...", end='\r')

    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        break

print(f"\n✅ Downloaded {len(all_ohlcv)} bars")

# Convert to DataFrame
df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Add basic indicators
print("📈 Computing indicators...")
df['rsi'] = 50.0  # Placeholder - will compute properly in backtest
df['atr'] = (df['high'] - df['low']).rolling(14).mean()
df['ma20'] = df['close'].rolling(20).mean()
df['ma200'] = df['close'].rolling(200).mean()

# Save
output_file = output_dir / 'BTCUSDT_1h.parquet'
df.to_parquet(output_file)

print(f"💾 Saved to: {output_file}")
print(f"📊 Shape: {df.shape}")
print(f"📅 Range: {df.index[0]} to {df.index[-1]}")
print(f"💰 Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
print()
print("✅ READY TO BACKTEST!")
