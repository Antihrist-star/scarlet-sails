import pandas as pd
import os

print("="*60)
print("VALIDATING DOWNLOADED DATA")
print("="*60)

total_bars = 0
summary = []

for file in sorted(os.listdir('data/raw')):
    if file.endswith('_FULL.parquet'):
        df = pd.read_parquet(f'data/raw/{file}')
        
        # Check for gaps
        if '15m' in file:
            expected_freq = '15min'
        elif '1h' in file:
            expected_freq = '1h'
        elif '4h' in file:
            expected_freq = '4h'
        else:
            expected_freq = '1d'
            
        gaps = df.index.to_series().diff()
        max_gap = gaps.max()
        
        years = (df.index[-1]-df.index[0]).days/365
        
        print(f"\n{file}:")
        print(f"  📊 Bars: {len(df):,}")
        print(f"  📅 Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  ⏰ Years: {years:.1f}")
        print(f"  🕳️ Max gap: {max_gap}")
        
        if max_gap > pd.Timedelta(expected_freq) * 10:
            print(f"  ⚠️ WARNING: Large gaps detected!")
        
        total_bars += len(df)
        summary.append({
            'file': file,
            'symbol': file.split('_')[0] + '_' + file.split('_')[1],
            'timeframe': file.split('_')[2],
            'bars': len(df),
            'years': years
        })

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\n📊 TOTAL BARS DOWNLOADED: {total_bars:,}")

# Group by symbol
for symbol in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']:
    symbol_data = [s for s in summary if s['symbol'] == symbol]
    print(f"\n{symbol}:")
    for s in symbol_data:
        print(f"  {s['timeframe']:4s}: {s['bars']:7,} bars ({s['years']:.1f} years)")

print("\n✅ VALIDATION COMPLETE!")
print("\nCOMPARISON WITH OLD DATA:")
print("Old: 70k bars (2 years) for 15m only")
print(f"New: {total_bars:,} bars (7+ years) for all timeframes")
print(f"IMPROVE