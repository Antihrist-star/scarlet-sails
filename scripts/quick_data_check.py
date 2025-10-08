import pandas as pd
import matplotlib.pyplot as plt

# Load BTC 1d для визуализации
df = pd.read_parquet('data/raw/BTC_USDT_1d_FULL.parquet')

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Price chart
axes[0].plot(df.index, df['close'])
axes[0].set_title('BTC/USDT Daily Price (Full History)')
axes[0].set_ylabel('Price USDT')
axes[0].grid(True)

# Volume
axes[1].bar(df.index, df['volume'], width=1)
axes[1].set_title('Daily Volume')
axes[1].set_ylabel('Volume')
axes[1].set_xlabel('Date')

plt.tight_layout()
plt.savefig('reports/btc_full_history.png')
plt.show()

# Key statistics
print("BTC HISTORICAL STATISTICS:")
print(f"Start Price: ${df['close'].iloc[0]:,.0f}")
print(f"End Price: ${df['close'].iloc[-1]:,.0f}")
print(f"Max Price: ${df['close'].max():,.0f}")
print(f"Min Price: ${df['close'].min():,.0f}")
print(f"Total Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.0f}%")

# Bear/Bull periods
returns = df['close'].pct_change(90)  # 3-month returns
bull_periods = (returns > 0.2).sum()
bear_periods = (returns < -0.2).sum()

print(f"\n3-MONTH PERIODS:")
print(f"Bull (>20% gain): {bull_periods}")
print(f"Bear (>20% loss): {bear_periods}")
print(f"Sideways: {len(returns) - bull_periods - bear_periods}")