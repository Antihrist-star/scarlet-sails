import pandas as pd
import numpy as np

def create_triple_barrier_labels(df, upper=0.02, lower=-0.01, horizon=96):
    """
    Triple-barrier method для крипто
    
    upper: 2% profit target
    lower: -1% stop loss
    horizon: 96 bars (24 hours для 15m)
    
    Returns:
    1 = UP (profit достигнут)
    -1 = DOWN (stop достигнут)  
    0 = NEUTRAL (ни один барьер не достигнут)
    """
    labels = []
    
    for i in range(len(df) - horizon):
        entry_price = df['close'].iloc[i]
        
        # Future prices
        future_prices = df['close'].iloc[i:i+horizon]
        future_returns = (future_prices - entry_price) / entry_price
        
        # Проверка барьеров
        upper_hit = future_returns[future_returns >= upper]
        lower_hit = future_returns[future_returns <= lower]
        
        if len(upper_hit) > 0 and len(lower_hit) > 0:
            # Какой первый?
            if upper_hit.index[0] < lower_hit.index[0]:
                label = 1  # UP first
            else:
                label = -1  # DOWN first
        elif len(upper_hit) > 0:
            label = 1
        elif len(lower_hit) > 0:
            label = -1
        else:
            label = 0  # NEUTRAL
            
        labels.append(label)
    
    # Pad with None for last rows
    labels += [None] * horizon
    
    return labels

if __name__ == "__main__":
    # Test
    df = pd.read_parquet('data/raw/BTC_USDT_15m_FULL.parquet')
    labels = create_triple_barrier_labels(df)
    
    print(f"Total: {len(labels)}")
    print(f"UP (1): {labels.count(1)} ({labels.count(1)/len([l for l in labels if l is not None])*100:.1f}%)")
    print(f"DOWN (-1): {labels.count(-1)} ({labels.count(-1)/len([l for l in labels if l is not None])*100:.1f}%)")
    print(f"NEUTRAL (0): {labels.count(0)} ({labels.count(0)/len([l for l in labels if l is not None])*100:.1f}%)")