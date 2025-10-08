"""
PLAN FOR MULTI-TIMEFRAME APPROACH

Goal: Use all timeframes (15m, 1h, 4h, 1d) simultaneously

Architecture:
1. Load all timeframes
2. Align to 15m resolution (forward fill)
3. Create hierarchical features:
   - 15m: micro movements
   - 1h: short trends  
   - 4h: medium trends
   - 1d: major trend

Features per timeframe:
- Price action: OHLCV
- Technical: RSI, EMA, BB
- Market regime: trend strength
- Cross-timeframe: alignment signals

Target Options:
A. Next 4h direction (simpler)
B. Profitable trade with stop loss (current)
C. Multi-horizon: 1h AND 4h predictions

Expected improvement:
- More context = better predictions
- Catch major trends from daily
- Filter noise using higher TFs
- Accuracy target: 65-75%
"""

# This will be implemented tomorrow in prepare_data_v4.py