"""
REGIME GATE - Первый фильтр системы

Based on forensics:
- Winners: 100% в BULL_TREND
- Losers: 90% в SIDEWAYS
- Edge exists только в trending markets

Strategy: Trade ONLY когда edge confirmed
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Tuple

class MarketRegime(Enum):
    """Market regime classification"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECOVERING_BEAR = "recovering_bear"

class RegimeGate:
    """
    Gate keeper - позволяет trading только в favorable regimes
    """
    
    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 200,
        momentum_period: int = 720,  # 30 days in hours
        volatility_threshold: float = 2.0
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.momentum_period = momentum_period
        self.volatility_threshold = volatility_threshold
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime
        
        Returns:
            MarketRegime enum
        """
        if len(df) < self.ma_long:
            return MarketRegime.SIDEWAYS  # Not enough data
        
        # Calculate indicators
        ma20 = df['close'].rolling(self.ma_short).mean().iloc[-1]
        ma200 = df['close'].rolling(self.ma_long).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        
        # Momentum (30-day return)
        if len(df) >= self.momentum_period:
            price_30d_ago = df['close'].iloc[-self.momentum_period]
            momentum_30d = (current_price - price_30d_ago) / price_30d_ago
        else:
            momentum_30d = 0
        
        # Volatility
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        atr_mean = df['atr'].rolling(100).mean().iloc[-1] if 'atr' in df.columns else 1
        volatility_ratio = atr / atr_mean if atr_mean > 0 else 1
        
        # BULL TREND criteria (from forensics)
        is_bull = (
            ma20 > ma200 * 1.05  # 5% spread between MAs
            and momentum_30d > 0.15  # 15% gain in 30 days
            and volatility_ratio < self.volatility_threshold  # Not too volatile
        )
        
        if is_bull:
            return MarketRegime.BULL_TREND
        
        # RECOVERING BEAR (from Test-3: +17% в bear!)
        # Deep oversold in bear market with capitulation volume
        is_recovering_bear = (
            ma20 < ma200  # Bear structure
            and df['rsi'].iloc[-1] < 25  # Deep oversold
            and df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5  # Volume spike
            and momentum_30d > -0.20  # Not in free fall
        )
        
        if is_recovering_bear:
            return MarketRegime.RECOVERING_BEAR
        
        # VOLATILE (avoid!)
        is_volatile = volatility_ratio > self.volatility_threshold * 1.5
        if is_volatile:
            return MarketRegime.VOLATILE
        
        # BEAR TREND
        is_bear = ma20 < ma200 * 0.95 and momentum_30d < -0.10
        if is_bear:
            return MarketRegime.BEAR_TREND
        
        # Default: SIDEWAYS (avoid!)
        return MarketRegime.SIDEWAYS
    
    def should_trade(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Decide if we should trade in current regime
        
        Returns:
            (should_trade: bool, confidence: float)
        """
        regime = self.detect_regime(df)
        
        # BULL TREND: Full confidence
        if regime == MarketRegime.BULL_TREND:
            return True, 1.0
        
        # RECOVERING BEAR: Half confidence
        elif regime == MarketRegime.RECOVERING_BEAR:
            return True, 0.5
        
        # All others: DON'T TRADE
        else:
            return False, 0.0
    
    def get_regime_stats(self, df: pd.DataFrame) -> dict:
        """
        Get detailed regime statistics
        """
        regimes = []
        for i in range(self.ma_long, len(df)):
            regime = self.detect_regime(df.iloc[:i+1])
            regimes.append(regime)
        
        regime_counts = pd.Series(regimes).value_counts()
        total = len(regimes)
        
        stats = {
            'total_bars': total,
            'regimes': {}
        }
        
        for regime, count in regime_counts.items():
            stats['regimes'][regime.value] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        return stats