"""
ENTRY CONFLUENCE - Multi-factor entry system

Based on forensic analysis:
- Winners: RSI 23.7, Volume 0.99x, Bull trend
- Losers: RSI 24.2, Volume 1.36x, Sideways

Key insights:
1. Normal volume (0.8-1.2x) better than high volume
2. Deep oversold (RSI < 25) better
3. Support proximity important
4. Bullish divergence powerful
5. Wyckoff accumulation (spring) strong signal
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class EntryConfluence:
    """
    Multi-factor entry scoring system
    Combines technical + forensic patterns
    """
    
    def __init__(
        self,
        rsi_threshold: float = 30,
        rsi_deep_threshold: float = 25,
        volume_normal_min: float = 0.8,
        volume_normal_max: float = 1.2,
        volume_high_threshold: float = 1.5,
        support_proximity_pct: float = 0.02
    ):
        self.rsi_threshold = rsi_threshold
        self.rsi_deep_threshold = rsi_deep_threshold
        self.volume_normal_min = volume_normal_min
        self.volume_normal_max = volume_normal_max
        self.volume_high_threshold = volume_high_threshold
        self.support_proximity_pct = support_proximity_pct
    
    def score_entry(
        self,
        df: pd.DataFrame,
        context: Optional[Dict] = None
    ) -> float:
        """
        Calculate entry confluence score (0.0 - 1.0)
        
        Higher score = better entry opportunity
        
        Args:
            df: Market data with indicators
            context: Optional context (regime, phase, etc)
        
        Returns:
            Confluence score (0.0 - 1.0)
        """
        if len(df) < 200:
            return 0.0
        
        score = 0.0
        current = df.iloc[-1]
        
        # FACTOR 1: RSI OVERSOLD (baseline)
        # From forensics: winners avg 23.7, losers 24.2
        rsi = current['rsi']
        
        if rsi < self.rsi_threshold:
            score += 0.25  # Base oversold
            
            if rsi < self.rsi_deep_threshold:
                # Deep oversold (like winners)
                score += 0.10
        else:
            # Not oversold = no entry
            return 0.0
        
        # FACTOR 2: VOLUME CONFIRMATION
        # CRITICAL: High volume (1.36x) in losers was BAD sign!
        volume_ratio = current['volume'] / df['volume'].rolling(20).mean().iloc[-1]
        
        if self.volume_normal_min < volume_ratio < self.volume_normal_max:
            # Normal volume (like winners: 0.99x)
            score += 0.20
        elif volume_ratio > self.volume_high_threshold:
            # High volume = RED FLAG (losers: 1.36x)
            score -= 0.15
        
        # FACTOR 3: SUPPORT PROXIMITY
        # Find recent support (lowest low in last 50 bars)
        support = df['low'].rolling(50).min().iloc[-1]
        support_distance = (current['close'] - support) / support
        
        if support_distance < self.support_proximity_pct:
            # Within 2% of support
            score += 0.15
        
        # FACTOR 4: MOMENTUM DIVERGENCE (bullish)
        # Price higher but RSI low = bullish divergence
        if len(df) >= 120:  # 5 days
            price_5d_ago = df['close'].iloc[-120]
            
            if rsi < 30 and current['close'] > price_5d_ago:
                # Bullish divergence: price up, RSI still low
                score += 0.20
        
        # FACTOR 5: WYCKOFF ACCUMULATION (if context provided)
        if context and context.get('phase') == 'ACCUMULATION':
            # Spring: false breakdown below support that bounces
            recent_low = df['low'].iloc[-20:].min()
            
            if recent_low < support and current['close'] > support:
                # Spring detected: strong accumulation signal
                score += 0.25
        
        # FACTOR 6: MA STRUCTURE (trend confirmation)
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma200 = df['close'].rolling(200).mean().iloc[-1]
        
        if ma20 > ma200:
            # Uptrend structure
            score += 0.10
        
        # Cap score at 1.0
        return min(score, 1.0)
    
    def get_entry_breakdown(
        self,
        df: pd.DataFrame,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Get detailed breakdown of entry factors
        """
        if len(df) < 200:
            return {'score': 0.0, 'factors': {}}
        
        current = df.iloc[-1]
        factors = {}
        
        # RSI
        rsi = current['rsi']
        factors['rsi'] = {
            'value': float(rsi),
            'oversold': rsi < self.rsi_threshold,
            'deep_oversold': rsi < self.rsi_deep_threshold,
            'score_contribution': 0.25 if rsi < self.rsi_threshold else 0.0
        }
        if rsi < self.rsi_deep_threshold:
            factors['rsi']['score_contribution'] += 0.10
        
        # Volume
        volume_ratio = current['volume'] / df['volume'].rolling(20).mean().iloc[-1]
        is_normal = self.volume_normal_min < volume_ratio < self.volume_normal_max
        is_high = volume_ratio > self.volume_high_threshold
        
        factors['volume'] = {
            'ratio': float(volume_ratio),
            'normal': is_normal,
            'high': is_high,
            'score_contribution': 0.20 if is_normal else (-0.15 if is_high else 0.0)
        }
        
        # Support
        support = df['low'].rolling(50).min().iloc[-1]
        support_distance = (current['close'] - support) / support
        near_support = support_distance < self.support_proximity_pct
        
        factors['support'] = {
            'level': float(support),
            'distance_pct': float(support_distance * 100),
            'near_support': near_support,
            'score_contribution': 0.15 if near_support else 0.0
        }
        
        # Divergence
        if len(df) >= 120:
            price_5d_ago = df['close'].iloc[-120]
            has_divergence = rsi < 30 and current['close'] > price_5d_ago
            
            factors['divergence'] = {
                'detected': has_divergence,
                'price_5d_ago': float(price_5d_ago),
                'current_price': float(current['close']),
                'score_contribution': 0.20 if has_divergence else 0.0
            }
        
        # MA structure
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma200 = df['close'].rolling(200).mean().iloc[-1]
        uptrend = ma20 > ma200
        
        factors['ma_structure'] = {
            'ma20': float(ma20),
            'ma200': float(ma200),
            'uptrend': uptrend,
            'score_contribution': 0.10 if uptrend else 0.0
        }
        
        # Total score
        total_score = self.score_entry(df, context)
        
        return {
            'score': total_score,
            'factors': factors
        }