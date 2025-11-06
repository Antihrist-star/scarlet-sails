"""
P0-3: 4-Factor Liquidity Monitoring System
===========================================

Detects liquidity crises like FTX collapse (November 2022).

Problem: Price-only monitoring misses liquidity freezes
- FTX Nov 2022: Exchange frozen, no liquidity
- Symptoms: Wide spreads, empty orderbook, no trades
- Result: Can't exit positions (catastrophic!)

Solution: 4-Factor Liquidity Detection
1. Bid-Ask Spread: >1% = illiquid (normally 0.1%)
2. Order Book Depth: <30% of median = thin
3. Volume: <20% of median = low activity
4. Trades Per Minute: <1 trade/min = frozen

Aggregation Rule:
- 3/4 factors triggered = LIQUIDITY CRISIS
- Action: HALT all trading on that asset

Philosophy: Better miss profits than get stuck in illiquid market

Author: Scarlet Sails Team
Date: 2025-11-05
Priority: P0 (CRITICAL)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum


class LiquidityAlert(Enum):
    """Liquidity alert levels"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"  # 1-2 factors triggered
    CRISIS = "CRISIS"     # 3+ factors triggered - HALT
    FROZEN = "FROZEN"     # All 4 factors triggered - EMERGENCY


class LiquidityMonitor:
    """
    4-Factor liquidity monitoring for exchange health checks.

    Designed to catch FTX-style liquidity crises.
    """

    def __init__(
        self,
        lookback_window: int = 672,  # 7 days for 15min data
        min_periods: int = 96  # Minimum 1 day of data
    ):
        """
        Initialize liquidity monitor.

        Args:
            lookback_window: Window for calculating medians (bars)
            min_periods: Minimum periods required
        """
        self.lookback_window = lookback_window
        self.min_periods = min_periods

        # Thresholds for each factor
        self.thresholds = {
            'spread': 0.01,        # 1% spread = WARNING
            'depth_ratio': 0.30,   # <30% of median depth = WARNING
            'volume_ratio': 0.20,  # <20% of median volume = WARNING
            'trades_per_min': 1.0  # <1 trade/minute = WARNING
        }

    def calculate_bid_ask_spread(
        self,
        bid: float,
        ask: float
    ) -> float:
        """
        Calculate bid-ask spread percentage.

        Args:
            bid: Best bid price
            ask: Best ask price

        Returns:
            Spread as percentage (e.g., 0.01 = 1%)
        """
        if bid <= 0 or ask <= 0:
            return float('inf')

        mid_price = (bid + ask) / 2
        spread = (ask - bid) / mid_price

        return spread

    def calculate_orderbook_depth(
        self,
        orderbook: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[str, float]:
        """
        Calculate orderbook depth metrics.

        Args:
            orderbook: Dict with 'bids' and 'asks', each a list of (price, size) tuples

        Returns:
            Dict with depth metrics:
                - bid_depth: Total bid liquidity (USD)
                - ask_depth: Total ask liquidity (USD)
                - total_depth: Combined depth
                - imbalance: Bid/ask imbalance ratio
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        # Calculate depth (total USD value)
        bid_depth = sum(price * size for price, size in bids)
        ask_depth = sum(price * size for price, size in asks)
        total_depth = bid_depth + ask_depth

        # Calculate imbalance
        if ask_depth > 0:
            imbalance = bid_depth / ask_depth
        else:
            imbalance = float('inf') if bid_depth > 0 else 0

        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total_depth,
            'imbalance': imbalance
        }

    def analyze_liquidity(
        self,
        current_data: Dict[str, any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, any]:
        """
        Analyze current liquidity state.

        Args:
            current_data: Dict with current market data:
                - bid: Best bid price
                - ask: Best ask price
                - orderbook: Dict with 'bids' and 'asks' lists
                - volume: Current bar volume
                - trades_count: Number of trades in current bar (15min)
                - timestamp: Current timestamp

            historical_data: DataFrame with historical data for baseline:
                - volume: Historical volumes
                - orderbook_depth: Historical depths
                - trades_count: Historical trade counts

        Returns:
            Dict with liquidity analysis:
                - alert_level: NORMAL/WARNING/CRISIS/FROZEN
                - factors: Dict of {factor: (value, threshold, triggered)}
                - summary: Human-readable summary
        """
        factors = {}
        triggered_factors = []

        # Factor 1: Bid-Ask Spread
        bid = current_data.get('bid', 0)
        ask = current_data.get('ask', 0)

        if bid > 0 and ask > 0:
            spread = self.calculate_bid_ask_spread(bid, ask)
            spread_triggered = spread >= self.thresholds['spread']

            factors['spread'] = {
                'value': spread,
                'threshold': self.thresholds['spread'],
                'triggered': spread_triggered,
                'description': f"{spread:.2%} spread {'⚠️ (too wide!)' if spread_triggered else ''}"
            }

            if spread_triggered:
                triggered_factors.append('spread')
        else:
            factors['spread'] = {
                'value': None,
                'threshold': self.thresholds['spread'],
                'triggered': True,
                'description': '⚠️ No bid/ask data (market frozen?)'
            }
            triggered_factors.append('spread')

        # Factor 2: Order Book Depth
        orderbook = current_data.get('orderbook', {})
        if orderbook:
            depth_metrics = self.calculate_orderbook_depth(orderbook)
            current_depth = depth_metrics['total_depth']

            # Compare to historical median
            if historical_data is not None and 'orderbook_depth' in historical_data.columns:
                median_depth = historical_data['orderbook_depth'].median()
                if median_depth > 0:
                    depth_ratio = current_depth / median_depth
                else:
                    depth_ratio = 0

                depth_triggered = depth_ratio < self.thresholds['depth_ratio']

                factors['depth'] = {
                    'value': depth_ratio,
                    'threshold': self.thresholds['depth_ratio'],
                    'triggered': depth_triggered,
                    'description': f"{depth_ratio:.1%} of median depth {'⚠️ (too thin!)' if depth_triggered else ''}"
                }

                if depth_triggered:
                    triggered_factors.append('depth')
            else:
                # No historical data - use absolute depth
                factors['depth'] = {
                    'value': current_depth,
                    'threshold': None,
                    'triggered': current_depth < 10000,  # $10k minimum
                    'description': f"${current_depth:,.0f} depth (no baseline)"
                }
        else:
            factors['depth'] = {
                'value': None,
                'threshold': self.thresholds['depth_ratio'],
                'triggered': True,
                'description': '⚠️ No orderbook data'
            }
            triggered_factors.append('depth')

        # Factor 3: Volume
        current_volume = current_data.get('volume', 0)

        if historical_data is not None and 'volume' in historical_data.columns:
            median_volume = historical_data['volume'].median()

            if median_volume > 0:
                volume_ratio = current_volume / median_volume
            else:
                volume_ratio = 0

            volume_triggered = volume_ratio < self.thresholds['volume_ratio']

            factors['volume'] = {
                'value': volume_ratio,
                'threshold': self.thresholds['volume_ratio'],
                'triggered': volume_triggered,
                'description': f"{volume_ratio:.1%} of median volume {'⚠️ (too low!)' if volume_triggered else ''}"
            }

            if volume_triggered:
                triggered_factors.append('volume')
        else:
            # No historical data
            factors['volume'] = {
                'value': current_volume,
                'threshold': None,
                'triggered': False,
                'description': f"{current_volume:,.0f} volume (no baseline)"
            }

        # Factor 4: Trades Per Minute
        trades_count = current_data.get('trades_count', 0)
        bar_minutes = 15  # Assuming 15min bars

        trades_per_min = trades_count / bar_minutes

        trades_triggered = trades_per_min < self.thresholds['trades_per_min']

        factors['trades_per_min'] = {
            'value': trades_per_min,
            'threshold': self.thresholds['trades_per_min'],
            'triggered': trades_triggered,
            'description': f"{trades_per_min:.1f} trades/min {'⚠️ (too few!)' if trades_triggered else ''}"
        }

        if trades_triggered:
            triggered_factors.append('trades_per_min')

        # Determine alert level
        n_triggered = len(triggered_factors)

        if n_triggered >= 4:
            alert_level = LiquidityAlert.FROZEN
            summary = "🛑 EMERGENCY: All 4 liquidity factors triggered - MARKET FROZEN"
        elif n_triggered >= 3:
            alert_level = LiquidityAlert.CRISIS
            summary = f"🚨 LIQUIDITY CRISIS: {n_triggered}/4 factors triggered - HALT TRADING"
        elif n_triggered >= 1:
            alert_level = LiquidityAlert.WARNING
            summary = f"⚠️ LIQUIDITY WARNING: {n_triggered}/4 factors triggered"
        else:
            alert_level = LiquidityAlert.NORMAL
            summary = "✅ Liquidity normal"

        return {
            'alert_level': alert_level,
            'factors': factors,
            'triggered_factors': triggered_factors,
            'n_triggered': n_triggered,
            'summary': summary,
            'timestamp': current_data.get('timestamp', datetime.now()),
            'recommendation': self._get_recommendation(alert_level)
        }

    def _get_recommendation(self, alert_level: LiquidityAlert) -> str:
        """Get trading recommendation based on alert level"""
        if alert_level == LiquidityAlert.FROZEN:
            return "HALT - Market frozen, cannot exit positions safely"
        elif alert_level == LiquidityAlert.CRISIS:
            return "HALT - Insufficient liquidity to trade safely"
        elif alert_level == LiquidityAlert.WARNING:
            return "CAUTION - Reduce position sizes, monitor closely"
        else:
            return "CONTINUE - Normal trading conditions"


def demo():
    """Demo showing liquidity monitoring"""
    print("="*60)
    print("4-Factor Liquidity Monitor Demo")
    print("="*60)

    monitor = LiquidityMonitor()

    # Simulate normal market conditions
    print("\n--- Scenario 1: Normal Market ---")
    normal_data = {
        'bid': 100.0,
        'ask': 100.1,  # 0.1% spread
        'orderbook': {
            'bids': [(100.0, 10), (99.9, 20), (99.8, 30)],
            'asks': [(100.1, 10), (100.2, 20), (100.3, 30)]
        },
        'volume': 5000,
        'trades_count': 30,  # 2 trades/minute
        'timestamp': datetime.now()
    }

    historical_normal = pd.DataFrame({
        'volume': [5000] * 100,
        'orderbook_depth': [12000] * 100,
        'trades_count': [30] * 100
    })

    result = monitor.analyze_liquidity(normal_data, historical_normal)

    print(f"Alert Level: {result['alert_level'].value}")
    print(f"Summary: {result['summary']}")
    print(f"Recommendation: {result['recommendation']}")
    print("\nFactors:")
    for name, factor in result['factors'].items():
        print(f"  {name}: {factor['description']}")

    # Simulate FTX-style liquidity crisis
    print("\n--- Scenario 2: FTX-Style Liquidity Crisis ---")
    crisis_data = {
        'bid': 100.0,
        'ask': 102.0,  # 2% spread (20x normal!)
        'orderbook': {
            'bids': [(100.0, 1), (99.5, 2)],  # Thin orderbook
            'asks': [(102.0, 1), (102.5, 2)]
        },
        'volume': 500,  # 10% of normal
        'trades_count': 3,  # 0.2 trades/minute
        'timestamp': datetime.now()
    }

    result_crisis = monitor.analyze_liquidity(crisis_data, historical_normal)

    print(f"Alert Level: {result_crisis['alert_level'].value}")
    print(f"Summary: {result_crisis['summary']}")
    print(f"Recommendation: {result_crisis['recommendation']}")
    print(f"Triggered: {result_crisis['n_triggered']}/4 factors")
    print("\nFactors:")
    for name, factor in result_crisis['factors'].items():
        status = "❌ TRIGGERED" if factor['triggered'] else "✅ OK"
        print(f"  {name}: {factor['description']} {status}")

    # Simulate complete market freeze
    print("\n--- Scenario 3: Complete Market Freeze ---")
    frozen_data = {
        'bid': 0,
        'ask': 0,
        'orderbook': {},
        'volume': 0,
        'trades_count': 0,
        'timestamp': datetime.now()
    }

    result_frozen = monitor.analyze_liquidity(frozen_data, historical_normal)

    print(f"Alert Level: {result_frozen['alert_level'].value}")
    print(f"Summary: {result_frozen['summary']}")
    print(f"Recommendation: {result_frozen['recommendation']}")
    print(f"Triggered: {result_frozen['n_triggered']}/4 factors")

    print("\n" + "="*60)
    print("Key Takeaway:")
    print("4-factor monitoring catches liquidity crises that")
    print("price-only systems miss. FTX-style freezes are now detected!")
    print("="*60)


if __name__ == "__main__":
    demo()
    print("\n✅ Liquidity monitoring module created successfully!")
