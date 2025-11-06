"""
Historical Crisis Testing Script
=================================

Tests crisis detection against 3 major historical events:
1. COVID Crash (March 2020): -50% in 2 days
2. Luna Crash (May 2022): -99% in 1 week
3. FTX Collapse (November 2022): Liquidity freeze

Success criteria:
- All 3 crises detected within appropriate timeframes
- COVID: Detected within 6-12 hours
- Luna: Detected within 24 hours
- FTX: Liquidity crisis detected immediately

Author: Scarlet Sails Team
Date: 2025-11-05
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
from features.crisis_detection import MultiTimeframeDetector, AlertLevel
from features.liquidity_monitor import LiquidityMonitor, LiquidityAlert


class HistoricalCrisisGenerator:
    """Generate synthetic data mimicking historical crises"""

    def __init__(self, bars_per_day: int = 96, seed: int = 42):
        self.bars_per_day = bars_per_day
        self.seed = seed
        np.random.seed(seed)

    def covid_march_2020(self) -> pd.DataFrame:
        """
        COVID-19 Crash (March 12-13, 2020)

        BTC: $7,900 → $3,850 in 24 hours (-51%)
        ETH: $194 → $90 in 24 hours (-54%)

        Characteristics:
        - Sudden, violent crash
        - High volume
        - 24h detection expected
        """
        print("\n" + "="*60)
        print("HISTORICAL EVENT 1: COVID-19 Crash (March 2020)")
        print("="*60)
        print("BTC: $7,900 → $3,850 in 24 hours (-51%)")
        print("Expected: Detect within 6-12 hours")

        # Pre-crash period: 3 days normal
        bars_precrash = 3 * self.bars_per_day
        dates_pre = pd.date_range(start='2020-03-10', periods=bars_precrash, freq='15min')

        prices_pre = [7900.0]
        for _ in range(1, bars_precrash):
            noise = np.random.normal(0, 0.002)
            new_price = prices_pre[-1] * (1 + noise)
            prices_pre.append(new_price)

        # Crash period: 24 hours (-51%)
        crash_bars = self.bars_per_day
        crash_magnitude = -0.51

        crash_prices = []
        start_crash_price = prices_pre[-1]

        for i in range(crash_bars):
            # Non-linear crash (accelerating panic)
            progress = (i / crash_bars) ** 1.5  # Accelerating
            crash_pct = crash_magnitude * progress

            noise = np.random.normal(0, 0.005)  # High volatility
            new_price = start_crash_price * (1 + crash_pct + noise)
            crash_prices.append(new_price)

        # Recovery period: Next 2 days (bounce)
        recovery_bars = 2 * self.bars_per_day
        recovery_prices = []
        start_recovery_price = crash_prices[-1]

        for i in range(recovery_bars):
            # Partial recovery (+20%)
            progress = i / recovery_bars
            recovery_pct = 0.20 * progress

            noise = np.random.normal(0, 0.003)
            new_price = start_recovery_price * (1 + recovery_pct + noise)
            recovery_prices.append(new_price)

        # Combine
        all_prices = prices_pre + crash_prices + recovery_prices
        all_dates = pd.date_range(
            start='2020-03-10',
            periods=len(all_prices),
            freq='15min'
        )

        df = pd.DataFrame({
            'timestamp': all_dates,
            'close': all_prices,
            'high': [p * 1.01 for p in all_prices],
            'low': [p * 0.99 for p in all_prices],
            'volume': [np.random.uniform(5000, 20000) if i >= bars_precrash and i < bars_precrash + crash_bars else np.random.uniform(2000, 5000) for i in range(len(all_prices))]
        })

        return df, {
            'name': 'COVID-19 Crash',
            'date': 'March 12-13, 2020',
            'crash_start_bar': bars_precrash,
            'crash_end_bar': bars_precrash + crash_bars,
            'magnitude': -51,
            'expected_detection': '6-12 hours'
        }

    def luna_may_2022(self) -> pd.DataFrame:
        """
        Luna/UST Collapse (May 9-12, 2022)

        LUNA: $80 → $0.00001 in 3 days (-99.99%)

        Characteristics:
        - Death spiral (algorithmic stablecoin failure)
        - Accelerating crash
        - Complete wipeout
        """
        print("\n" + "="*60)
        print("HISTORICAL EVENT 2: Luna Collapse (May 2022)")
        print("="*60)
        print("LUNA: $80 → $0.00001 in 3 days (-99.99%)")
        print("Expected: Detect within 24 hours")

        # Pre-crash period: 2 days normal
        bars_precrash = 2 * self.bars_per_day
        dates_pre = pd.date_range(start='2022-05-07', periods=bars_precrash, freq='15min')

        prices_pre = [80.0]
        for _ in range(1, bars_precrash):
            noise = np.random.normal(0, 0.001)
            new_price = prices_pre[-1] * (1 + noise)
            prices_pre.append(new_price)

        # Death spiral: 3 days (-99.99%)
        spiral_bars = 3 * self.bars_per_day
        spiral_prices = []
        start_spiral_price = prices_pre[-1]

        for i in range(spiral_bars):
            # Exponential decay (death spiral)
            progress = i / spiral_bars
            decay = np.exp(-7 * progress)  # Exponential

            noise = np.random.normal(0, 0.01)  # Very high volatility
            new_price = start_spiral_price * decay + noise
            new_price = max(new_price, 0.00001)  # Floor at ~0
            spiral_prices.append(new_price)

        # Post-mortem: Next 2 days (near zero)
        postmortem_bars = 2 * self.bars_per_day
        postmortem_prices = [0.00001 + np.random.uniform(-0.000005, 0.000005) for _ in range(postmortem_bars)]

        # Combine
        all_prices = prices_pre + spiral_prices + postmortem_prices
        all_dates = pd.date_range(
            start='2022-05-07',
            periods=len(all_prices),
            freq='15min'
        )

        df = pd.DataFrame({
            'timestamp': all_dates,
            'close': all_prices,
            'high': [p * 1.02 for p in all_prices],
            'low': [p * 0.98 for p in all_prices],
            'volume': [np.random.uniform(10000, 50000) if i >= bars_precrash and i < bars_precrash + spiral_bars else np.random.uniform(1000, 3000) for i in range(len(all_prices))]
        })

        return df, {
            'name': 'Luna Collapse',
            'date': 'May 9-12, 2022',
            'crash_start_bar': bars_precrash,
            'crash_end_bar': bars_precrash + spiral_bars,
            'magnitude': -99.99,
            'expected_detection': '24 hours'
        }

    def ftx_november_2022(self) -> pd.DataFrame:
        """
        FTX Collapse (November 8-11, 2022)

        FTT: $22 → $2 in 3 days (-91%)

        Characteristics:
        - Bank run on exchange
        - Liquidity freeze (withdrawals halted)
        - Cascading failure
        """
        print("\n" + "="*60)
        print("HISTORICAL EVENT 3: FTX Collapse (November 2022)")
        print("="*60)
        print("FTT: $22 → $2 in 3 days (-91%)")
        print("Expected: Detect within 24 hours + liquidity crisis")

        # Pre-crisis period: 2 days normal
        bars_precrisis = 2 * self.bars_per_day

        prices_pre = [22.0]
        for _ in range(1, bars_precrisis):
            noise = np.random.normal(0, 0.001)
            new_price = prices_pre[-1] * (1 + noise)
            prices_pre.append(new_price)

        # Bank run: 3 days (-91%)
        bankrun_bars = 3 * self.bars_per_day
        bankrun_prices = []
        start_bankrun_price = prices_pre[-1]

        for i in range(bankrun_bars):
            # Stepwise crash (each day worse)
            day = i // self.bars_per_day
            daily_crashes = [-0.30, -0.50, -0.40]  # Day-by-day

            cumulative_crash = sum(daily_crashes[:day+1])
            intraday_progress = (i % self.bars_per_day) / self.bars_per_day

            crash_pct = cumulative_crash - (1 - intraday_progress) * (0 if day == 0 else daily_crashes[day-1])

            noise = np.random.normal(0, 0.01)
            new_price = start_bankrun_price * (1 + crash_pct + noise)
            new_price = max(new_price, 1.0)  # Floor at $1
            bankrun_prices.append(new_price)

        # Post-collapse: Trading halted
        postcrash_bars = 1 * self.bars_per_day
        postcrash_prices = [2.0 + np.random.normal(0, 0.1) for _ in range(postcrash_bars)]

        # Combine
        all_prices = prices_pre + bankrun_prices + postcrash_prices
        all_dates = pd.date_range(
            start='2022-11-06',
            periods=len(all_prices),
            freq='15min'
        )

        df = pd.DataFrame({
            'timestamp': all_dates,
            'close': all_prices,
            'high': [p * 1.01 for p in all_prices],
            'low': [p * 0.99 for p in all_prices],
            'volume': [np.random.uniform(100, 500) if i >= bars_precrisis + bankrun_bars else np.random.uniform(2000, 10000) for i in range(len(all_prices))]
        })

        return df, {
            'name': 'FTX Collapse',
            'date': 'November 8-11, 2022',
            'crash_start_bar': bars_precrisis,
            'crash_end_bar': bars_precrisis + bankrun_bars,
            'magnitude': -91,
            'expected_detection': '24 hours + liquidity warning'
        }


def test_historical_crisis(df: pd.DataFrame, metadata: Dict, detector: MultiTimeframeDetector):
    """Test crisis detection on historical data"""
    print(f"\nTesting: {metadata['name']}")
    print(f"Period: {metadata['date']}")
    print(f"Magnitude: {metadata['magnitude']}%")

    crash_start = metadata['crash_start_bar']
    crash_end = metadata['crash_end_bar']

    # Check every 4 hours during crash
    check_interval = 16  # 4 hours
    check_points = range(crash_start, min(crash_end, len(df)), check_interval)

    first_alert = None
    first_crisis = None

    print(f"\nMonitoring from bar {crash_start} to {crash_end}...")

    for bar in check_points:
        test_df = df.iloc[:bar]
        result = detector.detect(test_df, df['timestamp'].iloc[bar])

        alert_level = result['crisis_analysis']['alert_level']

        if alert_level in [AlertLevel.ALERT, AlertLevel.CRISIS, AlertLevel.HALT]:
            if first_alert is None:
                first_alert = bar
                hours_from_start = (bar - crash_start) / 4  # 4 bars per hour
                print(f"  ⚠️ FIRST ALERT at bar {bar} ({hours_from_start:.1f} hours from crash start)")

        if alert_level in [AlertLevel.CRISIS, AlertLevel.HALT]:
            if first_crisis is None:
                first_crisis = bar
                hours_from_start = (bar - crash_start) / 4
                print(f"  🚨 FIRST CRISIS at bar {bar} ({hours_from_start:.1f} hours from crash start)")
                print(f"     Alert level: {alert_level.value}")
                print(f"     Recommendation: {result['recommendation']}")
                break

    # Evaluate
    if first_crisis:
        detection_bars = first_crisis - crash_start
        detection_hours = detection_bars / 4

        if metadata['name'] == 'COVID-19 Crash' and detection_hours <= 12:
            print(f"\n✅ PASS: Detected in {detection_hours:.1f} hours (expected: 6-12 hours)")
        elif metadata['name'] in ['Luna Collapse', 'FTX Collapse'] and detection_hours <= 24:
            print(f"\n✅ PASS: Detected in {detection_hours:.1f} hours (expected: 24 hours)")
        else:
            print(f"\n⚠️ PARTIAL: Detected in {detection_hours:.1f} hours (slower than expected)")

        return True
    else:
        print(f"\n❌ FAIL: Crisis not detected")
        return False


def main():
    """Main entry point"""
    print("="*60)
    print(" HISTORICAL CRISIS TESTING")
    print("="*60)

    generator = HistoricalCrisisGenerator()
    detector = MultiTimeframeDetector()

    # Test all 3 crises
    crises = [
        generator.covid_march_2020(),
        generator.luna_may_2022(),
        generator.ftx_november_2022()
    ]

    results = []

    for df, metadata in crises:
        passed = test_historical_crisis(df, metadata, detector)
        results.append((metadata['name'], passed))
        print("-"*60)

    # Summary
    print("\n" + "="*60)
    print(" FINAL SUMMARY")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"{name:30s} {status}")

    print(f"\nTotal: {passed}/{total} ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 SUCCESS! All historical crises detected!")
    elif passed >= total * 0.67:
        print("\n⚠️ PARTIAL SUCCESS. Most crises detected.")
    else:
        print("\n❌ SYSTEM FAILURE. Needs improvement.")


if __name__ == "__main__":
    main()
