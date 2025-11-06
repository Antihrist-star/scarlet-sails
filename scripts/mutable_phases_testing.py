"""
Mutable Phases Testing Script
==============================

Tests the crisis detection system against 5 challenging scenarios:
1. Flat→Crash: Sudden crash from sideways market (COVID-style)
2. False Breakout: Fake rally followed by crash
3. Vol Sideways: High volatility but no directional trend
4. Gradual Crisis: Slow burn decline (-5% daily over 5 days) ⭐ KEY TEST
5. Fake Recovery: False rally during bear market (bull trap)

Success criteria:
- Scenario 1: Should detect within 24h (PASS if detected)
- Scenario 4: Should detect within 3-4 days (CRITICAL - currently MISSED)
- Overall: 4/5 detection = 80% success rate (target)

Author: Scarlet Sails Team
Date: 2025-11-05
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from features.crisis_detection import MultiTimeframeDetector, AlertLevel, Regime
from features.post_entry_validator import PostEntryValidator
from typing import Dict, List, Tuple


class ScenarioGenerator:
    """Generate test scenarios for crisis detection validation"""

    def __init__(self, bars_per_day: int = 96, seed: int = 42):
        """
        Initialize scenario generator.

        Args:
            bars_per_day: Number of bars per day (96 for 15min data)
            seed: Random seed for reproducibility
        """
        self.bars_per_day = bars_per_day
        self.seed = seed
        np.random.seed(seed)

    def generate_base_data(
        self,
        days: int,
        start_price: float = 100.0,
        volatility: float = 0.001
    ) -> pd.DataFrame:
        """Generate base price data with random walk"""
        bars = days * self.bars_per_day
        dates = pd.date_range(start='2025-01-01', periods=bars, freq='15min')

        prices = [start_price]
        for i in range(1, bars):
            noise = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + noise)
            prices.append(new_price)

        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'volume': np.random.uniform(1000, 5000, bars)
        })

    def scenario_1_flat_to_crash(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Scenario 1: Flat→Crash (COVID-style sudden crash)

        Pattern:
        - Days 1-5: Sideways (±1%)
        - Day 6: Sudden crash (-25% in 6 hours)

        Expected:
        - ✅ Should trigger CRISIS/HALT within 6-12 hours
        - Detection: COVID March 2020 style
        """
        print("\n" + "="*60)
        print("SCENARIO 1: Flat→Crash (Sudden Crisis)")
        print("="*60)

        # Generate flat market for 5 days
        df = self.generate_base_data(days=5, volatility=0.0002)

        # Add crash on day 6 (6-hour crash = 24 bars)
        crash_bars = 24
        crash_magnitude = -0.25  # -25% total

        for i in range(crash_bars):
            # Distribute crash over 6 hours
            crash_per_bar = crash_magnitude / crash_bars
            last_price = df['close'].iloc[-1]
            new_price = last_price * (1 + crash_per_bar)

            new_row = pd.DataFrame({
                'timestamp': [df['timestamp'].iloc[-1] + timedelta(minutes=15)],
                'close': [new_price],
                'high': [new_price * 1.001],
                'low': [new_price * 0.999],
                'volume': [np.random.uniform(10000, 20000)]  # High volume during crash
            })
            df = pd.concat([df, new_row], ignore_index=True)

        # Add recovery period (next 24 hours)
        recovery_df = self.generate_base_data(days=1, start_price=df['close'].iloc[-1], volatility=0.001)
        recovery_df['timestamp'] = pd.date_range(
            start=df['timestamp'].iloc[-1] + timedelta(minutes=15),
            periods=len(recovery_df),
            freq='15min'
        )
        df = pd.concat([df, recovery_df], ignore_index=True)

        metadata = {
            'name': 'Flat→Crash',
            'description': 'Sudden -25% crash after sideways market',
            'crash_start_bar': 5 * self.bars_per_day,
            'crash_end_bar': 5 * self.bars_per_day + crash_bars,
            'expected_detection': 'Within 6-12 hours of crash',
            'target_alert': 'CRISIS or HALT'
        }

        return df, metadata

    def scenario_2_false_breakout(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Scenario 2: False Breakout (Bull trap)

        Pattern:
        - Days 1-3: Uptrend (+10%)
        - Days 4-5: Consolidation
        - Days 6-7: Sharp reversal (-18%)

        Expected:
        - ⚠️ Should detect by Day 7
        - Challenge: Distinguish fake breakout from real reversal
        """
        print("\n" + "="*60)
        print("SCENARIO 2: False Breakout (Bull Trap)")
        print("="*60)

        df = self.generate_base_data(days=3, volatility=0.001)

        # Rally phase (Days 1-3): +10%
        for i in range(len(df)):
            df.loc[i, 'close'] = df.loc[i, 'close'] * (1 + 0.10 * i / len(df))

        # Consolidation (Days 4-5)
        consolidation = self.generate_base_data(
            days=2,
            start_price=df['close'].iloc[-1],
            volatility=0.0003
        )
        consolidation['timestamp'] = pd.date_range(
            start=df['timestamp'].iloc[-1] + timedelta(minutes=15),
            periods=len(consolidation),
            freq='15min'
        )
        df = pd.concat([df, consolidation], ignore_index=True)

        # Sharp reversal (Days 6-7): -18%
        reversal_bars = 2 * self.bars_per_day
        reversal_magnitude = -0.18

        for i in range(reversal_bars):
            reversal_per_bar = reversal_magnitude / reversal_bars
            last_price = df['close'].iloc[-1]
            new_price = last_price * (1 + reversal_per_bar)

            new_row = pd.DataFrame({
                'timestamp': [df['timestamp'].iloc[-1] + timedelta(minutes=15)],
                'close': [new_price],
                'high': [new_price * 1.001],
                'low': [new_price * 0.999],
                'volume': [np.random.uniform(5000, 10000)]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        metadata = {
            'name': 'False Breakout',
            'description': 'Bull trap: +10% rally followed by -18% crash',
            'rally_end_bar': 3 * self.bars_per_day,
            'reversal_start_bar': 5 * self.bars_per_day,
            'expected_detection': 'Within 24-48h of reversal',
            'target_alert': 'CRISIS'
        }

        return df, metadata

    def scenario_3_volatile_sideways(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Scenario 3: Volatile Sideways (Choppy market)

        Pattern:
        - 7 days of high volatility (±5% intraday)
        - No directional trend (ends near start)

        Expected:
        - ❌ Should NOT trigger false alerts
        - Challenge: High volatility but not a crisis
        """
        print("\n" + "="*60)
        print("SCENARIO 3: Volatile Sideways (Choppy Market)")
        print("="*60)

        bars = 7 * self.bars_per_day
        dates = pd.date_range(start='2025-01-01', periods=bars, freq='15min')

        start_price = 100.0
        prices = [start_price]

        # Create oscillating pattern
        for i in range(1, bars):
            # Sine wave with noise
            cycle_progress = (i % self.bars_per_day) / self.bars_per_day
            oscillation = 0.05 * np.sin(2 * np.pi * cycle_progress)
            noise = np.random.normal(0, 0.002)

            new_price = start_price * (1 + oscillation + noise)
            prices.append(new_price)

        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'volume': np.random.uniform(2000, 8000, bars)
        })

        metadata = {
            'name': 'Volatile Sideways',
            'description': '±5% intraday swings, no directional trend',
            'expected_detection': 'Should NOT trigger false alerts',
            'target_alert': 'NORMAL or ALERT (not CRISIS)'
        }

        return df, metadata

    def scenario_4_gradual_crisis(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Scenario 4: Gradual Crisis (Slow Burn) ⭐ CRITICAL TEST

        Pattern:
        - Days 1-10: Consistent -5% daily decline
        - Total: -40% over 10 days
        - Each day individually: Only -5% (below -20% threshold)
        - Cumulative: -40% (CATASTROPHIC!)

        Expected:
        - ✅ MUST detect by Day 3-4 using 7d cumulative tracking
        - This is THE KEY FIX for P0-2

        Current System:
        - ❌ MISSES this completely (only checks 24h window)
        - Result: -40% loss undetected (UNACCEPTABLE!)

        Fixed System:
        - ✅ 7d window catches cumulative -30% by Day 7
        - ✅ Alert at -15% (Day 3-4)
        - ✅ Halt at -30% (Day 6-7)
        """
        print("\n" + "="*60)
        print("SCENARIO 4: Gradual Crisis ⭐ CRITICAL TEST")
        print("="*60)

        bars = 10 * self.bars_per_day
        dates = pd.date_range(start='2025-01-01', periods=bars, freq='15min')

        start_price = 100.0
        prices = [start_price]

        # Consistent -5% daily decline
        daily_decline = -0.05
        decline_per_bar = daily_decline / self.bars_per_day

        for i in range(1, bars):
            # Linear decline with small noise
            noise = np.random.normal(0, 0.0002)
            new_price = prices[-1] * (1 + decline_per_bar + noise)
            prices.append(new_price)

        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'volume': np.random.uniform(1000, 3000, bars)  # Declining volume
        })

        metadata = {
            'name': 'Gradual Crisis',
            'description': 'Consistent -5% daily decline over 10 days = -40% total',
            'daily_decline': daily_decline,
            'expected_detection': 'MUST detect by Day 3-4 (cumulative -15%)',
            'target_alert': 'ALERT by Day 3-4, CRISIS by Day 6-7',
            'critical': True,
            'note': 'This is THE test for P0-2 multi-timeframe fix'
        }

        return df, metadata

    def scenario_5_fake_recovery(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Scenario 5: Fake Recovery (Bear Market Rally)

        Pattern:
        - Days 1-3: Down -15%
        - Days 4-5: Rally +8% (fake recovery)
        - Days 6-8: Resume decline -12%
        - Total: -19% (but choppy)

        Expected:
        - ⚠️ Should detect overall bearish trend
        - Challenge: Don't get fooled by mid-crash rallies
        """
        print("\n" + "="*60)
        print("SCENARIO 5: Fake Recovery (Bear Market Rally)")
        print("="*60)

        # Initial decline (Days 1-3): -15%
        df = self.generate_base_data(days=3, volatility=0.001)
        decline_magnitude = -0.15

        for i in range(len(df)):
            df.loc[i, 'close'] = df.loc[0, 'close'] * (1 + decline_magnitude * i / len(df))

        # Fake rally (Days 4-5): +8%
        rally_bars = 2 * self.bars_per_day
        rally_magnitude = 0.08

        for i in range(rally_bars):
            rally_per_bar = rally_magnitude / rally_bars
            last_price = df['close'].iloc[-1]
            new_price = last_price * (1 + rally_per_bar)

            new_row = pd.DataFrame({
                'timestamp': [df['timestamp'].iloc[-1] + timedelta(minutes=15)],
                'close': [new_price],
                'high': [new_price * 1.001],
                'low': [new_price * 0.999],
                'volume': [np.random.uniform(3000, 6000)]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        # Resume decline (Days 6-8): -12%
        decline2_bars = 3 * self.bars_per_day
        decline2_magnitude = -0.12

        for i in range(decline2_bars):
            decline_per_bar = decline2_magnitude / decline2_bars
            last_price = df['close'].iloc[-1]
            new_price = last_price * (1 + decline_per_bar)

            new_row = pd.DataFrame({
                'timestamp': [df['timestamp'].iloc[-1] + timedelta(minutes=15)],
                'close': [new_price],
                'high': [new_price * 1.001],
                'low': [new_price * 0.999],
                'volume': [np.random.uniform(2000, 4000)]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        metadata = {
            'name': 'Fake Recovery',
            'description': '-15% → +8% rally → -12% = -19% total',
            'expected_detection': 'Detect overall bearish trend despite rally',
            'target_alert': 'CRISIS'
        }

        return df, metadata


class ScenarioTester:
    """Test crisis detection system against scenarios"""

    def __init__(self):
        self.detector = MultiTimeframeDetector()
        self.generator = ScenarioGenerator()
        # Longer observation window (3 hours) for catching false breakouts
        self.post_entry_validator = PostEntryValidator(observation_window_hours=3.0)

    def test_scenario(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        check_points: List[int] = None
    ) -> Dict:
        """
        Test a scenario and return results.

        Args:
            df: Price dataframe
            metadata: Scenario metadata
            check_points: Specific bars to check (default: every day)

        Returns:
            Dict with test results
        """
        if check_points is None:
            # Check every day by default
            check_points = list(range(96, len(df), 96))

            # For False Breakout: add hourly checks during observation period
            if metadata['name'] == 'False Breakout':
                reversal_start = metadata.get('reversal_start_bar', 0)
                entry_bar_target = reversal_start - 4

                # Add hourly checks (4 bars = 1 hour) around reversal
                hourly_checks = list(range(entry_bar_target, min(reversal_start + 100, len(df)), 4))
                check_points = sorted(set(check_points + hourly_checks))

        results = {
            'metadata': metadata,
            'detections': [],
            'first_alert_bar': None,
            'first_crisis_bar': None,
            'first_halt_bar': None,
            'total_alerts': 0,
            'total_crises': 0,
            'total_halts': 0,
            'post_entry_exit': None,  # For Scenario 2
            'resume_validations': [],  # For Scenario 5 - track resume validation results
            'fake_recovery_detected': False  # For Scenario 5 - if we stay cautious during rally
        }

        print(f"\nScenario: {metadata['name']}")
        print(f"Description: {metadata['description']}")
        print(f"Total bars: {len(df)}")
        print(f"Start price: ${df['close'].iloc[0]:.2f}")
        print(f"End price: ${df['close'].iloc[-1]:.2f}")
        print(f"Total return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
        print("\nRunning detection...")

        # Post-entry validation setup for Scenario 2 (False Breakout)
        post_entry_active = False
        trade_id = None
        entry_price_saved = None

        if metadata['name'] == 'False Breakout':
            # Entry at END OF CONSOLIDATION (just before reversal starts)
            # This simulates entering after consolidation breakout
            entry_bar = metadata.get('reversal_start_bar', 0) - 4  # 1 hour before reversal
            print(f"  📊 Will start post-entry observation at bar {entry_bar} (after consolidation)")
            print(f"  📊 Reversal starts at bar {metadata.get('reversal_start_bar', 0)}")
            print(f"  📊 Check points around entry: {[p for p in check_points if entry_bar - 10 <= p <= entry_bar + 100][:10]}")

        for bar in check_points:
            if bar >= len(df):
                continue

            test_df = df.iloc[:bar]
            detection = self.detector.detect(test_df, df['timestamp'].iloc[bar])

            alert_level = detection['crisis_analysis']['alert_level']

            # Track first occurrences
            if alert_level in [AlertLevel.ALERT, AlertLevel.CRISIS, AlertLevel.HALT]:
                if results['first_alert_bar'] is None:
                    results['first_alert_bar'] = bar
                    day = bar // 96
                    print(f"  ⚠️ FIRST ALERT at bar {bar} (Day {day})")

            if alert_level in [AlertLevel.CRISIS, AlertLevel.HALT]:
                if results['first_crisis_bar'] is None:
                    results['first_crisis_bar'] = bar
                    day = bar // 96
                    print(f"  🚨 FIRST CRISIS at bar {bar} (Day {day})")

            if alert_level == AlertLevel.HALT:
                if results['first_halt_bar'] is None:
                    results['first_halt_bar'] = bar
                    day = bar // 96
                    print(f"  🛑 FIRST HALT at bar {bar} (Day {day})")

            # Count totals
            if alert_level == AlertLevel.ALERT:
                results['total_alerts'] += 1
            elif alert_level == AlertLevel.CRISIS:
                results['total_crises'] += 1
            elif alert_level == AlertLevel.HALT:
                results['total_halts'] += 1

            # Store detection
            results['detections'].append({
                'bar': bar,
                'day': bar // 96,
                'price': test_df['close'].iloc[-1],
                'alert_level': alert_level.value,
                'regime': detection['confirmed_regime'].value,
                'recommendation': detection['recommendation']
            })

            # Resume validation tracking for Scenario 5
            if metadata['name'] == 'Fake Recovery' and detection.get('resume_analysis'):
                resume = detection['resume_analysis']
                results['resume_validations'].append({
                    'bar': bar,
                    'day': bar // 96,
                    'resume_level': resume['resume_level'].value,
                    'score': resume['score'],
                    'recommendation': resume['recommendation']
                })

                # Check if we correctly stayed cautious during fake rally (Days 4-5)
                # Rally phase is approximately bars 288-480 (Days 3-5)
                if 288 <= bar <= 480:  # During rally phase
                    if resume['resume_level'].value in ['HALT', 'CAUTIOUS']:
                        results['fake_recovery_detected'] = True
                        if bar == 288 or (bar % 96 == 0 and bar <= 480):  # Log once per day
                            print(f"  🎯 FAKE RECOVERY DETECTION at bar {bar} (Day {bar // 96}): {resume['resume_level'].value} ({resume['details']['summary']})")

            # Post-entry validation for Scenario 2
            if metadata['name'] == 'False Breakout':
                # Start observation at end of consolidation (before reversal)
                entry_bar_target = metadata.get('reversal_start_bar', 0) - 4
                if bar == entry_bar_target and not post_entry_active:
                    entry_price = test_df['close'].iloc[-1]
                    entry_price_saved = entry_price
                    entry_time = test_df['timestamp'].iloc[-1]
                    trade_id = f"FALSE_BREAKOUT_{bar}"

                    self.post_entry_validator.start_observation(
                        trade_id=trade_id,
                        entry_price=entry_price,
                        entry_time=entry_time,
                        direction='LONG'
                    )
                    post_entry_active = True
                    print(f"  📊 POST-ENTRY: Started observation at ${entry_price:.2f}")

                # Update observation if active (separate if, not elif!)
                entry_bar_target = metadata.get('reversal_start_bar', 0) - 4
                if post_entry_active and trade_id and bar > entry_bar_target:
                    current_price = test_df['close'].iloc[-1]
                    current_volume = test_df['volume'].iloc[-1]
                    timestamp = test_df['timestamp'].iloc[-1]

                    # Debug
                    if bar <= 520 and entry_price_saved:  # First few updates
                        print(f"  🔍 Update observation at bar {bar}: ${current_price:.2f} (move: {((current_price / entry_price_saved) - 1) * 100:.1f}%)")

                    validation_result = self.post_entry_validator.update_observation(
                        trade_id=trade_id,
                        current_price=current_price,
                        current_volume=current_volume,
                        timestamp=timestamp
                    )

                    # Debug validation result
                    if bar <= 520:
                        print(f"      → Action: {validation_result['action']}, Reason: {validation_result['reason']}")

                    if validation_result['action'] == 'EXIT':
                        print(f"  🎯 POST-ENTRY EXIT at bar {bar} (Day {bar // 96}): {validation_result['reason']}")
                        results['post_entry_exit'] = {
                            'bar': bar,
                            'day': bar // 96,
                            'reason': validation_result['reason'],
                            'confidence': validation_result['confidence'],
                            'details': validation_result['details']
                        }
                        post_entry_active = False
                        # Mark this as a detection (counts as PASS for Scenario 2)
                        if results['first_crisis_bar'] is None:
                            results['first_crisis_bar'] = bar

        return results

    def evaluate_results(self, results: Dict) -> str:
        """
        Evaluate if scenario test passed.

        Returns:
            "PASS", "FAIL", or "PARTIAL"
        """
        metadata = results['metadata']
        name = metadata['name']

        # Scenario-specific evaluation
        if name == 'Flat→Crash':
            # Should detect within 24h (96 bars) of crash start
            if results['first_crisis_bar'] or results['first_halt_bar']:
                crash_start = metadata['crash_start_bar']
                detection_bar = results['first_crisis_bar'] or results['first_halt_bar']
                detection_delay = detection_bar - crash_start

                if detection_delay <= 96:  # Within 24h
                    return "✅ PASS"
                else:
                    return "⚠️ PARTIAL (detected but slow)"
            return "❌ FAIL"

        elif name == 'Gradual Crisis':
            # CRITICAL: Must detect by Day 3-4 (cumulative tracking)
            if results['first_alert_bar']:
                day = results['first_alert_bar'] // 96
                if day <= 4:
                    return "✅ PASS (detected early!)"
                elif day <= 7:
                    return "⚠️ PARTIAL (detected late)"
                else:
                    return "❌ FAIL (detected too late)"
            return "❌ FAIL (not detected)"

        elif name == 'False Breakout':
            # Should detect bull trap via post-entry validation
            if results['post_entry_exit']:
                day = results['post_entry_exit']['day']
                if day <= 5:  # Within 2 days of rally end (Day 3)
                    return "✅ PASS (bull trap detected by post-entry!)"
                else:
                    return "⚠️ PARTIAL (detected late)"
            # Fallback: regular crisis detection
            elif results['first_crisis_bar']:
                return "✅ PASS"
            return "❌ FAIL"

        elif name == 'Volatile Sideways':
            # Should NOT trigger false crises
            if results['total_crises'] == 0 and results['total_halts'] == 0:
                return "✅ PASS (no false alarms)"
            elif results['total_crises'] <= 2:
                return "⚠️ PARTIAL (some false alarms)"
            else:
                return "❌ FAIL (too many false alarms)"

        elif name == 'Fake Recovery':
            # Should either:
            # 1. Stay cautious during fake rally (Days 4-5) via resume validation, OR
            # 2. Detect second crash (Days 6-8) quickly

            if results['fake_recovery_detected']:
                # Best case: Resume validator rejected fake recovery
                return "✅ PASS (fake recovery detected by resume validator!)"
            elif results['first_crisis_bar']:
                # Second best: Detected the second crash
                crisis_day = results['first_crisis_bar'] // 96
                if crisis_day <= 6:
                    return "✅ PASS (detected second crash)"
                elif crisis_day <= 8:
                    return "⚠️ PARTIAL (detected second crash late)"
                else:
                    return "⚠️ PARTIAL (detected too late)"
            elif results['first_alert_bar']:
                return "⚠️ PARTIAL (only alert, no crisis)"
            else:
                return "❌ FAIL (fake recovery not detected)"

        else:
            # Default: Any detection = PASS
            if results['first_crisis_bar'] or results['first_halt_bar']:
                return "✅ PASS"
            elif results['first_alert_bar']:
                return "⚠️ PARTIAL"
            return "❌ FAIL"

    def run_all_scenarios(self):
        """Run all 5 scenarios and generate report"""
        print("\n" + "="*60)
        print(" MUTABLE PHASES TESTING - 5 SCENARIOS")
        print("="*60)

        scenarios = [
            self.generator.scenario_1_flat_to_crash(),
            self.generator.scenario_2_false_breakout(),
            self.generator.scenario_3_volatile_sideways(),
            self.generator.scenario_4_gradual_crisis(),
            self.generator.scenario_5_fake_recovery()
        ]

        all_results = []

        for df, metadata in scenarios:
            results = self.test_scenario(df, metadata)
            evaluation = self.evaluate_results(results)

            all_results.append({
                'results': results,
                'evaluation': evaluation
            })

            print(f"\nResult: {evaluation}")
            print("-"*60)

        # Summary report
        print("\n" + "="*60)
        print(" FINAL SUMMARY")
        print("="*60)

        passed = sum(1 for r in all_results if r['evaluation'].startswith("✅"))
        partial = sum(1 for r in all_results if r['evaluation'].startswith("⚠️"))
        failed = sum(1 for r in all_results if r['evaluation'].startswith("❌"))

        for i, result in enumerate(all_results, 1):
            name = result['results']['metadata']['name']
            eval_str = result['evaluation']
            print(f"{i}. {name:20s} {eval_str}")

        print("\n" + "-"*60)
        print(f"PASSED: {passed}/5 ({passed/5*100:.0f}%)")
        print(f"PARTIAL: {partial}/5")
        print(f"FAILED: {failed}/5")

        # Overall assessment
        if passed >= 4:
            print("\n🎉 SUCCESS! System detects 80%+ of scenarios")
        elif passed >= 3:
            print("\n⚠️ PARTIAL SUCCESS. Needs improvement.")
        else:
            print("\n❌ SYSTEM FAILURE. Major fixes needed.")

        # Scenario 4 specific check
        scenario_4_result = all_results[3]
        if scenario_4_result['evaluation'].startswith("✅"):
            print("\n✅ SCENARIO 4 (Gradual Crisis) DETECTED! P0-2 FIX WORKING! 🚀")
        else:
            print("\n❌ SCENARIO 4 (Gradual Crisis) STILL MISSED - P0-2 needs more work")

        return all_results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test crisis detection against mutable phases')
    parser.add_argument('--scenario', type=str, help='Test specific scenario (1-5 or name)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    tester = ScenarioTester()

    if args.scenario:
        # Test specific scenario
        scenario_map = {
            '1': tester.generator.scenario_1_flat_to_crash,
            '2': tester.generator.scenario_2_false_breakout,
            '3': tester.generator.scenario_3_volatile_sideways,
            '4': tester.generator.scenario_4_gradual_crisis,
            '5': tester.generator.scenario_5_fake_recovery,
            'flat_crash': tester.generator.scenario_1_flat_to_crash,
            'false_breakout': tester.generator.scenario_2_false_breakout,
            'volatile': tester.generator.scenario_3_volatile_sideways,
            'gradual_crisis': tester.generator.scenario_4_gradual_crisis,
            'fake_recovery': tester.generator.scenario_5_fake_recovery
        }

        if args.scenario in scenario_map:
            df, metadata = scenario_map[args.scenario]()
            results = tester.test_scenario(df, metadata)
            evaluation = tester.evaluate_results(results)
            print(f"\n{evaluation}")
        else:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: 1-5 or {list(scenario_map.keys())}")
    else:
        # Run all scenarios
        tester.run_all_scenarios()


if __name__ == "__main__":
    main()
