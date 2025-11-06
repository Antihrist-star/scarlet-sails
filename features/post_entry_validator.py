"""
Post-Entry Validation Module
=============================

Validates trade quality AFTER entry to detect false signals (bull traps).

Problem: Entry signals can look valid but turn out to be traps
Solution: Observe price behavior for 1-2 hours after entry, exit if fake

Key patterns detected:
1. Bull trap: Price pumps +5-10%, then reverses (fake breakout)
2. Immediate adverse move: Wrong entry, price goes against immediately
3. Volume drying up: Initial volume spike, then nothing (fake move)
4. Too good to be true: Unrealistic gains (+10% in 1h) → take profit

Author: Scarlet Sails Team
Date: 2025-11-05
Priority: P1 (Scenario 2 fix)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class TradeObservation:
    """Data structure for tracking trade after entry"""
    trade_id: str
    entry_price: float
    entry_time: datetime
    direction: str  # 'LONG' or 'SHORT'
    bars_observed: int
    max_favorable_move: float
    current_price: float
    volume_profile: List[float]
    status: str  # 'OBSERVING', 'CONFIRMED', 'EXITED'


class PostEntryValidator:
    """
    Validates trade quality after entry.

    Workflow:
    1. Trade enters → start_observation()
    2. Each new bar → update_observation()
    3. Check rules → exit if trap detected
    4. After 2 hours → confirm or exit

    Philosophy: "Trust but verify"
    """

    def __init__(
        self,
        observation_window_hours: float = 2.0,
        bars_per_hour: int = 4  # 15min bars
    ):
        """
        Initialize post-entry validator.

        Args:
            observation_window_hours: How long to observe (default: 2 hours)
            bars_per_hour: Bars per hour (4 for 15min data)
        """
        self.observation_window = observation_window_hours
        self.bars_per_hour = bars_per_hour
        self.min_confirmation_bars = int(observation_window_hours * bars_per_hour)

        # Active observations
        self.active_observations: Dict[str, TradeObservation] = {}

        # Thresholds
        self.thresholds = {
            'pump_threshold': 0.05,      # 5% move = significant pump
            'reversal_ratio': 0.5,       # If drops to 50% of max = reversal
            'adverse_threshold': -0.02,  # -2% adverse move = bad entry
            'volume_drop_ratio': 0.3,    # Volume drops to 30% = drying up
            'tgtbt_threshold': 0.10      # +10% in 1h = too good to be true
        }

    def start_observation(
        self,
        trade_id: str,
        entry_price: float,
        entry_time: datetime,
        direction: str
    ) -> None:
        """
        Start observing a trade after entry.

        Args:
            trade_id: Unique trade identifier
            entry_price: Price at entry
            entry_time: Timestamp of entry
            direction: 'LONG' or 'SHORT'
        """
        self.active_observations[trade_id] = TradeObservation(
            trade_id=trade_id,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            bars_observed=0,
            max_favorable_move=0.0,
            current_price=entry_price,
            volume_profile=[],
            status='OBSERVING'
        )

        print(f"📊 Started observation: {trade_id} at ${entry_price:.2f} ({direction})")

    def update_observation(
        self,
        trade_id: str,
        current_price: float,
        current_volume: float,
        timestamp: datetime
    ) -> Dict[str, any]:
        """
        Update observation with new bar data.

        Args:
            trade_id: Trade identifier
            current_price: Current price
            current_volume: Current volume
            timestamp: Current timestamp

        Returns:
            Dict with:
                - action: 'HOLD' or 'EXIT'
                - reason: Explanation
                - confidence: 0-1
                - details: Additional info
        """
        if trade_id not in self.active_observations:
            return {
                'action': 'HOLD',
                'reason': 'No active observation',
                'confidence': 0.0,
                'details': {}
            }

        obs = self.active_observations[trade_id]
        obs.bars_observed += 1
        obs.current_price = current_price
        obs.volume_profile.append(current_volume)

        # Calculate move from entry
        if obs.direction == 'LONG':
            move_pct = (current_price - obs.entry_price) / obs.entry_price
        else:
            move_pct = (obs.entry_price - current_price) / obs.entry_price

        # Track max favorable move
        if move_pct > obs.max_favorable_move:
            obs.max_favorable_move = move_pct

        # Check if observation period complete
        time_elapsed = (timestamp - obs.entry_time).total_seconds() / 3600

        if time_elapsed < self.observation_window:
            # Still observing - check for early exit signals

            # Debug Rule 2
            if obs.bars_observed >= 2:
                print(f"      DEBUG Rule 2: bars={obs.bars_observed}, max_fav={obs.max_favorable_move:.4f}, current={move_pct:.4f}")

            exit_check = self._check_early_exit(obs, move_pct)

            if exit_check['should_exit']:
                obs.status = 'EXITED'
                del self.active_observations[trade_id]

                return {
                    'action': 'EXIT',
                    'reason': exit_check['reason'],
                    'confidence': exit_check['confidence'],
                    'details': {
                        'bars_observed': obs.bars_observed,
                        'max_move': f"{obs.max_favorable_move:.2%}",
                        'current_move': f"{move_pct:.2%}",
                        'time_elapsed': f"{time_elapsed:.1f}h"
                    }
                }

            return {
                'action': 'HOLD',
                'reason': 'Observing',
                'confidence': 0.5,
                'details': {
                    'bars_observed': obs.bars_observed,
                    'current_move': f"{move_pct:.2%}",
                    'max_move': f"{obs.max_favorable_move:.2%}"
                }
            }

        else:
            # Observation complete - trade confirmed
            obs.status = 'CONFIRMED'
            del self.active_observations[trade_id]

            return {
                'action': 'HOLD',
                'reason': 'Trade confirmed (observation complete)',
                'confidence': 1.0,
                'details': {
                    'bars_observed': obs.bars_observed,
                    'final_move': f"{move_pct:.2%}",
                    'max_move': f"{obs.max_favorable_move:.2%}"
                }
            }

    def _check_early_exit(
        self,
        obs: TradeObservation,
        current_move: float
    ) -> Dict[str, any]:
        """
        Check if trade should be exited early.

        Returns:
            Dict with:
                - should_exit: bool
                - reason: str
                - confidence: float
        """
        # Rule 1: Reversal after pump (BULL TRAP!)
        if obs.max_favorable_move > self.thresholds['pump_threshold']:
            # Had a significant pump (+5%+)
            if current_move < obs.max_favorable_move * self.thresholds['reversal_ratio']:
                # Now only 50% of max = reversal!
                return {
                    'should_exit': True,
                    'reason': f'🎣 Bull trap detected! Pump {obs.max_favorable_move:.1%} → now {current_move:.1%}',
                    'confidence': 0.9
                }

        # Rule 2: Continuous decline without pump (FALSE BREAKOUT)
        # If no significant upward movement but consistent decline
        # Check after minimum observations
        if obs.bars_observed >= 2:  # At least 2 updates
            if obs.max_favorable_move <= 0.005 and current_move <= -0.003:
                # No meaningful pump (<=0.5%) but already -0.30%+ decline = false breakout
                return {
                    'should_exit': True,
                    'reason': f'🎪 False breakout detected! No rally (max: {obs.max_favorable_move:.1%}), declining {current_move:.1%}',
                    'confidence': 0.85
                }

        # Rule 3: Immediate adverse move (BAD ENTRY)
        if obs.bars_observed >= self.bars_per_hour:  # After 1 hour
            if current_move < self.thresholds['adverse_threshold']:
                # -2% from entry = wrong
                return {
                    'should_exit': True,
                    'reason': f'⚠️ Immediate adverse move: {current_move:.1%}',
                    'confidence': 0.8
                }

        # Rule 4: Volume drying up (FAKE MOVE)
        if len(obs.volume_profile) >= self.bars_per_hour:
            recent_vol = np.mean(obs.volume_profile[-2:])
            initial_vol = np.mean(obs.volume_profile[:2])

            if initial_vol > 0:
                vol_ratio = recent_vol / initial_vol

                if vol_ratio < self.thresholds['volume_drop_ratio']:
                    # Volume dropped 70%+
                    return {
                        'should_exit': True,
                        'reason': f'📉 Volume drying up: {vol_ratio:.1%} of initial',
                        'confidence': 0.7
                    }

        # Rule 5: Too good to be true (SUSPICIOUS)
        if obs.bars_observed >= self.bars_per_hour:
            if current_move > self.thresholds['tgtbt_threshold']:
                # +10% in 1 hour = suspicious, take profit
                return {
                    'should_exit': True,
                    'reason': f'🤔 Too good to be true: {current_move:.1%} in 1h - take profit!',
                    'confidence': 0.6
                }

        # No exit signal
        return {
            'should_exit': False,
            'reason': '',
            'confidence': 0.0
        }

    def get_active_observations(self) -> Dict[str, TradeObservation]:
        """Get all active observations"""
        return self.active_observations.copy()

    def clear_observation(self, trade_id: str) -> None:
        """Manually clear an observation"""
        if trade_id in self.active_observations:
            del self.active_observations[trade_id]


def demo():
    """Demo showing bull trap detection"""
    print("="*60)
    print("Post-Entry Validator Demo - Bull Trap Detection")
    print("="*60)

    validator = PostEntryValidator()

    # Scenario: Bull trap
    print("\n--- Bull Trap Scenario ---")
    print("Entry: $100 LONG")

    validator.start_observation(
        trade_id='DEMO_TRAP',
        entry_price=100.0,
        entry_time=datetime(2025, 1, 1, 10, 0),
        direction='LONG'
    )

    # Phase 1: Pump (bars 1-4)
    print("\nPhase 1: Pump")
    prices_pump = [102, 105, 108, 110]
    volumes_pump = [5000, 6000, 5500, 5000]

    for i, (p, v) in enumerate(zip(prices_pump, volumes_pump)):
        timestamp = datetime(2025, 1, 1, 10, 0) + timedelta(minutes=(i+1)*15)
        result = validator.update_observation('DEMO_TRAP', p, v, timestamp)

        print(f"  Bar {i+1}: ${p} - {result['reason']}")
        if result['action'] == 'EXIT':
            print(f"  🚨 EXIT: {result['reason']}")
            return

    # Phase 2: Dump (bull trap reversal)
    print("\nPhase 2: Dump (Reversal)")
    prices_dump = [108, 105, 100, 95]
    volumes_dump = [7000, 8000, 9000, 10000]

    for i, (p, v) in enumerate(zip(prices_dump, volumes_dump)):
        timestamp = datetime(2025, 1, 1, 10, 0) + timedelta(minutes=(i+5)*15)
        result = validator.update_observation('DEMO_TRAP', p, v, timestamp)

        print(f"  Bar {i+5}: ${p} - {result['reason']}")

        if result['action'] == 'EXIT':
            print(f"\n🎯 EXIT SIGNAL: {result['reason']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Details: {result['details']}")
            print("\n✅ Bull trap successfully detected!")
            return

    print("\n❌ Failed to detect bull trap")


if __name__ == "__main__":
    demo()
    print("\n✅ Post-entry validator module created successfully!")
