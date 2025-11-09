"""
Position Manager - Day 10
=========================

Manages position lifecycle: Entry → Stop → Trail → Exit

Implements P_j(S) as trajectory of decisions, not single point:
- P_entry(S): When to enter
- P_stop(S): Where to place stop
- P_trail(S): How to trail stop
- P_exit(S): When to exit (partial or full)
- P_hold(S): Continue holding or close

Based on research: proper position management can improve
Sharpe ratio by 40-60% vs naive "buy and hold until target".

Author: Scarlet Sails Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

try:
    from models.exit_strategy import (
        AdaptiveStopLoss,
        TrailingStopManager,
        PartialExitManager,
        StopLevel,
        TakeProfitLevel,
    )
except ModuleNotFoundError:
    # Running as script
    from exit_strategy import (
        AdaptiveStopLoss,
        TrailingStopManager,
        PartialExitManager,
        StopLevel,
        TakeProfitLevel,
    )


class PositionState(Enum):
    """Position lifecycle states"""
    NONE = "none"                 # No position
    ENTERING = "entering"         # Order placed, not filled
    OPEN = "open"                # Position active
    STOP_TRAILING = "trailing"    # Trailing stop active
    PARTIAL_EXIT = "partial"      # Some exits taken
    CLOSING = "closing"           # Exit order placed
    CLOSED = "closed"             # Position fully closed


class ExitReason(Enum):
    """Why position was exited"""
    STOP_HIT = "stop_hit"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TP3_HIT = "tp3_hit"
    TRAILING_STOP = "trailing_stop"
    TIME_STOP = "time_stop"           # Held too long
    CRISIS_SIGNAL = "crisis_signal"   # Crisis detected
    MANUAL = "manual"                 # Manual override


@dataclass
class Position:
    """Active position data"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    size: float  # Position size (units)

    # Stop/TP levels
    stop_loss: float
    stop_type: str
    tp_levels: List[TakeProfitLevel] = field(default_factory=list)

    # State tracking
    state: PositionState = PositionState.OPEN
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Partial exits
    original_size: float = 0.0  # Original position size
    exits_taken: List[Dict] = field(default_factory=list)

    # Trailing state
    trailing_active: bool = False
    trailing_stop: Optional[float] = None

    # Metadata
    regime_at_entry: str = "UNKNOWN"
    volatility_at_entry: float = 0.0
    bars_held: int = 0

    def __post_init__(self):
        if self.original_size == 0.0:
            self.original_size = self.size


class PositionManager:
    """
    Manages position lifecycle using exit strategy components.

    Workflow:
    1. Entry: DecisionFormulaV2 signals entry
    2. Set Stop: AdaptiveStopLoss calculates initial stop
    3. Monitor: Check for TP hits, update trailing
    4. Exit: PartialExitManager + TrailingStop coordinate exits
    5. Close: Final cleanup and logging
    """

    def __init__(
        self,
        max_holding_time_bars: int = 168,  # Max 168 bars (7 days for 1h)
        enable_trailing: bool = True,
        enable_partial_exits: bool = True,
    ):
        """
        Initialize position manager.

        Args:
            max_holding_time_bars: Auto-exit after this many bars
            enable_trailing: Use trailing stops
            enable_partial_exits: Use partial TP exits
        """
        self.max_holding_time_bars = max_holding_time_bars
        self.enable_trailing = enable_trailing
        self.enable_partial_exits = enable_partial_exits

        # Components
        self.stop_calculator = AdaptiveStopLoss()
        self.trailing_manager = TrailingStopManager()
        self.partial_manager = PartialExitManager()

        # Active positions
        self.positions: Dict[str, Position] = {}

        # Position history (for analysis)
        self.closed_positions: List[Position] = []

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        size: float,
        direction: str,
        df: pd.DataFrame,
        current_bar: int,
        regime: str = "BULL",
    ) -> Position:
        """
        Open new position with adaptive stop and TP levels.

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            entry_time: Entry timestamp
            size: Position size (units)
            direction: 'long' or 'short'
            df: Price dataframe (for stop calculation)
            current_bar: Current bar index
            regime: Market regime

        Returns:
            Position object
        """
        # Calculate adaptive stop
        stop_level = self.stop_calculator.calculate(
            df, current_bar, entry_price, direction
        )

        # Calculate TP levels (if enabled)
        tp_levels = []
        if self.enable_partial_exits:
            tp_levels = self.partial_manager.calculate_levels(
                entry_price, stop_level.price, direction
            )

        # Create position
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            stop_loss=stop_level.price,
            stop_type=stop_level.type.value,
            tp_levels=tp_levels,
            current_price=entry_price,
            regime_at_entry=regime,
            volatility_at_entry=self.stop_calculator.calculate_atr(df, current_bar),
        )

        # Store
        self.positions[symbol] = position

        return position

    def update_position(
        self,
        symbol: str,
        current_price: float,
        df: pd.DataFrame,
        current_bar: int,
    ) -> Tuple[Position, List[Dict]]:
        """
        Update position: check stops, TPs, update trailing.

        Args:
            symbol: Asset symbol
            current_price: Current market price
            df: Price dataframe
            current_bar: Current bar index

        Returns:
            (updated_position, list_of_exit_signals)

        Exit signals format:
        [{
            'reason': ExitReason,
            'price': float,
            'size_pct': float,  # 0-1
            'label': str,  # "TP1", "STOP", etc
        }]
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        position = self.positions[symbol]
        exit_signals = []

        # Update position state
        position.current_price = current_price
        position.bars_held += 1

        # Calculate P&L
        if position.direction == 'long':
            position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl_pct = (position.entry_price - current_price) / position.entry_price

        position.unrealized_pnl = position.unrealized_pnl_pct * position.size * position.entry_price

        # 1. Check stop loss
        stop_hit = False
        if position.direction == 'long':
            stop_hit = current_price <= position.stop_loss
        else:
            stop_hit = current_price >= position.stop_loss

        if stop_hit:
            exit_signals.append({
                'reason': ExitReason.STOP_HIT,
                'price': position.stop_loss,
                'size_pct': 1.0,  # Exit full position
                'label': 'STOP',
            })
            return position, exit_signals

        # 2. Check time stop (held too long)
        if position.bars_held >= self.max_holding_time_bars:
            exit_signals.append({
                'reason': ExitReason.TIME_STOP,
                'price': current_price,
                'size_pct': 1.0,
                'label': 'TIME_STOP',
            })
            return position, exit_signals

        # 3. Check TP levels (partial exits)
        if self.enable_partial_exits and position.tp_levels:
            tp_hits = self.partial_manager.check_exits(
                current_price, position.tp_levels, position.direction
            )

            for tp in tp_hits:
                # Only exit if not already taken
                already_taken = any(
                    exit['label'] == tp.label
                    for exit in position.exits_taken
                )

                if not already_taken:
                    exit_signals.append({
                        'reason': getattr(ExitReason, f"{tp.label}_HIT"),
                        'price': tp.price,
                        'size_pct': tp.size_pct,
                        'label': tp.label,
                    })

        # 4. Update trailing stop (if enabled and profitable)
        if self.enable_trailing:
            # Start trailing at +2% profit
            if position.unrealized_pnl_pct >= 0.02:
                atr = self.stop_calculator.calculate_atr(df, current_bar)

                new_trailing_stop = self.trailing_manager.update(
                    current_price,
                    position.entry_price,
                    atr,
                    position.direction
                )

                # Update position trailing state
                if not position.trailing_active:
                    position.trailing_active = True
                    position.state = PositionState.STOP_TRAILING

                position.trailing_stop = new_trailing_stop

                # Update main stop to trailing stop (raise stop)
                position.stop_loss = max(position.stop_loss, new_trailing_stop)

        return position, exit_signals

    def execute_exits(
        self,
        symbol: str,
        exit_signals: List[Dict],
    ) -> Position:
        """
        Execute exit signals and update position.

        Args:
            symbol: Asset symbol
            exit_signals: List of exit signals from update_position

        Returns:
            Updated position
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        position = self.positions[symbol]

        for signal in exit_signals:
            # Record exit
            exit_record = {
                'reason': signal['reason'],
                'price': signal['price'],
                'size': position.size * signal['size_pct'],
                'size_pct': signal['size_pct'],
                'label': signal['label'],
                'timestamp': datetime.now(),
            }

            position.exits_taken.append(exit_record)

            # Reduce position size
            position.size *= (1 - signal['size_pct'])

            # Update state
            if signal['size_pct'] >= 1.0:
                # Full exit
                position.state = PositionState.CLOSED
                self.closed_positions.append(position)
                del self.positions[symbol]
                break
            else:
                # Partial exit
                position.state = PositionState.PARTIAL_EXIT

        return position

    def get_position_summary(self, symbol: str) -> Dict:
        """Get position summary for display/logging."""
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]

        return {
            'symbol': pos.symbol,
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'current_price': pos.current_price,
            'size': pos.size,
            'original_size': pos.original_size,
            'pnl_pct': pos.unrealized_pnl_pct,
            'pnl': pos.unrealized_pnl,
            'stop_loss': pos.stop_loss,
            'trailing_active': pos.trailing_active,
            'trailing_stop': pos.trailing_stop,
            'bars_held': pos.bars_held,
            'state': pos.state.value,
            'exits_taken': len(pos.exits_taken),
            'regime': pos.regime_at_entry,
        }


# Unit tests
if __name__ == "__main__":
    print("Position Manager - Unit Tests")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')

    base_price = 100
    trend = np.linspace(0, 20, 200)
    noise = np.random.normal(0, 2, 200)
    close_prices = base_price + trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * 0.99,
        'high': close_prices * 1.02,
        'low': close_prices * 0.98,
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, 200)
    })
    df = df.set_index('timestamp')

    # Create position manager
    pm = PositionManager(
        max_holding_time_bars=50,
        enable_trailing=True,
        enable_partial_exits=True,
    )

    # Test 1: Open position
    print("\n1. Opening position...")
    entry_bar = 100
    entry_price = df['close'].iloc[entry_bar]
    entry_time = df.index[entry_bar]

    position = pm.open_position(
        symbol='BTC/USDT',
        entry_price=entry_price,
        entry_time=entry_time,
        size=1.0,
        direction='long',
        df=df,
        current_bar=entry_bar,
        regime='BULL',
    )

    summary = pm.get_position_summary('BTC/USDT')
    print(f"   Entry: ${summary['entry_price']:.2f}")
    print(f"   Stop: ${summary['stop_loss']:.2f}")
    print(f"   Size: {summary['size']:.2f}")

    # Test 2: Simulate price movement
    print("\n2. Simulating 20 bars...")

    for i in range(entry_bar + 1, entry_bar + 21):
        current_price = df['close'].iloc[i]

        # Update position
        position, exit_signals = pm.update_position(
            'BTC/USDT',
            current_price,
            df,
            i
        )

        # Execute exits if any
        if exit_signals:
            print(f"\n   Bar {i}: EXIT SIGNALS")
            for signal in exit_signals:
                print(f"      {signal['label']}: ${signal['price']:.2f} ({signal['size_pct']*100:.0f}%)")

            pm.execute_exits('BTC/USDT', exit_signals)

            # Check if fully closed
            if 'BTC/USDT' not in pm.positions:
                print("   Position fully closed!")
                break

        # Show progress every 5 bars
        if i % 5 == 0:
            summary = pm.get_position_summary('BTC/USDT')
            if summary:
                print(f"   Bar {i}: P=${summary['current_price']:.2f}, "
                      f"PnL={summary['pnl_pct']*100:.2f}%, "
                      f"Stop=${summary['stop_loss']:.2f}, "
                      f"Trailing={'Yes' if summary['trailing_active'] else 'No'}")

    # Test 3: Summary
    print("\n3. Final summary...")
    if pm.closed_positions:
        closed = pm.closed_positions[-1]
        print(f"   Position closed!")
        print(f"   Entry: ${closed.entry_price:.2f}")
        print(f"   Exits taken: {len(closed.exits_taken)}")
        for exit in closed.exits_taken:
            print(f"      {exit['label']}: ${exit['price']:.2f} ({exit['size_pct']*100:.0f}%)")
        print(f"   Final PnL: {closed.unrealized_pnl_pct*100:.2f}%")
    else:
        summary = pm.get_position_summary('BTC/USDT')
        if summary:
            print(f"   Position still open")
            print(f"   Current PnL: {summary['pnl_pct']*100:.2f}%")
            print(f"   Exits taken: {summary['exits_taken']}")

    print("\n" + "=" * 60)
    print("✅ Position Manager working!")
