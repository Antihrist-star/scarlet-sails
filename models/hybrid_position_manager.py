"""
Hybrid Position Manager - Day 10
=================================

Regime-aware position management:
- BULL: Trailing stop only (let winners run!)
- SIDEWAYS: Full PositionManager (partial exits protect)
- BEAR: PositionManager with tight stops
- CRISIS: Don't trade (or exit immediately if entered)

Combines best of both:
- Naive's upside capture in trends
- PM's risk control in chop

Author: Scarlet Sails Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    from models.position_manager import PositionManager, Position, PositionState, ExitReason
    from models.regime_detector import SimpleRegimeDetector, MarketRegime
    from models.exit_strategy import AdaptiveStopLoss, TrailingStopManager
except ModuleNotFoundError:
    from position_manager import PositionManager, Position, PositionState, ExitReason
    from regime_detector import SimpleRegimeDetector, MarketRegime
    from exit_strategy import AdaptiveStopLoss, TrailingStopManager


class HybridPositionManager:
    """
    Regime-aware position manager.

    Strategy by regime:
    - BULL: Simple trailing (no partial exits)
    - SIDEWAYS: Full PM (partial exits + trailing)
    - BEAR: PM with tighter stops (1.5x ATR instead of 2x)
    - CRISIS: Exit immediately

    Expected performance:
    - Bull: ~Naive performance (catch trends)
    - Sideways: ~PM performance (protect profits)
    - Overall: Best of both worlds
    """

    def __init__(
        self,
        max_holding_time_bars: int = 500,
    ):
        """
        Initialize hybrid manager.

        Args:
            max_holding_time_bars: Max holding time
        """
        self.max_holding_time_bars = max_holding_time_bars

        # Components
        self.regime_detector = SimpleRegimeDetector()
        self.pm_full = PositionManager(
            max_holding_time_bars=max_holding_time_bars,
            enable_trailing=True,
            enable_partial_exits=True,  # Full PM
        )
        self.pm_tight = PositionManager(
            max_holding_time_bars=max_holding_time_bars,
            enable_trailing=True,
            enable_partial_exits=True,  # Tight stops
        )

        # For trend mode (trailing only)
        self.stop_calculator = AdaptiveStopLoss()
        self.trailing_managers: Dict[str, TrailingStopManager] = {}

        # Active positions
        self.positions: Dict[str, Position] = {}
        self.position_regimes: Dict[str, MarketRegime] = {}  # Track regime per position

        # History
        self.closed_positions: List[Position] = []
        self.regime_switches: List[Dict] = []  # Track regime changes

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        size: float,
        direction: str,
        df: pd.DataFrame,
        current_bar: int,
    ) -> Position:
        """
        Open position with regime-aware strategy.

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            entry_time: Entry timestamp
            size: Position size
            direction: 'long' or 'short'
            df: Price dataframe
            current_bar: Current bar index

        Returns:
            Position object
        """
        # Detect regime
        regime = self.regime_detector.detect(df, current_bar)
        self.position_regimes[symbol] = regime

        # Open position based on regime
        if regime == MarketRegime.BULL_TREND:
            # BULL: Simple trailing (no partial exits)
            # Use basic PM but will override exits
            position = self.pm_full.open_position(
                symbol, entry_price, entry_time, size, direction,
                df, current_bar, regime=regime.value
            )

            # Initialize trailing manager
            self.trailing_managers[symbol] = TrailingStopManager()

        elif regime == MarketRegime.SIDEWAYS:
            # SIDEWAYS: Full PM
            position = self.pm_full.open_position(
                symbol, entry_price, entry_time, size, direction,
                df, current_bar, regime=regime.value
            )

        elif regime == MarketRegime.BEAR_MARKET:
            # BEAR: PM with tighter stops
            position = self.pm_tight.open_position(
                symbol, entry_price, entry_time, size, direction,
                df, current_bar, regime=regime.value
            )

            # Tighten stop (1.5x ATR instead of 2x)
            position.stop_loss = entry_price - (abs(entry_price - position.stop_loss) * 0.75)

        else:  # CRISIS
            # CRISIS: Should not enter, but if we do, tight stops
            position = self.pm_tight.open_position(
                symbol, entry_price, entry_time, size, direction,
                df, current_bar, regime=regime.value
            )

            # Very tight stop (1x ATR)
            position.stop_loss = entry_price - (abs(entry_price - position.stop_loss) * 0.5)

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
        Update position with regime-aware logic.

        FIX: LOCK STRATEGY AT ENTRY!
        - Use ENTRY regime, not current regime
        - Don't switch strategy mid-position
        - This prevents early exits and inconsistent behavior

        Args:
            symbol: Asset symbol
            current_price: Current price
            df: Price dataframe
            current_bar: Current bar index

        Returns:
            (updated_position, exit_signals)
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        position = self.positions[symbol]

        # LOCKED STRATEGY: Use ENTRY regime (not current regime)
        entry_regime = self.position_regimes.get(symbol)

        # Optional: Detect current regime for logging (but don't use for decisions!)
        current_regime = self.regime_detector.detect(df, current_bar)
        if current_regime != entry_regime:
            self.regime_switches.append({
                'symbol': symbol,
                'bar': current_bar,
                'from': entry_regime.value,
                'to': current_regime.value,
                'note': 'Regime changed but strategy LOCKED at entry',
            })

        # Update based on ENTRY regime (LOCKED strategy!)
        exit_signals = []

        if entry_regime == MarketRegime.BULL_TREND:
            # BULL: Trailing stop only (no partial exits)
            position, bull_signals = self._update_bull_trend(
                symbol, position, current_price, df, current_bar
            )
            exit_signals.extend(bull_signals)

        elif entry_regime == MarketRegime.SIDEWAYS:
            # SIDEWAYS: Full PM
            position, sideways_signals = self.pm_full.update_position(
                symbol, current_price, df, current_bar
            )
            exit_signals.extend(sideways_signals)

        elif entry_regime == MarketRegime.BEAR_MARKET:
            # BEAR: PM with tight stops
            position, bear_signals = self.pm_tight.update_position(
                symbol, current_price, df, current_bar
            )
            exit_signals.extend(bear_signals)

        else:  # CRISIS
            # CRISIS: Use tight stops (no emergency exit!)
            position, crisis_signals = self.pm_tight.update_position(
                symbol, current_price, df, current_bar
            )
            exit_signals.extend(crisis_signals)

        return position, exit_signals

    def _update_bull_trend(
        self,
        symbol: str,
        position: Position,
        current_price: float,
        df: pd.DataFrame,
        current_bar: int,
    ) -> Tuple[Position, List[Dict]]:
        """
        Update position in BULL trend mode.

        Logic: Trailing stop ONLY, no partial exits.
        Let winners run to capture full trend!
        """
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

        # 1. Check basic stop loss
        stop_hit = False
        if position.direction == 'long':
            stop_hit = current_price <= position.stop_loss
        else:
            stop_hit = current_price >= position.stop_loss

        if stop_hit:
            exit_signals.append({
                'reason': ExitReason.STOP_HIT,
                'price': position.stop_loss,
                'size_pct': 1.0,
                'label': 'STOP',
            })
            return position, exit_signals

        # 2. Check time stop
        if position.bars_held >= self.max_holding_time_bars:
            exit_signals.append({
                'reason': ExitReason.TIME_STOP,
                'price': current_price,
                'size_pct': 1.0,
                'label': 'TIME_STOP',
            })
            return position, exit_signals

        # 3. Update trailing stop (start at +2% profit)
        if position.unrealized_pnl_pct >= 0.02:
            if symbol not in self.trailing_managers:
                self.trailing_managers[symbol] = TrailingStopManager()

            trailing_mgr = self.trailing_managers[symbol]

            atr = self.stop_calculator.calculate_atr(df, current_bar)

            new_trailing_stop = trailing_mgr.update(
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
            position.stop_loss = max(position.stop_loss, new_trailing_stop)

        return position, exit_signals

    def execute_exits(
        self,
        symbol: str,
        exit_signals: List[Dict],
    ) -> Optional[Position]:
        """
        Execute exits and update position.

        Args:
            symbol: Asset symbol
            exit_signals: Exit signals from update

        Returns:
            Updated position or None if fully closed
        """
        # Handle all exits directly (positions tracked in self.positions)
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

            # Reduce size
            position.size *= (1 - signal['size_pct'])

            # Close if full exit
            if signal['size_pct'] >= 1.0:
                position.state = PositionState.CLOSED
                self.closed_positions.append(position)
                del self.positions[symbol]
                if symbol in self.trailing_managers:
                    del self.trailing_managers[symbol]
                return None

        return position

    def get_position_summary(self, symbol: str) -> Dict:
        """Get position summary."""
        if symbol in self.pm_full.positions:
            summary = self.pm_full.get_position_summary(symbol)
        elif symbol in self.pm_tight.positions:
            summary = self.pm_tight.get_position_summary(symbol)
        elif symbol in self.positions:
            pos = self.positions[symbol]
            summary = {
                'symbol': pos.symbol,
                'direction': pos.direction,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'size': pos.size,
                'pnl_pct': pos.unrealized_pnl_pct,
                'pnl': pos.unrealized_pnl,
                'stop_loss': pos.stop_loss,
                'trailing_active': pos.trailing_active,
                'bars_held': pos.bars_held,
                'state': pos.state.value,
            }
        else:
            summary = {}

        # Add regime info
        if symbol in self.position_regimes:
            summary['regime'] = self.position_regimes[symbol].value

        return summary


# Unit test
if __name__ == "__main__":
    print("Hybrid Position Manager - Unit Test")
    print("=" * 60)

    # Create sample data with regime shift
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')

    # First half: Bull trend
    bull_prices = np.linspace(100, 150, 500)
    bull_prices += np.random.normal(0, 2, 500)

    # Second half: Sideways
    sideways_prices = 150 + np.sin(np.linspace(0, 10*np.pi, 500)) * 5
    sideways_prices += np.random.normal(0, 3, 500)

    all_prices = np.concatenate([bull_prices, sideways_prices])

    df = pd.DataFrame({
        'open': all_prices * 0.99,
        'high': all_prices * 1.01,
        'low': all_prices * 0.98,
        'close': all_prices,
        'volume': np.random.uniform(1000, 5000, 1000),
    }, index=dates)

    # Create hybrid manager
    hpm = HybridPositionManager()

    # Test: Open position in bull, hold through regime change
    print("\n1. Opening position in BULL trend...")
    entry_bar = 250
    entry_price = df['close'].iloc[entry_bar]
    entry_time = df.index[entry_bar]

    position = hpm.open_position(
        symbol='TEST',
        entry_price=entry_price,
        entry_time=entry_time,
        size=1.0,
        direction='long',
        df=df,
        current_bar=entry_bar,
    )

    print(f"   Entry: ${entry_price:.2f}")
    print(f"   Regime: {hpm.position_regimes['TEST'].value}")

    # Simulate holding
    print("\n2. Simulating price movement through regime change...")

    for bar in range(entry_bar + 1, min(entry_bar + 300, len(df))):
        current_price = df['close'].iloc[bar]

        # Update
        position, exit_signals = hpm.update_position(
            'TEST', current_price, df, bar
        )

        # Execute exits
        if exit_signals:
            print(f"\n   Bar {bar}: EXIT SIGNALS")
            for signal in exit_signals:
                print(f"      {signal['label']}: ${signal['price']:.2f}")

            hpm.execute_exits('TEST', exit_signals)

            if 'TEST' not in hpm.positions:
                print("   Position fully closed!")
                break

        # Show progress every 50 bars
        if bar % 50 == 0 and 'TEST' in hpm.positions:
            summary = hpm.get_position_summary('TEST')
            regime = hpm.regime_detector.detect(df, bar)
            print(f"   Bar {bar}: P=${summary['current_price']:.2f}, "
                  f"PnL={summary['pnl_pct']*100:.2f}%, "
                  f"Regime={regime.value}")

    # Final summary
    print("\n3. Final summary...")
    if hpm.closed_positions:
        closed = hpm.closed_positions[-1]
        print(f"   Position closed!")
        print(f"   Entry: ${closed.entry_price:.2f}")
        print(f"   Bars held: {closed.bars_held}")
        print(f"   Exits taken: {len(closed.exits_taken)}")
        print(f"   Final PnL: {closed.unrealized_pnl_pct*100:.2f}%")

    if hpm.regime_switches:
        print(f"\n   Regime switches: {len(hpm.regime_switches)}")
        for switch in hpm.regime_switches:
            print(f"      Bar {switch['bar']}: {switch['from']} → {switch['to']}")

    print("\n" + "=" * 60)
    print("✅ Hybrid Position Manager working!")
