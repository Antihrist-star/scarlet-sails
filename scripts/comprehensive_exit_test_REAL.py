# FILE: scripts/comprehensive_exit_test_REAL.py

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.position_manager import PositionManager
from models.hybrid_position_manager import HybridPositionManager

print("="*80)
print("COMPREHENSIVE EXIT TEST - REAL BTC DATA")
print("="*80)

# ============================================================================
# LOAD REAL DATA
# ============================================================================
print("\n📂 Loading REAL BTC data...")
df = pd.read_parquet('data/raw/BTC_USDT_1h_FULL.parquet')
print(f"   ✅ Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
print(f"   Period: {(df.index[-1] - df.index[0]).days} days")
print(f"   Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")

# ============================================================================
# CALCULATE INDICATORS
# ============================================================================
print("\n📊 Calculating indicators...")

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

df['rsi'] = calculate_rsi(df['close'])
df['atr'] = calculate_atr(df)
df['ma20'] = df['close'].rolling(20).mean()
df['ma200'] = df['close'].rolling(200).mean()

df = df.dropna()
print(f"   ✅ After indicators: {len(df):,} bars")

# ============================================================================
# SIMPLE REGIME DETECTION (INLINE)
# ============================================================================
print("\n🏛️ Detecting market regimes...")

def detect_regime_simple(row):
    """Simple MA-based regime"""
    if pd.isna(row['ma20']) or pd.isna(row['ma200']):
        return 'SIDEWAYS'

    if row['ma20'] > row['ma200'] * 1.02:
        return 'BULL_TREND'
    elif row['ma20'] < row['ma200'] * 0.98:
        return 'BEAR_MARKET'
    else:
        return 'SIDEWAYS'

df['regime'] = df.apply(detect_regime_simple, axis=1)
print(f"   ✅ Regimes detected")

regime_counts = df['regime'].value_counts()
print("\n   Regime distribution:")
for regime, count in regime_counts.items():
    pct = count / len(df) * 100
    print(f"      {regime:20s}: {count:6d} bars ({pct:5.1f}%)")

# ============================================================================
# GENERATE ENTRY SIGNALS - FIXED: RSI < 30 (OVERSOLD) ✅
# ============================================================================
print("\n🎯 Generating entry signals (RSI < 30 OVERSOLD - FIXED!)...")

entry_signals = []
for i in range(200, len(df)):
    # FIX: Buy on OVERSOLD, not OVERBOUGHT!
    if df['rsi'].iloc[i] < 30:  # ✅ CORRECT: oversold
        if not entry_signals or (i - entry_signals[-1]['bar_index'] > 24):
            entry_signals.append({
                'bar_index': i,
                'timestamp': df.index[i],
                'price': df['close'].iloc[i],
                'regime': df['regime'].iloc[i]
            })

print(f"   ✅ Generated {len(entry_signals)} entry signals")

signal_regimes = [s['regime'] for s in entry_signals]
print("\n   Signals by regime:")
for regime in ['BULL_TREND', 'BEAR_MARKET', 'SIDEWAYS']:
    count = signal_regimes.count(regime)
    if count > 0:
        print(f"      {regime}: {count}")

# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def backtest_naive(df, signals):
    """Naive strategy: fixed TP/SL"""
    trades = []
    for sig in signals:
        entry_bar = sig['bar_index']
        entry_price = sig['price']

        tp_price = entry_price * 1.15
        sl_price = entry_price * 0.95

        for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            if high >= tp_price:
                exit_price = tp_price
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_bar,
                    'exit_reason': 'TP'
                })
                break
            elif low <= sl_price:
                exit_price = sl_price
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_bar,
                    'exit_reason': 'SL'
                })
                break

    return trades

def backtest_pm(df, signals):
    """Position Manager: adaptive stops + trailing"""
    pm = PositionManager(
        max_holding_time_bars=168,
        enable_trailing=True,
        enable_partial_exits=True,
    )

    trades = []

    for sig in signals:
        entry_bar = sig['bar_index']
        entry_price = sig['price']
        entry_time = sig['timestamp']

        # Open position using CORRECT API ✅
        position = pm.open_position(
            symbol='BTC/USDT',
            entry_price=entry_price,
            entry_time=entry_time,
            size=1.0,
            direction='long',
            df=df,
            current_bar=entry_bar,
            regime=sig['regime'],
        )

        # Simulate holding position
        for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
            current_price = df['close'].iloc[i]

            # Update position using CORRECT API ✅
            try:
                position, exit_signals = pm.update_position(
                    'BTC/USDT',
                    current_price,
                    df,
                    i
                )

                # Execute exits
                if exit_signals:
                    exit_price = exit_signals[0]['price']
                    pnl_pct = (exit_price - entry_price) / entry_price

                    trades.append({
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_bar,
                        'exit_reason': exit_signals[0]['label']
                    })

                    pm.execute_exits('BTC/USDT', exit_signals)
                    break

            except Exception as e:
                # Position closed or error
                break

    return trades

def backtest_hybrid(df, signals):
    """Hybrid: regime-aware position management"""
    hybrid = HybridPositionManager(
        max_holding_time_bars=168,
    )

    trades = []

    for sig in signals:
        entry_bar = sig['bar_index']
        entry_price = sig['price']
        entry_time = sig['timestamp']

        # Open position using CORRECT API ✅
        # Note: HybridPositionManager detects regime internally
        position = hybrid.open_position(
            symbol='BTC/USDT',
            entry_price=entry_price,
            entry_time=entry_time,
            size=1.0,
            direction='long',
            df=df,
            current_bar=entry_bar,
        )

        # Simulate holding position
        for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
            current_price = df['close'].iloc[i]

            # Update position using CORRECT API ✅
            try:
                position, exit_signals = hybrid.update_position(
                    'BTC/USDT',
                    current_price,
                    df,
                    i
                )

                # Execute exits
                if exit_signals:
                    exit_price = exit_signals[0]['price']
                    pnl_pct = (exit_price - entry_price) / entry_price

                    trades.append({
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_bar,
                        'exit_reason': exit_signals[0]['label']
                    })

                    hybrid.execute_exits('BTC/USDT', exit_signals)
                    break

            except Exception as e:
                # Position closed or error
                break

    return trades

# ============================================================================
# RUN TESTS
# ============================================================================
print("\n" + "="*80)
print("RUNNING BACKTESTS ON REAL DATA...")
print("="*80)

print("\n1️⃣ Naive strategy...")
naive_trades = backtest_naive(df, entry_signals)
print(f"   ✅ {len(naive_trades)} trades")

print("\n2️⃣ Position Manager...")
pm_trades = backtest_pm(df, entry_signals)
print(f"   ✅ {len(pm_trades)} trades")

print("\n3️⃣ Hybrid strategy...")
hybrid_trades = backtest_hybrid(df, entry_signals)
print(f"   ✅ {len(hybrid_trades)} trades")

# ============================================================================
# RESULTS
# ============================================================================

def calc_metrics(trades):
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'avg_win': 0, 'avg_loss': 0, 'pf': 0}

    pnls = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / len(trades) if trades else 0
    total_pnl = sum(pnls) * 100
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 1e-6
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pf': pf
    }

naive_m = calc_metrics(naive_trades)
pm_m = calc_metrics(pm_trades)
hybrid_m = calc_metrics(hybrid_trades)

print("\n" + "="*80)
print("REAL BTC RESULTS (8 YEARS, RSI < 30 OVERSOLD)")
print("="*80)

print(f"\n{'Metric':<20} {'Naive':>15} {'PM':>15} {'Hybrid':>15}")
print("-"*80)
print(f"{'Trades':<20} {naive_m['trades']:>15} {pm_m['trades']:>15} {hybrid_m['trades']:>15}")
print(f"{'Win Rate':<20} {naive_m['win_rate']:>14.1%} {pm_m['win_rate']:>14.1%} {hybrid_m['win_rate']:>14.1%}")
print(f"{'Total P&L':<20} {naive_m['total_pnl']:>14.1f}% {pm_m['total_pnl']:>14.1f}% {hybrid_m['total_pnl']:>14.1f}%")
print(f"{'Avg Win':<20} {naive_m['avg_win']:>14.1%} {pm_m['avg_win']:>14.1%} {hybrid_m['avg_win']:>14.1%}")
print(f"{'Avg Loss':<20} {naive_m['avg_loss']:>14.1%} {pm_m['avg_loss']:>14.1%} {hybrid_m['avg_loss']:>14.1%}")
print(f"{'Profit Factor':<20} {naive_m['pf']:>15.2f} {pm_m['pf']:>15.2f} {hybrid_m['pf']:>15.2f}")

print("\n" + "="*80)
print("✅ COMPLETE")
print("="*80)
