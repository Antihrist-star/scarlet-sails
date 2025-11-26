"""
TEST WEEK 1 IMPROVEMENTS

Compare:
1. Baseline (original strategy)
2. With Regime Gate only
3. With Regime Gate + Entry Confluence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.regime_gate import RegimeGate, MarketRegime
from models.entry_confluence import EntryConfluence

print("="*80)
print("WEEK 1 IMPROVEMENTS TEST")
print("="*80)
print("\nComparing 3 strategies:")
print("1. BASELINE: RSI < 30 (original)")
print("2. REGIME GATE: Only trade in bull/recovering bear")
print("3. FULL SYSTEM: Regime Gate + Entry Confluence")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading data...")
df = pd.read_parquet('data/raw/BTC_USDT_1h_FULL.parquet')
print(f"✅ Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

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
print(f"✅ After indicators: {len(df):,} bars")

# ============================================================================
# STRATEGY 1: BASELINE (RSI < 30)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY 1: BASELINE (RSI < 30, trade everywhere)")
print("="*80)

baseline_signals = []
for i in range(200, len(df)):
    if df['rsi'].iloc[i] < 30:
        if not baseline_signals or (i - baseline_signals[-1]['bar_index'] > 24):
            baseline_signals.append({
                'bar_index': i,
                'timestamp': df.index[i],
                'price': df['close'].iloc[i],
                'rsi': df['rsi'].iloc[i]
            })

print(f"✅ Generated {len(baseline_signals)} baseline signals")

# Simulate trades
def simulate_trades(df, signals, strategy_name):
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
                pnl_pct = (tp_price - entry_price) / entry_price
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': tp_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_bar,
                    'exit_reason': 'TP'
                })
                break
            elif low <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': sl_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_bar,
                    'exit_reason': 'SL'
                })
                break
    
    return trades

baseline_trades = simulate_trades(df, baseline_signals, 'BASELINE')

# Calculate metrics
def calculate_metrics(trades):
    if not trades:
        return None
    
    pnls = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    
    total_pnl_pct = sum(pnls) * 100
    win_rate = len(winners) / len(trades) if trades else 0
    
    # Costs
    fees_per_trade = 0.001 * 2  # 0.1% each side
    slippage_per_trade = 0.0005 * 2  # 0.05% each side
    total_cost_per_trade = fees_per_trade + slippage_per_trade
    
    gross_pnl_pct = total_pnl_pct
    costs_pct = len(trades) * total_cost_per_trade * 100
    net_pnl_pct = gross_pnl_pct - costs_pct
    
    # Annual return
    period_years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = net_pnl_pct / period_years
    monthly_return = annual_return / 12
    
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'gross_pnl': gross_pnl_pct,
        'costs': costs_pct,
        'net_pnl': net_pnl_pct,
        'annual_return': annual_return,
        'monthly_return': monthly_return
    }

baseline_metrics = calculate_metrics(baseline_trades)

print(f"\n📊 BASELINE Results:")
print(f"   Trades: {baseline_metrics['trades']:,}")
print(f"   Win Rate: {baseline_metrics['win_rate']:.1%}")
print(f"   Gross P&L: {baseline_metrics['gross_pnl']:.1f}%")
print(f"   Costs: -{baseline_metrics['costs']:.1f}%")
print(f"   Net P&L: {baseline_metrics['net_pnl']:.1f}%")
print(f"   Annual Return: {baseline_metrics['annual_return']:.1f}%")
print(f"   Monthly Return: {baseline_metrics['monthly_return']:.1f}%")

# ============================================================================
# STRATEGY 2: REGIME GATE
# ============================================================================
print("\n" + "="*80)
print("STRATEGY 2: REGIME GATE (only bull/recovering bear)")
print("="*80)

regime_gate = RegimeGate()

regime_filtered_signals = []
for sig in baseline_signals:
    bar_idx = sig['bar_index']
    should_trade, confidence = regime_gate.should_trade(df.iloc[:bar_idx+1])
    
    if should_trade:
        sig_with_confidence = sig.copy()
        sig_with_confidence['regime_confidence'] = confidence
        regime_filtered_signals.append(sig_with_confidence)

print(f"✅ Filtered to {len(regime_filtered_signals)} signals (from {len(baseline_signals)})")
print(f"   Reduction: {(1 - len(regime_filtered_signals)/len(baseline_signals)):.1%}")

regime_trades = simulate_trades(df, regime_filtered_signals, 'REGIME_GATE')
regime_metrics = calculate_metrics(regime_trades)

print(f"\n📊 REGIME GATE Results:")
print(f"   Trades: {regime_metrics['trades']:,}")
print(f"   Win Rate: {regime_metrics['win_rate']:.1%}")
print(f"   Net P&L: {regime_metrics['net_pnl']:.1f}%")
print(f"   Annual Return: {regime_metrics['annual_return']:.1f}%")
print(f"   Monthly Return: {regime_metrics['monthly_return']:.1f}%")

improvement_regime = regime_metrics['annual_return'] / baseline_metrics['annual_return']
print(f"\n   🚀 Improvement vs Baseline: {improvement_regime:.1f}x")

# ============================================================================
# STRATEGY 3: FULL SYSTEM (Regime + Confluence)
# ============================================================================
print("\n" + "="*80)
print("STRATEGY 3: FULL SYSTEM (Regime Gate + Entry Confluence)")
print("="*80)

confluence = EntryConfluence()

full_system_signals = []
for sig in regime_filtered_signals:
    bar_idx = sig['bar_index']
    
    # Score entry quality
    score = confluence.score_entry(df.iloc[:bar_idx+1])
    
    # Only take high-quality entries (score > 0.5)
    if score >= 0.5:
        sig_with_score = sig.copy()
        sig_with_score['confluence_score'] = score
        full_system_signals.append(sig_with_score)

print(f"✅ Filtered to {len(full_system_signals)} signals (from {len(regime_filtered_signals)})")
print(f"   Reduction: {(1 - len(full_system_signals)/len(regime_filtered_signals)):.1%}")

full_trades = simulate_trades(df, full_system_signals, 'FULL_SYSTEM')
full_metrics = calculate_metrics(full_trades)

print(f"\n📊 FULL SYSTEM Results:")
print(f"   Trades: {full_metrics['trades']:,}")
print(f"   Win Rate: {full_metrics['win_rate']:.1%}")
print(f"   Net P&L: {full_metrics['net_pnl']:.1f}%")
print(f"   Annual Return: {full_metrics['annual_return']:.1f}%")
print(f"   Monthly Return: {full_metrics['monthly_return']:.1f}%")

improvement_full = full_metrics['annual_return'] / baseline_metrics['annual_return']
print(f"\n   🚀 Improvement vs Baseline: {improvement_full:.1f}x")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: ALL 3 STRATEGIES")
print("="*80)

print(f"\n{'Metric':<25} {'Baseline':>15} {'Regime Gate':>15} {'Full System':>15} {'Winner':>15}")
print("-"*85)

metrics_to_compare = [
    ('Trades', 'trades', '{:,}'),
    ('Win Rate', 'win_rate', '{:.1%}'),
    ('Net P&L %', 'net_pnl', '{:.1f}%'),
    ('Annual Return', 'annual_return', '{:.1f}%'),
    ('Monthly Return', 'monthly_return', '{:.1f}%')
]

for metric_name, key, fmt in metrics_to_compare:
    baseline_val = baseline_metrics[key]
    regime_val = regime_metrics[key]
    full_val = full_metrics[key]
    
    # Format values
    b_str = fmt.format(baseline_val)
    r_str = fmt.format(regime_val)
    f_str = fmt.format(full_val)
    
    # Determine winner
    if key == 'trades':
        winner = 'N/A'
    else:
        if full_val >= regime_val >= baseline_val:
            winner = 'Full System'
        elif regime_val >= full_val >= baseline_val:
            winner = 'Regime Gate'
        elif full_val >= baseline_val >= regime_val:
            winner = 'Full System'
        else:
            winner = 'Mixed'
    
    print(f"{metric_name:<25} {b_str:>15} {r_str:>15} {f_str:>15} {winner:>15}")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("WEEK 1 VERDICT")
print("="*80)

print(f"\n🎯 TARGET: Improve from 24.6% annual to 80-120% annual")
print(f"\n✅ ACHIEVED: {full_metrics['annual_return']:.1f}% annual")

if full_metrics['annual_return'] >= 80:
    print(f"   ✅ TARGET MET! ({improvement_full:.1f}x improvement)")
    print(f"   Ready for Week 2 (MDP + ML)")
elif full_metrics['annual_return'] >= 50:
    print(f"   ⚠️  PROGRESS ({improvement_full:.1f}x improvement)")
    print(f"   Need more optimization before Week 2")
else:
    print(f"   ❌ INSUFFICIENT ({improvement_full:.1f}x improvement)")
    print(f"   Need to revisit approach")

print("\n" + "="*80)
print("✅ WEEK 1 TEST COMPLETE")
print("="*80)