"""
MASTER COMPREHENSIVE AUDIT - FINAL TRUTH

Test EVERYTHING:
- 14 assets
- 4 timeframes
- Hybrid strategy (working baseline!)
- Full historical data
- Generate complete picture

NO MORE PARTIAL TESTS. THIS IS IT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

print("="*100)
print("MASTER COMPREHENSIVE AUDIT - FINAL 24H DELIVERY")
print("="*100)
print("\nScope: 14 assets × 4 timeframes × Hybrid strategy = 56 combinations")
print("Timeline: Full historical data (8+ years)")
print("Goal: COMPLETE PICTURE of where edge exists")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

ASSETS = [
    'BTC', 'ETH', 'SOL',           # Top 3
    'LINK', 'LDO', 'SUI', 'HBAR',  # Mid caps (from ВАЛЬКИРИЯ)
    'ENA', 'ALGO', 'AVAX',         # Additional
    'DOT', 'LTC', 'ONDO', 'UNI'    # More
]

TIMEFRAMES = ['15m', '1h', '4h', '1d']

DATA_DIR = Path('data/raw')
FEATURES_DIR = Path('data/features')
OUTPUT_DIR = Path('reports/master_audit')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# HYBRID STRATEGY (from Day 11)
# ============================================================================

class HybridStrategy:
    """
    Working baseline from Day 11 (+247% on BTC)
    """
    
    def __init__(self):
        self.atr_multipliers = {
            'BULL': 3.0,
            'BEAR': 2.0,
            'SIDEWAYS': 1.5
        }
        self.trailing_activation = 0.08  # Activate at +8%
        self.partial_exits = [0.33, 0.33, 0.34]  # TP levels
        self.tp_levels = [1.05, 1.10, 1.15]  # +5%, +10%, +15%
    
    def generate_signals(self, df):
        """Generate entry signals (RSI < 30)"""
        signals = []
        for i in range(200, len(df)):
            if df['rsi'].iloc[i] < 30:
                if not signals or (i - signals[-1]['bar_index'] > 24):
                    signals.append({
                        'bar_index': i,
                        'timestamp': df.index[i],
                        'price': df['close'].iloc[i],
                        'rsi': df['rsi'].iloc[i],
                        'regime': self.detect_regime(df.iloc[:i+1])
                    })
        return signals
    
    def detect_regime(self, df):
        """Simple regime detection"""
        if len(df) < 200:
            return 'SIDEWAYS'
        
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma200 = df['close'].rolling(200).mean().iloc[-1]
        
        if ma20 > ma200 * 1.05:
            return 'BULL'
        elif ma20 < ma200 * 0.95:
            return 'BEAR'
        else:
            return 'SIDEWAYS'
    
    def backtest(self, df, signals):
        """Hybrid backtest with adaptive exits"""
        trades = []
        
        for sig in signals:
            entry_bar = sig['bar_index']
            entry_price = sig['price']
            regime = sig['regime']
            
            # Adaptive stop based on regime
            atr = df['atr'].iloc[entry_bar]
            stop_distance = atr * self.atr_multipliers.get(regime, 2.0)
            current_stop = entry_price - stop_distance
            
            # Track position
            peak_price = entry_price
            remaining_size = 1.0
            total_pnl = 0.0
            
            # Search for exit
            for i in range(entry_bar + 1, min(entry_bar + 500, len(df))):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                close = df['close'].iloc[i]
                
                # Update peak
                if close > peak_price:
                    peak_price = close
                
                # Trailing stop
                profit_pct = (peak_price - entry_price) / entry_price
                if profit_pct > self.trailing_activation:
                    # Parabolic trailing
                    trail_distance = stop_distance * (1 - profit_pct * 0.5)
                    new_stop = peak_price - trail_distance
                    current_stop = max(current_stop, new_stop)
                
                # Check stop hit
                if low <= current_stop:
                    exit_price = current_stop
                    pnl = (exit_price - entry_price) / entry_price * remaining_size
                    total_pnl += pnl
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': total_pnl,
                        'bars_held': i - entry_bar,
                        'exit_reason': 'STOP',
                        'regime': regime
                    })
                    break
                
                # Check partial exits
                for idx, (tp_pct, exit_pct) in enumerate(zip(self.tp_levels, self.partial_exits)):
                    tp_price = entry_price * tp_pct
                    if high >= tp_price and remaining_size > 0:
                        # Partial exit
                        exit_size = exit_pct if remaining_size >= exit_pct else remaining_size
                        pnl = (tp_price - entry_price) / entry_price * exit_size
                        total_pnl += pnl
                        remaining_size -= exit_size
                        
                        if remaining_size <= 0:
                            trades.append({
                                'entry_price': entry_price,
                                'exit_price': tp_price,
                                'pnl_pct': total_pnl,
                                'bars_held': i - entry_bar,
                                'exit_reason': f'TP{idx+1}',
                                'regime': regime
                            })
                            break
        
        return trades

# ============================================================================
# DATA LOADER
# ============================================================================

def load_and_prepare_data(asset, timeframe):
    """Load and prepare data with indicators"""
    
    # Try to load from features first
    feature_file = FEATURES_DIR / f"{asset}_USDT_{timeframe}_features.parquet"
    if feature_file.exists():
        try:
            df = pd.read_parquet(feature_file)
            # Check if has required columns
            required = ['close', 'high', 'low', 'volume', 'rsi', 'atr']
            if all(col in df.columns for col in required):
                return df
        except:
            pass
    
    # Load from raw
    raw_file = DATA_DIR / f"{asset}_USDT_{timeframe}.parquet"
    if not raw_file.exists():
        raw_file = DATA_DIR / f"{asset}_USDT_{timeframe}_FULL.parquet"
    
    if not raw_file.exists():
        return None
    
    df = pd.read_parquet(raw_file)
    
    # Calculate indicators
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
    df = df.dropna()
    
    return df

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_comprehensive_metrics(trades, df):
    """Calculate all relevant metrics"""
    
    if not trades:
        return None
    
    pnls = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    
    # Basic metrics
    total_pnl_pct = sum(pnls) * 100
    win_rate = len(winners) / len(trades) if trades else 0
    
    # Costs
    fees_per_trade = 0.001 * 2  # 0.1% each side
    slippage_per_trade = 0.0005 * 2  # 0.05% each side
    total_cost_per_trade = fees_per_trade + slippage_per_trade
    
    gross_pnl_pct = total_pnl_pct
    costs_pct = len(trades) * total_cost_per_trade * 100
    net_pnl_pct = gross_pnl_pct - costs_pct
    
    # Time-based returns
    period_years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = net_pnl_pct / period_years if period_years > 0 else 0
    monthly_return = annual_return / 12
    
    # Risk metrics
    if losers:
        avg_loss = np.mean(losers)
        max_loss = min(losers)
    else:
        avg_loss = 0
        max_loss = 0
    
    if winners:
        avg_win = np.mean(winners)
        max_win = max(winners)
        profit_factor = sum(winners) / abs(sum(losers)) if losers else 999
    else:
        avg_win = 0
        max_win = 0
        profit_factor = 0
    
    # Sharpe (simplified)
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    else:
        sharpe = 0
    
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'gross_pnl': gross_pnl_pct,
        'costs': costs_pct,
        'net_pnl': net_pnl_pct,
        'annual_return': annual_return,
        'monthly_return': monthly_return,
        'avg_win': avg_win * 100,
        'avg_loss': avg_loss * 100,
        'max_win': max_win * 100,
        'max_loss': max_loss * 100,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'period_years': period_years
    }

# ============================================================================
# MASTER AUDIT EXECUTION
# ============================================================================

print("\n" + "="*100)
print("STARTING MASTER AUDIT")
print("="*100)

results = []
strategy = HybridStrategy()

total_combinations = len(ASSETS) * len(TIMEFRAMES)
completed = 0

for asset in ASSETS:
    for timeframe in TIMEFRAMES:
        completed += 1
        
        print(f"\n[{completed}/{total_combinations}] Testing {asset} {timeframe}...")
        
        # Load data
        df = load_and_prepare_data(asset, timeframe)
        
        if df is None or len(df) < 500:
            print(f"   ⚠️  Skipped (no data or insufficient)")
            results.append({
                'asset': asset,
                'timeframe': timeframe,
                'status': 'NO_DATA',
                'metrics': None
            })
            continue
        
        print(f"   📊 Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        
        # Generate signals
        signals = strategy.generate_signals(df)
        print(f"   🎯 Generated {len(signals)} signals")
        
        if len(signals) < 10:
            print(f"   ⚠️  Skipped (too few signals)")
            results.append({
                'asset': asset,
                'timeframe': timeframe,
                'status': 'TOO_FEW_SIGNALS',
                'signals': len(signals),
                'metrics': None
            })
            continue
        
        # Backtest
        trades = strategy.backtest(df, signals)
        print(f"   💰 Executed {len(trades)} trades")
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(trades, df)
        
        if metrics:
            print(f"   ✅ Win Rate: {metrics['win_rate']:.1%}, Annual: {metrics['annual_return']:.1f}%")
            
            results.append({
                'asset': asset,
                'timeframe': timeframe,
                'status': 'SUCCESS',
                'signals': len(signals),
                'metrics': metrics
            })
        else:
            print(f"   ❌ Failed to calculate metrics")
            results.append({
                'asset': asset,
                'timeframe': timeframe,
                'status': 'METRICS_FAILED',
                'signals': len(signals),
                'metrics': None
            })

# ============================================================================
# GENERATE COMPREHENSIVE REPORT
# ============================================================================

print("\n" + "="*100)
print("GENERATING COMPREHENSIVE REPORT")
print("="*100)

# Save raw results
with open(OUTPUT_DIR / 'raw_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Create matrix
successful_results = [r for r in results if r['metrics'] is not None]

print(f"\n✅ Successfully tested: {len(successful_results)}/{total_combinations} combinations")

# Best performers
if successful_results:
    sorted_by_annual = sorted(successful_results, key=lambda x: x['metrics']['annual_return'], reverse=True)
    
    print("\n" + "="*100)
    print("TOP 10 BEST PERFORMERS (by annual return)")
    print("="*100)
    print(f"\n{'Asset':<8} {'TF':<6} {'Trades':>8} {'Win%':>8} {'Annual':>10} {'Monthly':>10} {'Sharpe':>8}")
    print("-"*70)
    
    for r in sorted_by_annual[:10]:
        m = r['metrics']
        print(f"{r['asset']:<8} {r['timeframe']:<6} {m['trades']:>8} {m['win_rate']:>7.1%} "
              f"{m['annual_return']:>9.1f}% {m['monthly_return']:>9.1f}% {m['sharpe']:>7.2f}")
    
    # Save detailed report
    report_lines = []
    report_lines.append("="*100)
    report_lines.append("MASTER COMPREHENSIVE AUDIT - COMPLETE RESULTS")
    report_lines.append("="*100)
    report_lines.append(f"\nGenerated: {datetime.now()}")
    report_lines.append(f"Total combinations tested: {len(successful_results)}/{total_combinations}")
    report_lines.append("\n" + "="*100)
    report_lines.append("DETAILED RESULTS BY ASSET")
    report_lines.append("="*100)
    
    for asset in ASSETS:
        asset_results = [r for r in successful_results if r['asset'] == asset]
        if not asset_results:
            continue
        
        report_lines.append(f"\n### {asset} ###")
        report_lines.append(f"{'TF':<6} {'Trades':>8} {'Win%':>8} {'Annual':>10} {'Monthly':>10} {'Sharpe':>8} {'PF':>8}")
        report_lines.append("-"*70)
        
        for r in sorted(asset_results, key=lambda x: x['metrics']['annual_return'], reverse=True):
            m = r['metrics']
            report_lines.append(
                f"{r['timeframe']:<6} {m['trades']:>8} {m['win_rate']:>7.1%} "
                f"{m['annual_return']:>9.1f}% {m['monthly_return']:>9.1f}% "
                f"{m['sharpe']:>7.2f} {m['profit_factor']:>7.2f}"
            )
    
    # Portfolio potential
    report_lines.append("\n" + "="*100)
    report_lines.append("PORTFOLIO POTENTIAL")
    report_lines.append("="*100)
    
    profitable = [r for r in successful_results if r['metrics']['annual_return'] > 10]
    report_lines.append(f"\nProfitable combinations (>10% annual): {len(profitable)}")
    
    if profitable:
        avg_annual = np.mean([r['metrics']['annual_return'] for r in profitable])
        report_lines.append(f"Average annual return: {avg_annual:.1f}%")
        report_lines.append(f"\nWith portfolio of {len(profitable)} uncorrelated strategies:")
        report_lines.append(f"Expected portfolio return: {avg_annual * 0.7:.1f}% (assuming 0.7 correlation)")
    
    # Save report
    with open(OUTPUT_DIR / 'comprehensive_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("\n" + "="*100)
    print("✅ MASTER AUDIT COMPLETE")
    print("="*100)
    print(f"\nReports saved to: {OUTPUT_DIR}")
    print(f"- raw_results.json")
    print(f"- comprehensive_report.txt")
    print("="*100)

else:
    print("\n❌ No successful results to analyze")