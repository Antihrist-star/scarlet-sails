"""
SIMPLE BACKTESTER FOR RULE-BASED STRATEGY
Uses signal == 1 for BUY, simple exit logic

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
Version: Simple & Fast
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append('.')

from strategies.rule_based_v2 import RuleBasedStrategy


def prepare_data_for_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Map advanced features to classic indicators"""
    df = df.copy()
    
    if 'norm_rsi_pctile' in df.columns:
        df['RSI_14'] = df['norm_rsi_pctile'] * 100
    
    if 'norm_macd_zscore' in df.columns:
        df['MACD_12_26_9'] = df['norm_macd_zscore']
    
    if 'deriv_macd_diff1' in df.columns:
        df['MACDs_12_26_9'] = df['deriv_macd_diff1']
    
    if 'norm_bb_width_pctile' in df.columns and 'close' in df.columns:
        bb_width = df['norm_bb_width_pctile'] * df['close'] * 0.02
        df['BBM_20_2.0'] = df['close']
        df['BBL_20_2.0'] = df['close'] - bb_width
        df['BBU_20_2.0'] = df['close'] + bb_width
    
    if 'norm_atr_pctile' in df.columns and 'close' in df.columns:
        df['ATRr_14'] = df['norm_atr_pctile'] * df['close'] * 0.02
    
    return df


class SimpleBacktester:
    """Simple backtester for signal-based strategies"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0 = no position, 1 = long
        self.position_size = 0
        self.entry_price = 0
        self.entry_bar = 0
        
        self.equity_curve = []
        self.trades = []
        
        # Simple exit rules
        self.hold_period = 20  # Exit after 20 bars (5 hours on 15m)
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
    
    def backtest(self, df: pd.DataFrame, signals: pd.DataFrame):
        """
        Run backtest
        
        Parameters:
        -----------
        df : DataFrame with OHLCV data
        signals : DataFrame with 'signal' column (0 or 1)
        """
        print(f"Running backtest on {len(df)} bars...")
        
        # Align signals with data
        signals = signals.reindex(df.index, fill_value=0)
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            signal = signals.loc[timestamp, 'signal'] if timestamp in signals.index else 0
            
            # Check exit conditions if in position
            if self.position == 1:
                bars_held = i - self.entry_bar
                pnl_pct = (price / self.entry_price - 1)
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # 1. Hold period
                if bars_held >= self.hold_period:
                    should_exit = True
                    exit_reason = "hold_period"
                
                # 2. Stop loss
                elif pnl_pct <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # 3. Take profit
                elif pnl_pct >= self.take_profit_pct:
                    should_exit = True
                    exit_reason = "take_profit"
                
                if should_exit:
                    # Exit position
                    exit_value = self.position_size * price
                    pnl = exit_value - (self.position_size * self.entry_price)
                    
                    self.capital += exit_value
                    self.capital -= exit_value * 0.0015  # Commission
                    
                    self.trades.append({
                        'entry_time': df.index[self.entry_bar],
                        'exit_time': timestamp,
                        'entry_price': self.entry_price,
                        'exit_price': price,
                        'size': self.position_size,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'bars_held': bars_held,
                        'exit_reason': exit_reason
                    })
                    
                    self.position = 0
                    self.position_size = 0
                    self.entry_price = 0
                    self.entry_bar = 0
            
            # Check entry conditions if no position
            if self.position == 0 and signal == 1:
                # Enter position
                position_value = self.capital * 0.1  # 10% of capital per trade
                self.position_size = position_value / price
                self.entry_price = price
                self.entry_bar = i
                self.position = 1
                
                self.capital -= position_value * 0.0015  # Commission
            
            # Calculate equity
            if self.position == 1:
                equity = self.capital + (self.position_size * price)
            else:
                equity = self.capital
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'price': price,
                'position': self.position
            })
        
        # Close any open position at the end
        if self.position == 1:
            final_price = df.iloc[-1]['close']
            exit_value = self.position_size * final_price
            pnl = exit_value - (self.position_size * self.entry_price)
            
            self.capital += exit_value
            
            self.trades.append({
                'entry_time': df.index[self.entry_bar],
                'exit_time': df.index[-1],
                'entry_price': self.entry_price,
                'exit_price': final_price,
                'size': self.position_size,
                'pnl': pnl,
                'pnl_pct': (final_price / self.entry_price - 1) * 100,
                'bars_held': len(df) - self.entry_bar,
                'exit_reason': 'end_of_data'
            })
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        # Total return
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 96)  # 15min bars
        else:
            sharpe = 0.0
        
        # Max drawdown
        cummax = equity_df['equity'].cummax()
        drawdowns = (equity_df['equity'] - cummax) / cummax * 100
        max_drawdown = drawdowns.min()
        
        # Trade statistics
        if len(trades_df) > 0:
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(wins) / len(trades_df) * 100
            
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 0
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            avg_bars_held = trades_df['bars_held'].mean()
        else:
            win_rate = 0
            profit_factor = 0
            avg_bars_held = 0
        
        metrics = {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_bars_held': avg_bars_held,
            'final_capital': equity_df['equity'].iloc[-1]
        }
        
        return metrics, equity_df, trades_df


def main():
    """Run simple backtest"""
    print("="*80)
    print("SIMPLE BACKTESTER - RULE-BASED STRATEGY")
    print("="*80)
    print()
    
    # Load data
    print("STEP 1: LOADING DATA")
    print("-"*80)
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    df = pd.read_parquet(data_file)
    
    # Use 2024 data
    test_df = df[df.index >= '2024-01-01'].copy()
    test_df = prepare_data_for_strategy(test_df)
    
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Bars: {len(test_df)}")
    print()
    
    # Generate signals
    print("STEP 2: GENERATING SIGNALS")
    print("-"*80)
    strategy = RuleBasedStrategy()
    signals = strategy.generate_signals(test_df)
    
    buy_signals = (signals['signal'] == 1).sum()
    print(f"Buy signals: {buy_signals}")
    print()
    
    # Run backtest
    print("STEP 3: RUNNING BACKTEST")
    print("-"*80)
    backtester = SimpleBacktester(initial_capital=10000)
    backtester.backtest(test_df, signals)
    
    # Calculate metrics
    metrics, equity_df, trades_df = backtester.calculate_metrics()
    
    print()
    print("="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print()
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Avg Hold Period: {metrics['avg_bars_held']:.1f} bars")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print()
    
    # Compare with Buy & Hold
    bh_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1) * 100
    print("COMPARISON WITH BUY & HOLD:")
    print("-"*80)
    print(f"Buy & Hold Return: {bh_return:.2f}%")
    print(f"Strategy Return: {metrics['total_return_pct']:.2f}%")
    print(f"Outperformance: {metrics['total_return_pct'] - bh_return:.2f}%")
    print()
    
    # Trade breakdown
    if len(trades_df) > 0:
        print("TRADE BREAKDOWN:")
        print("-"*80)
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}")
        print()
        
        print("SAMPLE TRADES:")
        print(trades_df[['entry_time', 'exit_price', 'pnl_pct', 'exit_reason']].head(10).to_string())
    
    print()
    print("="*80)
    print("✅ BACKTEST COMPLETE!")
    print("="*80)
    
    # Save results
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save trades
    if len(trades_df) > 0:
        trades_df.to_csv(output_dir / 'rule_based_trades.csv', index=False)
        print(f"\nTrades saved to: {output_dir / 'rule_based_trades.csv'}")
    
    # Save equity curve
    equity_df.to_csv(output_dir / 'rule_based_equity.csv', index=False)
    print(f"Equity curve saved to: {output_dir / 'rule_based_equity.csv'}")
    
    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Equity curve
    ax1.plot(equity_df['timestamp'], equity_df['equity'], label='Strategy', linewidth=2)
    ax1.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Rule-Based Strategy - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Price with positions
    ax2.plot(test_df.index, test_df['close'], label='BTC Price', alpha=0.7)
    
    # Mark entry points
    if len(trades_df) > 0:
        entries = trades_df['entry_time']
        entry_prices = trades_df['entry_price']
        ax2.scatter(entries, entry_prices, color='green', marker='^', s=100, label='Buy', zorder=5)
    
    ax2.set_title('BTC Price with Entry Points', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rule_based_backtest.png', dpi=150)
    print(f"Chart saved to: {output_dir / 'rule_based_backtest.png'}")
    
    print()


if __name__ == "__main__":
    main()