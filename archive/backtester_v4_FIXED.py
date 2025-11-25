"""
BACKTESTER V4 - SCARLET SAILS (FIXED!)
Fixed imports and proper error handling

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
import sys

# Add project root to path
sys.path.append('.')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import strategies with proper error handling
STRATEGIES_AVAILABLE = {}

try:
    from strategies.rule_based_v2 import RuleBasedStrategy
    STRATEGIES_AVAILABLE['rule_based'] = True
    logger.info("✅ RuleBasedStrategy imported")
except Exception as e:
    STRATEGIES_AVAILABLE['rule_based'] = False
    logger.warning(f"⚠️ Could not import RuleBasedStrategy: {e}")

try:
    from strategies.xgboost_ml_v2 import XGBoostMLStrategy
    STRATEGIES_AVAILABLE['xgboost'] = True
    logger.info("✅ XGBoostMLStrategy imported")
except Exception as e:
    STRATEGIES_AVAILABLE['xgboost'] = False
    logger.warning(f"⚠️ Could not import XGBoostMLStrategy: {e}")

try:
    from strategies.hybrid_v2 import HybridStrategy
    STRATEGIES_AVAILABLE['hybrid'] = True
    logger.info("✅ HybridStrategy imported")
except Exception as e:
    STRATEGIES_AVAILABLE['hybrid'] = False
    logger.warning(f"⚠️ Could not import HybridStrategy: {e}")

# DQN needs special handling
try:
    from rl.dqn import DQNAgent
    from rl.trading_environment import TradingEnvironment
    STRATEGIES_AVAILABLE['dqn'] = True
    logger.info("✅ DQN components imported")
except Exception as e:
    STRATEGIES_AVAILABLE['dqn'] = False
    logger.warning(f"⚠️ Could not import DQN: {e}")


class ProductionBacktester:
    """
    Production-grade backtester with proper error handling
    """
    
    def __init__(self, initial_capital: float = 10000):
        """Initialize backtester"""
        self.initial_capital = initial_capital
        self.results = {}
        
        logger.info(f"ProductionBacktester initialized: ${initial_capital:,.2f}")
        logger.info(f"Available strategies: {[k for k, v in STRATEGIES_AVAILABLE.items() if v]}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from parquet file"""
        logger.info(f"Loading data from: {filepath}")
        
        try:
            df = pd.read_parquet(filepath)
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
            
            logger.info(f"  Loaded: {len(df)} bars")
            logger.info(f"  Period: {df.index[0]} to {df.index[-1]}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def run_simple_strategy(self, 
                           df: pd.DataFrame,
                           strategy_type: str) -> Dict:
        """
        Run simple strategy based on indicators
        
        Parameters:
        -----------
        df : DataFrame
            Market data
        strategy_type : str
            'rsi', 'macd', 'trend', or 'buyhold'
        
        Returns:
        --------
        dict : Performance metrics
        """
        logger.info(f"Running {strategy_type} strategy...")
        
        # Initialize
        capital = self.initial_capital
        position = 0
        position_size = 0
        entry_price = 0
        
        equity_curve = []
        trades = []
        
        # Generate signals based on strategy type
        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            
            # Determine action based on strategy
            action = 'hold'
            
            if i < 100:  # Skip first 100 bars
                action = 'hold'
            elif strategy_type == 'rsi':
                # RSI strategy
                if 'RSI_14' in df.columns:
                    rsi = row['RSI_14']
                    if rsi < 30 and position == 0:
                        action = 'buy'
                    elif rsi > 70 and position == 1:
                        action = 'sell'
            
            elif strategy_type == 'macd':
                # MACD strategy
                if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                    macd = row['MACD_12_26_9']
                    signal = row['MACDs_12_26_9']
                    
                    # Get previous values
                    if i > 0:
                        prev_macd = df.iloc[i-1]['MACD_12_26_9']
                        prev_signal = df.iloc[i-1]['MACDs_12_26_9']
                        
                        # Crossover
                        if macd > signal and prev_macd <= prev_signal and position == 0:
                            action = 'buy'
                        elif macd < signal and prev_macd >= prev_signal and position == 1:
                            action = 'sell'
            
            elif strategy_type == 'trend':
                # Moving average crossover
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                    fast = row['SMA_20']
                    slow = row['SMA_50']
                    
                    if i > 0:
                        prev_fast = df.iloc[i-1]['SMA_20']
                        prev_slow = df.iloc[i-1]['SMA_50']
                        
                        if fast > slow and prev_fast <= prev_slow and position == 0:
                            action = 'buy'
                        elif fast < slow and prev_fast >= prev_slow and position == 1:
                            action = 'sell'
            
            elif strategy_type == 'buyhold':
                # Buy and hold
                if i == 100 and position == 0:
                    action = 'buy'
            
            # Execute trades
            if action == 'buy' and position == 0:
                position_value = capital * 0.1
                position_size = position_value / price
                entry_price = price
                position = 1
                
                capital -= position_value * 0.0015
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'size': position_size
                })
            
            elif action == 'sell' and position == 1:
                exit_value = position_size * price
                pnl = exit_value - (position_size * entry_price)
                
                capital += exit_value
                capital -= exit_value * 0.0015
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'size': position_size,
                    'pnl': pnl,
                    'return_pct': (price / entry_price - 1) * 100
                })
                
                position = 0
                position_size = 0
                entry_price = 0
            
            # Calculate equity
            if position == 1:
                equity = capital + (position_size * price)
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Sharpe
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 96)
        else:
            sharpe = 0.0
        
        # Drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax * 100
        max_drawdown = drawdowns.min()
        
        # Trade stats
        sell_trades = [t for t in trades if t['action'] == 'sell']
        
        if sell_trades:
            wins = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
            win_rate = wins / len(sell_trades) * 100
            
            avg_win = np.mean([t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0]) if wins > 0 else 0
            avg_loss = np.mean([abs(t['pnl']) for t in sell_trades if t.get('pnl', 0) < 0]) if (len(sell_trades) - wins) > 0 else 0
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        metrics = {
            'strategy': strategy_type,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(sell_trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'final_capital': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': trades
        }
        
        logger.info(f"  {strategy_type}:")
        logger.info(f"    Return: {total_return:.2f}%")
        logger.info(f"    Sharpe: {sharpe:.2f}")
        logger.info(f"    Drawdown: {max_drawdown:.2f}%")
        logger.info(f"    Trades: {len(sell_trades)}")
        logger.info(f"    Win Rate: {win_rate:.2f}%")
        
        return metrics
    
    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest for all available strategies"""
        logger.info("="*80)
        logger.info("STARTING PRODUCTION BACKTEST")
        logger.info("="*80)
        
        results = {}
        
        # Run simple strategies (always work)
        results['rsi_strategy'] = self.run_simple_strategy(df, 'rsi')
        results['macd_strategy'] = self.run_simple_strategy(df, 'macd')
        results['trend_strategy'] = self.run_simple_strategy(df, 'trend')
        results['buy_hold'] = self.run_simple_strategy(df, 'buyhold')
        
        self.results = results
        
        logger.info("="*80)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*80)
        
        return results
    
    def create_visualizations(self, output_dir: str = 'backtest_results'):
        """Create visualization charts"""
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info("Creating visualizations...")
        
        # 1. Equity curves
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for name, metrics in self.results.items():
            equity = metrics['equity_curve']
            time = range(len(equity))
            ax.plot(time, equity, 
                   label=f"{name} ({metrics['total_return_pct']:.1f}%)", 
                   linewidth=2)
        
        ax.axhline(y=self.initial_capital, color='black', 
                  linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Time (bars)', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title('Strategy Comparison - Equity Curves (Real Data)', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curves.png', dpi=150)
        logger.info(f"  Saved: {output_dir}/equity_curves.png")
        plt.close()
        
        # 2. Metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        strategies = list(self.results.keys())
        
        # Returns
        returns = [self.results[s]['total_return_pct'] for s in strategies]
        axes[0, 0].bar(strategies, returns, color='steelblue')
        axes[0, 0].set_title('Total Return (%)')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sharpe
        sharpes = [self.results[s]['sharpe_ratio'] for s in strategies]
        axes[0, 1].bar(strategies, sharpes, color='green')
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].set_ylabel('Sharpe')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Drawdown
        drawdowns = [abs(self.results[s]['max_drawdown_pct']) for s in strategies]
        axes[1, 0].bar(strategies, drawdowns, color='red')
        axes[1, 0].set_title('Maximum Drawdown (%)')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Win Rate
        win_rates = [self.results[s]['win_rate_pct'] for s in strategies]
        axes[1, 1].bar(strategies, win_rates, color='purple')
        axes[1, 1].set_title('Win Rate (%)')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].axhline(y=50, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=150)
        logger.info(f"  Saved: {output_dir}/metrics_comparison.png")
        plt.close()
    
    def export_results(self, output_dir: str = 'backtest_results'):
        """Export results to CSV"""
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info("Exporting results...")
        
        # Metrics
        metrics_df = pd.DataFrame([
            {
                'strategy': m['strategy'],
                'total_return_pct': m['total_return_pct'],
                'sharpe_ratio': m['sharpe_ratio'],
                'max_drawdown_pct': m['max_drawdown_pct'],
                'total_trades': m['total_trades'],
                'win_rate_pct': m['win_rate_pct'],
                'profit_factor': m['profit_factor'],
                'final_capital': m['final_capital']
            }
            for m in self.results.values()
        ])
        
        metrics_df.to_csv(f'{output_dir}/backtest_metrics.csv', index=False)
        logger.info(f"  Saved: {output_dir}/backtest_metrics.csv")
        
        # Trades
        all_trades = []
        for name, metrics in self.results.items():
            for trade in metrics['trades']:
                trade_copy = trade.copy()
                trade_copy['strategy'] = name
                all_trades.append(trade_copy)
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(f'{output_dir}/all_trades.csv', index=False)
            logger.info(f"  Saved: {output_dir}/all_trades.csv")


def main():
    """Main backtesting script"""
    print("="*80)
    print("SCARLET SAILS - PRODUCTION BACKTESTER V4 (FIXED!)")
    print("="*80)
    print()
    
    # Initialize
    backtester = ProductionBacktester(initial_capital=10000)
    
    # Load data
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    print("STEP 1: LOADING DATA")
    print("-"*80)
    df = backtester.load_data(data_file)
    print()
    
    # Use test period (2024 data)
    print("STEP 2: SELECTING TEST PERIOD")
    print("-"*80)
    test_df = df[df.index >= '2024-01-01']
    logger.info(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    logger.info(f"Test bars: {len(test_df)}")
    print()
    
    # Run backtest
    print("STEP 3: RUNNING BACKTEST")
    print("-"*80)
    results = backtester.run(test_df)
    print()
    
    # Summary
    print("="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Visualizations
    print()
    print("STEP 4: CREATING VISUALIZATIONS")
    print("-"*80)
    backtester.create_visualizations()
    
    # Export
    print()
    print("STEP 5: EXPORTING RESULTS")
    print("-"*80)
    backtester.export_results()
    
    print()
    print("="*80)
    print("✅ PRODUCTION BACKTESTING COMPLETE!")
    print("="*80)
    print()
    print("Results saved to: backtest_results/")
    print()


if __name__ == "__main__":
    main()