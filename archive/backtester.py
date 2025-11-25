"""
BACKTESTER - SCARLET SAILS
Comprehensive backtesting framework for all trading strategies

Features:
- Compares all 4 strategies (Rule-Based, XGBoost, Hybrid, DQN)
- Uses Orchestrator for unified management
- Calculates key metrics: PnL, Sharpe, Drawdown, Win Rate
- Creates visualizations: Equity curve, Trades timeline
- Exports results to CSV

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import strategies
from strategies.rule_based_v2 import RuleBasedStrategy
from strategies.xgboost_ml_v2 import XGBoostMLStrategy
from strategies.hybrid_v2 import HybridStrategy
from rl.dqn import DQNAgent
from rl.trading_environment import TradingEnvironment

# Import orchestrator
from orchestrator import StrategyOrchestrator, Portfolio, RiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
import warnings
warnings.filterwarnings('ignore')


def generate_backtest_data(n_bars=2000, seed=42):
    """
    Generate realistic market data for backtesting
    
    Parameters:
    -----------
    n_bars : int
        Number of bars
    seed : int
        Random seed
    
    Returns:
    --------
    DataFrame : OHLCV data
    """
    np.random.seed(seed)
    
    logger.info(f"Generating {n_bars} bars of backtest data...")
    
    # Multiple market regimes
    regimes = [
        (500, 0.0003, 0.015, "Bull"),
        (500, -0.0002, 0.025, "Bear"),
        (500, 0.0002, 0.018, "Recovery"),
        (500, -0.0001, 0.030, "Sideways")
    ]
    
    close_prices = [50000]
    
    for regime_len, trend, vol, name in regimes:
        for _ in range(regime_len):
            ret = np.random.normal(trend, vol)
            new_price = close_prices[-1] * (1 + ret)
            close_prices.append(new_price)
    
    close_prices = np.array(close_prices[:n_bars])
    close_prices = np.maximum(close_prices, 1000)
    
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='h')
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.0003, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.001, 0.002, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.001, 0.002, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)
    
    logger.info(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def extract_dqn_state(data, current_idx):
    """
    Extract state for DQN agent
    
    Parameters:
    -----------
    data : DataFrame
        Market data
    current_idx : int
        Current bar index
    
    Returns:
    --------
    np.array : State vector (12 features)
    """
    if current_idx < 50:
        return np.zeros(12)
    
    window = data.iloc[current_idx-50:current_idx+1]
    
    # Calculate features
    close = window['close'].iloc[-1]
    price_norm = (close - window['close'].mean()) / window['close'].std() if window['close'].std() > 0 else 0
    
    # RSI
    delta = window['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss if loss.iloc[-1] != 0 else 0
    rsi = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50
    
    # MACD
    ema12 = window['close'].ewm(span=12).mean().iloc[-1]
    ema26 = window['close'].ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    signal_line = window['close'].ewm(span=9).mean().iloc[-1]
    
    # Bollinger Bands
    sma20 = window['close'].rolling(20).mean().iloc[-1]
    std20 = window['close'].rolling(20).std().iloc[-1]
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # Volume
    volume_ratio = window['volume'].iloc[-1] / window['volume'].mean() if window['volume'].mean() > 0 else 1
    
    # Volatility
    volatility = window['close'].pct_change().std()
    
    # Momentum
    momentum = (window['close'].iloc[-1] / window['close'].iloc[0] - 1) if window['close'].iloc[0] > 0 else 0
    
    # Trend strength (ADX-like)
    trend_strength = abs(window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0] if window['close'].iloc[0] > 0 else 0
    
    state = np.array([
        price_norm,
        rsi / 100,
        macd / close if close > 0 else 0,
        signal_line / close if close > 0 else 0,
        bb_upper / close if close > 0 else 0,
        bb_lower / close if close > 0 else 0,
        min(volume_ratio, 5) / 5,  # Cap at 5x
        min(volatility, 0.1) / 0.1,  # Cap at 10%
        min(abs(momentum), 0.5) / 0.5,  # Cap at 50%
        min(trend_strength, 0.5) / 0.5,  # Cap at 50%
        0,  # position_status (0 = no position)
        0   # current_pnl (normalized)
    ])
    
    return state


def run_simple_backtest(strategy, data, strategy_name, initial_capital=10000):
    """
    Run simple backtest for a single strategy
    
    Parameters:
    -----------
    strategy : Strategy object
        Strategy to test
    data : DataFrame
        Market data
    strategy_name : str
        Name of strategy
    initial_capital : float
        Starting capital
    
    Returns:
    --------
    dict : Backtest results
    """
    logger.info(f"Running backtest: {strategy_name}")
    
    # Initialize portfolio
    portfolio = Portfolio(initial_capital)
    position = None
    trades = []
    equity_curve = []
    
    symbol = 'BTC/USDT'
    
    # Run through data
    for i in range(50, len(data)):
        current_time = data.index[i]
        current_price = data['close'].iloc[i]
        window_data = data.iloc[:i+1]
        
        # Get signal - handle different strategy interfaces
        signal = 0
        confidence = 0
        signal_result = None
        
        try:
            # Try different method names (check most common first)
            if hasattr(strategy, 'generate_signals'):  # PLURAL! Most strategies use this
                signal_result = strategy.generate_signals(window_data)
            elif hasattr(strategy, 'generate_signal'):  # Singular
                signal_result = strategy.generate_signal(window_data)
            elif hasattr(strategy, 'get_signal'):
                signal_result = strategy.get_signal(window_data)
            elif hasattr(strategy, 'select_action'):
                # DQN uses select_action with state
                state = extract_dqn_state(data, i)
                action = strategy.select_action(state, training=False)
                # Convert action to signal: 0=sell, 1=buy, 2=hold
                if action == 1:
                    signal = 1
                elif action == 0:
                    signal = -1
                else:
                    signal = 0
                confidence = 0.5
                # Already processed, skip parsing below
                continue
            else:
                # Strategy doesn't have recognized interface
                if i == 50:
                    methods = [m for m in dir(strategy) if not m.startswith('_')]
                    logger.warning(f"Strategy {strategy_name} has no signal method. Methods: {', '.join(methods[:10])}")
                # Skip this bar - no signal
                signal = 0
                confidence = 0
            
            # Parse signal result (if not already processed by DQN)
            if signal_result is not None:
                if signal_result is None or signal_result == 0:
                    signal = 0
                    confidence = 0
                elif isinstance(signal_result, dict):
                    signal = signal_result.get('signal', 0)
                    confidence = signal_result.get('confidence', 0.5)
                elif isinstance(signal_result, (int, float)):
                    # Numeric signal
                    if signal_result > 0:
                        signal = 1
                    elif signal_result < 0:
                        signal = -1
                    else:
                        signal = 0
                    confidence = 0.5
                else:
                    signal = int(signal_result) if signal_result else 0
                    confidence = 0.5
            
        except Exception as e:
            # Only log error once per strategy, not every bar
            if i == 50:
                logger.warning(f"Strategy {strategy_name} error: {e}")
            signal = 0
            confidence = 0
        
        # Execute trades
        if signal == 1 and position is None:
            # Buy signal
            position_value = portfolio.cash * 0.1  # 10% of cash
            size = position_value / current_price
            
            success = portfolio.open_position(
                symbol=symbol,
                side='long',
                size=size,
                price=current_price,
                timestamp=current_time
            )
            
            if success:
                position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'size': size
                }
        
        elif signal == -1 and position is not None:
            # Sell signal
            pnl = portfolio.close_position(symbol, current_price, current_time)
            
            if pnl is not None:
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': (current_price / position['entry_price'] - 1) * 100
                })
                position = None
        
        # Record equity
        current_prices = {symbol: current_price}
        portfolio.record_equity(current_time, current_prices)
        equity_curve.append(portfolio.equity(current_prices))
    
    # Close any open position
    if position is not None:
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        pnl = portfolio.close_position(symbol, current_price, current_time)
        if pnl is not None:
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'return_pct': (current_price / position['entry_price'] - 1) * 100
            })
    
    # Calculate metrics
    metrics = calculate_metrics(
        initial_capital=initial_capital,
        equity_curve=equity_curve,
        trades=trades
    )
    
    return {
        'strategy_name': strategy_name,
        'metrics': metrics,
        'trades': trades,
        'equity_curve': equity_curve
    }


def calculate_metrics(initial_capital, equity_curve, trades):
    """
    Calculate performance metrics
    
    Returns:
    --------
    dict : Performance metrics
    """
    if not equity_curve:
        return {
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'total_trades': 0,
            'win_rate_pct': 0,
            'avg_trade_pct': 0,
            'profit_factor': 0,
            'final_equity': initial_capital
        }
    
    equity_array = np.array(equity_curve)
    
    # Total return
    total_return = (equity_array[-1] / initial_capital - 1) * 100
    
    # Sharpe ratio (annualized, assuming hourly data)
    returns = np.diff(equity_array) / equity_array[:-1]
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)
    else:
        sharpe = 0.0
    
    # Maximum drawdown
    cummax = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array - cummax) / cummax * 100
    max_drawdown = drawdowns.min()
    
    # Trade statistics
    total_trades = len(trades)
    
    if total_trades > 0:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        avg_trade = np.mean([t['return_pct'] for t in trades])
        
        # Profit factor
        total_wins = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
    else:
        win_rate = 0
        avg_trade = 0
        profit_factor = 0
    
    return {
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'total_trades': total_trades,
        'win_rate_pct': win_rate,
        'avg_trade_pct': avg_trade,
        'profit_factor': profit_factor,
        'final_equity': equity_array[-1]
    }


def create_visualizations(results, output_dir='backtest_results'):
    """
    Create visualization charts
    
    Parameters:
    -----------
    results : list
        List of backtest results
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Creating visualizations...")
    
    # 1. Equity curves comparison
    plt.figure(figsize=(14, 8))
    
    for result in results:
        strategy_name = result['strategy_name']
        equity_curve = result['equity_curve']
        plt.plot(equity_curve, label=strategy_name, linewidth=2)
    
    plt.title('Equity Curves Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Time (bars)', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'equity_curves.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"  Saved: {filepath}")
    
    # 2. Metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics_to_plot = [
        ('total_return_pct', 'Total Return (%)', 0),
        ('sharpe_ratio', 'Sharpe Ratio', 1),
        ('max_drawdown_pct', 'Max Drawdown (%)', 2),
        ('total_trades', 'Total Trades', 3),
        ('win_rate_pct', 'Win Rate (%)', 4),
        ('profit_factor', 'Profit Factor', 5)
    ]
    
    for metric_key, metric_name, idx in metrics_to_plot:
        ax = axes[idx // 3, idx % 3]
        
        strategies = [r['strategy_name'] for r in results]
        values = [r['metrics'][metric_key] for r in results]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ax.bar(strategies, values, color=colors[:len(strategies)])
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"  Saved: {filepath}")
    
    logger.info("Visualizations complete!")


def export_results(results, output_dir='backtest_results'):
    """
    Export results to CSV
    
    Parameters:
    -----------
    results : list
        List of backtest results
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Exporting results to CSV...")
    
    # Metrics summary
    metrics_data = []
    for result in results:
        row = {'Strategy': result['strategy_name']}
        row.update(result['metrics'])
        metrics_data.append(row)
    
    df_metrics = pd.DataFrame(metrics_data)
    filepath = os.path.join(output_dir, 'backtest_metrics.csv')
    df_metrics.to_csv(filepath, index=False)
    logger.info(f"  Saved: {filepath}")
    
    # All trades
    all_trades = []
    for result in results:
        for trade in result['trades']:
            trade['strategy'] = result['strategy_name']
            all_trades.append(trade)
    
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        filepath = os.path.join(output_dir, 'all_trades.csv')
        df_trades.to_csv(filepath, index=False)
        logger.info(f"  Saved: {filepath}")


def main():
    """
    Main backtesting pipeline
    """
    print("\n" + "="*80)
    print("SCARLET SAILS - BACKTESTING FRAMEWORK")
    print("="*80)
    print()
    
    # Generate data
    print("STEP 1: DATA PREPARATION")
    print("-"*80)
    data = generate_backtest_data(n_bars=2000)
    print(f"✅ Data ready: {len(data)} bars")
    print()
    
    # Initialize strategies
    print("STEP 2: STRATEGY INITIALIZATION")
    print("-"*80)
    
    strategies = {
        'Rule-Based': RuleBasedStrategy(),
        'XGBoost ML': XGBoostMLStrategy(),
        'Hybrid': HybridStrategy(),
    }
    
    # Add DQN
    try:
        dqn_agent = DQNAgent(
            state_dim=12,
            action_dim=3,
            model_path='models/dqn_best_pnl.pth'
        )
        strategies['DQN RL'] = dqn_agent
        logger.info("✅ All 4 strategies initialized")
    except Exception as e:
        logger.warning(f"DQN initialization failed: {e}")
        logger.info("✅ 3 strategies initialized (no DQN)")
    
    print(f"✅ Strategies initialized: {list(strategies.keys())}")
    print()
    
    # Run backtests
    print("STEP 3: RUNNING BACKTESTS")
    print("-"*80)
    print("This may take a few minutes...")
    print()
    
    results = []
    for name, strategy in strategies.items():
        result = run_simple_backtest(strategy, data, name, initial_capital=10000)
        results.append(result)
        
        # Print summary
        metrics = result['metrics']
        print(f"\n{name}:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    print()
    
    # Create visualizations
    print("STEP 4: CREATING VISUALIZATIONS")
    print("-"*80)
    create_visualizations(results)
    print()
    
    # Export results
    print("STEP 5: EXPORTING RESULTS")
    print("-"*80)
    export_results(results)
    print()
    
    # Summary
    print("="*80)
    print("✅ BACKTESTING COMPLETE!")
    print("="*80)
    print()
    print("Results saved to: backtest_results/")
    print("  - equity_curves.png")
    print("  - metrics_comparison.png")
    print("  - backtest_metrics.csv")
    print("  - all_trades.csv")
    print()
    print("Next steps:")
    print("  1. Review results in backtest_results/")
    print("  2. Analyze best performing strategy")
    print("  3. Push to GitHub")
    print()


if __name__ == "__main__":
    main()