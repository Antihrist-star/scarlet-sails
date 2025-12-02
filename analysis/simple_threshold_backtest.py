"""
Simple Threshold Backtest Module
Evaluates trading thresholds using pre-computed fee-adjusted returns.
No look-ahead bias - uses only available data at decision time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 35040) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year (default: 35040 for 15m bars)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown percentage.
    
    Args:
        cumulative_returns: Cumulative return series
    
    Returns:
        Maximum drawdown as percentage
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    cumulative_wealth = np.cumprod(1 + cumulative_returns)
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_dd = np.min(drawdown) * 100  # Convert to percentage
    return float(max_dd)


def evaluate_threshold(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    threshold: float
) -> Dict:
    """
    Evaluate a single threshold on the given dataset.
    
    Args:
        df: DataFrame with predictions and fee-adjusted returns
        proba_col: Name of column with model probabilities
        fee_ret_col: Name of column with fee-adjusted returns
        threshold: Probability threshold for trade signal
    
    Returns:
        Dictionary with performance metrics
    """
    # Generate signals based on threshold
    signals = (df[proba_col] >= threshold).astype(int)
    
    # Select only trades where signal = 1
    trade_mask = signals == 1
    trade_returns = df.loc[trade_mask, fee_ret_col].values
    
    if len(trade_returns) == 0:
        return {
            'threshold': threshold,
            'n_trades': 0,
            'total_return': 0.0,
            'mean_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0,
        }
    
    # Calculate metrics
    total_return = np.sum(trade_returns)
    mean_return = np.mean(trade_returns)
    sharpe = calculate_sharpe_ratio(trade_returns)
    max_dd = calculate_max_drawdown(trade_returns)
    win_rate = np.mean(trade_returns > 0) * 100
    
    return {
        'threshold': threshold,
        'n_trades': int(len(trade_returns)),
        'total_return': float(total_return),
        'mean_return': float(mean_return),
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'win_rate': float(win_rate),
    }


def evaluate_thresholds(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    thresholds: List[float]
) -> Dict[float, Dict]:
    """
    Evaluate multiple thresholds on the given dataset.
    
    Args:
        df: DataFrame with predictions and fee-adjusted returns
        proba_col: Name of column with model probabilities
        fee_ret_col: Name of column with fee-adjusted returns
        thresholds: List of probability thresholds to evaluate
    
    Returns:
        Dictionary mapping threshold to performance metrics
    """
    results = {}
    
    for threshold in thresholds:
        results[threshold] = evaluate_threshold(
            df=df,
            proba_col=proba_col,
            fee_ret_col=fee_ret_col,
            threshold=threshold
        )
    
    return results


def select_optimal_threshold(
    threshold_results: Dict[float, Dict],
    max_dd_limit: float = 20.0,
    min_trades: int = 10
) -> Dict:
    """
    Select optimal threshold based on Sharpe ratio with constraints.
    
    Args:
        threshold_results: Results from evaluate_thresholds
        max_dd_limit: Maximum allowed drawdown percentage
        min_trades: Minimum number of trades required
    
    Returns:
        Dictionary with optimal threshold and its metrics
    """
    best_threshold = None
    best_sharpe = -np.inf
    best_metrics = {}
    
    for threshold, metrics in threshold_results.items():
        # Check constraints
        if metrics['n_trades'] < min_trades:
            continue
        if metrics['max_drawdown_pct'] < -max_dd_limit:
            continue
        
        # Update best if Sharpe is higher
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_threshold = threshold
            best_metrics = metrics
    
    if best_threshold is None:
        # Fallback: select threshold with most trades if no valid candidates
        best_threshold = max(threshold_results.keys(), 
                            key=lambda t: threshold_results[t]['n_trades'])
        best_metrics = threshold_results[best_threshold]
        best_sharpe = best_metrics['sharpe_ratio']
    
    return {
        'threshold': best_threshold,
        'sharpe': best_sharpe,
        'backtest_metrics': best_metrics,
    }


if __name__ == "__main__":
    # Simple test
    print("Simple Threshold Backtest Module")
    print("Usage:")
    print("  from analysis.simple_threshold_backtest import evaluate_thresholds, select_optimal_threshold")
    print("  results = evaluate_thresholds(df, 'proba', 'fee_ret', [0.5, 0.6, 0.7, 0.8, 0.9])")
    print("  optimal = select_optimal_threshold(results, max_dd_limit=20.0)")
