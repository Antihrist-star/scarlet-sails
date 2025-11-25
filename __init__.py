"""
Scarlet Sails Backtesting Framework - Core Module
"""

from .data_loader import load_market_data, AVAILABLE_COINS, AVAILABLE_TIMEFRAMES
from .backtest_engine import BacktestEngine
from .metrics_calculator import MetricsCalculator
from .position_sizer import PositionSizer, RiskManager
from .trade_logger import TradeLogger, Trade

__all__ = [
    'load_market_data',
    'AVAILABLE_COINS',
    'AVAILABLE_TIMEFRAMES',
    'BacktestEngine',
    'MetricsCalculator',
    'PositionSizer',
    'RiskManager',
    'TradeLogger',
    'Trade',
]