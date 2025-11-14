#!/usr/bin/env python3
"""
BACKTEST ALL COINS & TIMEFRAMES
================================

Запускает P_j(S) фреймворк на всех 14 монетах и 4 таймфреймах.
Результаты сохраняются в CSV для анализа.

ИСПОЛЬЗОВАНИЕ (Windows):
  python backtest_all_coins.py

ОЖИДАЕМОЕ ВРЕМЯ:
  ~5-10 минут для всех 56 комбинаций (14 монет × 4 таймфрейма)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from backtesting.backtest_pjs_framework import (
    PjSBacktestEngine,
    BacktestConfig,
    RuleBasedStrategy
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Все монеты (14)
COINS = [
    'BTC', 'ETH', 'SOL', 'AVAX', 'DOT', 'LINK', 'LTC',
    'ALGO', 'HBAR', 'LDO', 'ENA', 'ONDO', 'SUI', 'UNI'
]

# Все таймфреймы (4)
TIMEFRAMES = ['15m', '1h', '4h', '1d']

# Оптимальные параметры из ФАЗА 2A
OPTIMAL_TP = 0.03   # 3.0%
OPTIMAL_SL = 0.012  # 1.2%

# ============================================================================
# MAIN BACKTEST FUNCTION
# ============================================================================

def run_backtest_for_coin(coin: str, timeframe: str, data_dir: Path) -> dict:
    """Run backtest for single coin+timeframe combination"""

    # Определить имя файла
    filename = f"{coin}_USDT_{timeframe}.parquet"
    filepath = data_dir / filename

    # Проверить файл
    if not filepath.exists():
        logger.warning(f"⚠️  File not found: {filepath}")
        return {
            'coin': coin,
            'timeframe': timeframe,
            'status': 'MISSING',
            'trades': 0,
            'win_rate': 0,
            'pf': 0,
            'pnl': 0,
            'return': 0
        }

    try:
        # Загрузить данные
        ohlcv = pd.read_parquet(filepath)

        if len(ohlcv) < 200:
            logger.warning(f"⚠️  {coin}_{timeframe}: Too few bars ({len(ohlcv)})")
            return {
                'coin': coin,
                'timeframe': timeframe,
                'status': 'INSUFFICIENT_DATA',
                'trades': 0,
                'win_rate': 0,
                'pf': 0,
                'pnl': 0,
                'return': 0
            }

        # Генерировать сигналы
        strategy = RuleBasedStrategy(rsi_threshold=30, period=14)
        signals = strategy.generate_signals(ohlcv)

        signal_count = np.sum(signals)
        if signal_count == 0:
            logger.warning(f"⚠️  {coin}_{timeframe}: No signals generated")
            return {
                'coin': coin,
                'timeframe': timeframe,
                'status': 'NO_SIGNALS',
                'trades': 0,
                'win_rate': 0,
                'pf': 0,
                'pnl': 0,
                'return': 0
            }

        # Конфигурация с оптимальными параметрами
        config = BacktestConfig(
            initial_capital=100000,
            position_size_pct=0.95,
            take_profit=OPTIMAL_TP,      # 3.0%
            stop_loss=OPTIMAL_SL,        # 1.2%
            max_hold_bars=288,
            commission=0.001,
            slippage=0.0005,
            cooldown_bars=10,
            ml_enabled=False,
            filters_enabled=True,        # Используем фильтры
            opportunity_enabled=True,
            cost_enabled=True,
            risk_penalty_enabled=True,
        )

        # Запустить backtest
        ml_scores = signals.astype(float) * 0.7
        backtest = PjSBacktestEngine(config)
        results = backtest.run(ohlcv=ohlcv, raw_signals=signals, ml_scores=ml_scores)

        return {
            'coin': coin,
            'timeframe': timeframe,
            'status': 'OK',
            'bars': len(ohlcv),
            'signals': signal_count,
            'trades': results['trades'],
            'wins': results['wins'],
            'losses': results['losses'],
            'win_rate': results['win_rate'],
            'pf': results['profit_factor'],
            'pnl': results['total_pnl'],
            'return': results['total_pnl_pct'],
            'capital_final': results['final_capital']
        }

    except Exception as e:
        logger.error(f"❌ {coin}_{timeframe}: {str(e)[:100]}")
        return {
            'coin': coin,
            'timeframe': timeframe,
            'status': f'ERROR: {str(e)[:50]}',
            'trades': 0,
            'win_rate': 0,
            'pf': 0,
            'pnl': 0,
            'return': 0
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*100)
    print("BACKTEST ALL COINS & TIMEFRAMES")
    print("="*100)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Определить путь к данным
    # Для Windows: C:\Users\Dmitriy\scarlet-sails\data\raw\
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"

    print(f"Data directory: {data_dir}")
    print(f"Coins: {len(COINS)} ({', '.join(COINS)})")
    print(f"Timeframes: {len(TIMEFRAMES)} ({', '.join(TIMEFRAMES)})")
    print(f"Total combinations: {len(COINS) * len(TIMEFRAMES)}")
    print(f"Parameters: TP={OPTIMAL_TP*100:.1f}%, SL={OPTIMAL_SL*100:.2f}%")

    # Проверить наличие директории
    if not data_dir.exists():
        print(f"\n❌ ERROR: Data directory not found: {data_dir}")
        print(f"Expected path: {data_dir}")
        return

    # Запустить все комбинации
    results = []
    total = len(COINS) * len(TIMEFRAMES)
    current = 0

    print(f"\n{'Coin':<6} {'TF':<4} {'Status':<15} {'Bars':<8} {'Trades':<8} {'WR %':<8} {'PF':<8} {'P&L $':<12} {'Return %':<10}")
    print("-" * 110)

    for coin in COINS:
        for tf in TIMEFRAMES:
            current += 1
            logger.info(f"[{current}/{total}] Testing {coin}_{tf}...")

            result = run_backtest_for_coin(coin, tf, data_dir)
            results.append(result)

            # Вывести результат
            status = result['status']
            if status == 'OK':
                print(f"{coin:<6} {tf:<4} {status:<15} {result['bars']:<8} {result['trades']:<8} "
                      f"{result['win_rate']:>7.1f}% {result['pf']:>7.2f} "
                      f"${result['pnl']:>10,.0f} {result['return']:>9.2f}%")
            else:
                print(f"{coin:<6} {tf:<4} {status:<15}")

    # Создать DataFrame результатов
    results_df = pd.DataFrame(results)

    # Анализ
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)

    successful = results_df[results_df['status'] == 'OK']
    print(f"\nSuccessful backtests: {len(successful)}/{len(results_df)}")

    if len(successful) > 0:
        print(f"\nAverage results (successful only):")
        print(f"  Trades:      {successful['trades'].mean():.1f}")
        print(f"  Win Rate:    {successful['win_rate'].mean():.1f}%")
        print(f"  Profit Factor: {successful['pf'].mean():.2f}")
        print(f"  Avg Return:  {successful['return'].mean():.2f}%")
        print(f"  Total P&L:   ${successful['pnl'].sum():,.0f}")

        print(f"\nTop 5 by Win Rate:")
        top_wr = successful.nlargest(5, 'win_rate')[['coin', 'timeframe', 'trades', 'win_rate', 'pf', 'return']]
        for idx, row in top_wr.iterrows():
            print(f"  {row['coin']}_{row['timeframe']}: {row['win_rate']:.1f}% WR, {row['pf']:.2f} PF, {row['return']:.2f}% return")

        print(f"\nTop 5 by Return:")
        top_ret = successful.nlargest(5, 'return')[['coin', 'timeframe', 'trades', 'win_rate', 'pf', 'return']]
        for idx, row in top_ret.iterrows():
            print(f"  {row['coin']}_{row['timeframe']}: {row['return']:.2f}% return, {row['win_rate']:.1f}% WR, {row['pf']:.2f} PF")

    # Сохранить результаты в CSV
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = reports_dir / f"backtest_all_coins_{timestamp}.csv"

    results_df.to_csv(csv_file, index=False)
    print(f"\n✅ Results saved: {csv_file}")

    # Также сохранить Summary
    summary_file = reports_dir / f"backtest_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("BACKTEST ALL COINS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: TP={OPTIMAL_TP*100:.1f}%, SL={OPTIMAL_SL*100:.2f}%\n")
        f.write(f"Total combinations: {len(results_df)}\n")
        f.write(f"Successful: {len(successful)}\n\n")
        f.write(results_df.to_string(index=False))

    print(f"✅ Summary saved: {summary_file}")

    print(f"\n" + "="*100)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
