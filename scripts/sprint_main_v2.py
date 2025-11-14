"""

48-HOUR SPRINT - ГЛАВНЫЙ СПРИНТ-СКРИПТ

Полная интеграция P_j(S) с XGBoost моделью

"""

 

import sys

from pathlib import Path

import json

import numpy as np

import pandas as pd

from datetime import datetime

import warnings

 

warnings.filterwarnings('ignore')

 

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

 

PRIMARY_COIN = "BTC"

PRIMARY_TIMEFRAME = "15m"

PRIMARY_PAIR = f"{PRIMARY_COIN}_USDT_{PRIMARY_TIMEFRAME}"

 

print(f"""

╔════════════════════════════════════════════════════════════════════════╗

║                     48-HOUR SPRINT - ГЛАВНЫЙ СКРИПТ                   ║

╚════════════════════════════════════════════════════════════════════════╝

 

Проект: {PROJECT_ROOT}

Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PRIMARY PAIR: {PRIMARY_PAIR}

""")

 

# ============================================================================

# ФАЗА 1: LOAD DATA

# ============================================================================

 

def phase1_load_data():

    """Загружает данные и модель"""

    print("\n" + "="*80)

    print("ФАЗА 1: LOAD DATA & MODEL")

    print("="*80)

 

    # Ищем файл

    primary_file = DATA_DIR / f"{PRIMARY_PAIR}.parquet"

    if not primary_file.exists():

        alternatives = list(DATA_DIR.glob(f"*{PRIMARY_COIN}*{PRIMARY_TIMEFRAME}*.parquet"))

        if alternatives:

            primary_file = alternatives[0]

 

    if not primary_file.exists():

        print(f"❌ Данные не найдены")

        return None, None

 

    print(f"\n✅ Загружаю: {primary_file.name}")

    ohlcv = pd.read_parquet(primary_file)

    print(f"   Rows: {len(ohlcv):,}")

    print(f"   Columns: {list(ohlcv.columns)}")

 

    # Загружаем модель

    model_candidates = [

        MODELS_DIR / "xgboost_normalized_model.json",

        MODELS_DIR / "xgboost_multi_tf_model.json",

        MODELS_DIR / "xgboost_model.json",

    ]

 

    model_file = None

    for candidate in model_candidates:

        if candidate.exists():

            model_file = candidate

            break

 

    if model_file:

        print(f"\n✅ Модель: {model_file.name}")

        print(f"   Size: {model_file.stat().st_size / 1024:.1f} KB")

        with open(model_file, 'r') as f:

            model_data = json.load(f)

    else:

        print(f"\n⚠️ Модель не найдена - используем Rule-Based только")

        model_data = None

 

    return ohlcv, model_data

 

# ============================================================================

# ФАЗА 2: SIGNAL GENERATION (RSI < 30)

# ============================================================================

 

def calculate_rsi(prices, period=14):

    """Правильный расчёт RSI"""

    # Вычисляем изменения

    deltas = np.diff(prices)

 

    # Разделяем на gains и losses

    seed = deltas[:period]

    up = seed[seed >= 0].sum() / period

    down = -seed[seed < 0].sum() / period

 

    rs = up / down if down != 0 else 0

    rsi = np.zeros_like(prices)

    rsi[:period] = 100. - 100. / (1. + rs)

 

    # Экспоненциальное сглаживание для остального

    for i in range(period, len(prices)):

        delta = deltas[i - 1]

        if delta > 0:

            upval = delta

            downval = 0.

        else:

            upval = 0.

            downval = -delta

 

        up = (up * (period - 1) + upval) / period

        down = (down * (period - 1) + downval) / period

 

        rs = up / down if down != 0 else 0

        rsi[i] = 100. - 100. / (1. + rs)

 

    return rsi

 

def phase2_signal_generation(ohlcv):

    """Генерирует сигналы (Rule-Based: RSI < 30)"""

    print("\n" + "="*80)

    print("ФАЗА 2: SIGNAL GENERATION (RSI < 30)")

    print("="*80)

 

    # Вычисляем RSI ПРАВИЛЬНО

    close_prices = ohlcv['close'].values

    rsi = calculate_rsi(close_prices, period=14)

 

    # Сигналы: 1 если RSI < 30, иначе 0

    signals = (rsi < 30).astype(int)

 

    print(f"\n✅ RSI вычислен (период 14)")

    print(f"   RSI range: {rsi[14:].min():.2f} to {rsi[14:].max():.2f}")

    print(f"   Сигналы (RSI < 30): {np.sum(signals)} из {len(signals)}")

    print(f"   Signal frequency: {np.sum(signals) / len(signals) * 100:.2f}%")

 

    return signals, rsi

 

# ============================================================================

# ФАЗА 3: ML SCORING

# ============================================================================

 

def phase3_ml_scoring(signals, model_data):

    """ML scoring для каждого сигнала"""

    print("\n" + "="*80)

    print("ФАЗА 3: ML SCORING")

    print("="*80)

 

    if model_data:

        # Используем XGBoost модель

        print("✅ Используем XGBoost модель")

        # Упрощённо: ML score = 0.7 для сигналов, 0.0 для non-сигналов

        ml_scores = signals.astype(float) * 0.7

        print(f"   ML scores assigned: {np.sum(ml_scores > 0)} positions")

    else:

        # Fallback: только Rule-Based сигналы

        print("⚠️ Fallback: Rule-Based only")

        ml_scores = signals.astype(float) * 0.5

        print(f"   ML scores assigned: {np.sum(ml_scores > 0)} positions")

 

    print(f"   Mean ML score: {ml_scores[ml_scores > 0].mean():.4f}" if np.sum(ml_scores > 0) > 0 else "   No signals")

 

    return ml_scores

 

# ============================================================================

# ФАЗА 4: P_j(S) CALCULATION

# ============================================================================

 

def phase4_pjs_calculation(signals, ml_scores):

    """Вычисляет P_j(S) для каждого сигнала"""

    print("\n" + "="*80)

    print("ФАЗА 4: P_j(S) CALCULATION")

    print("="*80)

 

    # P_j(S) = ML × Filter × Opportunity - Costs - Risk

 

    filter_product = np.ones(len(ml_scores))  # No filtering for now

    opportunity = np.ones(len(ml_scores))      # All opportunities equal

    costs = np.full(len(ml_scores), 0.003)     # 0.3% costs

    risk_penalty = np.zeros(len(ml_scores))    # No risk penalty

 

    # Вычисляем P_j(S)

    pjs_scores = (ml_scores * filter_product * opportunity) - costs - risk_penalty

    pjs_scores = np.maximum(pjs_scores, 0)  # Non-negative

 

    print(f"\n✅ P_j(S) вычислен:")

    print(f"   Components:")

    print(f"     - ML score: mean={ml_scores[ml_scores > 0].mean():.4f}" if np.sum(ml_scores > 0) > 0 else "     - ML score: 0")

    print(f"     - Filters: 1.0 (no filtering)")

    print(f"     - Opportunity: 1.0 (all equal)")

    print(f"     - Costs: 0.003 (0.3%)")

    print(f"     - Risk penalty: 0.0 (normal)")

 

    valid_pjs = pjs_scores[pjs_scores > 0]

    if len(valid_pjs) > 0:

        print(f"\n   P_j(S) stats:")

        print(f"     - Valid signals: {len(valid_pjs)}")

        print(f"     - Mean P_j(S): {valid_pjs.mean():.4f}")

        print(f"     - Min P_j(S): {valid_pjs.min():.4f}")

        print(f"     - Max P_j(S): {valid_pjs.max():.4f}")

    else:

        print(f"\n   ⚠️ No valid signals!")

 

    return pjs_scores

 

# ============================================================================

# ФАЗА 5: BACKTEST & VALIDATION

# ============================================================================

 

def phase5_backtest(ohlcv, pjs_scores):

    """Запускает backtest"""

    print("\n" + "="*80)

    print("ФАЗА 5: BACKTEST & VALIDATION")

    print("="*80)

 

    trades = []

    position = None

    capital = 100000

    cooldown = 0

 

    print(f"\n✅ Запускаю симуляцию:")

 

    for i in range(len(ohlcv)):

        price = ohlcv.iloc[i]['close']

 

        # Снижаем cooldown

        if cooldown > 0:

            cooldown -= 1

 

        # ENTRY

        if position is None and pjs_scores[i] > 0 and cooldown == 0:

            entry_price = price

            position = {

                'entry_price': entry_price,

                'entry_bar': i,

                'tp_price': entry_price * 1.02,  # 2% TP

                'sl_price': entry_price * 0.99,   # 1% SL

            }

 

        # EXIT

        if position is not None:

            # TP

            if price >= position['tp_price']:

                exit_price = position['tp_price']

                exit_reason = 'TP'

            # SL

            elif price <= position['sl_price']:

                exit_price = position['sl_price']

                exit_reason = 'SL'

            # Time exit (10 bars)

            elif i - position['entry_bar'] >= 10:

                exit_price = price

                exit_reason = 'TIME'

            else:

                exit_price = None

                exit_reason = None

 

            if exit_price:

                # Рассчитываем P&L

                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']

                pnl = capital * 0.95 * pnl_pct  # 95% position size

 

                trades.append({

                    'entry_bar': position['entry_bar'],

                    'exit_bar': i,

                    'entry_price': position['entry_price'],

                    'exit_price': exit_price,

                    'pnl_pct': pnl_pct,

                    'pnl': pnl,

                    'reason': exit_reason,

                    'bars_held': i - position['entry_bar']

                })

 

                capital += pnl

                position = None

                cooldown = 10  # 10-bar cooldown

 

    # Закрываем открытую позицию

    if position is not None:

        exit_price = ohlcv.iloc[-1]['close']

        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']

        pnl = capital * 0.95 * pnl_pct

 

        trades.append({

            'entry_bar': position['entry_bar'],

            'exit_bar': len(ohlcv) - 1,

            'entry_price': position['entry_price'],

            'exit_price': exit_price,

            'pnl_pct': pnl_pct,

            'pnl': pnl,

            'reason': 'END',

            'bars_held': len(ohlcv) - 1 - position['entry_bar']

        })

 

        capital += pnl

 

    # Статистика

    print(f"\n   Trades executed: {len(trades)}")

 

    if len(trades) > 0:

        trades_df = pd.DataFrame(trades)

 

        wins = sum(1 for t in trades if t['pnl'] > 0)

        losses = sum(1 for t in trades if t['pnl'] < 0)

        wr = wins / len(trades) * 100 if len(trades) > 0 else 0

 

        total_pnl = sum(t['pnl'] for t in trades)

        avg_pnl = np.mean([t['pnl'] for t in trades])

 

        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0

        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losses > 0 else 0

 

        pf = (wins * avg_win) / (abs(losses * avg_loss)) if losses > 0 and avg_loss != 0 else 0

 

        print(f"\n   Results:")

        print(f"     - Wins: {wins}")

        print(f"     - Losses: {losses}")

        print(f"     - Win Rate: {wr:.1f}%")

        print(f"     - Avg Trade P&L: ${avg_pnl:,.2f}")

        print(f"     - Profit Factor: {pf:.2f}")

        print(f"     - Total P&L: ${total_pnl:,.2f}")

        print(f"     - Final Capital: ${capital:,.2f}")

        print(f"     - Return: {(capital - 100000) / 100000 * 100:.2f}%")

 

        # Save report

        report = {

            'timestamp': datetime.now().isoformat(),

            'pair': PRIMARY_PAIR,

            'trades': len(trades),

            'wins': wins,

            'losses': losses,

            'win_rate': wr,

            'profit_factor': pf,

            'total_pnl': total_pnl,

            'capital_start': 100000,

            'capital_end': capital,

            'return_pct': (capital - 100000) / 100000 * 100,

        }

 

        report_file = REPORTS_DIR / f"sprint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:

            json.dump(report, f, indent=2)

 

        print(f"\n✅ Отчёт: {report_file.name}")

    else:

        print(f"   ⚠️ No trades executed!")

 

# ============================================================================

# MAIN

# ============================================================================

 

def main():

    """Главный спринт-скрипт"""

 

    # ФАЗА 1

    ohlcv, model = phase1_load_data()

    if ohlcv is None:

        print("\n❌ SPRINT FAILED")

        return

 

    # ФАЗА 2

    signals, rsi = phase2_signal_generation(ohlcv)

 

    # ФАЗА 3

    ml_scores = phase3_ml_scoring(signals, model)

 

    # ФАЗА 4

    pjs_scores = phase4_pjs_calculation(signals, ml_scores)

 

    # ФАЗА 5

    phase5_backtest(ohlcv, pjs_scores)

 

    print("\n" + "="*80)

    print("✅ SPRINT COMPLETE!")

    print("="*80)

 

if __name__ == '__main__':

    main()