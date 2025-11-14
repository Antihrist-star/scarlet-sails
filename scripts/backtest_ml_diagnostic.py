#!/usr/bin/env python3
"""
DAY 2 - ЧЕСТНОЕ СРАВНЕНИЕ ДВУХ БЭКТЕСТОВ
===========================================

Цель: Понять ИМЕННО как модели считали результаты

ТЕСТ 1: Comprehensive backtest (как в audit)
- Использует максимальную цену за 96 баров
- Никакого SL
- Никаких издержек
- Сравнивает потенциал сигналов

ТЕСТ 2: Realistic backtest (как в реальности)
- Использует TP +1.0% и SL -0.5%
- Издержки 0.3%
- Exit при ПЕРВОМ событии
- Реальные торговые условия

На ОДНИХ И ТЕХ ЖЕ данных → видим точную разницу
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import joblib
import xgboost as xgb

# ============================================================================
# КОНФИГ
# ============================================================================
CONFIG = {
    'asset': 'BTC',
    'timeframe': '15m',
    'forward_window': 96,  # 24 часов на 15m
    'tp_percent': 1.0,
    'sl_percent': -0.5,
    'costs_percent': 0.3,
    'ml_threshold': 0.50,
}

# ============================================================================
# ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ============================================================================

def load_ml_model_and_data():
    """Загружаем обученную ML модель и данные"""
    print("=" * 80)
    print("LOADING ML MODEL AND DATA")
    print("=" * 80)

    model_path = Path("models/xgboost_normalized_model.json")
    scaler_path = Path("models/xgboost_normalized_scaler.pkl")
    features_path = Path("models/xgboost_normalized_features.json")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None, None, None

    # Загружаем модель
    model = xgb.Booster()
    model.load_model(str(model_path))
    print(f"✅ Model loaded: {model_path}")

    # Загружаем scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded: {scaler_path}")

    # Загружаем features
    with open(features_path) as f:
        features_info = json.load(f)
    feature_names = features_info['features']
    print(f"✅ Features loaded: {len(feature_names)} features")

    # Загружаем данные
    data_path = Path(f"data/{CONFIG['asset']}_{CONFIG['timeframe']}_normalized.parquet")
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return None, None, None, None

    df = pd.read_parquet(data_path)
    print(f"✅ Data loaded: {len(df)} bars")

    return model, scaler, feature_names, df


def extract_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Извлекаем фичи из данных"""
    print("\nExtracting features...")

    # Для этой диагностики просто берём первые N фич которые есть в данных
    available_features = [f for f in feature_names if f in df.columns]
    print(f"Available features: {len(available_features)}/{len(feature_names)}")

    features_df = df[available_features].copy()
    return features_df


# ============================================================================
# ТЕСТ 1: COMPREHENSIVE BACKTEST (max price за 96 баров)
# ============================================================================

def comprehensive_backtest(model, scaler, features_df: pd.DataFrame, df: pd.DataFrame,
                          feature_names: List[str]) -> Dict:
    """
    COMPREHENSIVE: использует максимальную цену за 96 баров
    (как в оригинальном audit)
    """
    print("\n" + "=" * 80)
    print("TEST 1: COMPREHENSIVE BACKTEST (Max Price за 96 баров)")
    print("=" * 80)

    forward_window = CONFIG['forward_window']
    tp_target = CONFIG['tp_percent'] / 100

    results = {
        'entries': [],
        'wins': 0,
        'losses': 0,
        'trades': 0,
        'ml_probabilities': [],
        'exit_prices': [],
        'profit_percents': [],
    }

    # Нужно иметь достаточно данных для ML
    min_samples = max(100, len(features_df.columns))

    for i in range(min_samples, len(df) - forward_window - 1):
        # Получаем фичи для текущего бара
        try:
            # Берём фичи с правильным индексом
            current_idx = min(i, len(features_df) - 1)
            X_current = features_df.iloc[current_idx].values.reshape(1, -1)

            # Масштабируем
            X_scaled = scaler.transform(X_current)

            # Предсказание ML
            ml_prob = model.predict(xgb.DMatrix(X_scaled))[0]

            # Проверяем сигнал
            if ml_prob < CONFIG['ml_threshold']:
                continue

            # Entry price
            entry_price = df.iloc[i]['close']

            # COMPREHENSIVE: берём МАКСИМАЛЬНУЮ цену за forward_window баров
            max_price = df.iloc[i:i+forward_window]['high'].max()
            profit = (max_price - entry_price) / entry_price

            # Результат
            if profit >= tp_target:
                result = 'WIN'
                results['wins'] += 1
            else:
                result = 'LOSS'
                results['losses'] += 1

            results['trades'] += 1
            results['ml_probabilities'].append(ml_prob)
            results['exit_prices'].append(max_price)
            results['profit_percents'].append(profit * 100)

            if results['trades'] % 10000 == 0:
                print(f"  Processed {results['trades']} trades...")

        except Exception as e:
            continue

    # Статистика
    wr = results['wins'] / results['trades'] * 100 if results['trades'] > 0 else 0
    avg_win = np.mean([p for p in results['profit_percents'] if p > 0]) if results['wins'] > 0 else 0
    avg_loss = np.mean([p for p in results['profit_percents'] if p < 0]) if results['losses'] > 0 else 0
    pf = (results['wins'] * avg_win) / (results['losses'] * abs(avg_loss)) if results['losses'] > 0 and avg_loss != 0 else 0

    print(f"\n📊 COMPREHENSIVE RESULTS:")
    print(f"  Trades: {results['trades']:,}")
    print(f"  Wins: {results['wins']:,}")
    print(f"  Losses: {results['losses']:,}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Avg Win: {avg_win:.2f}%")
    print(f"  Avg Loss: {avg_loss:.2f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  ML Probability: min={np.min(results['ml_probabilities']):.3f}, " +
          f"avg={np.mean(results['ml_probabilities']):.3f}, " +
          f"max={np.max(results['ml_probabilities']):.3f}")

    return {
        'type': 'comprehensive',
        'trades': results['trades'],
        'wins': results['wins'],
        'losses': results['losses'],
        'wr': wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pf': pf,
    }


# ============================================================================
# ТЕСТ 2: REALISTIC BACKTEST (TP/SL с издержками)
# ============================================================================

def realistic_backtest(model, scaler, features_df: pd.DataFrame, df: pd.DataFrame,
                      feature_names: List[str]) -> Dict:
    """
    REALISTIC: используем TP/SL и издержки
    (как в реальной торговле)
    """
    print("\n" + "=" * 80)
    print("TEST 2: REALISTIC BACKTEST (TP/SL + Costs)")
    print("=" * 80)

    forward_window = CONFIG['forward_window']
    tp_pct = CONFIG['tp_percent'] / 100
    sl_pct = CONFIG['sl_percent'] / 100
    costs = CONFIG['costs_percent'] / 100

    results = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'tp_exits': 0,
        'sl_exits': 0,
        'time_exits': 0,
        'profit_percents': [],
        'ml_probabilities': [],
    }

    min_samples = max(100, len(features_df.columns))

    for i in range(min_samples, len(df) - forward_window - 1):
        try:
            current_idx = min(i, len(features_df) - 1)
            X_current = features_df.iloc[current_idx].values.reshape(1, -1)
            X_scaled = scaler.transform(X_current)

            ml_prob = model.predict(xgb.DMatrix(X_scaled))[0]

            if ml_prob < CONFIG['ml_threshold']:
                continue

            entry_price = df.iloc[i]['close']

            # Вычисляем TP и SL уровни с учётом издержек
            tp_level = entry_price * (1 + tp_pct + costs)
            sl_level = entry_price * (1 + sl_pct - costs)

            # Проходим по барам forward_window и ищем первый exit
            exit_reason = None
            exit_price = None
            profit = None

            for j in range(1, forward_window + 1):
                if i + j >= len(df):
                    break

                bar = df.iloc[i + j]

                # Проверяем TP
                if bar['high'] >= tp_level:
                    exit_reason = 'TP'
                    exit_price = tp_level
                    profit = (tp_level - entry_price) / entry_price * 100
                    results['tp_exits'] += 1
                    break

                # Проверяем SL
                elif bar['low'] <= sl_level:
                    exit_reason = 'SL'
                    exit_price = sl_level
                    profit = (sl_level - entry_price) / entry_price * 100
                    results['sl_exits'] += 1
                    break

            # Если не выбили TP/SL, выходим по Time
            if exit_reason is None:
                exit_reason = 'TIME'
                exit_price = df.iloc[min(i + forward_window, len(df) - 1)]['close']
                profit = (exit_price - entry_price) / entry_price * 100
                results['time_exits'] += 1

            # Результат
            if profit >= 0:
                results['wins'] += 1
            else:
                results['losses'] += 1

            results['trades'] += 1
            results['profit_percents'].append(profit)
            results['ml_probabilities'].append(ml_prob)

            if results['trades'] % 10000 == 0:
                print(f"  Processed {results['trades']} trades...")

        except Exception as e:
            continue

    # Статистика
    wr = results['wins'] / results['trades'] * 100 if results['trades'] > 0 else 0
    avg_profit = np.mean(results['profit_percents']) if results['profit_percents'] else 0

    print(f"\n📊 REALISTIC RESULTS:")
    print(f"  Trades: {results['trades']:,}")
    print(f"  Wins: {results['wins']:,}")
    print(f"  Losses: {results['losses']:,}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  TP exits: {results['tp_exits']:,}")
    print(f"  SL exits: {results['sl_exits']:,}")
    print(f"  TIME exits: {results['time_exits']:,}")
    print(f"  Avg Profit: {avg_profit:.2f}%")

    return {
        'type': 'realistic',
        'trades': results['trades'],
        'wins': results['wins'],
        'losses': results['losses'],
        'wr': wr,
        'tp_exits': results['tp_exits'],
        'sl_exits': results['sl_exits'],
        'time_exits': results['time_exits'],
    }


# ============================================================================
# СРАВНЕНИЕ
# ============================================================================

def compare_results(comp_result: Dict, real_result: Dict):
    """Сравниваем результаты"""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ: Comprehensive vs Realistic")
    print("=" * 80)

    print(f"\n📊 WIN RATE:")
    print(f"  Comprehensive: {comp_result['wr']:.1f}%")
    print(f"  Realistic:     {real_result['wr']:.1f}%")
    print(f"  差异 (difference): {comp_result['wr'] - real_result['wr']:.1f}%")

    print(f"\n📊 TRADES:")
    print(f"  Comprehensive: {comp_result['trades']:,}")
    print(f"  Realistic:     {real_result['trades']:,}")

    if real_result['trades'] > 0:
        print(f"\n📊 EXIT REASONS (Realistic):")
        print(f"  TP:   {real_result['tp_exits']:,} ({real_result['tp_exits']/real_result['trades']*100:.1f}%)")
        print(f"  SL:   {real_result['sl_exits']:,} ({real_result['sl_exits']/real_result['trades']*100:.1f}%)")
        print(f"  TIME: {real_result['time_exits']:,} ({real_result['time_exits']/real_result['trades']*100:.1f}%)")

    print(f"\n💡 ВЫВОДЫ:")
    wr_diff = comp_result['wr'] - real_result['wr']

    if wr_diff > 20:
        print(f"  ⚠️  БОЛЬШАЯ РАЗНИЦА ({wr_diff:.1f}%)")
        print(f"  Проблема: SL срабатывает слишком часто?")
        print(f"  Решение: Увеличить SL, или убрать SL совсем?")
    elif wr_diff > 10:
        print(f"  ⚠️  Заметная разница ({wr_diff:.1f}%)")
        print(f"  Нормально для TP/SL, но нужно улучшить")
    else:
        print(f"  ✅ Приемлемая разница ({wr_diff:.1f}%)")
        print(f"  Модель работает стабильно")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("DAY 2: ML MODEL DIAGNOSTIC - COMPREHENSIVE vs REALISTIC")
    print("=" * 80)
    print(f"Asset: {CONFIG['asset']}")
    print(f"Timeframe: {CONFIG['timeframe']}")
    print(f"ML Threshold: {CONFIG['ml_threshold']}")
    print(f"TP: {CONFIG['tp_percent']}% | SL: {CONFIG['sl_percent']}% | Costs: {CONFIG['costs_percent']}%")

    # Загружаем
    model, scaler, feature_names, df = load_ml_model_and_data()
    if model is None:
        print("❌ Failed to load data")
        return

    # Извлекаем фичи
    features_df = extract_features(df, feature_names)

    # Тест 1: Comprehensive
    comp_result = comprehensive_backtest(model, scaler, features_df, df, feature_names)

    # Тест 2: Realistic
    real_result = realistic_backtest(model, scaler, features_df, df, feature_names)

    # Сравниваем
    compare_results(comp_result, real_result)

    # Сохраняем результаты
    results_file = Path("reports/ml_diagnostic_comprehensive_vs_realistic.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'comprehensive': comp_result,
            'realistic': real_result,
        }, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")


if __name__ == '__main__':
    main()
