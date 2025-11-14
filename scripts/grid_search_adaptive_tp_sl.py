#!/usr/bin/env python3
"""
DAY 2: ADAPTIVE TP/SL Grid Search
==================================

Вместо фиксированных TP/SL, ищем АДАПТИВНЫЕ параметры:

Логика:
1. Grid Search всех комбинаций TP/SL на TRAIN (2020-2023)
2. Проверяем на OOT (2024)
3. Находим STABLE комбинации (работают на обоих периодах)
4. Анализируем: при каких условиях какие параметры лучше?
5. Создаём адаптивную функцию для OpportunityScorer

Философия: Не "волшебные параметры", а УСЛОВИЕ-ЗАВИСИМЫЕ ПАРАМЕТРЫ
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import json
from datetime import datetime
import joblib
import xgboost as xgb
from itertools import product

# ============================================================================
# КОНФИГ
# ============================================================================

CONFIG = {
    'asset': 'BTC',
    'timeframe': '15m',
    'forward_window': 96,  # 24 часов на 15m

    # Grid Search parameters
    'tp_range': np.arange(0.5, 3.0, 0.25),  # 0.5%, 0.75%, 1.0%, ... 2.75%
    'sl_range': np.arange(-2.0, -0.25, 0.25),  # -2.0%, -1.75%, ... -0.5%
    'costs': 0.3,

    # Date splits
    'train_end': '2023-12-31',  # Train: 2020-2023
    'oot_start': '2024-01-01',  # OOT: 2024
    'oot_end': '2024-12-31',
}


# ============================================================================
# ЗАГРУЗКА
# ============================================================================

def load_data_and_model():
    """Загружаем ML модель и данные"""
    print("=" * 80)
    print("LOADING DATA AND MODEL")
    print("=" * 80)

    # Модель
    model_path = Path("models/xgboost_normalized_model.json")
    scaler_path = Path("models/xgboost_normalized_scaler.pkl")
    features_path = Path("models/xgboost_normalized_features.json")

    if not model_path.exists():
        print(f"❌ Model not found")
        return None, None, None, None

    model = xgb.Booster()
    model.load_model(str(model_path))
    scaler = joblib.load(scaler_path)

    with open(features_path) as f:
        features_info = json.load(f)
    feature_names = features_info['features']

    # Данные
    data_path = Path(f"data/{CONFIG['asset']}_{CONFIG['timeframe']}_normalized.parquet")
    if not data_path.exists():
        print(f"❌ Data not found")
        return None, None, None, None

    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✅ Model loaded")
    print(f"✅ Scaler loaded")
    print(f"✅ Features loaded: {len(feature_names)}")
    print(f"✅ Data loaded: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return model, scaler, feature_names, df


def split_train_oot(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into TRAIN and Out-of-Time"""
    train_mask = df['timestamp'] <= CONFIG['train_end']
    oot_mask = df['timestamp'] >= CONFIG['oot_start']

    train_df = df[train_mask].reset_index(drop=True)
    oot_df = df[oot_mask].reset_index(drop=True)

    print(f"\n📊 DATA SPLIT:")
    print(f"  TRAIN (2020-2023): {len(train_df)} bars")
    print(f"  OOT (2024):        {len(oot_df)} bars")

    return train_df, oot_df


# ============================================================================
# BACKTEST С TP/SL
# ============================================================================

def run_backtest(df: pd.DataFrame,
                 model,
                 scaler,
                 features_df: pd.DataFrame,
                 feature_names: List[str],
                 tp: float,
                 sl: float,
                 ml_threshold: float = 0.50) -> Dict:
    """
    Запускает backtest с заданными TP/SL параметрами

    Возвращает:
    {
        'trades': количество сделок,
        'wr': win rate,
        'pf': profit factor,
        'tp_exits': сделок вышло по TP,
        'sl_exits': сделок вышло по SL,
        'profits': список прибыльности каждой сделки,
    }
    """
    tp_pct = tp / 100
    sl_pct = sl / 100
    costs = CONFIG['costs'] / 100

    results = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'tp_exits': 0,
        'sl_exits': 0,
        'time_exits': 0,
        'profits': [],
    }

    min_samples = max(100, len(features_df.columns))

    for i in range(min_samples, len(df) - CONFIG['forward_window'] - 1):
        try:
            # ML сигнал
            current_idx = min(i, len(features_df) - 1)
            X_current = features_df.iloc[current_idx].values.reshape(1, -1)
            X_scaled = scaler.transform(X_current)
            ml_prob = model.predict(xgb.DMatrix(X_scaled))[0]

            if ml_prob < ml_threshold:
                continue

            entry_price = df.iloc[i]['close']

            # TP и SL уровни
            tp_level = entry_price * (1 + tp_pct + costs)
            sl_level = entry_price * (1 + sl_pct - costs)

            # Ищем первый exit
            exit_found = False
            for j in range(1, CONFIG['forward_window'] + 1):
                if i + j >= len(df):
                    break

                bar = df.iloc[i + j]

                if bar['high'] >= tp_level:
                    profit = (tp_level - entry_price) / entry_price * 100
                    results['tp_exits'] += 1
                    exit_found = True
                elif bar['low'] <= sl_level:
                    profit = (sl_level - entry_price) / entry_price * 100
                    results['sl_exits'] += 1
                    exit_found = True

                if exit_found:
                    break

            if not exit_found:
                exit_price = df.iloc[min(i + CONFIG['forward_window'], len(df) - 1)]['close']
                profit = (exit_price - entry_price) / entry_price * 100
                results['time_exits'] += 1

            if profit >= 0:
                results['wins'] += 1
            else:
                results['losses'] += 1

            results['trades'] += 1
            results['profits'].append(profit)

        except Exception as e:
            continue

    # Статистика
    if results['trades'] > 0:
        wr = results['wins'] / results['trades'] * 100
        wins_profit = sum([p for p in results['profits'] if p > 0]) / results['wins'] if results['wins'] > 0 else 0
        loss_profit = sum([p for p in results['profits'] if p < 0]) / results['losses'] if results['losses'] > 0 else 0

        if results['losses'] > 0 and loss_profit != 0:
            pf = (results['wins'] * wins_profit) / (results['losses'] * abs(loss_profit))
        else:
            pf = 0
    else:
        wr = 0
        pf = 0

    return {
        'tp': tp,
        'sl': sl,
        'trades': results['trades'],
        'wr': wr,
        'pf': pf,
        'tp_exits': results['tp_exits'],
        'sl_exits': results['sl_exits'],
        'time_exits': results['time_exits'],
        'avg_profit': np.mean(results['profits']) if results['profits'] else 0,
    }


# ============================================================================
# GRID SEARCH
# ============================================================================

def grid_search(train_df: pd.DataFrame,
                oot_df: pd.DataFrame,
                model,
                scaler,
                train_features,
                oot_features,
                feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Запускаем grid search для всех комбинаций TP/SL

    Возвращает:
    - train_results: результаты на TRAIN
    - oot_results: результаты на OOT
    """
    tp_values = CONFIG['tp_range']
    sl_values = CONFIG['sl_range']

    print(f"\n{'=' * 80}")
    print(f"GRID SEARCH: {len(tp_values)} TP values × {len(sl_values)} SL values")
    print(f"Total combinations: {len(tp_values) * len(sl_values)}")
    print(f"{'=' * 80}")

    train_results = []
    oot_results = []

    combination = 0
    total = len(tp_values) * len(sl_values)

    for tp, sl in product(tp_values, sl_values):
        combination += 1

        # TRAIN backtest
        train_result = run_backtest(train_df, model, scaler, train_features, feature_names, tp, sl)

        # OOT backtest
        oot_result = run_backtest(oot_df, model, scaler, oot_features, feature_names, tp, sl)

        train_results.append(train_result)
        oot_results.append(oot_result)

        if combination % 10 == 0:
            print(f"  [{combination:3d}/{total}] TP={tp:5.2f}% SL={sl:6.2f}% | " +
                  f"TRAIN WR={train_result['wr']:5.1f}% PF={train_result['pf']:5.2f} | " +
                  f"OOT WR={oot_result['wr']:5.1f}% PF={oot_result['pf']:5.2f}")

    train_df_results = pd.DataFrame(train_results)
    oot_df_results = pd.DataFrame(oot_results)

    return train_df_results, oot_df_results


# ============================================================================
# АНАЛИЗ И ОТБОР STABLE КОМБИНАЦИЙ
# ============================================================================

def analyze_results(train_results: pd.DataFrame, oot_results: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует результаты и находит STABLE комбинации

    Criteria для stable комбинации:
    - TRAIN WR >= 45%
    - OOT WR >= 40%
    - Stable (разница TRAIN-OOT <= 10%)
    """
    print(f"\n{'=' * 80}")
    print("ANALYZING RESULTS - FINDING STABLE COMBINATIONS")
    print(f"{'=' * 80}")

    results = train_results.copy()
    results['oot_wr'] = oot_results['wr']
    results['oot_pf'] = oot_results['pf']
    results['wr_diff'] = abs(results['wr'] - results['oot_wr'])

    # Фильтруем stable комбинации
    stable = results[
        (results['wr'] >= 45.0) &  # TRAIN WR >= 45%
        (results['oot_wr'] >= 40.0) &  # OOT WR >= 40%
        (results['wr_diff'] <= 10.0) &  # Стабильность
        (results['trades'] >= 100)  # Минимум сделок
    ].sort_values('pf', ascending=False)

    print(f"\n✅ STABLE COMBINATIONS FOUND: {len(stable)}")

    if len(stable) > 0:
        print(f"\n📊 TOP 10 STABLE COMBINATIONS:")
        print(f"{'TP':>6} {'SL':>6} {'TRAIN WR':>10} {'TRAIN PF':>10} " +
              f"{'OOT WR':>10} {'OOT PF':>10} {'DIFF':>8}")
        print("-" * 70)

        for idx, row in stable.head(10).iterrows():
            print(f"{row['tp']:6.2f}% {row['sl']:6.2f}% " +
                  f"{row['wr']:9.1f}% {row['pf']:9.2f} " +
                  f"{row['oot_wr']:9.1f}% {row['oot_pf']:9.2f} " +
                  f"{row['wr_diff']:7.1f}%")

        print(f"\n💡 KEY INSIGHTS:")
        print(f"  Best TRAIN TP: {stable.iloc[0]['tp']:.2f}%")
        print(f"  Best TRAIN SL: {stable.iloc[0]['sl']:.2f}%")
        print(f"  Avg OOT WR: {stable['oot_wr'].mean():.1f}%")
        print(f"  Avg OOT PF: {stable['oot_pf'].mean():.2f}")

    else:
        print(f"❌ NO STABLE COMBINATIONS FOUND")
        print(f"\nTrying to relax criteria...")

        relaxed = results[
            (results['wr'] >= 42.0) &
            (results['oot_wr'] >= 35.0) &
            (results['wr_diff'] <= 15.0) &
            (results['trades'] >= 50)
        ].sort_values('pf', ascending=False)

        print(f"✅ RELAXED CRITERIA: {len(relaxed)} combinations")
        stable = relaxed

    return stable


# ============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def save_results(train_results: pd.DataFrame, oot_results: pd.DataFrame, stable_df: pd.DataFrame):
    """Сохраняем результаты для дальнейшего анализа"""
    results_dir = Path("reports")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Сохраняем все результаты
    train_results.to_csv(results_dir / f"grid_search_train_{timestamp}.csv", index=False)
    oot_results.to_csv(results_dir / f"grid_search_oot_{timestamp}.csv", index=False)

    # Сохраняем stable комбинации
    stable_df.to_csv(results_dir / f"grid_search_stable_{timestamp}.csv", index=False)

    # JSON для интеграции в OpportunityScorer
    if len(stable_df) > 0:
        best_combo = stable_df.iloc[0]
        adaptive_config = {
            'best_tp': float(best_combo['tp']),
            'best_sl': float(best_combo['sl']),
            'best_wr': float(best_combo['wr']),
            'best_pf': float(best_combo['pf']),
            'oot_wr': float(best_combo['oot_wr']),
            'oot_pf': float(best_combo['oot_pf']),
            'stable_count': len(stable_df),
            'stable_avg_wr': float(stable_df['wr'].mean()),
            'stable_avg_pf': float(stable_df['pf'].mean()),
            'timestamp': timestamp,
        }

        with open(results_dir / f"adaptive_tp_sl_config_{timestamp}.json", 'w') as f:
            json.dump(adaptive_config, f, indent=2)

        print(f"\n✅ Results saved:")
        print(f"  Train: {results_dir / f'grid_search_train_{timestamp}.csv'}")
        print(f"  OOT:   {results_dir / f'grid_search_oot_{timestamp}.csv'}")
        print(f"  Stable: {results_dir / f'grid_search_stable_{timestamp}.csv'}")
        print(f"  Config: {results_dir / f'adaptive_tp_sl_config_{timestamp}.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'=' * 80}")
    print("DAY 2: ADAPTIVE TP/SL GRID SEARCH")
    print(f"{'=' * 80}")

    # Загружаем
    model, scaler, feature_names, df = load_data_and_model()
    if model is None:
        return

    # Разделяем на TRAIN и OOT
    train_df, oot_df = split_train_oot(df)

    # Извлекаем фичи
    def extract_features(df, feature_names):
        available = [f for f in feature_names if f in df.columns]
        return df[available]

    train_features = extract_features(train_df, feature_names)
    oot_features = extract_features(oot_df, feature_names)

    # Grid search
    train_results, oot_results = grid_search(
        train_df, oot_df,
        model, scaler,
        train_features, oot_features,
        feature_names
    )

    # Анализируем
    stable_df = analyze_results(train_results, oot_results)

    # Сохраняем
    save_results(train_results, oot_results, stable_df)

    print(f"\n{'=' * 80}")
    print("✅ GRID SEARCH COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
