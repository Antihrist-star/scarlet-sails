#!/usr/bin/env python3
"""
ПОЛНЫЙ ИНВЕНТАРЬ ВСЕХ РЕСУРСОВ
================================

Сканирует и анализирует:
1. Все OHLCV данные (14 монет × 4 таймфрейма)
2. Все XGBoost модели
3. Диапазоны дат и пропуски
4. Структуру проекта

Результат: Полная картина того, с чем мы работаем
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIRS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "datasets",
]
MODEL_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "model",
    PROJECT_ROOT / "ml_models",
]

# Ожидаемые монеты
COINS = ["BTC", "ETH", "SOL", "ALGO", "AVAX", "DOT", "ENA", "HBAR", "LDO", "LINK", "LTC", "ONDO", "SUI", "UNI"]
TIMEFRAMES = ["15m", "1h", "4h", "1d", "1M", "5m", "15min", "1hour", "4hour", "1day"]

# ============================================================================
# UTILS
# ============================================================================

def find_files(root_dir, extensions=['.parquet', '.csv', '.pkl', '.feather']):
    """Найти все файлы данных"""
    files = []
    if not root_dir.exists():
        return files

    for ext in extensions:
        files.extend(root_dir.glob(f"**/*{ext}"))
    return files

def parse_filename(filename):
    """Попытаться распарсить имя файла на монету и таймфрейм"""
    name = filename.stem.upper()

    coin = None
    timeframe = None

    # Проверяем монеты
    for c in COINS:
        if c in name:
            coin = c
            break

    # Проверяем таймфреймы
    for tf in TIMEFRAMES:
        if tf.upper() in name:
            timeframe = tf
            break

    return coin, timeframe

def analyze_ohlcv(filepath):
    """Анализирует OHLCV файл"""
    try:
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath, nrows=10000)  # Не грузим всё для скорости
        else:
            return None

        if len(df) == 0:
            return {'status': 'EMPTY', 'rows': 0}

        # Определяем столбцы с временем
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower()]

        info = {
            'status': 'OK',
            'rows': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
        }

        # Попытаемся определить диапазон дат
        if time_cols:
            time_col = time_cols[0]
            try:
                dates = pd.to_datetime(df[time_col])
                info['date_min'] = str(dates.min())
                info['date_max'] = str(dates.max())
                info['date_range_days'] = (dates.max() - dates.min()).days
            except:
                pass

        # Проверяем наличие OHLCV
        required = ['open', 'high', 'low', 'close', 'volume']
        has_ohlcv = [col.lower() in [c.lower() for c in df.columns] for col in required]
        info['has_ohlcv'] = all(has_ohlcv)

        return info
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

def check_model_file(filepath):
    """Проверяет файл ML модели"""
    try:
        # Проверяем расширение
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return {
                'status': 'OK',
                'type': 'XGBoost JSON',
                'size_kb': filepath.stat().st_size / 1024,
                'keys': list(data.keys()) if isinstance(data, dict) else 'Not a dict'
            }
        elif filepath.suffix == '.pkl':
            return {
                'status': 'OK',
                'type': 'Pickle',
                'size_kb': filepath.stat().st_size / 1024,
            }
        elif filepath.suffix == '.joblib':
            return {
                'status': 'OK',
                'type': 'Joblib',
                'size_kb': filepath.stat().st_size / 1024,
            }
        else:
            return {
                'status': 'UNKNOWN',
                'type': filepath.suffix,
                'size_kb': filepath.stat().st_size / 1024,
            }
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}

# ============================================================================
# MAIN SCANNING
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ПОЛНЫЙ ИНВЕНТАРЬ РЕСУРСОВ SCARLET-SAILS")
    print("="*80)
    print(f"Время сканирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}\n")

    # ─────────────────────────────────────────────────────────────────────
    # PART 1: ДАННЫЕ (OHLCV)
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "-"*80)
    print("PART 1: OHLCV DATA FILES")
    print("-"*80)

    ohlcv_files = []
    for data_dir in DATA_DIRS:
        ohlcv_files.extend(find_files(data_dir, ['.parquet', '.csv', '.feather']))

    if not ohlcv_files:
        print("❌ Не найдены файлы данных!")
    else:
        print(f"✅ Найдено файлов: {len(ohlcv_files)}\n")

        # Группируем по монетам и таймфреймам
        data_by_pair = {}

        for filepath in sorted(ohlcv_files):
            coin, timeframe = parse_filename(filepath)
            info = analyze_ohlcv(filepath)

            if info is None:
                continue

            pair = f"{coin}_{timeframe}" if coin and timeframe else filepath.name

            if pair not in data_by_pair:
                data_by_pair[pair] = {
                    'file': filepath,
                    'info': info,
                    'coin': coin,
                    'timeframe': timeframe
                }

        # Выводим таблицу
        print(f"{'Pair':<20} {'Rows':<10} {'Date Min':<15} {'Date Max':<15} {'Days':<8} {'Status':<10}")
        print("-" * 90)

        for pair, data in sorted(data_by_pair.items()):
            info = data['info']
            rows = info.get('rows', '?')
            date_min = info.get('date_min', '?')[:10]
            date_max = info.get('date_max', '?')[:10]
            days = info.get('date_range_days', '?')
            status = info.get('status', 'UNKNOWN')

            status_icon = "✅" if status == "OK" else "❌"
            print(f"{pair:<20} {rows:<10} {date_min:<15} {date_max:<15} {days:<8} {status_icon}")

        # Статистика
        print("\n📊 СТАТИСТИКА ДАННЫХ:")
        total_files = len(data_by_pair)
        ok_files = sum(1 for d in data_by_pair.values() if d['info'].get('status') == 'OK')
        empty_files = sum(1 for d in data_by_pair.values() if d['info'].get('status') == 'EMPTY')
        error_files = sum(1 for d in data_by_pair.values() if d['info'].get('status') == 'ERROR')

        print(f"  Всего файлов: {total_files}")
        print(f"  ✅ OK: {ok_files}")
        print(f"  🟡 EMPTY: {empty_files}")
        print(f"  ❌ ERROR: {error_files}")

        # Какие монеты есть
        coins_found = set(d['coin'] for d in data_by_pair.values() if d['coin'])
        timeframes_found = set(d['timeframe'] for d in data_by_pair.values() if d['timeframe'])

        print(f"\n  Найденные монеты ({len(coins_found)}): {', '.join(sorted(coins_found))}")
        print(f"  Найденные таймфреймы ({len(timeframes_found)}): {', '.join(sorted(timeframes_found))}")

        # Проверяем покрытие
        print(f"\n  Ожидаемых пар: {len(COINS)} монет × {len(TIMEFRAMES)} таймфреймов = {len(COINS) * len(TIMEFRAMES)}")
        print(f"  Найдено пар с данными: {ok_files}")
        coverage = (ok_files / (len(COINS) * len(TIMEFRAMES))) * 100 if len(COINS) * len(TIMEFRAMES) > 0 else 0
        print(f"  Покрытие: {coverage:.1f}%")

    # ─────────────────────────────────────────────────────────────────────
    # PART 2: ML МОДЕЛИ
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "-"*80)
    print("PART 2: ML MODELS")
    print("-"*80)

    model_files = []
    for model_dir in MODEL_DIRS:
        model_files.extend(find_files(model_dir, ['.json', '.pkl', '.joblib', '.h5']))

    if not model_files:
        print("❌ Не найдены файлы моделей!")
    else:
        print(f"✅ Найдено моделей: {len(model_files)}\n")

        print(f"{'Model Name':<50} {'Type':<15} {'Size KB':<10}")
        print("-" * 80)

        for filepath in sorted(model_files):
            info = check_model_file(filepath)
            name = filepath.name
            model_type = info.get('type', '?')
            size = info.get('size_kb', 0)

            status_icon = "✅" if info.get('status') == 'OK' else "❌"
            print(f"{status_icon} {name:<48} {model_type:<15} {size:<10.1f}")

    # ─────────────────────────────────────────────────────────────────────
    # PART 3: СТРУКТУРА ПРОЕКТА
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "-"*80)
    print("PART 3: PROJECT STRUCTURE")
    print("-"*80)

    key_dirs = {
        'data': PROJECT_ROOT / 'data',
        'models': PROJECT_ROOT / 'models',
        'scripts': PROJECT_ROOT / 'scripts',
        'backtesting': PROJECT_ROOT / 'backtesting',
        'features': PROJECT_ROOT / 'features',
        'lib': PROJECT_ROOT / 'lib',
        'reports': PROJECT_ROOT / 'reports',
    }

    print("\nДиректории проекта:")
    for name, path in key_dirs.items():
        if path.exists():
            file_count = len(list(path.glob('*')))
            print(f"  ✅ {name:<15} ({file_count} files)")
        else:
            print(f"  ❌ {name:<15} (NOT FOUND)")

    # ─────────────────────────────────────────────────────────────────────
    # PART 4: ВАЖНЫЕ ФАЙЛЫ
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "-"*80)
    print("PART 4: KEY PROJECT FILES")
    print("-"*80)

    important_files = {
        'backtest_pjs_framework.py': PROJECT_ROOT / 'backtesting' / 'backtest_pjs_framework.py',
        'opportunity_scorer.py': PROJECT_ROOT / 'lib' / 'opportunity_scorer.py',
        'test_pjs_framework_v1.py': PROJECT_ROOT / 'scripts' / 'test_pjs_framework_v1.py',
    }

    for name, path in important_files.items():
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {name:<40} ({size:>10,} bytes)")
        else:
            print(f"  ❌ {name:<40} NOT FOUND")

    # ─────────────────────────────────────────────────────────────────────
    # PART 5: SUMMARY & RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    if ok_files > 0:
        print(f"\n✅ Найдено {ok_files} файлов OHLCV с реальными данными")
        print(f"✅ Найдено {len(model_files)} ML моделей")
        print("\n🚀 ГОТОВЫ К СПРИНТУ!")
        print("\nСледующие шаги:")
        print("  1. Выбрать главный таймфрейм для тестирования (рекомендуется 15m или 1h)")
        print("  2. Выбрать основную монету для разработки (рекомендуется BTC)")
        print("  3. Загрузить данные для train/test split (2020-2023 train, 2024 test)")
        print("  4. Запустить V1 baseline тест на реальных данных")
        print("  5. Интегрировать 48-hour sprint план")
    else:
        print("\n⚠️ ПРОБЛЕМА: Не найдено достаточно данных для спринта!")
        print("\nНужно:")
        print("  1. Загрузить данные из DVC (git lfs pull, dvc pull)")
        print("  2. Убедиться что файлы распакованы в data/raw/")
        print("  3. Запустить этот скрипт ещё раз")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
