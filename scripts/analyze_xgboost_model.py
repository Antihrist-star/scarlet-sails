#!/usr/bin/env python3
"""
АНАЛИЗ XGBOOST МОДЕЛИ
=======================

Загружает XGBoost модель и выводит всю информацию:
- Архитектура модели
- Количество признаков
- Параметры обучения
- Названия признаков
- Важность признаков
"""

import sys
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATHS = [
    PROJECT_ROOT / "models" / "xgboost_model.json",
    PROJECT_ROOT / "model" / "xgboost_model.json",
]

# ============================================================================
# АНАЛИЗ JSON ФАЙЛА (для XGBoost JSON формата)
# ============================================================================

def analyze_xgboost_json():
    """Анализирует XGBoost модель в JSON формате"""

    print("\n" + "="*80)
    print("АНАЛИЗ XGBOOST МОДЕЛИ (JSON формат)")
    print("="*80)

    # Ищем файл
    model_file = None
    for path in MODEL_PATHS:
        if path.exists():
            model_file = path
            break

    if not model_file:
        print(f"❌ XGBoost модель не найдена в следующих местах:")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        return False

    print(f"\n✅ Найдена модель: {model_file}")
    print(f"   Размер: {model_file.stat().st_size / 1024:.1f} KB")

    try:
        with open(model_file, 'r') as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка при загрузке JSON: {e}")
        return False

    # ─────────────────────────────────────────────────────────────────────
    # АНАЛИЗИРУЕМ СТРУКТУРУ
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "-"*80)
    print("СТРУКТУРА МОДЕЛИ")
    print("-"*80)

    print(f"\nТип данных: {type(model_data)}")

    if isinstance(model_data, dict):
        print(f"Ключи на верхнем уровне: {list(model_data.keys())}")

        # Проверяем основные структуры
        if 'learner' in model_data:
            print("\n✅ Это XGBoost JSON (scikit-learn формат)")
            learner = model_data['learner']

            # Параметры
            if 'attributes' in learner:
                attrs = learner['attributes']
                print(f"\n📊 ПАРАМЕТРЫ:")
                for key, val in attrs.items():
                    print(f"  {key}: {val}")

            # Feature names
            if 'feature_names' in learner:
                features = learner['feature_names']
                print(f"\n📋 ПРИЗНАКИ ({len(features)} шт):")
                for i, feat in enumerate(features[:20]):  # Первые 20
                    print(f"  {i:2d}. {feat}")
                if len(features) > 20:
                    print(f"  ... и ещё {len(features)-20} признаков")

            # Feature types
            if 'feature_types' in learner:
                ftypes = learner['feature_types']
                print(f"\n🔧 ТИПЫ ПРИЗНАКОВ:")
                print(f"  {ftypes}")

            # Object list (деревья)
            if 'gradient_booster' in learner:
                gb = learner['gradient_booster']
                if 'model' in gb:
                    model_info = gb['model']
                    if 'gbtree_model_param' in model_info:
                        params = model_info['gbtree_model_param']
                        print(f"\n🌳 ИНФОРМАЦИЯ О ДЕРЕВЬЯХ:")
                        for key, val in params.items():
                            print(f"  {key}: {val}")

                    if 'trees' in model_info:
                        trees = model_info['trees']
                        print(f"\n  Количество деревьев: {len(trees)}")

                    if 'tree_sizes' in model_info:
                        sizes = model_info['tree_sizes']
                        print(f"  Размеры деревьев: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")

        elif 'tree_sizes' in model_data or 'trees' in model_data:
            print("\n✅ Это XGBoost JSON (другой формат)")

            if 'trees' in model_data:
                print(f"  Количество деревьев: {len(model_data['trees'])}")

            if 'feature_names' in model_data:
                features = model_data['feature_names']
                print(f"\n📋 ПРИЗНАКИ ({len(features)} шт):")
                for i, feat in enumerate(features[:20]):
                    print(f"  {i:2d}. {feat}")
                if len(features) > 20:
                    print(f"  ... и ещё {len(features)-20} признаков")

        else:
            print("\n⚠️ Неизвестный формат XGBoost JSON")
            print(f"Доступные ключи: {list(model_data.keys())[:10]}")

    elif isinstance(model_data, list):
        print("\n✅ Модель представлена как список")
        print(f"Элементов: {len(model_data)}")

    return True

# ============================================================================
# АНАЛИЗ ДРУГИХ ФАЙЛОВ МОДЕЛИ
# ============================================================================

def analyze_scalers():
    """Анализирует файлы скейлеров"""

    print("\n" + "-"*80)
    print("АНАЛИЗ СКЕЙЛЕРОВ")
    print("-"*80)

    scaler_x = PROJECT_ROOT / "models" / "scaler_X_v3.pkl"
    scaler_y = PROJECT_ROOT / "models" / "scaler_y.pkl"

    if scaler_x.exists():
        print(f"\n✅ Найден scaler_X_v3.pkl ({scaler_x.stat().st_size / 1024:.1f} KB)")
        try:
            import pickle
            with open(scaler_x, 'rb') as f:
                scaler_data = pickle.load(f)
            print(f"  Тип: {type(scaler_data)}")
            if hasattr(scaler_data, 'n_features_in_'):
                print(f"  Количество признаков: {scaler_data.n_features_in_}")
            if hasattr(scaler_data, 'scale_'):
                print(f"  Mean shape: {scaler_data.mean_.shape if hasattr(scaler_data, 'mean_') else '?'}")
        except Exception as e:
            print(f"  ⚠️ Ошибка при анализе: {e}")
    else:
        print(f"❌ scaler_X_v3.pkl не найден")

    if scaler_y.exists():
        print(f"\n✅ Найден scaler_y.pkl ({scaler_y.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"❌ scaler_y.pkl не найден")

def analyze_config():
    """Анализирует конфиг TP/SL"""

    print("\n" + "-"*80)
    print("АНАЛИЗ КОНФИГА TP/SL")
    print("-"*80)

    config_file = PROJECT_ROOT / "models" / "best_tp_sl_config.json"

    if config_file.exists():
        print(f"\n✅ Найден best_tp_sl_config.json")
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            print(f"  Содержимое:")
            if isinstance(config, dict):
                for key, val in config.items():
                    print(f"    {key}: {val}")
            elif isinstance(config, list):
                print(f"  Список с {len(config)} элементами")
                for i, item in enumerate(config[:3]):
                    print(f"    [{i}]: {item}")
                if len(config) > 3:
                    print(f"    ...")
        except Exception as e:
            print(f"  ⚠️ Ошибка при анализе: {e}")
    else:
        print(f"❌ best_tp_sl_config.json не найден")

# ============================================================================
# РЕКОМЕНДАЦИИ
# ============================================================================

def print_recommendations():
    """Выводит рекомендации"""

    print("\n" + "="*80)
    print("РЕКОМЕНДАЦИИ ДЛЯ СПРИНТА")
    print("="*80)

    print("""
✅ ЧТО ЕСТЬ:
  1. XGBoost модель (xgboost_model.json) - ГОТОВА К ИСПОЛЬЗОВАНИЮ
  2. Скейлеры для нормализации (scaler_X_v3.pkl, scaler_y.pkl)
  3. Конфиг TP/SL (best_tp_sl_config.json)

❌ ЧТО НУЖНО:
  1. Загрузить РЕАЛЬНЫЕ OHLCV данные из DVC
  2. Имеет смысл проверить: есть ли другие модели для других таймфреймов?

🚀 СЛЕДУЮЩИЕ ШАГИ:
  1. Загрузить данные: git lfs pull && dvc pull
  2. Запустить скрипт inventory_all_resources.py ещё раз
  3. Загрузить XGBoost модель в backtest framework
  4. Запустить V1 тест на реальных данных с ML scoring
  5. Продолжить 48-hour sprint план

📝 КОМАНДА ДЛЯ ЗАГРУЗКИ ДАННЫХ:
  cd /home/user/scarlet-sails
  git lfs pull
  dvc pull
  python3 scripts/inventory_all_resources.py
    """)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ПОЛНЫЙ АНАЛИЗ XGBOOST МОДЕЛИ")
    print("="*80)
    print(f"Проект: {PROJECT_ROOT}")

    # Анализируем модель
    if analyze_xgboost_json():
        # Анализируем доп. файлы
        analyze_scalers()
        analyze_config()
        print_recommendations()
    else:
        print("\n⚠️ Не удалось загрузить XGBoost модель")
        print("Убедитесь что модель находится в /models/ директории")

if __name__ == '__main__':
    main()
