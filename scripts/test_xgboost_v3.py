"""
Test XGBoost v3 End-to-End
==========================

Проверка что модель работает от загрузки до сигнала.

Использование:
    python scripts/test_xgboost_v3.py
"""

import pandas as pd
from pathlib import Path


def test_end_to_end():
    """Тест: загрузка данных → модель → сигнал."""
    
    print("\n" + "="*60)
    print("🧪 TEST: XGBoost v3 End-to-End")
    print("="*60)
    
    # 1. Проверить что файлы существуют
    print("\n1️⃣ Проверка файлов...")
    
    data_path = Path("data/features/BTC_USDT_15m_features.parquet")
    model_path = Path("models/xgboost_v3_btc_15m.json")
    strategy_path = Path("strategies/xgboost_ml_v3.py")
    
    if not data_path.exists():
        print(f"   ❌ Данные не найдены: {data_path}")
        return False
    print(f"   ✅ Данные: {data_path}")
    
    if not model_path.exists():
        print(f"   ❌ Модель не найдена: {model_path}")
        print(f"   → Сначала запусти: python scripts/train_xgboost_v3.py")
        return False
    print(f"   ✅ Модель: {model_path}")
    
    if not strategy_path.exists():
        print(f"   ❌ Стратегия не найдена: {strategy_path}")
        return False
    print(f"   ✅ Стратегия: {strategy_path}")
    
    # 2. Импортировать стратегию
    print("\n2️⃣ Импорт стратегии...")
    try:
        from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3
        print("   ✅ Import успешен")
    except ImportError as e:
        print(f"   ❌ Import ошибка: {e}")
        return False
    
    # 3. Загрузить модель
    print("\n3️⃣ Загрузка модели...")
    try:
        strategy = XGBoostMLStrategyV3(str(model_path))
        print(f"   ✅ Модель загружена: {strategy}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки: {e}")
        return False
    
    # 4. Загрузить данные
    print("\n4️⃣ Загрузка данных...")
    try:
        df = pd.read_parquet(data_path)
        print(f"   ✅ Загружено {len(df):,} строк")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки данных: {e}")
        return False
    
    # 5. Получить features для последнего бара
    print("\n5️⃣ Подготовка features...")
    try:
        features = df.drop(columns=['target']).iloc[-1:]
        print(f"   ✅ Features shape: {features.shape}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False
    
    # 6. Генерировать сигнал
    print("\n6️⃣ Генерация сигнала...")
    try:
        result = strategy.generate_signal(features)
        print(f"   ✅ Сигнал получен:")
        print(f"      Signal:      {result['signal']}")
        print(f"      Probability: {result['probability']:.4f}")
        print(f"      P_ml:        {result['P_ml']:.4f}")
        print(f"      Threshold:   {result['threshold']}")
        print(f"      Filters OK:  {result['filters_pass']}")
    except Exception as e:
        print(f"   ❌ Ошибка генерации: {e}")
        return False
    
    # 7. Тест batch prediction
    print("\n7️⃣ Batch prediction (последние 100 баров)...")
    try:
        test_df = df.tail(100)
        result_df = strategy.generate_signals_batch(test_df, threshold=0.5)
        signals_count = result_df['ml_signal'].sum()
        print(f"   ✅ Сигналов: {signals_count} из 100")
        print(f"   Средняя probability: {result_df['ml_proba'].mean():.4f}")
    except Exception as e:
        print(f"   ❌ Ошибка batch: {e}")
        return False
    
    # 8. Тест evaluate
    print("\n8️⃣ Evaluate на test set...")
    try:
        # Последние 20% данных как test
        split = int(len(df) * 0.8)
        X_test = df.drop(columns=['target']).iloc[split:]
        y_test = df['target'].iloc[split:]
        
        metrics = strategy.evaluate(X_test, y_test, threshold=0.5)
        print(f"   ✅ Метрики:")
        print(f"      AUC:       {metrics['auc']:.4f}")
        print(f"      F1:        {metrics['f1']:.4f}")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall:    {metrics['recall']:.4f}")
    except Exception as e:
        print(f"   ❌ Ошибка evaluate: {e}")
        return False
    
    # 9. Тест optimal threshold
    print("\n9️⃣ Optimal threshold...")
    try:
        opt = strategy.find_optimal_threshold(X_test, y_test)
        print(f"   ✅ Optimal threshold: {opt['optimal_threshold']:.3f}")
        print(f"      Best F1: {opt['best_f1']:.4f}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False
    
    # ИТОГ
    print("\n" + "="*60)
    print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("="*60)
    print("\nModel 2 (XGBoost) работает end-to-end:")
    print("   ✅ Загрузка модели")
    print("   ✅ Single prediction")
    print("   ✅ Batch prediction")
    print("   ✅ Evaluate")
    print("   ✅ Optimal threshold")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_end_to_end()
    exit(0 if success else 1)
