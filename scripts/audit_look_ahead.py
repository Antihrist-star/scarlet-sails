# scripts/audit_look_ahead.py

import pandas as pd
import numpy as np

def audit_look_ahead_bias(df):
    """
    Проверка на утечку будущих данных
    """
    
    print("=== LOOK-AHEAD BIAS AUDIT ===\n")
    
    # 1. Проверка multi-timeframe features
    print("1. Checking multi-timeframe features...")
    
    # Пример правильного подхода:
    # Для 15m свечи в 10:00 используем ЗАВЕРШЕННУЮ 1h свечу до 10:00
    
    # НЕПРАВИЛЬНО:
    # df['1h_close'] = df.resample('1h')['close'].last()
    
    # ПРАВИЛЬНО:
    # df['1h_close'] = df.resample('1h')['close'].last().shift(1)
    #                                                      ^^^^^^^^ КРИТИЧНО!
    
    # Проверяем все колонки с "_1h", "_4h", "_1d"
    tf_columns = [col for col in df.columns if any(x in col for x in ['_1h', '_4h', '_1d'])]
    
    for col in tf_columns:
        # Проверяем: есть ли корреляция с БУДУЩИМИ ценами?
        future_corr = df[col].corr(df['close'].shift(-96))  # 24h вперед
        
        if abs(future_corr) > 0.3:  # Подозрительно высокая корреляция
            print(f"⚠️ WARNING: {col} has {future_corr:.3f} correlation with future!")
        else:
            print(f"✅ {col}: OK ({future_corr:.3f})")
    
    
    # 2. Проверка индикаторов
    print("\n2. Checking indicators...")
    
    indicator_cols = [col for col in df.columns if any(x in col for x in ['ema', 'sma', 'rsi', 'macd'])]
    
    for col in indicator_cols:
        # Индикаторы должны использовать только ПРОШЛЫЕ данные
        # Проверяем: первые N значений должны быть NaN (где N = период индикатора)
        
        first_valid_idx = df[col].first_valid_index()
        
        if first_valid_idx == 0:
            print(f"⚠️ WARNING: {col} has no initial NaN values - possible look-ahead!")
        else:
            print(f"✅ {col}: OK (starts at index {first_valid_idx})")
    
    
    # 3. Проверка labels
    print("\n3. Checking labels...")
    
    # Labels должны смотреть В БУДУЩЕЕ от текущей свечи
    # Но features НЕ должны содержать информацию из того же будущего
    
    # Тест: обучаем dummy model только на последней свече
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Берём только последний close как feature
    X_test = df[['close']].iloc[1000:2000].values
    y_test = df['label'].iloc[1000:2000].values
    
    # Обучаем на предыдущих данных
    X_train = df[['close']].iloc[:1000].values
    y_train = df['label'].iloc[:1000].values
    
    dummy_model = LogisticRegression()
    dummy_model.fit(X_train, y_train)
    
    dummy_acc = accuracy_score(y_test, dummy_model.predict(X_test))
    
    print(f"\nDummy model accuracy (only close): {dummy_acc:.3f}")
    
    if dummy_acc > 0.65:
        print("⚠️ WARNING: Labels might be leaking - too easy to predict!")
    else:
        print("✅ Labels: OK")
    
    print("\n=== AUDIT COMPLETE ===")


# ЗАПУСК:
df = pd.read_csv('data/processed/processed_data.csv')
audit_look_ahead_bias(df)