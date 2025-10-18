# scripts/run_backtest.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from backtesting.honest_backtest import HonestBacktestEngine

# --- Пути ---
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)

# Путь к модели и скалеру
model_path = os.path.join(project_root, "models", "logistic_baseline_clean_2d.pth")
scaler_path = os.path.join(project_root, "models", "scaler_clean_2d.pkl")

# Путь к ОРИГИНАЛЬНЫМ OHLCV данным (замените на ваш реальный путь)
data_path = os.path.join(project_root, "data", "raw", "BTC_USDT_15m_FULL.parquet") # Укажите правильный путь!

# --- Загрузка модели и скалера ---
print("Loading model and scaler...")
from models.logistic_baseline import LogisticBaseline
import joblib

# Загрузка архитектуры модели
input_dim = 31
model = LogisticBaseline(input_dim=input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

scaler = joblib.load(scaler_path)

# --- Загрузка тестовых данных (X_test_clean.pt) ---
print("Loading test features...")
X_test_clean = torch.load(os.path.join(project_root, "models", "X_test_clean.pt")).numpy()
y_test_clean = torch.load(os.path.join(project_root, "models", "y_test_clean.pt")).numpy() # Для сравнения

# --- Загрузка OHLCV данных ---
print(f"Loading OHLCV data from {data_path}...")
if not os.path.exists(data_path):
    print(f"Error: OHLCV data file not found at {data_path}")
    print("Please specify the correct path to your raw OHLCV data.")
    sys.exit(1)

try:
    data_raw = pd.read_parquet(data_path)
except Exception as e:
    print(f"Error loading OHLCV data: {e}")
    sys.exit(1)

# Убедимся, что колонки есть и в правильном регистре
required_cols = ['open', 'high', 'low', 'close']
data_raw.columns = data_raw.columns.str.lower()
if not all(col in data_raw.columns for col in required_cols):
    print(f"Error: OHLCV data must contain columns: {required_cols}")
    sys.exit(1)

# --- Проверка длины данных ---
if len(data_raw) < len(X_test_clean) + 2:
    print(f"Error: OHLCV data length ({len(data_raw)}) is insufficient for test features length ({len(X_test_clean)}) + 2 for entry/exit.")
    sys.exit(1)

# Возьмём последние N баров, соответствующих X_test_clean + 2 для обеспечения возможности покупки и продажи
# signals[0..M-1] -> buy on open[1..M] -> sell on open[2..M+1] (из ohlcv_subset)
# Для M сигналов нужно M+2 баров в подмножестве, чтобы последняя покупка (по сигналу M-1) могла продаться на баре M+1
ohlcv_subset = data_raw.tail(len(X_test_clean) + 2).reset_index(drop=True)

# --- Получение предсказаний модели ---
print("Generating predictions...")
with torch.no_grad():
    # НОРМАЛИЗУЕМ тестовые данные с помощью ЗАГРУЖЕННОГО скалера
    X_test_scaled = scaler.transform(X_test_clean) # X_test_clean уже 2D, нормализуем его
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_pred_proba = model.predict_proba(X_test_tensor)
    OPTIMAL_THRESHOLD = 0.69 # Замените на актуальное значение из model_comparison.py
    y_pred_signals = (y_pred_proba[:, 1] > OPTIMAL_THRESHOLD).astype(int) # <-- Сначала определяем y_pred_signals

    # --- ДОБАВЛЕННЫЕ СТРОКИ ДИАГНОСТИКИ ---
    print(f"Number of BUY signals (1): {(y_pred_signals == 1).sum()}")
    print(f"Number of DOWN signals (0): {(y_pred_signals == 0).sum()}")
    print(f"Max probability UP: {y_pred_proba[:, 1].max():.4f}")
    print(f"Min probability UP: {y_pred_proba[:, 1].min():.4f}")
    print(f"Mean probability UP: {y_pred_proba[:, 1].mean():.4f}")
    # --- КОНЕЦ ДОБАВЛЕННЫХ СТРОК ---

# Эти строки уже были, но теперь y_pred_signals определена
print(f"Number of signals generated: {len(y_pred_signals)}")
print(f"Number of OHLCV bars available in subset: {len(ohlcv_subset)}")

# --- Запуск честного бэктеста ---
print("Running honest backtest...")
engine = HonestBacktestEngine(commission_rate=0.001, slippage_rate=0.0005)

# ВАЖНО: Длина OHLCV данных должна быть ровно len(signals) + 2
# signals[0..M-1] -> покупка на open[1..M], продажа на open[2..M+1]
# Требуется len(data) >= len(signals) + 2, как проверено в honest_backtest.py
# Мы взяли ohlcv_subset длины len(signals) + 2, теперь нужно взять первые len(signals) + 2 баров
ohlcv_for_backtest = ohlcv_subset.head(len(y_pred_signals) + 2)

print(f"Adjusted OHLCV bars for backtest: {len(ohlcv_for_backtest)} (signals: {len(y_pred_signals)})")

results = engine.run_backtest(
    data=ohlcv_for_backtest,
    signals=y_pred_signals,
    initial_capital=100_000.0,
    position_size_pct=0.1
)

# --- Вывод результатов ---
print("\n" + "="*80)
print("HONEST BACKTEST RESULTS")
print("="*80)
print(f"Total Return: {results['total_return_pct']:.4f}%")
print(f"Total PnL: ${results['total_pnl']:.2f}")
print(f"Number of Trades: {results['num_trades']}")
print(f"Win Rate: {results['win_rate']*100:.2f}%")
print(f"Profit Factor: {results['profit_factor']:.4f}")
print(f"Max Drawdown: {results['max_drawdown_pct']:.4f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
print(f"Calmar Ratio: {results['calmar_ratio']:.4f}")
print("="*80)

if results['num_trades'] > 0:
    print("\nFirst 5 Trades (example):")
    for i, trade in enumerate(results['trades'][:5]):
        print(f"  Trade {i+1}: Entry Bar {trade['entry_bar']}, Exit Bar {trade['exit_bar']}, "
              f"P&L {trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
else:
    print("\nNo trades were executed during the backtest.")

print("\nBacktest completed.")
