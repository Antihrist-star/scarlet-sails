#!/usr/bin/env python3
from pathlib import Path
import json

PROJECT_ROOT = Path("C:\\Users\\Dmitriy\\scarlet-sails")
data_dir = PROJECT_ROOT / "data" / "raw"
models_dir = PROJECT_ROOT / "models"

# Данные
parquet_files = list(data_dir.glob("*.parquet"))
print(f"\n✅ OHLCV файлы: {len(parquet_files)}")
for f in sorted(parquet_files)[:5]:
    print(f"   {f.name}")

# Модели XGBoost
xgb_files = list(models_dir.glob("xgboost*.json"))
print(f"\n✅ XGBoost модели: {len(xgb_files)}")
for f in xgb_files:
    size = f.stat().st_size / 1024
    print(f"   {f.name} ({size:.0f} KB)")

# Компоненты
py_files = list(models_dir.glob("*.py"))
print(f"\n✅ Python компоненты: {len(py_files)}")
for f in sorted(py_files)[:5]:
    print(f"   {f.name}")

print(f"\n📊 ИТОГО:")
print(f"   OHLCV: {len(parquet_files)} файлов")
print(f"   XGBoost модели: {len(xgb_files)}")
print(f"   Компоненты: {len(py_files)}")