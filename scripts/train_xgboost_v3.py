"""Train XGBoost v3
================

Обучение XGBoost модели на 74 features (single timeframe).

Использование:
    python scripts/train_xgboost_v3.py

Результат:
    models/xgboost_v3_{coin}_{tf}.json
    models/xgboost_v3_{coin}_{tf}_metadata.json
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, 
    recall_score, accuracy_score, confusion_matrix,
    precision_recall_curve
)


def load_data(parquet_path: str) -> tuple:
    """
    Загрузить данные из parquet.
    
    Returns
    -------
    tuple
        (X, y, feature_names, df)
    """
    print(f"📂 Загрузка: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"   Размер: {len(df):,} строк, {len(df.columns)} колонок")
    print(f"   Период: {df.index[0]} — {df.index[-1]}")
    
    if 'target' not in df.columns:
        raise ValueError("Колонка 'target' не найдена!")
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    inf_count = np.isinf(X.values).sum()
    nan_count = np.isnan(X.values).sum()
    
    if inf_count > 0 or nan_count > 0:
        print(f"   ⚠️ Найдено inf: {inf_count}, nan: {nan_count}")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Class balance: {y.mean():.2%} (class 1)")
    
    return X, y, list(X.columns), df


def temporal_split(X, y, train_ratio: float = 0.8) -> tuple:
    """
    Временной split (без перемешивания).
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n📄 Split:")
    print(f"   Train: {len(X_train):,} samples ({train_ratio:.0%})")
    print(f"   Test:  {len(X_test):,} samples ({1-train_ratio:.0%})")
    print(f"   Train class 1: {y_train.mean():.2%}")
    print(f"   Test class 1:  {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, split_idx


def train_model(X_train, y_train, X_test, y_test, params: dict = None) -> xgb.XGBClassifier:
    """
    Обучить XGBoost модель.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 500,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    if params:
        default_params.update(params)
    
    print(f"\n🔧 Параметры:")
    for k, v in default_params.items():
        if k not in ['n_jobs', 'random_state']:
            print(f"   {k}: {v}")
    
    print(f"\n🚀 Обучение...")
    
    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    
    return model

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Оценить качество модели.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "threshold": threshold
    }
    
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])
    
    return metrics


def find_optimal_threshold(model, X_test, y_test) -> dict:
    """
    Найти оптимальный threshold по F1.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    
    return {
        "optimal_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "precision_at_best": float(precision[best_idx]),
        "recall_at_best": float(recall[best_idx])
    }


def save_model(model: xgb.XGBClassifier, output_path: str, feature_names: list, metrics: dict, threshold_info: dict, parquet_path: str):
    """
    Сохранить модель и metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"\n💾 Модель: {output_path}")
    
    metadata = {
        "created_at": datetime.now().isoformat(),
        "source_data": parquet_path,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
        "optimal_threshold": threshold_info,
        "model_params": model.get_params()
    }
    
    metadata_path = output_path.parent / (output_path.stem + '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"💾 Metadata: {metadata_path}")


def print_results(metrics: dict, threshold_info: dict, criteria: dict):
    """
    Вывести результаты и проверить критерии.
    """
    print("\n" + "="*60)
    print("📄 РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"\n   Threshold 0.5:")
    print(f"   AUC:       {metrics['auc']:.4f}")
    print(f"   F1:        {metrics['f1']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"\n   Optimal threshold ({threshold_info['optimal_threshold']:.3f}):")
    print(f"   F1:        {threshold_info['best_f1']:.4f}")
    print(f"   Precision: {threshold_info['precision_at_best']:.4f}")
    print(f"   Recall:    {threshold_info['recall_at_best']:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {metrics['true_negatives']:,}  FP: {metrics['false_positives']:,}")
    print(f"   FN: {metrics['false_negatives']:,}  TP: {metrics['true_positives']:,}")
    print("\n" + "="*60)
    print("✅ КРИТЕРИИ")
    print("="*60)
    checks = {
        "AUC > 0.60": metrics['auc'] > criteria.get('auc', 0.60),
        "F1 > 0.50": threshold_info['best_f1'] > criteria.get('f1', 0.50),
        "Precision > 0.45": threshold_info['precision_at_best'] > criteria.get('precision', 0.45),
        "Recall > 0.40": threshold_info['recall_at_best'] > criteria.get('recall', 0.40)
    }
    all_pass = True
    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {name}")
        if not passed:
            all_pass = False
    print("\n" + "="*60)
    if all_pass:
        print("🎉 ВСЕ КРИТЕРИИ ПРОЙДЕНЫ!")
    else:
        print("⚠️  НЕ ВСЕ КРИТЕРИИ ПРОЙДЕНЫ")
    print("="*60)
    return all_pass


def main():
    """Главная функция."""
    print("\n" + "="*60)
    print("🚀 XGBOOST TRAINING v3")
    print("="*60)
    
    PARQUET_PATH = "data/features/BTC_USDT_15m_features.parquet"
    OUTPUT_PATH = "models/xgboost_v3_btc_15m.json"
    CRITERIA = {
        "auc": 0.60,
        "f1": 0.50,
        "precision": 0.45,
        "recall": 0.40
    }
    
    X, y, feature_names, df = load_data(PARQUET_PATH)
    X_train, X_test, y_train, y_test, split_idx = temporal_split(X, y, 0.8)
    model = train_model(X_train, y_train, X_test, y_test)
    metrics = evaluate_model(model, X_test, y_test, threshold=0.5)
    threshold_info = find_optimal_threshold(model, X_test, y_test)
    all_pass = print_results(metrics, threshold_info, CRITERIA)
    
    save_model(
        model=model,
        output_path=OUTPUT_PATH,
        feature_names=feature_names,
        metrics=metrics,
        threshold_info=threshold_info,
        parquet_path=PARQUET_PATH
    )
    
    print(f"\n✅ Готово!")
    return all_pass


if __name__ == "__main__":
    main()
