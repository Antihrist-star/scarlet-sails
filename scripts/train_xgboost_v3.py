"""
Train XGBoost v3 with temporal split and fee-adjusted targets.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from analysis.simple_threshold_backtest import (
    evaluate_thresholds,
    select_optimal_threshold,
)
from core.feature_engine_v2 import FeatureSpecV3
from core.feature_loader import FeatureLoader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost v3 model")
    parser.add_argument("--config-path", type=str, default="configs/model2_training.yaml")
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--coin", type=str, help="Override coin from config (e.g., BTC)")
    parser.add_argument("--tf", type=str, help="Override timeframe from config (e.g., 15m)")
    parser.add_argument("--no-backtest", action="store_true", help="Skip threshold optimization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def compute_targets(df: pd.DataFrame, horizon: int, commission: float, slippage: float, target_type: str) -> pd.DataFrame:
    df = df.copy()
    entry_price = df["open"].shift(-1)
    exit_price = df["close"].shift(-horizon)

    raw_ret = (exit_price - entry_price) / entry_price
    round_trip_cost = (commission + slippage) * 2
    fee_ret = raw_ret - round_trip_cost

    df["raw_ret"] = raw_ret
    df["fee_ret"] = fee_ret
    df["rapnl"] = fee_ret  # Placeholder until RAPnL is formalised

    if target_type == "fee_ret":
        target_series = (df["fee_ret"] > 0).astype(int)
    elif target_type == "raw_ret":
        target_series = (df["raw_ret"] > 0).astype(int)
    else:
        target_series = (df["rapnl"] > 0).astype(int)

    df["target"] = target_series
    df = df.iloc[:-horizon]  # drop tail with NaNs/unknown targets

    return df.dropna(subset=["target"])


def temporal_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_end: str,
    test_end: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test sets by temporal boundaries.
    
    Handles both tz-aware and tz-naive datetime indices correctly.
    
    Args:
        df: DataFrame with DatetimeIndex
        train_start: Start of training period (ISO format string)
        train_end: End of training period (start of validation)
        val_end: End of validation period (start of test)
        test_end: End of test period (None = use all remaining data)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    
    Raises:
        TypeError: If df.index is not a DatetimeIndex
    """
    df_sorted = df.sort_index()
    
    if not isinstance(df_sorted.index, pd.DatetimeIndex):
        raise TypeError(f"temporal_split expects a DatetimeIndex, got {type(df_sorted.index)}")
    
    idx = df_sorted.index
    
    # Handle timezone-aware vs timezone-naive indices
    if idx.tz is not None:
        # Index is tz-aware (e.g., UTC) -> convert boundaries to same timezone
        tz = idx.tz
        logger.info(f"Index is tz-aware (timezone: {tz})")
        
        train_start_ts = pd.Timestamp(train_start, tz=tz)
        train_end_ts = pd.Timestamp(train_end, tz=tz)
        val_end_ts = pd.Timestamp(val_end, tz=tz)
        test_end_ts = pd.Timestamp(test_end, tz=tz) if test_end else None
    else:
        # Index is tz-naive -> keep boundaries tz-naive
        logger.info("Index is tz-naive")
        
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)
        test_end_ts = pd.Timestamp(test_end) if test_end else None
    
    # Log data range and split boundaries
    logger.info(f"Data range: {idx.min()} to {idx.max()}")
    logger.info(f"Train period: {train_start_ts} to {train_end_ts}")
    logger.info(f"Val period: {train_end_ts} to {val_end_ts}")
    if test_end_ts:
        logger.info(f"Test period: {val_end_ts} to {test_end_ts}")
    else:
        logger.info(f"Test period: {val_end_ts} to end")
    
    # Create masks for each period
    train_mask = (idx >= train_start_ts) & (idx < train_end_ts)
    val_mask = (idx >= train_end_ts) & (idx < val_end_ts)
    
    if test_end_ts is not None:
        test_mask = (idx >= val_end_ts) & (idx < test_end_ts)
    else:
        test_mask = idx >= val_end_ts
    
    train_df = df_sorted[train_mask]
    val_df = df_sorted[val_mask]
    test_df = df_sorted[test_mask]
    
    # Log split sizes
    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Warn if any split is empty
    if len(train_df) == 0:
        logger.warning("Training set is empty! Check date ranges.")
    if len(val_df) == 0:
        logger.warning("Validation set is empty! Check date ranges.")
    if len(test_df) == 0:
        logger.warning("Test set is empty! Check date ranges.")
    
    return train_df, val_df, test_df


def build_dmatrices(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, spec: FeatureSpecV3):
    X_train = spec.enforce(train_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")
    X_val = spec.enforce(val_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")
    X_test = spec.enforce(test_df, raise_on_missing=True).drop(columns=["target"], errors="ignore")

    y_train = train_df["target"]
    y_val = val_df["target"]
    y_test = test_df["target"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict) -> xgb.XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    params = params.copy()
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("scale_pos_weight", scale_pos_weight)
    params.setdefault("random_state", 42)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    return {
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
        "threshold": threshold,
        "samples": int(len(y)),
        "class_balance": float(y.mean()),
    }


def save_model(model: xgb.XGBClassifier, output_path: Path, feature_spec: FeatureSpecV3, metadata: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    meta_path = output_path.with_name(output_path.stem + "_metadata.json")
    metadata = metadata.copy()
    metadata.update({
        "feature_names": feature_spec.feature_names,
        "n_features": feature_spec.n_features,
    })
    meta_path.write_text(json.dumps(metadata, indent=2))


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cfg = load_config(args.config_path)

    model_cfg = cfg["model2"]
    costs_cfg = cfg.get("costs", {})
    backtest_cfg = cfg.get("backtest", {})

    # Override coin and timeframe from CLI if provided
    coin = args.coin if args.coin else model_cfg.get("coin", "BTC")
    timeframe = args.tf if args.tf else model_cfg.get("timeframe", "15m")

    print(f"\n{'='*60}")
    print(f"Training XGBoost v3 Model")
    print(f"{'='*60}")
    print(f"Coin: {coin}")
    print(f"Timeframe: {timeframe}")
    print(f"Config: {args.config_path}")
    print(f"{'='*60}\n")

    loader = FeatureLoader(data_dir=model_cfg.get("data_dir", "data/features"))
    file_path = loader.get_file_path(coin, timeframe)
    df = loader.load_features(
        coin=coin,
        timeframe=timeframe,
        start_date=model_cfg["train_start"],
        end_date=model_cfg.get("test_end"),
        validate=True
    )

    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Data index type: {type(df.index)}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"Timezone: {df.index.tz if df.index.tz else 'tz-naive'}")
        print(f"Data range: {df.index.min()} to {df.index.max()}\n")

    df_with_targets = compute_targets(
        df,
        horizon=model_cfg["horizon_bars"],
        commission=costs_cfg.get("commission", 0.001),
        slippage=costs_cfg.get("slippage", 0.0005),
        target_type=model_cfg.get("target_type", "fee_ret"),
    )

    print("Splitting data into train/val/test sets...")
    train_df, val_df, test_df = temporal_split(
        df_with_targets,
        train_start=model_cfg["train_start"],
        train_end=model_cfg["train_end"],
        val_end=model_cfg["val_end"],
        test_end=model_cfg.get("test_end"),
    )
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples\n")

    feature_spec = FeatureSpecV3.from_dataframe(train_df)
    if feature_spec.n_features != 74:
        raise ValueError(f"Expected 74 features, got {feature_spec.n_features}")

    X_train, X_val, X_test, y_train, y_val, y_test = build_dmatrices(train_df, val_df, test_df, feature_spec)

    print("Training XGBoost model...")
    model_params = model_cfg.get("xgboost_params", {})
    model = train_model(X_train, y_train, X_val, y_val, model_params)
    print("Model training complete.\n")

    print("Evaluating model performance...")
    metrics_train = evaluate_model(model, X_train, y_train, threshold=0.5)
    metrics_val = evaluate_model(model, X_val, y_val, threshold=0.5)
    metrics_test = evaluate_model(model, X_test, y_test, threshold=0.5)
    print(f"  Train AUC: {metrics_train['auc']:.4f}")
    print(f"  Val AUC:   {metrics_val['auc']:.4f}")
    print(f"  Test AUC:  {metrics_test['auc']:.4f}\n")

    # Threshold optimization on validation set
    optimal = {"threshold": 0.5, "sharpe": 0.0, "backtest_metrics": {}}
    
    if not args.no_backtest:
        print("Optimizing threshold on validation set...")
        val_prob = model.predict_proba(X_val)[:, 1]
        
        # Add probabilities to validation dataframe
        val_df_with_proba = val_df.copy()
        val_df_with_proba["P_ml"] = val_prob
        
        # Evaluate threshold grid
        thresholds = backtest_cfg.get("threshold_grid", [round(x, 2) for x in np.linspace(0.5, 0.9, 5)])
        bt_results = evaluate_thresholds(
            df=val_df_with_proba,
            proba_col="P_ml",
            fee_ret_col="fee_ret",
            thresholds=thresholds
        )
        
        # Select optimal threshold
        optimal = select_optimal_threshold(
            threshold_results=bt_results,
            max_dd_limit=backtest_cfg.get("max_dd_pct", 20.0),
            min_trades=backtest_cfg.get("min_trades", 10)
        )
        
        print(f"\nThreshold optimization complete:")
        print(f"  Best threshold: {optimal['threshold']:.2f}")
        print(f"  Sharpe ratio: {optimal['sharpe']:.4f}")
        print(f"  Trades: {optimal['backtest_metrics'].get('n_trades', 0)}")
        print(f"  Win rate: {optimal['backtest_metrics'].get('win_rate', 0):.2f}%")
        print(f"  Max DD: {optimal['backtest_metrics'].get('max_drawdown_pct', 0):.2f}%\n")
    else:
        print("Skipping threshold optimization (--no-backtest flag)\n")

    output_path = Path(model_cfg.get("output_path", f"models/xgboost_v3_{coin.lower()}_{timeframe}.json"))
    metadata = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "coin": coin,
        "timeframe": timeframe,
        "experiment": args.experiment_name,
        "source_data": str(file_path),
        "metrics": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "optimal_threshold_trading": optimal,
        "target_type": model_cfg.get("target_type", "fee_ret"),
        "horizon_bars": model_cfg.get("horizon_bars"),
    }

    save_model(model, output_path, feature_spec, metadata)

    summary = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "best_threshold": optimal,
    }
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    print(f"\nModel saved to: {output_path}")
    print(f"Metadata saved to: {output_path.with_name(output_path.stem + '_metadata.json')}")


if __name__ == "__main__":
    main()
