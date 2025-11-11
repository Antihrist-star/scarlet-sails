#!/usr/bin/env python3
"""
MODEL COMPARATOR - Compare all 3 trading systems
Запускает все три модели на ОДИНАКОВЫХ данных:
1. Rule-Based
2. ML XGBoost
3. Hybrid

Показывает:
- Какие сигналы каждая взяла
- Где совпали, где разошлись
- Win Rate и Profit Factor по каждой
- Уникальные сигналы каждой модели
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.xgboost_model import XGBoostModel

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Constants
TARGET_TF = "15m"
FORWARD_BARS = 96  # 24 hours
PROFIT_THRESHOLD = 0.01  # 1%
RSI_THRESHOLD = 30.0


class RuleBasedModel:
    """Rule-based система: RSI < 30"""

    def __init__(self, rsi_threshold: float = 30.0):
        self.rsi_threshold = rsi_threshold

    def should_enter(self, df: pd.DataFrame, bar_index: int) -> Tuple[bool, str]:
        """Проверка входа по правилам"""

        # Find RSI column
        rsi_col = None
        for col in ['rsi', 'RSI_14', '15m_RSI_14']:
            if col in df.columns:
                rsi_col = col
                break

        if rsi_col is None:
            rsi_candidates = [col for col in df.columns if 'RSI' in col.upper()]
            if rsi_candidates:
                rsi_col = rsi_candidates[0]

        if rsi_col is None:
            return False, "RSI column not found"

        rsi = df[rsi_col].iloc[bar_index]

        if rsi < self.rsi_threshold:
            return True, f"RSI {rsi:.2f} < {self.rsi_threshold}"
        else:
            return False, f"RSI {rsi:.2f} >= {self.rsi_threshold}"


class MLModel:
    """ML-only система: XGBoost без Layer 1 фильтра"""

    def __init__(self, model_path: Path, scaler_path: Path, threshold: float):
        self.model = XGBoostModel()
        self.model.load(str(model_path))
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold
        self.extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    def should_enter(
        self,
        all_timeframes: Dict,
        bar_index: int
    ) -> Tuple[bool, str, float]:
        """Проверка входа по ML"""

        features = self.extractor.extract_features_at_bar(
            all_timeframes,
            TARGET_TF,
            bar_index
        )

        if features is None:
            return False, "Features extraction failed", 0.0

        # Scale
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        prob = self.model.predict_proba(features_scaled)[0]
        ml_score = prob[1]

        if ml_score >= self.threshold:
            return True, f"ML score {ml_score:.3f} >= {self.threshold}", ml_score
        else:
            return False, f"ML score {ml_score:.3f} < {self.threshold}", ml_score


class HybridModel:
    """Hybrid система: Rule + ML + Crisis"""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        ml_threshold: float,
        rsi_threshold: float = 30.0
    ):
        self.rsi_threshold = rsi_threshold
        self.model = XGBoostModel()
        self.model.load(str(model_path))
        self.scaler = joblib.load(scaler_path)
        self.ml_threshold = ml_threshold
        self.extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    def should_enter(
        self,
        df: pd.DataFrame,
        all_timeframes: Dict,
        bar_index: int
    ) -> Tuple[bool, str, Dict]:
        """Проверка входа через 3 слоя"""

        details = {
            'layer1_passed': False,
            'layer2_passed': False,
            'layer3_passed': False,
            'ml_score': None
        }

        # Layer 1: RSI
        rsi_col = None
        for col in ['rsi', 'RSI_14', '15m_RSI_14']:
            if col in df.columns:
                rsi_col = col
                break

        if rsi_col is None:
            rsi_candidates = [col for col in df.columns if 'RSI' in col.upper()]
            if rsi_candidates:
                rsi_col = rsi_candidates[0]

        if rsi_col is None:
            return False, "RSI column not found", details

        rsi = df[rsi_col].iloc[bar_index]

        if rsi >= self.rsi_threshold:
            return False, f"Layer 1 rejected: RSI {rsi:.2f} >= {self.rsi_threshold}", details

        details['layer1_passed'] = True

        # Layer 2: ML
        features = self.extractor.extract_features_at_bar(
            all_timeframes,
            TARGET_TF,
            bar_index
        )

        if features is None:
            return False, "Layer 2: Features extraction failed", details

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prob = self.model.predict_proba(features_scaled)[0]
        ml_score = prob[1]
        details['ml_score'] = ml_score

        if ml_score < self.ml_threshold:
            return False, f"Layer 2 rejected: ML {ml_score:.3f} < {self.ml_threshold}", details

        details['layer2_passed'] = True

        # Layer 3: Crisis (placeholder - always passes for now)
        details['layer3_passed'] = True

        return True, f"All layers passed (RSI={rsi:.2f}, ML={ml_score:.3f})", details


def calculate_trade_result(df: pd.DataFrame, entry_index: int) -> Optional[Dict]:
    """Вычислить результат трейда"""

    exit_index = entry_index + FORWARD_BARS

    if exit_index >= len(df):
        return None

    entry_price = df['close'].iloc[entry_index]
    exit_price = df['close'].iloc[exit_index]

    pnl_pct = (exit_price - entry_price) / entry_price * 100
    profit = pnl_pct > PROFIT_THRESHOLD * 100

    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'profit': profit
    }


def compare_models(
    asset: str = "BTC",
    max_bars: int = 10000
) -> Dict:
    """
    Сравнить все 3 модели на одинаковых данных

    Returns:
        {
            'rule_based': {...},
            'ml': {...},
            'hybrid': {...},
            'overlap_analysis': {...}
        }
    """

    print("\n" + "="*100)
    print("MODEL COMPARISON - All 3 Systems")
    print("="*100)

    # ═════════════════════════════════════════════════════════
    # LOAD MODELS
    # ═════════════════════════════════════════════════════════
    print(f"\n📦 Loading models...")

    model_path = MODELS_DIR / "xgboost_multi_tf_model.json"
    scaler_path = MODELS_DIR / "xgboost_multi_tf_scaler.pkl"
    threshold_path = MODELS_DIR / "xgboost_multi_tf_threshold.txt"

    if not all([model_path.exists(), scaler_path.exists()]):
        print("❌ Model files not found")
        return None

    with open(threshold_path, 'r') as f:
        ml_threshold = float(f.read().strip())

    rule_model = RuleBasedModel(rsi_threshold=RSI_THRESHOLD)
    ml_model = MLModel(model_path, scaler_path, threshold=ml_threshold)
    hybrid_model = HybridModel(model_path, scaler_path, ml_threshold=ml_threshold, rsi_threshold=RSI_THRESHOLD)

    print(f"✅ Rule-Based: RSI < {RSI_THRESHOLD}")
    print(f"✅ ML: XGBoost threshold {ml_threshold}")
    print(f"✅ Hybrid: 3-layer system")

    # ═════════════════════════════════════════════════════════
    # LOAD DATA
    # ═════════════════════════════════════════════════════════
    print(f"\n📊 Loading data for {asset}...")

    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
    all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, TARGET_TF)

    print(f"✅ Loaded {len(primary_df)} bars")

    # Limit bars for testing
    if max_bars and len(primary_df) > max_bars:
        print(f"⚠️  Limiting to {max_bars} bars (set max_bars=None for full test)")
        primary_df = primary_df.iloc[-max_bars:].copy()

        # Update all_timeframes as well
        start_time = primary_df.index[0]
        end_time = primary_df.index[-1]
        for tf_key in all_tf:
            all_tf[tf_key] = all_tf[tf_key][
                (all_tf[tf_key].index >= start_time) &
                (all_tf[tf_key].index <= end_time)
            ].copy()

    # ═════════════════════════════════════════════════════════
    # RUN BACKTEST FOR EACH MODEL
    # ═════════════════════════════════════════════════════════
    print(f"\n🔬 Testing all 3 models on {len(primary_df)} bars...")

    results = {
        'rule_based': {'signals': [], 'trades': []},
        'ml': {'signals': [], 'trades': []},
        'hybrid': {'signals': [], 'trades': []}
    }

    # Test each bar
    for bar_index in range(len(primary_df) - FORWARD_BARS):
        if bar_index % 1000 == 0:
            print(f"   Progress: {bar_index}/{len(primary_df) - FORWARD_BARS}")

        timestamp = primary_df.index[bar_index]
        price = primary_df['close'].iloc[bar_index]

        # Rule-Based
        rule_enter, rule_reason = rule_model.should_enter(primary_df, bar_index)
        if rule_enter:
            trade_result = calculate_trade_result(primary_df, bar_index)
            if trade_result:
                results['rule_based']['signals'].append(bar_index)
                results['rule_based']['trades'].append({
                    'bar_index': bar_index,
                    'timestamp': timestamp,
                    'price': price,
                    **trade_result
                })

        # ML
        ml_enter, ml_reason, ml_score = ml_model.should_enter(all_tf, bar_index)
        if ml_enter:
            trade_result = calculate_trade_result(primary_df, bar_index)
            if trade_result:
                results['ml']['signals'].append(bar_index)
                results['ml']['trades'].append({
                    'bar_index': bar_index,
                    'timestamp': timestamp,
                    'price': price,
                    'ml_score': ml_score,
                    **trade_result
                })

        # Hybrid
        hybrid_enter, hybrid_reason, hybrid_details = hybrid_model.should_enter(
            primary_df, all_tf, bar_index
        )
        if hybrid_enter:
            trade_result = calculate_trade_result(primary_df, bar_index)
            if trade_result:
                results['hybrid']['signals'].append(bar_index)
                results['hybrid']['trades'].append({
                    'bar_index': bar_index,
                    'timestamp': timestamp,
                    'price': price,
                    'ml_score': hybrid_details['ml_score'],
                    **trade_result
                })

    print(f"✅ Backtest complete")

    # ═════════════════════════════════════════════════════════
    # CALCULATE STATISTICS
    # ═════════════════════════════════════════════════════════
    print(f"\n📈 Calculating statistics...")

    for model_name in ['rule_based', 'ml', 'hybrid']:
        trades = results[model_name]['trades']

        if len(trades) == 0:
            results[model_name]['stats'] = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
            continue

        wins = [t for t in trades if t['profit']]
        losses = [t for t in trades if not t['profit']]

        total_pnl = sum(t['pnl_pct'] for t in trades)
        win_rate = len(wins) / len(trades) * 100

        total_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0.01  # Avoid div by zero
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        results[model_name]['stats'] = {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades),
            'avg_win': total_profit / len(wins) if wins else 0,
            'avg_loss': -total_loss / len(losses) if losses else 0
        }

    # ═════════════════════════════════════════════════════════
    # OVERLAP ANALYSIS
    # ═════════════════════════════════════════════════════════
    print(f"\n🔍 Analyzing signal overlaps...")

    rule_signals = set(results['rule_based']['signals'])
    ml_signals = set(results['ml']['signals'])
    hybrid_signals = set(results['hybrid']['signals'])

    overlap = {
        'rule_only': rule_signals - ml_signals - hybrid_signals,
        'ml_only': ml_signals - rule_signals - hybrid_signals,
        'hybrid_only': hybrid_signals - rule_signals - ml_signals,
        'rule_and_ml': (rule_signals & ml_signals) - hybrid_signals,
        'rule_and_hybrid': (rule_signals & hybrid_signals) - ml_signals,
        'ml_and_hybrid': (ml_signals & hybrid_signals) - rule_signals,
        'all_three': rule_signals & ml_signals & hybrid_signals
    }

    results['overlap_analysis'] = {
        'rule_signals_total': len(rule_signals),
        'ml_signals_total': len(ml_signals),
        'hybrid_signals_total': len(hybrid_signals),
        'rule_only': len(overlap['rule_only']),
        'ml_only': len(overlap['ml_only']),
        'hybrid_only': len(overlap['hybrid_only']),
        'rule_and_ml': len(overlap['rule_and_ml']),
        'rule_and_hybrid': len(overlap['rule_and_hybrid']),
        'ml_and_hybrid': len(overlap['ml_and_hybrid']),
        'all_three': len(overlap['all_three'])
    }

    # Analyze unique signals
    for signal_type, indices in overlap.items():
        if len(indices) == 0:
            continue

        # Get trades for these indices
        if 'rule' in signal_type and signal_type != 'ml_only' and signal_type != 'hybrid_only':
            model_trades = results['rule_based']['trades']
        elif 'ml' in signal_type and signal_type != 'rule_only' and signal_type != 'hybrid_only':
            model_trades = results['ml']['trades']
        elif 'hybrid' in signal_type and signal_type != 'rule_only' and signal_type != 'ml_only':
            model_trades = results['hybrid']['trades']
        else:
            continue

        # Filter trades
        filtered_trades = [t for t in model_trades if t['bar_index'] in indices]

        if len(filtered_trades) > 0:
            wins = [t for t in filtered_trades if t['profit']]
            wr = len(wins) / len(filtered_trades) * 100
            total_pnl = sum(t['pnl_pct'] for t in filtered_trades)

            results['overlap_analysis'][f'{signal_type}_wr'] = wr
            results['overlap_analysis'][f'{signal_type}_pnl'] = total_pnl

    return results


def print_comparison_results(results: Dict):
    """Красиво печатает результаты сравнения"""

    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)

    # Individual model stats
    print("\n" + "─"*100)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("─"*100)

    print(f"\n{'Model':<20} {'Trades':<10} {'Win Rate':<12} {'Profit Factor':<15} {'Total PnL':<12} {'Avg PnL':<12}")
    print("─"*100)

    for model_name in ['rule_based', 'ml', 'hybrid']:
        stats = results[model_name]['stats']
        display_name = {
            'rule_based': '1. Rule-Based',
            'ml': '2. ML XGBoost',
            'hybrid': '3. Hybrid 3-Layer'
        }[model_name]

        print(f"{display_name:<20} "
              f"{stats['total_trades']:<10} "
              f"{stats['win_rate']:>6.2f}% {' '*4} "
              f"{stats['profit_factor']:>6.2f} {' '*8} "
              f"{stats['total_pnl']:>+7.2f}% {' '*3} "
              f"{stats['avg_pnl']:>+7.3f}%")

    # Overlap analysis
    print("\n" + "─"*100)
    print("SIGNAL OVERLAP ANALYSIS")
    print("─"*100)

    overlap = results['overlap_analysis']

    print(f"\nTotal signals:")
    print(f"  Rule-Based: {overlap['rule_signals_total']}")
    print(f"  ML:         {overlap['ml_signals_total']}")
    print(f"  Hybrid:     {overlap['hybrid_signals_total']}")

    print(f"\nVenn Diagram breakdown:")
    print(f"  Rule-only:          {overlap['rule_only']:>6} signals")
    print(f"  ML-only:            {overlap['ml_only']:>6} signals")
    print(f"  Hybrid-only:        {overlap['hybrid_only']:>6} signals")
    print(f"  Rule ∩ ML:          {overlap['rule_and_ml']:>6} signals")
    print(f"  Rule ∩ Hybrid:      {overlap['rule_and_hybrid']:>6} signals")
    print(f"  ML ∩ Hybrid:        {overlap['ml_and_hybrid']:>6} signals")
    print(f"  All three:          {overlap['all_three']:>6} signals")

    # Key insights
    print("\n" + "─"*100)
    print("KEY INSIGHTS")
    print("─"*100)

    # Which model is best?
    models_stats = {
        'Rule-Based': results['rule_based']['stats'],
        'ML': results['ml']['stats'],
        'Hybrid': results['hybrid']['stats']
    }

    best_wr = max(models_stats.items(), key=lambda x: x[1]['win_rate'])
    best_pf = max(models_stats.items(), key=lambda x: x[1]['profit_factor'])
    best_total_pnl = max(models_stats.items(), key=lambda x: x[1]['total_pnl'])
    most_trades = max(models_stats.items(), key=lambda x: x[1]['total_trades'])

    print(f"\n🏆 Best Win Rate:      {best_wr[0]} ({best_wr[1]['win_rate']:.2f}%)")
    print(f"🏆 Best Profit Factor: {best_pf[0]} ({best_pf[1]['profit_factor']:.2f})")
    print(f"🏆 Best Total PnL:     {best_total_pnl[0]} ({best_total_pnl[1]['total_pnl']:+.2f}%)")
    print(f"📊 Most Trades:        {most_trades[0]} ({most_trades[1]['total_trades']})")

    # Are models different?
    rule_total = overlap['rule_signals_total']
    ml_total = overlap['ml_signals_total']
    hybrid_total = overlap['hybrid_signals_total']
    all_three_overlap = overlap['all_three']

    if rule_total > 0:
        rule_unique_pct = (rule_total - all_three_overlap) / rule_total * 100
    else:
        rule_unique_pct = 0

    if ml_total > 0:
        ml_unique_pct = (ml_total - all_three_overlap) / ml_total * 100
    else:
        ml_unique_pct = 0

    if hybrid_total > 0:
        hybrid_unique_pct = (hybrid_total - all_three_overlap) / hybrid_total * 100
    else:
        hybrid_unique_pct = 0

    print(f"\n🔍 Signal Uniqueness:")
    print(f"   Rule-Based: {rule_unique_pct:.1f}% unique signals (not shared with all 3)")
    print(f"   ML:         {ml_unique_pct:.1f}% unique signals")
    print(f"   Hybrid:     {hybrid_unique_pct:.1f}% unique signals")

    if all([rule_unique_pct > 50, ml_unique_pct > 50, hybrid_unique_pct > 50]):
        print(f"\n✅ VERDICT: Модели ФУНДАМЕНТАЛЬНО РАЗНЫЕ (>50% уникальных сигналов)")
        print(f"   → Имеет смысл держать все 3 модели (ловят разные паттерны)")
    elif hybrid_total > 0 and all_three_overlap / hybrid_total > 0.8:
        print(f"\n⚠️  VERDICT: Hybrid похожа на другие модели (>80% overlap)")
        print(f"   → Возможно, достаточно одной модели")
    else:
        print(f"\n🤔 VERDICT: Частичное пересечение")
        print(f"   → Требуется дальнейший анализ")


def main():
    """Main comparison"""

    print("="*100)
    print("MODEL COMPARISON TOOL")
    print("="*100)
    print("\nThis script compares all 3 trading systems on the same data:")
    print("  1. Rule-Based (RSI < 30)")
    print("  2. ML XGBoost (31 features)")
    print("  3. Hybrid 3-Layer (Rules + ML + Crisis)")

    # Run comparison
    results = compare_models(
        asset="BTC",
        max_bars=10000  # Test on 10K bars (~104 days of 15m data)
    )

    if results is None:
        return

    # Print results
    print_comparison_results(results)

    # Save to JSON
    output_file = PROJECT_ROOT / "reports" / "model_comparison.json"
    output_file.parent.mkdir(exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {}
    for model_name in ['rule_based', 'ml', 'hybrid']:
        json_results[model_name] = {
            'stats': results[model_name]['stats'],
            'signal_count': len(results[model_name]['signals'])
        }
    json_results['overlap_analysis'] = results['overlap_analysis']

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
