#!/usr/bin/env python3
"""
TRADE EXPLAINER - HYBRID MODEL
Объясняет каждый трейд Hybrid модели в деталях:
- Все 31 признак (raw + scaled)
- Решение каждого из 3 слоев
- XGBoost prediction breakdown
- Результат трейда
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from typing import Dict, List, Optional, Tuple
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


class HybridTradeExplainer:
    """Объясняет каждое решение Hybrid модели"""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        features_path: Path,
        threshold_path: Path,
        rsi_threshold: float = 30.0
    ):
        self.model = XGBoostModel()
        self.model.load(str(model_path))

        self.scaler = joblib.load(scaler_path)

        with open(features_path, 'r') as f:
            meta = json.load(f)
            self.feature_names = meta['feature_names']

        with open(threshold_path, 'r') as f:
            self.ml_threshold = float(f.read().strip())

        self.rsi_threshold = rsi_threshold

        print(f"✅ Loaded Hybrid model:")
        print(f"   ML threshold: {self.ml_threshold}")
        print(f"   RSI threshold: {self.rsi_threshold}")
        print(f"   Features: {len(self.feature_names)}")

    def explain_trade(
        self,
        all_timeframes: Dict[str, pd.DataFrame],
        primary_df: pd.DataFrame,
        bar_index: int,
        asset: str
    ) -> Dict:
        """
        Полное объяснение решения для одного бара

        Returns:
            {
                'bar_index': int,
                'timestamp': str,
                'price': float,
                'layer1': {...},  # RSI check
                'layer2': {...},  # ML check
                'layer3': {...},  # Crisis check
                'final_decision': bool,
                'trade_result': {...}  # если был трейд
            }
        """
        timestamp = primary_df.index[bar_index]
        price = primary_df['close'].iloc[bar_index]

        explanation = {
            'bar_index': bar_index,
            'timestamp': str(timestamp),
            'price': price,
            'asset': asset
        }

        # ════════════════════════════════════════════════════════
        # LAYER 1: RULE-BASED FILTER (RSI)
        # ════════════════════════════════════════════════════════
        layer1 = self._explain_layer1(primary_df, bar_index)
        explanation['layer1'] = layer1

        if not layer1['passed']:
            explanation['final_decision'] = False
            explanation['rejection_layer'] = 1
            explanation['rejection_reason'] = layer1['reason']
            return explanation

        # ════════════════════════════════════════════════════════
        # LAYER 2: ML FILTER
        # ════════════════════════════════════════════════════════
        layer2 = self._explain_layer2(all_timeframes, bar_index)
        explanation['layer2'] = layer2

        if not layer2['passed']:
            explanation['final_decision'] = False
            explanation['rejection_layer'] = 2
            explanation['rejection_reason'] = layer2['reason']
            # Still calculate what would happen
            explanation['trade_result'] = self._calculate_trade_result(primary_df, bar_index)
            return explanation

        # ════════════════════════════════════════════════════════
        # LAYER 3: CRISIS GATE (placeholder - not implemented yet)
        # ════════════════════════════════════════════════════════
        layer3 = self._explain_layer3(primary_df, bar_index)
        explanation['layer3'] = layer3

        if not layer3['passed']:
            explanation['final_decision'] = False
            explanation['rejection_layer'] = 3
            explanation['rejection_reason'] = layer3['reason']
            explanation['trade_result'] = self._calculate_trade_result(primary_df, bar_index)
            return explanation

        # ════════════════════════════════════════════════════════
        # ALL LAYERS PASSED → ENTER TRADE
        # ════════════════════════════════════════════════════════
        explanation['final_decision'] = True
        explanation['rejection_layer'] = None
        explanation['rejection_reason'] = None
        explanation['trade_result'] = self._calculate_trade_result(primary_df, bar_index)

        return explanation

    def _explain_layer1(self, df: pd.DataFrame, bar_index: int) -> Dict:
        """Объяснение Layer 1: RSI filter"""

        # Find RSI column
        rsi_col = None
        for col in ['rsi', 'RSI_14', '15m_RSI_14']:
            if col in df.columns:
                rsi_col = col
                break

        if rsi_col is None:
            # Fallback search
            rsi_candidates = [col for col in df.columns if 'RSI' in col.upper()]
            if rsi_candidates:
                rsi_col = rsi_candidates[0]

        if rsi_col is None:
            return {
                'passed': False,
                'reason': 'RSI column not found',
                'rsi_value': None,
                'rsi_threshold': self.rsi_threshold
            }

        rsi_value = df[rsi_col].iloc[bar_index]
        passed = rsi_value < self.rsi_threshold

        return {
            'passed': passed,
            'reason': f"RSI {rsi_value:.2f} {'<' if passed else '>='} {self.rsi_threshold}" +
                     (" → Oversold ✅" if passed else " → Not oversold ❌"),
            'rsi_value': float(rsi_value),
            'rsi_threshold': self.rsi_threshold,
            'rsi_column': rsi_col
        }

    def _explain_layer2(self, all_timeframes: Dict, bar_index: int) -> Dict:
        """Объяснение Layer 2: ML filter"""

        extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

        # Extract raw features
        features_raw = extractor.extract_features_at_bar(
            all_timeframes,
            TARGET_TF,
            bar_index
        )

        if features_raw is None:
            return {
                'passed': False,
                'reason': 'Could not extract features (NaN)',
                'features_raw': None,
                'features_scaled': None,
                'ml_score': None
            }

        # Scale features
        features_scaled = self.scaler.transform(features_raw.reshape(1, -1))[0]

        # ML prediction
        prob = self.model.predict_proba(features_raw.reshape(1, -1))[0]
        ml_score = prob[1]  # Probability of UP

        passed = ml_score >= self.ml_threshold

        # Feature breakdown
        feature_breakdown = []
        for i, (name, raw_val, scaled_val) in enumerate(
            zip(self.feature_names, features_raw, features_scaled)
        ):
            feature_breakdown.append({
                'index': i,
                'name': name,
                'raw_value': float(raw_val),
                'scaled_value': float(scaled_val)
            })

        # Top 5 most extreme scaled features
        abs_scaled = np.abs(features_scaled)
        top_indices = np.argsort(abs_scaled)[-5:][::-1]
        top_features = [feature_breakdown[i] for i in top_indices]

        return {
            'passed': passed,
            'reason': f"ML score {ml_score:.3f} {'>=' if passed else '<'} {self.ml_threshold}" +
                     (f" → High confidence UP ✅" if passed else f" → Low confidence ❌"),
            'ml_score': float(ml_score),
            'ml_threshold': self.ml_threshold,
            'prob_down': float(prob[0]),
            'prob_up': float(prob[1]),
            'features_raw': [float(x) for x in features_raw],
            'features_scaled': [float(x) for x in features_scaled],
            'feature_breakdown': feature_breakdown,
            'top_5_features': top_features
        }

    def _explain_layer3(self, df: pd.DataFrame, bar_index: int) -> Dict:
        """Объяснение Layer 3: Crisis gate"""

        # Placeholder - crisis detection not implemented yet
        return {
            'passed': True,
            'reason': 'Crisis gate disabled (not implemented)',
            'crisis_detected': False
        }

    def _calculate_trade_result(self, df: pd.DataFrame, bar_index: int) -> Optional[Dict]:
        """Вычислить результат трейда (если бы вошли)"""

        entry_price = df['close'].iloc[bar_index]

        # Exit after FORWARD_BARS
        exit_index = bar_index + FORWARD_BARS

        if exit_index >= len(df):
            return None  # Not enough data for exit

        exit_price = df['close'].iloc[exit_index]

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        profit = pnl_pct > PROFIT_THRESHOLD * 100

        return {
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'exit_index': exit_index,
            'exit_timestamp': str(df.index[exit_index]),
            'pnl_pct': float(pnl_pct),
            'profit': profit,
            'holding_period_bars': FORWARD_BARS,
            'holding_period_hours': FORWARD_BARS * 15 / 60
        }


def print_explanation(explanation: Dict, verbose: bool = True):
    """Красиво печатает объяснение трейда"""

    print("\n" + "="*100)
    print(f"TRADE EXPLANATION - {explanation['asset']}")
    print("="*100)

    print(f"\n📍 Bar: {explanation['bar_index']}")
    print(f"📅 Time: {explanation['timestamp']}")
    print(f"💰 Price: ${explanation['price']:,.2f}")

    # Layer 1
    print("\n" + "─"*100)
    print("LAYER 1: RULE-BASED FILTER (RSI)")
    print("─"*100)
    layer1 = explanation['layer1']
    print(f"RSI Value: {layer1['rsi_value']:.2f}")
    print(f"Threshold: {layer1['rsi_threshold']}")
    print(f"Result: {layer1['reason']}")

    if not layer1['passed']:
        print(f"\n❌ REJECTED at Layer 1")
        print(f"Reason: {explanation['rejection_reason']}")
        return

    # Layer 2
    print("\n" + "─"*100)
    print("LAYER 2: ML FILTER (XGBOOST)")
    print("─"*100)
    layer2 = explanation['layer2']

    if layer2['features_raw'] is None:
        print("⚠️  Could not extract features")
    else:
        print(f"ML Score (P(UP)): {layer2['ml_score']:.3f}")
        print(f"Threshold: {layer2['ml_threshold']}")
        print(f"Prob(DOWN): {layer2['prob_down']:.3f}")
        print(f"Prob(UP):   {layer2['prob_up']:.3f}")
        print(f"Result: {layer2['reason']}")

        if verbose:
            print("\n🔍 Top 5 Most Important Features (by scaled magnitude):")
            for i, feat in enumerate(layer2['top_5_features'], 1):
                print(f"  {i}. {feat['name']:20s} = {feat['raw_value']:12.4f} (scaled: {feat['scaled_value']:7.3f})")

    if not layer2['passed']:
        print(f"\n❌ REJECTED at Layer 2")
        print(f"Reason: {explanation['rejection_reason']}")

        # Show what would have happened
        if explanation['trade_result']:
            result = explanation['trade_result']
            print(f"\n💭 What would have happened if we entered:")
            print(f"   Entry: ${result['entry_price']:,.2f}")
            print(f"   Exit:  ${result['exit_price']:,.2f} (after {result['holding_period_hours']:.1f}h)")
            print(f"   PnL:   {result['pnl_pct']:+.2f}%")
            if result['profit']:
                print(f"   ✅ Would have been PROFITABLE (ML saved us from missing this!)")
            else:
                print(f"   ❌ Would have been a LOSS (ML saved us! 🛡️)")
        return

    # Layer 3
    print("\n" + "─"*100)
    print("LAYER 3: CRISIS GATE")
    print("─"*100)
    layer3 = explanation['layer3']
    print(f"Result: {layer3['reason']}")

    if not layer3['passed']:
        print(f"\n❌ REJECTED at Layer 3")
        print(f"Reason: {explanation['rejection_reason']}")
        return

    # Final decision
    print("\n" + "="*100)
    print("✅ ALL LAYERS PASSED → ENTER LONG")
    print("="*100)

    # Trade result
    if explanation['trade_result']:
        result = explanation['trade_result']
        print(f"\n📈 TRADE RESULT:")
        print(f"   Entry:  ${result['entry_price']:,.2f}")
        print(f"   Exit:   ${result['exit_price']:,.2f} (after {result['holding_period_hours']:.1f}h)")
        print(f"   PnL:    {result['pnl_pct']:+.2f}%")

        if result['profit']:
            print(f"   ✅ PROFITABLE TRADE!")
        else:
            print(f"   ❌ LOSS")


def main():
    """Анализ нескольких трейдов Hybrid модели"""

    print("="*100)
    print("HYBRID MODEL - TRADE-BY-TRADE EXPLAINER")
    print("="*100)

    # Load model
    model_path = MODELS_DIR / "xgboost_multi_tf_model.json"
    scaler_path = MODELS_DIR / "xgboost_multi_tf_scaler.pkl"
    features_path = MODELS_DIR / "xgboost_multi_tf_features.json"
    threshold_path = MODELS_DIR / "xgboost_multi_tf_threshold.txt"

    if not all([model_path.exists(), scaler_path.exists(), features_path.exists()]):
        print("❌ Model files not found. Please train the model first:")
        print("   python scripts/train_xgboost_multi_tf.py")
        return

    explainer = HybridTradeExplainer(
        model_path=model_path,
        scaler_path=scaler_path,
        features_path=features_path,
        threshold_path=threshold_path
    )

    # Load data
    asset = "BTC"
    print(f"\n📊 Loading data for {asset}...")

    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))
    all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, TARGET_TF)

    print(f"✅ Loaded {len(primary_df)} bars")

    # Find some interesting bars to explain
    # 1. Find bars where RSI < 30
    rsi_col = None
    for col in ['rsi', 'RSI_14', '15m_RSI_14']:
        if col in primary_df.columns:
            rsi_col = col
            break

    if rsi_col is None:
        rsi_candidates = [col for col in primary_df.columns if 'RSI' in col.upper()]
        if rsi_candidates:
            rsi_col = rsi_candidates[0]

    if rsi_col is None:
        print("❌ RSI column not found")
        return

    # Find bars with RSI < 30
    oversold_bars = primary_df[primary_df[rsi_col] < 30].index

    if len(oversold_bars) == 0:
        print("⚠️  No oversold bars found in this dataset")
        return

    print(f"\n✅ Found {len(oversold_bars)} oversold bars (RSI < 30)")

    # Sample 10 bars from RECENT data (skip first 10,000 bars to ensure enough history)
    # This avoids NaN issues with multi-timeframe features
    MIN_BAR_INDEX = 10000  # ~104 days of 15m data, ensures all TF features available

    sample_indices = []
    for timestamp in oversold_bars[-200:]:  # Take from last 200 oversold bars
        bar_index = primary_df.index.get_loc(timestamp)
        # Make sure we have enough history AND enough data for exit
        if bar_index >= MIN_BAR_INDEX and bar_index + FORWARD_BARS < len(primary_df):
            sample_indices.append(bar_index)
            if len(sample_indices) >= 10:  # Stop after 10 samples
                break

    if len(sample_indices) == 0:
        print(f"⚠️  No oversold bars found with sufficient history (bar_index > {MIN_BAR_INDEX})")
        print(f"   Try running on more recent data or reducing MIN_BAR_INDEX")
        return

    print(f"\n🔬 Analyzing {len(sample_indices)} sample trades (from recent data)...\n")

    # Explain each trade
    explanations = []

    for i, bar_index in enumerate(sample_indices, 1):
        print(f"\n{'='*100}")
        print(f"TRADE {i}/{len(sample_indices)}")
        print('='*100)

        explanation = explainer.explain_trade(
            all_timeframes=all_tf,
            primary_df=primary_df,
            bar_index=bar_index,
            asset=asset
        )

        print_explanation(explanation, verbose=True)

        explanations.append(explanation)

    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    total = len(explanations)
    layer1_rejected = sum(1 for e in explanations if e.get('rejection_layer') == 1)
    layer2_rejected = sum(1 for e in explanations if e.get('rejection_layer') == 2)
    layer3_rejected = sum(1 for e in explanations if e.get('rejection_layer') == 3)
    all_passed = sum(1 for e in explanations if e['final_decision'])

    print(f"\nTotal analyzed: {total}")
    print(f"Layer 1 rejected: {layer1_rejected} ({layer1_rejected/total*100:.1f}%)")
    print(f"Layer 2 rejected: {layer2_rejected} ({layer2_rejected/total*100:.1f}%)")
    print(f"Layer 3 rejected: {layer3_rejected} ({layer3_rejected/total*100:.1f}%)")
    print(f"All layers passed: {all_passed} ({all_passed/total*100:.1f}%)")

    # Trade results for those that passed all layers
    entered_trades = [e for e in explanations if e['final_decision'] and e.get('trade_result')]

    if entered_trades:
        profitable = sum(1 for e in entered_trades if e['trade_result']['profit'])
        total_pnl = sum(e['trade_result']['pnl_pct'] for e in entered_trades)

        print(f"\nTrades entered: {len(entered_trades)}")
        print(f"Profitable: {profitable} ({profitable/len(entered_trades)*100:.1f}%)")
        print(f"Losing: {len(entered_trades)-profitable}")
        print(f"Total PnL: {total_pnl:+.2f}%")
        print(f"Avg PnL per trade: {total_pnl/len(entered_trades):+.2f}%")

    # ML saves analysis
    layer2_rejects_with_result = [
        e for e in explanations
        if e.get('rejection_layer') == 2 and e.get('trade_result')
    ]

    if layer2_rejects_with_result:
        would_be_profitable = sum(
            1 for e in layer2_rejects_with_result
            if e['trade_result']['profit']
        )
        would_be_loss = len(layer2_rejects_with_result) - would_be_profitable

        print(f"\n🛡️  ML FILTER ANALYSIS (Layer 2 rejects):")
        print(f"   Total rejected: {len(layer2_rejects_with_result)}")
        print(f"   Would have been profitable: {would_be_profitable}")
        print(f"   Would have been losses: {would_be_loss}")

        if would_be_loss > would_be_profitable:
            print(f"   ✅ ML SAVED US from more losses than missed profits!")
        else:
            print(f"   ⚠️  ML rejected more profitable trades than losses")

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
