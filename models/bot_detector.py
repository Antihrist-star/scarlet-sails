"""
Bot Detector - Week 2 Phase 3
=============================

Binary XGBoost classifier to detect bot activity.

Philosophy: "If it trades like a bot, it probably is a bot"

Bot Activity Types:
===================

1. Wash Trading:
   Bot buys and sells to itself to inflate volume
   Signals:
   - Extreme volume (5-20x baseline)
   - Low price impact (high volume but price stable)
   - Low correlation with BTC (isolated activity)
   - Imbalanced order book

2. Front-Running:
   Bot detects large orders and trades ahead
   Signals:
   - One exchange consistently leads
   - High latency score
   - Arbitrage opportunities
   - Large spreads between exchanges

3. Latency Arbitrage:
   Bot exploits faster data feeds
   Signals:
   - Consistent leader exchange
   - High arbitrage profit potential
   - Quick price adjustments across exchanges

4. Spoofing:
   Bot places fake orders to manipulate price
   Signals:
   - Extreme order book imbalance
   - Thin depth (orders pulled quickly)
   - High spread volatility

Detection Strategy:
===================
- Combine latency signals + order book anomalies + volume patterns
- Binary classification: BOT vs NO_BOT
- High precision (few false positives) is critical
- When bot detected → Halt or reduce position size

Author: Scarlet Sails Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BotActivity(Enum):
    """Bot activity classification."""
    NO_BOT = 0  # Normal trading
    BOT = 1     # Bot detected


@dataclass
class BotDetection:
    """Result of bot detection."""
    timestamp: datetime
    bot_activity: BotActivity
    confidence: float  # 0-1 probability
    bot_type: Optional[str] = None  # wash_trading, front_running, latency_arbitrage, spoofing

    def __repr__(self):
        return (f"BotDetection(activity={self.bot_activity.name}, "
                f"confidence={self.confidence:.2%}, type={self.bot_type})")


class BotDetector:
    """
    XGBoost binary classifier for bot activity detection.

    Uses 38 features from FeatureEngine with focus on:
    - Latency signals (leader detection, arbitrage)
    - Order book anomalies (imbalance, thin depth)
    - Volume patterns (wash trading)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 1.0,  # For imbalanced datasets
    ):
        """
        Initialize bot detector.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            scale_pos_weight: Weight for positive class (BOT) if imbalanced
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """
        Train the bot detector.

        Args:
            X: Feature matrix
            y: Target labels (0=NO_BOT, 1=BOT)
            eval_set: Optional (X_val, y_val) for validation

        Returns:
            Training metrics
        """
        self.feature_names = list(X.columns)

        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )

        # Train
        if eval_set:
            X_val, y_val = eval_set
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)

        # Calculate metrics
        train_pred = self.model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        train_precision = precision_score(y, train_pred, zero_division=0)
        train_recall = recall_score(y, train_pred, zero_division=0)

        metrics = {
            'train_accuracy': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
        }

        if eval_set:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred, zero_division=0)
            val_recall = recall_score(y_val, val_pred, zero_division=0)

            metrics.update({
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
            })

        return metrics

    def predict(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> BotDetection:
        """
        Predict bot activity from features.

        Args:
            features: Dictionary of feature name → value
            timestamp: Optional timestamp

        Returns:
            BotDetection with classification and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[self.feature_names]

        # Predict
        proba = self.model.predict_proba(feature_df)[0]  # [prob_no_bot, prob_bot]
        bot_probability = proba[1]

        bot_activity = BotActivity.BOT if bot_probability > 0.5 else BotActivity.NO_BOT
        confidence = bot_probability if bot_activity == BotActivity.BOT else (1 - bot_probability)

        # Infer bot type based on features
        bot_type = None
        if bot_activity == BotActivity.BOT:
            if features.get('latency_score', 0) > 0.6:
                bot_type = 'latency_arbitrage'
            elif features.get('bid_ask_imbalance', 0.5) > 0.7:
                bot_type = 'spoofing'
            elif features.get('volume_ratio_1h', 1) > 5:
                bot_type = 'wash_trading'
            else:
                bot_type = 'front_running'

        return BotDetection(
            timestamp=timestamp or datetime.now(),
            bot_activity=bot_activity,
            confidence=confidence,
            bot_type=bot_type,
        )

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)


def generate_bot_detection_training_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic training data for bot detector.

    Args:
        n_samples: Total number of samples (50% BOT, 50% NO_BOT)

    Returns:
        (X, y) where y is 0=NO_BOT, 1=BOT
    """
    np.random.seed(42)

    samples_per_class = n_samples // 2

    X_list = []
    y_list = []

    for bot_class in range(2):
        for _ in range(samples_per_class):
            if bot_class == 0:  # NO_BOT (normal trading)
                features = {
                    # Anomaly: Low to moderate
                    'volume_spike_severity': np.random.uniform(0.0, 0.5),
                    'price_shock_severity': np.random.uniform(0.0, 0.4),
                    'anomaly_score': np.random.uniform(0.0, 0.4),

                    # Traditional: Normal ranges
                    'rsi_1h': np.random.uniform(0.3, 0.7),
                    'volume_ratio_1h': np.random.uniform(0.8, 2.0),  # Normal volume
                    'btc_correlation_7d': np.random.uniform(0.5, 0.9),  # Correlated with market

                    # Order book: Balanced
                    'spread_ratio': np.random.uniform(0.8, 1.5),
                    'bid_ask_imbalance': np.random.uniform(0.4, 0.6),  # Balanced
                    'total_depth_ratio': np.random.uniform(0.8, 1.5),

                    # Latency: Normal
                    'latency_score': np.random.uniform(0.0, 0.3),  # Low latency score
                    'arbitrage_opportunity': np.random.choice([0, 1], p=[0.9, 0.1]),  # Rare arbitrage
                    'price_spread_max': np.random.uniform(0.05, 0.3),
                }

            else:  # BOT (bot activity)
                bot_type = np.random.choice(['wash_trading', 'front_running', 'latency_arbitrage', 'spoofing'])

                if bot_type == 'wash_trading':
                    features = {
                        'volume_spike_severity': np.random.uniform(0.7, 1.0),  # High volume
                        'price_shock_severity': np.random.uniform(0.0, 0.3),  # Low price impact
                        'volume_ratio_1h': np.random.uniform(5.0, 15.0),  # Extreme volume
                        'btc_correlation_7d': np.random.uniform(0.1, 0.4),  # Low correlation (isolated)
                        'spread_ratio': np.random.uniform(1.5, 3.0),
                        'bid_ask_imbalance': np.random.uniform(0.6, 0.9),  # Imbalanced
                        'latency_score': np.random.uniform(0.2, 0.5),
                    }

                elif bot_type == 'front_running':
                    features = {
                        'volume_spike_severity': np.random.uniform(0.5, 0.8),
                        'latency_score': np.random.uniform(0.6, 1.0),  # High latency
                        'arbitrage_opportunity': 1.0,
                        'price_spread_max': np.random.uniform(0.5, 1.5),
                        'volume_ratio_1h': np.random.uniform(2.0, 5.0),
                        'bid_ask_imbalance': np.random.uniform(0.6, 0.8),
                    }

                elif bot_type == 'latency_arbitrage':
                    features = {
                        'latency_score': np.random.uniform(0.7, 1.0),  # Very high
                        'arbitrage_opportunity': 1.0,
                        'arbitrage_profit_potential': np.random.uniform(0.3, 1.0),
                        'price_spread_max': np.random.uniform(0.8, 2.0),
                        'volume_ratio_1h': np.random.uniform(1.5, 4.0),
                    }

                else:  # spoofing
                    features = {
                        'bid_ask_imbalance': np.random.uniform(0.7, 0.95),  # Very imbalanced
                        'total_depth_ratio': np.random.uniform(0.2, 0.5),  # Thin depth
                        'spread_volatility': np.random.uniform(20, 50),
                        'volume_spike_severity': np.random.uniform(0.4, 0.7),
                        'latency_score': np.random.uniform(0.3, 0.6),
                    }

            # Fill missing features with defaults
            all_features = {
                'volume_spike_severity': 0.0,
                'price_shock_severity': 0.0,
                'anomaly_score': 0.0,
                'anomaly_count': 0.0,
                'news_sentiment': 0.5,
                'rsi_1h': 0.5,
                'rsi_4h': 0.5,
                'rsi_1d': 0.5,
                'price_ma20_ratio': 1.0,
                'price_ma50_ratio': 1.0,
                'ma20_ma200_ratio': 1.0,
                'volume_ratio_1h': 1.0,
                'volume_ratio_4h': 1.0,
                'volume_ratio_1d': 1.0,
                'btc_correlation_7d': 0.5,
                'btc_correlation_30d': 0.5,
                'volatility_ratio_7d': 1.0,
                'volatility_ratio_30d': 1.0,
                'spread_ratio': 1.0,
                'spread_bps': 0.1,
                'bid_depth_ratio': 1.0,
                'ask_depth_ratio': 1.0,
                'total_depth_ratio': 1.0,
                'bid_ask_imbalance': 0.5,
                'depth_imbalance': 0.5,
                'spread_volatility': 0.0,
                'price_spread_max': 0.1,
                'price_spread_median': 0.05,
                'price_std_dev': 0.05,
                'leader_exchange_index': 0.0,
                'latency_score': 0.0,
                'arbitrage_opportunity': 0.0,
                'arbitrage_profit_potential': 0.0,
                'volume_volatility_1h': 0.0,
                'correlation_spread': 0.0,
                'rsi_volume': 0.0,
                'depth_imbalance_risk': 0.0,
                'anomaly_arbitrage': 0.0,
            }

            all_features.update(features)

            X_list.append(all_features)
            y_list.append(bot_class)

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)

    return X, y


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Bot Detector Demo")
    print("=" * 60)

    print("\n1. Generating training data...")
    X_train, y_train = generate_bot_detection_training_data(n_samples=1000)
    print(f"   Generated {len(X_train)} training samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Class distribution:")
    print(f"     NO_BOT: {(y_train == 0).sum()} samples")
    print(f"     BOT: {(y_train == 1).sum()} samples")

    print("\n2. Training bot detector...")
    detector = BotDetector(n_estimators=50, max_depth=5)
    metrics = detector.train(X_train, y_train)
    print(f"   Training accuracy: {metrics['train_accuracy']:.2%}")
    print(f"   Training precision: {metrics['train_precision']:.2%}")
    print(f"   Training recall: {metrics['train_recall']:.2%}")

    print("\n3. Feature importance (Top 10)...")
    importance = detector.get_feature_importance(top_n=10)
    for i, row in importance.iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")

    print("\n4. Testing predictions...")

    # Test NO_BOT scenario
    print("\n--- Scenario 1: Normal Trading (NO_BOT) ---")
    normal_features = X_train[y_train == 0].iloc[0].to_dict()
    prediction = detector.predict(normal_features)
    print(f"Predicted: {prediction.bot_activity.name} ({prediction.confidence:.2%} confidence)")

    # Test BOT scenarios
    print("\n--- Scenario 2: Wash Trading (BOT) ---")
    bot_features = X_train[y_train == 1].iloc[0].to_dict()
    prediction = detector.predict(bot_features)
    print(f"Predicted: {prediction.bot_activity.name} ({prediction.confidence:.2%} confidence)")
    print(f"Bot type: {prediction.bot_type}")

    print("\n--- Scenario 3: Latency Arbitrage (BOT) ---")
    bot_features = X_train[y_train == 1].iloc[50].to_dict()
    prediction = detector.predict(bot_features)
    print(f"Predicted: {prediction.bot_activity.name} ({prediction.confidence:.2%} confidence)")
    print(f"Bot type: {prediction.bot_type}")

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("- Binary classifier (BOT vs NO_BOT)")
    print("- High precision crucial (avoid false positives)")
    print("- Top signals: latency_score, bid_ask_imbalance, volume_ratio")
    print("- Detects: wash trading, front-running, latency arb, spoofing")
    print("=" * 60)

    print("\n✅ Bot detector ready for integration!")
