"""
Opportunity Scorer - Week 2 Phase 3
===================================

XGBoost regressor to score profit opportunities (0-1).

Philosophy: "Crisis = danger + opportunity. Score the opportunity."

Opportunity Scoring:
====================

Input: Market state (38 features)
Output: Opportunity score 0-1

Score Interpretation:
- 0.0-0.3: LOW opportunity (avoid, high risk)
- 0.3-0.6: MODERATE opportunity (small position, test waters)
- 0.6-0.8: GOOD opportunity (normal position size)
- 0.8-1.0: EXCELLENT opportunity (maximum position)

Factors that increase opportunity score:
----------------------------------------
1. Oversold RSI (RSI < 30) → Mean reversion likely
2. Moderate volume spike (2-3x) → Interest but not panic
3. Low BTC correlation → Independent asset, not market-wide crash
4. Positive news sentiment → Overreaction to FUD
5. Quick recovery signals → V-shaped pattern forming
6. Low bot activity → Real humans trading (less manipulation)
7. Normal order book depth → Liquidity available

Factors that decrease opportunity score:
----------------------------------------
1. High bot activity → Manipulation risk
2. CRASH crisis type → Real fundamental problem
3. High BTC correlation + BTC down → Market-wide crash
4. Negative news (CRITICAL impact) → Real bad news
5. Thin order book → Illiquid, hard to exit
6. Extreme volume (>5x) → Panic selling

Use Cases:
==========
1. Buy the dip: Score >0.7 + crisis_type=OPPORTUNITY → BUY
2. Avoid traps: Score <0.3 + crisis_type=CRASH → DO NOT BUY
3. Position sizing: Use score to scale position (0.8 score = 80% position)

Author: Scarlet Sails Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class OpportunityScore:
    """Result of opportunity scoring."""
    timestamp: datetime
    score: float  # 0-1
    confidence: float  # Model confidence (based on prediction variance)
    factors: Dict[str, float]  # Key factors contributing to score

    def get_recommendation(self) -> str:
        """Get trading recommendation based on score."""
        if self.score >= 0.8:
            return "EXCELLENT - Max position"
        elif self.score >= 0.6:
            return "GOOD - Normal position"
        elif self.score >= 0.3:
            return "MODERATE - Small position"
        else:
            return "LOW - Avoid"

    def __repr__(self):
        return (f"OpportunityScore(score={self.score:.2f}, "
                f"recommendation={self.get_recommendation()})")


class OpportunityScorer:
    """
    XGBoost regressor for opportunity scoring.

    Predicts profit potential (0-1) from market features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        """
        Initialize opportunity scorer.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """
        Train the opportunity scorer.

        Args:
            X: Feature matrix
            y: Target scores (0-1 continuous)
            eval_set: Optional (X_val, y_val) for validation

        Returns:
            Training metrics
        """
        self.feature_names = list(X.columns)

        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
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
        train_mse = mean_squared_error(y, train_pred)
        train_mae = mean_absolute_error(y, train_pred)
        train_r2 = r2_score(y, train_pred)

        metrics = {
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
        }

        if eval_set:
            val_pred = self.model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)

            metrics.update({
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2,
            })

        return metrics

    def predict(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> OpportunityScore:
        """
        Predict opportunity score from features.

        Args:
            features: Dictionary of feature name → value
            timestamp: Optional timestamp

        Returns:
            OpportunityScore with score and recommendation
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[self.feature_names]

        # Predict
        score = self.model.predict(feature_df)[0]

        # Clip to 0-1 range
        score = np.clip(score, 0.0, 1.0)

        # Calculate confidence (simplified: based on feature quality)
        # Higher confidence if features are in normal ranges
        confidence = 0.8  # Default confidence

        # Analyze key factors
        factors = {
            'rsi_oversold': max(0, 0.3 - features.get('rsi_1h', 0.5)),  # Lower RSI = more oversold
            'volume_moderate': 1.0 - abs(features.get('volume_ratio_1h', 1.0) - 2.0) / 2.0,
            'low_correlation': 1.0 - features.get('btc_correlation_7d', 0.5),
            'news_positive': features.get('news_sentiment', 0.5),
            'depth_healthy': features.get('total_depth_ratio', 1.0),
            'low_bot_activity': 1.0 - features.get('anomaly_arbitrage', 0.0),
        }

        return OpportunityScore(
            timestamp=timestamp or datetime.now(),
            score=score,
            confidence=confidence,
            factors=factors,
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


def generate_opportunity_training_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic training data for opportunity scorer.

    Target (y) is opportunity score 0-1 based on feature combinations.

    Args:
        n_samples: Number of samples

    Returns:
        (X, y) where y is continuous score 0-1
    """
    np.random.seed(42)

    X_list = []
    y_list = []

    for _ in range(n_samples):
        # Generate features
        rsi_1h = np.random.uniform(0.1, 0.9)
        volume_ratio_1h = np.random.uniform(0.5, 10.0)
        btc_correlation_7d = np.random.uniform(0.0, 1.0)
        news_sentiment = np.random.uniform(0.0, 1.0)
        total_depth_ratio = np.random.uniform(0.3, 1.5)
        anomaly_score = np.random.uniform(0.0, 1.0)
        bid_ask_imbalance = np.random.uniform(0.0, 1.0)
        price_shock_severity = np.random.uniform(0.0, 1.0)
        latency_score = np.random.uniform(0.0, 1.0)

        # Calculate opportunity score based on factors
        score = 0.5  # Base score

        # RSI oversold (+)
        if rsi_1h < 0.3:
            score += 0.2
        elif rsi_1h < 0.4:
            score += 0.1

        # Volume moderate (+)
        if 1.5 <= volume_ratio_1h <= 3.0:
            score += 0.15
        elif volume_ratio_1h > 5.0:
            score -= 0.2  # Panic

        # Low BTC correlation (+)
        if btc_correlation_7d < 0.4:
            score += 0.15

        # Positive news (+)
        if news_sentiment > 0.6:
            score += 0.1
        elif news_sentiment < 0.3:
            score -= 0.15

        # Healthy depth (+)
        if total_depth_ratio > 0.8:
            score += 0.1
        elif total_depth_ratio < 0.5:
            score -= 0.15

        # Low anomaly (-)
        if anomaly_score > 0.7:
            score -= 0.2

        # Low bot activity (+)
        if latency_score < 0.3:
            score += 0.1
        elif latency_score > 0.7:
            score -= 0.15

        # Clip to 0-1
        score = np.clip(score, 0.0, 1.0)

        # Add noise
        score += np.random.normal(0, 0.05)
        score = np.clip(score, 0.0, 1.0)

        # Build feature dict
        features = {
            'volume_spike_severity': np.random.uniform(0.0, 0.5) if volume_ratio_1h < 5 else np.random.uniform(0.5, 1.0),
            'price_shock_severity': price_shock_severity,
            'anomaly_score': anomaly_score,
            'anomaly_count': np.random.uniform(0, 2),
            'news_sentiment': news_sentiment,
            'rsi_1h': rsi_1h,
            'rsi_4h': np.random.uniform(0.2, 0.8),
            'rsi_1d': np.random.uniform(0.3, 0.7),
            'price_ma20_ratio': np.random.uniform(0.8, 1.2),
            'price_ma50_ratio': np.random.uniform(0.8, 1.2),
            'ma20_ma200_ratio': np.random.uniform(0.9, 1.1),
            'volume_ratio_1h': volume_ratio_1h,
            'volume_ratio_4h': np.random.uniform(0.8, 3.0),
            'volume_ratio_1d': np.random.uniform(0.8, 2.5),
            'btc_correlation_7d': btc_correlation_7d,
            'btc_correlation_30d': np.random.uniform(0.3, 0.8),
            'volatility_ratio_7d': np.random.uniform(0.5, 2.0),
            'volatility_ratio_30d': np.random.uniform(0.6, 1.8),
            'spread_ratio': np.random.uniform(0.8, 2.5),
            'spread_bps': np.random.uniform(5, 50),
            'bid_depth_ratio': np.random.uniform(0.5, 1.5),
            'ask_depth_ratio': np.random.uniform(0.5, 1.5),
            'total_depth_ratio': total_depth_ratio,
            'bid_ask_imbalance': bid_ask_imbalance,
            'depth_imbalance': np.random.uniform(0.3, 0.7),
            'spread_volatility': np.random.uniform(0, 30),
            'price_spread_max': np.random.uniform(0.05, 1.5),
            'price_spread_median': np.random.uniform(0.02, 0.8),
            'price_std_dev': np.random.uniform(0.02, 0.8),
            'leader_exchange_index': np.random.uniform(0.0, 1.0),
            'latency_score': latency_score,
            'arbitrage_opportunity': np.random.choice([0.0, 1.0], p=[0.7, 0.3]),
            'arbitrage_profit_potential': np.random.uniform(0.0, 0.5),
            'volume_volatility_1h': np.random.uniform(0.0, 3.0),
            'correlation_spread': np.random.uniform(0.0, 0.5),
            'rsi_volume': rsi_1h * volume_ratio_1h / 10.0,
            'depth_imbalance_risk': np.random.uniform(0.0, 0.5),
            'anomaly_arbitrage': anomaly_score * latency_score,
        }

        X_list.append(features)
        y_list.append(score)

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)

    return X, y


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Opportunity Scorer Demo")
    print("=" * 60)

    print("\n1. Generating training data...")
    X_train, y_train = generate_opportunity_training_data(n_samples=1000)
    print(f"   Generated {len(X_train)} training samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Target range: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"   Target mean: {y_train.mean():.2f}")

    print("\n2. Training opportunity scorer...")
    scorer = OpportunityScorer(n_estimators=100, max_depth=6)
    metrics = scorer.train(X_train, y_train)
    print(f"   Training MSE: {metrics['train_mse']:.4f}")
    print(f"   Training MAE: {metrics['train_mae']:.4f}")
    print(f"   Training R²: {metrics['train_r2']:.4f}")

    print("\n3. Feature importance (Top 10)...")
    importance = scorer.get_feature_importance(top_n=10)
    for i, row in importance.iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")

    print("\n4. Testing predictions...")

    # Test high opportunity
    print("\n--- Scenario 1: High Opportunity (Oversold, Low Risk) ---")
    high_opp_features = {
        'rsi_1h': 0.25,  # Oversold
        'volume_ratio_1h': 2.0,  # Moderate volume
        'btc_correlation_7d': 0.6,  # Some correlation
        'news_sentiment': 0.6,  # Slightly positive
        'total_depth_ratio': 1.2,  # Healthy depth
        'anomaly_score': 0.3,  # Low anomaly
        'latency_score': 0.2,  # Low bot activity
    }
    # Fill remaining features
    for feat in X_train.columns:
        if feat not in high_opp_features:
            high_opp_features[feat] = X_train[feat].median()

    prediction = scorer.predict(high_opp_features)
    print(f"Score: {prediction.score:.2f}")
    print(f"Recommendation: {prediction.get_recommendation()}")
    print("Key factors:")
    for factor, value in list(prediction.factors.items())[:3]:
        print(f"  {factor}: {value:.2f}")

    # Test low opportunity
    print("\n--- Scenario 2: Low Opportunity (Crash, High Risk) ---")
    low_opp_features = {
        'rsi_1h': 0.15,  # Very oversold (crash)
        'volume_ratio_1h': 8.0,  # Panic volume
        'btc_correlation_7d': 0.9,  # High correlation (market crash)
        'news_sentiment': 0.2,  # Very negative
        'total_depth_ratio': 0.4,  # Thin depth
        'anomaly_score': 0.9,  # High anomaly
        'latency_score': 0.1,
    }
    for feat in X_train.columns:
        if feat not in low_opp_features:
            low_opp_features[feat] = X_train[feat].median()

    prediction = scorer.predict(low_opp_features)
    print(f"Score: {prediction.score:.2f}")
    print(f"Recommendation: {prediction.get_recommendation()}")

    # Test moderate opportunity
    print("\n--- Scenario 3: Moderate Opportunity (Mixed Signals) ---")
    mod_opp_features = {
        'rsi_1h': 0.35,  # Slightly oversold
        'volume_ratio_1h': 1.8,  # Normal-ish volume
        'btc_correlation_7d': 0.5,  # Medium correlation
        'news_sentiment': 0.5,  # Neutral
        'total_depth_ratio': 0.9,
        'anomaly_score': 0.4,
        'latency_score': 0.3,
    }
    for feat in X_train.columns:
        if feat not in mod_opp_features:
            mod_opp_features[feat] = X_train[feat].median()

    prediction = scorer.predict(mod_opp_features)
    print(f"Score: {prediction.score:.2f}")
    print(f"Recommendation: {prediction.get_recommendation()}")

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("- Regression model (continuous score 0-1)")
    print("- Combines multiple factors (RSI, volume, correlation, news)")
    print("- Scores >0.7: Good buying opportunities")
    print("- Scores <0.3: Avoid (high risk, low reward)")
    print("- Use score for position sizing (0.8 score = 80% position)")
    print("=" * 60)

    print("\n✅ Opportunity scorer ready for integration!")
