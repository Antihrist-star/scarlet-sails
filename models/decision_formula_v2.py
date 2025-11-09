# FILE: models/decision_formula_v2.py
# PURPOSE: Unified decision formula with all 4 layers explicit

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MarketState:
    """Current market state snapshot"""
    features: np.ndarray  # 38 features
    volatility: float     # Current volatility
    volatility_regime: str  # HIGH/NORMAL/LOW
    regime: str  # BULL/BEAR/SIDEWAYS
    regime_confidence: float  # 0-1
    regime_transition_probs: Dict[str, float]  # Prob of next regime
    correlations: np.ndarray  # Asset correlation matrix
    recent_slippage: float  # Current slippage estimate
    data_quality: float  # 0-1
    timestamp: pd.Timestamp

@dataclass
class PortfolioState:
    """Current portfolio state"""
    open_positions: Dict[str, float]  # Asset: position_size
    total_exposure: float  # Sum of |positions|
    current_drawdown: float  # Current DD %
    daily_pnl: float  # Today's P&L
    last_position: Dict[str, float]  # Previous position
    portfolio_correlation_matrix: np.ndarray
    portfolio_heat: float  # Effective exposure
    risk_budget_remaining: float  # 0-1


class DecisionFormulaV2:
    """
    Complete decision formula with all 4 layers explicit:

    P_j(t) = argmax_p [
        F_j(θ*) · ∏𝟙_k · O*
        - C_trans - C_switch - λU + γE[V]
    ]
    """

    def __init__(self,
                 ml_models: Dict,  # {crisis_clf, bot_detector, opp_scorer}
                 risk_params: Dict,
                 cost_params: Dict):
        """
        Initialize with trained ML models and parameters

        Args:
            ml_models: {
                'crisis_classifier': XGBClassifier,
                'bot_detector': XGBClassifier,
                'opportunity_scorer': XGBRegressor,
            }
            risk_params: {
                'max_drawdown': 0.15,
                'max_exposure': 0.10,
                'max_portfolio_heat': 0.80,
                'lambda_base': 0.5,  # Risk aversion base
                'gamma': 0.95,  # Discount factor per 24h
            }
            cost_params: {
                'transaction_fee': 0.001,  # 0.1%
                'slippage_multiplier': 1.0,  # Adjust based on volatility
                'switching_frequency_penalty': 0.01,
            }
        """
        self.crisis_clf = ml_models['crisis_classifier']
        self.bot_detector = ml_models['bot_detector']
        self.opp_scorer = ml_models['opportunity_scorer']

        self.risk_params = risk_params
        self.cost_params = cost_params

        # Regime-dependent risk aversion
        self.lambda_by_regime = {
            'BULL': 0.3,      # Risk-on in bull
            'BEAR': 0.8,      # Risk-off in bear
            'SIDEWAYS': 0.5,  # Neutral
        }

    # ======================== LAYER 1 ========================

    def compute_base_score(self, features: np.ndarray) -> Tuple[float, int]:
        """
        F_j(θ*) - Base ML score

        Returns: (score, crisis_class)
                 crisis_class: 0=CRASH, 1=NORMAL
        """
        crisis_probs = self.crisis_clf.predict_proba(features.reshape(1, -1))[0]
        crisis_class = self.crisis_clf.classes_[np.argmax(crisis_probs)]
        base_score = np.max(crisis_probs)

        return base_score, int(crisis_class)

    # ======================== LAYER 2 ========================

    def evaluate_filters(self,
                        features: np.ndarray,
                        base_score: float,
                        crisis_class: int,
                        portfolio_state: PortfolioState,
                        market_state: MarketState) -> Tuple[float, Dict[str, bool]]:
        """
        ∏ 𝟙_k - All 7 multiplicative filters

        Returns: (filters_product, filter_details)
        """
        filters = {}

        # 1. Safety filter
        filters['safety'] = (
            portfolio_state.current_drawdown < self.risk_params['max_drawdown'] and
            portfolio_state.total_exposure < self.risk_params['max_exposure'] and
            portfolio_state.daily_pnl > -self.risk_params['max_drawdown'] * 1000  # Assume $1000
        )

        # 2. Edge filter
        filters['edge'] = base_score > 0.60  # Minimum confidence

        # 3. Crisis-type filter (binary classification)
        # If CRASH (0) is predicted with high confidence -> HALT
        # If NORMAL (1) is predicted -> TRADE
        if crisis_class == 0:  # CRASH
            filters['crisis_type'] = False  # Block trading (will trigger HALT)
        else:  # NORMAL
            filters['crisis_type'] = True   # Allow trading

        # 4. Bot-free filter
        bot_prob = self.bot_detector.predict_proba(features.reshape(1, -1))[0, 1]
        filters['bot_free'] = bot_prob < 0.5

        # 5. Regime filter (simple: all regimes allowed, but confidence check)
        filters['regime'] = market_state.regime_confidence > 0.5

        # 6. Correlation filter
        filters['correlation'] = (
            portfolio_state.portfolio_heat < self.risk_params['max_portfolio_heat']
        )

        # 7. Data quality filter
        filters['data_quality'] = market_state.data_quality > 0.8

        # Compute product (all must be true, else 0)
        filters_product = float(all(filters.values()))

        return filters_product, filters

    # ======================== LAYER 3 ========================

    def compute_opportunity_score(self, features: np.ndarray) -> float:
        """
        O*(p) - Opportunity multiplier

        Returns: 0.1 (very low) to 1.5+ (very high)
        """
        opp_pred = self.opp_scorer.predict(features.reshape(1, -1))[0]
        # Normalize to [0.1, 1.5] range
        opp_scaled = 0.1 + opp_pred * 1.4
        return np.clip(opp_scaled, 0.1, 1.5)

    # ======================== LAYER 4 ========================

    def compute_transaction_cost(self,
                                 position_size: float,
                                 volatility: float,
                                 slippage: float) -> float:
        """
        C_transaction - Transaction costs

        Explicit formula:
        C = base_fee * position_size + slippage_estimate
        """
        base_fee = self.cost_params['transaction_fee']

        # Slippage scales with volatility
        slippage_cost = slippage * self.cost_params['slippage_multiplier'] * (volatility / 0.02)

        total_cost = base_fee * abs(position_size) + slippage_cost

        return total_cost

    def compute_switching_cost(self,
                               prev_position: Dict[str, float],
                               new_position: Dict[str, float]) -> float:
        """
        C_switching - Cost of changing positions

        Explicit formula:
        C_switch = frequency_penalty * |position_change|
        """
        position_change = sum(
            abs(new_position.get(asset, 0) - prev_position.get(asset, 0))
            for asset in set(list(prev_position.keys()) + list(new_position.keys()))
        )

        switching_cost = self.cost_params['switching_frequency_penalty'] * position_change

        return switching_cost

    def compute_uncertainty_penalty(self,
                                    base_score: float,
                                    regime_confidence: float,
                                    data_quality: float) -> float:
        """
        U(p, Σ_t, c_t) - Uncertainty measure

        Explicit formula:
        U = (1 - base_score) + (1 - regime_confidence) + (1 - data_quality)
        """
        uncertainty = (
            (1 - base_score) * 0.5 +
            (1 - regime_confidence) * 0.3 +
            (1 - data_quality) * 0.2
        )

        return np.clip(uncertainty, 0, 1)

    def compute_dynamic_lambda(self, regime: str, volatility_regime: str) -> float:
        """
        λ (dynamic risk aversion)

        Explicit formula:
        λ = lambda_by_regime[regime] * vol_multiplier
        """
        base_lambda = self.lambda_by_regime.get(regime, 0.5)

        vol_multiplier = {
            'LOW': 0.7,
            'NORMAL': 1.0,
            'HIGH': 1.5,
        }.get(volatility_regime, 1.0)

        return base_lambda * vol_multiplier

    def compute_forward_value(self,
                             regime_transition_probs: Dict[str, float],
                             features: np.ndarray,
                             periods_ahead: int = 24) -> float:
        """
        γ · E[V_{t+1}] - Forward-looking value

        Explicit formula:
        E[V] = sum over next regimes:
               P(regime_next) * E[return | regime_next]
        """
        # Expected returns by regime (historical)
        regime_returns = {
            'BULL': 0.02,      # 2% per 24h in bull
            'BEAR': -0.01,     # -1% per 24h in bear
            'SIDEWAYS': 0.003, # 0.3% per 24h sideways
        }

        expected_value = sum(
            prob * regime_returns.get(regime, 0)
            for regime, prob in regime_transition_probs.items()
        )

        # Discount by periods ahead
        gamma = self.risk_params['gamma']
        discounted_value = expected_value * (gamma ** periods_ahead)

        return discounted_value

    # ======================== FINAL FORMULA ========================

    def compute_decision_score(self,
                              features: np.ndarray,
                              portfolio_state: PortfolioState,
                              market_state: MarketState,
                              position_size_candidate: float = 0.05) -> Tuple[float, Dict]:
        """
        COMPLETE FORMULA:

        P_j(t) = [
            F_j(θ*) · ∏𝟙_k · O*
            - C_trans - C_switch - λU
            + γE[V]
        ]
        """

        # LAYER 1: Base score
        base_score, crisis_class = self.compute_base_score(features)

        # LAYER 2: Filters
        filters_product, filter_details = self.evaluate_filters(
            features, base_score, crisis_class, portfolio_state, market_state
        )

        # LAYER 3: Opportunity
        opp_score = self.compute_opportunity_score(features)

        # LAYER 4: Costs & Risk
        c_trans = self.compute_transaction_cost(
            position_size_candidate,
            market_state.volatility,
            market_state.recent_slippage
        )

        c_switch = self.compute_switching_cost(
            portfolio_state.last_position,
            {'new_position': position_size_candidate}
        )

        uncertainty = self.compute_uncertainty_penalty(
            base_score,
            market_state.regime_confidence,
            market_state.data_quality
        )

        lambda_param = self.compute_dynamic_lambda(
            market_state.regime,
            market_state.volatility_regime
        )

        forward_value = self.compute_forward_value(
            market_state.regime_transition_probs,
            features,
            periods_ahead=24
        )

        # FINAL COMPUTATION
        if filters_product == 0:
            # If any filter fails, score = 0 (HALT)
            final_score = 0.0
        else:
            final_score = (
                base_score * opp_score
                - c_trans
                - c_switch
                - (lambda_param * uncertainty)
                + self.risk_params['gamma'] * forward_value
            )

        # Clip to reasonable range
        final_score = np.clip(final_score, -1, 2)

        return final_score, {
            'base_score': base_score,
            'crisis_class': crisis_class,
            'filters_product': filters_product,
            'filter_details': filter_details,
            'opportunity_score': opp_score,
            'c_transaction': c_trans,
            'c_switching': c_switch,
            'uncertainty': uncertainty,
            'lambda': lambda_param,
            'forward_value': forward_value,
            'final_score': final_score,
        }

    def make_decision(self,
                     features: np.ndarray,
                     portfolio_state: PortfolioState,
                     market_state: MarketState) -> Tuple[str, float, Dict]:
        """
        Make trading decision and position sizing

        Returns: (action, position_size, details)
        """

        score, details = self.compute_decision_score(
            features, portfolio_state, market_state
        )

        # Decision thresholds
        if score <= 0:
            action = 'HALT'
            position_size = 0
        elif score < 0.3:
            action = 'REDUCE'
            position_size = 0.02  # 2% of portfolio
        elif score < 0.6:
            action = 'SMALL_TRADE'
            position_size = 0.05  # 5% of portfolio
        else:
            action = 'TRADE'
            position_size = 0.10  # 10% of portfolio

        return action, position_size, details
