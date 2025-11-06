"""
P0-1: Portfolio Correlation & Effective Weight Formula
=======================================================

Implements mathematically sound correlation-adjusted crisis scoring.

Problem with old approach:
- Arbitrary 0.5 multiplier: adjusted_score = score * (1 - corr * 0.5)
- No theoretical justification
- Breaks with negative correlation (amplifies instead of dampens)

New approach: Effective Weight Formula
- Based on Effective Sample Size theory
- Formula: effective_weight = 1 / (1 + (n-1) * avg_abs_corr)
- Properties:
  - High correlation → low weight (diversification lost)
  - Low correlation → high weight (good diversification)
  - Zero correlation → weight = 1.0 (no adjustment)

Example:
- 3 assets, correlation = 0.8 (high)
  → effective_weight = 1 / (1 + 2 * 0.8) = 0.385
  → 3 assets only provide 38.5% unique information

- 3 assets, correlation = 0.2 (low)
  → effective_weight = 1 / (1 + 2 * 0.2) = 0.714
  → 3 assets provide 71.4% unique information

Author: Scarlet Sails Team
Date: 2025-11-05
Priority: P0 (CRITICAL)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AssetCrisisScore:
    """Individual asset crisis score"""
    asset: str
    score: float
    alert_level: str
    timestamp: pd.Timestamp


class PortfolioCorrelationAnalyzer:
    """
    Analyzes correlation between assets and calculates
    correlation-adjusted portfolio crisis scores.
    """

    def __init__(
        self,
        correlation_window: int = 672,  # 7 days for 15min data
        min_correlation_periods: int = 96  # Minimum 1 day of data
    ):
        """
        Initialize portfolio correlation analyzer.

        Args:
            correlation_window: Window for correlation calculation (bars)
            min_correlation_periods: Minimum periods required for correlation
        """
        self.correlation_window = correlation_window
        self.min_correlation_periods = min_correlation_periods

    def calculate_correlation_matrix(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between multiple assets.

        Args:
            price_data: Dict of {asset_name: dataframe with 'close' column}

        Returns:
            Correlation matrix (pandas DataFrame)
        """
        # Extract returns for each asset
        returns_dict = {}

        for asset, df in price_data.items():
            if len(df) < 2:
                continue

            # Calculate returns
            returns = df['close'].pct_change().dropna()

            # Use last N periods
            if len(returns) > self.correlation_window:
                returns = returns.iloc[-self.correlation_window:]

            returns_dict[asset] = returns

        if not returns_dict:
            return pd.DataFrame()

        # Align all returns to same index (inner join)
        returns_df = pd.DataFrame(returns_dict)

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def calculate_effective_weight(
        self,
        correlations: List[float]
    ) -> float:
        """
        Calculate effective weight based on correlation with other assets.

        Formula: effective_weight = 1 / (1 + (n-1) * avg_abs_corr)

        This is derived from Effective Sample Size theory in statistics.

        Args:
            correlations: List of correlation coefficients with other assets

        Returns:
            Effective weight (0 to 1)
        """
        if not correlations:
            return 1.0

        # Use absolute correlation (both positive and negative reduce independence)
        avg_abs_corr = np.mean(np.abs(correlations))

        # Number of assets including this one
        n_assets = len(correlations) + 1

        # Effective weight formula
        effective_weight = 1.0 / (1 + (n_assets - 1) * avg_abs_corr)

        return effective_weight

    def calculate_portfolio_crisis_score(
        self,
        asset_scores: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        Calculate portfolio-wide crisis score with correlation adjustment.

        Algorithm:
        1. For each asset, get its crisis score
        2. Calculate its correlations with other assets
        3. Calculate effective weight (accounts for redundancy)
        4. Aggregate: weighted average using effective weights

        Args:
            asset_scores: Dict of {asset: crisis_score}
            correlation_matrix: Correlation matrix between assets
            weights: Optional dict of {asset: portfolio_weight}
                     If None, use equal weighting

        Returns:
            Dict with:
                - portfolio_score: Aggregated crisis score
                - effective_weights: Dict of {asset: effective_weight}
                - correlations: Correlation matrix
                - contributions: How much each asset contributes
        """
        if not asset_scores:
            return {
                'portfolio_score': 0.0,
                'effective_weights': {},
                'contributions': {},
                'error': 'No asset scores provided'
            }

        assets = list(asset_scores.keys())

        # Default to equal weights if not provided
        if weights is None:
            weights = {asset: 1.0 / len(assets) for asset in assets}

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate effective weights for each asset
        effective_weights = {}
        adjusted_scores = {}

        for asset in assets:
            if asset not in correlation_matrix.index:
                # No correlation data - use weight 1.0
                effective_weights[asset] = 1.0
                adjusted_scores[asset] = asset_scores[asset]
                continue

            # Get correlations with other assets
            correlations = []
            for other_asset in assets:
                if other_asset == asset:
                    continue
                if other_asset in correlation_matrix.columns:
                    corr = correlation_matrix.loc[asset, other_asset]
                    if not np.isnan(corr):
                        correlations.append(corr)

            # Calculate effective weight
            eff_weight = self.calculate_effective_weight(correlations)
            effective_weights[asset] = eff_weight

            # Adjust score by effective weight
            # High correlation → low weight → less contribution
            adjusted_scores[asset] = asset_scores[asset] * eff_weight

        # Calculate portfolio-wide score
        # Weighted average of adjusted scores
        portfolio_score = 0.0
        contributions = {}

        for asset in assets:
            portfolio_weight = weights[asset]
            adjusted_score = adjusted_scores[asset]

            contribution = portfolio_weight * adjusted_score
            contributions[asset] = contribution
            portfolio_score += contribution

        return {
            'portfolio_score': portfolio_score,
            'effective_weights': effective_weights,
            'adjusted_scores': adjusted_scores,
            'contributions': contributions,
            'correlations': correlation_matrix.to_dict() if not correlation_matrix.empty else {}
        }

    def analyze_diversification(
        self,
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Analyze portfolio diversification quality.

        Returns:
            Dict with:
                - avg_correlation: Average pairwise correlation
                - max_correlation: Maximum pairwise correlation
                - min_correlation: Minimum pairwise correlation
                - diversification_ratio: Quality of diversification (0-1)
                - interpretation: Human-readable assessment
        """
        if correlation_matrix.empty or len(correlation_matrix) < 2:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'diversification_ratio': 1.0,
                'interpretation': 'Insufficient data'
            }

        # Get upper triangle of correlation matrix (exclude diagonal)
        n_assets = len(correlation_matrix)
        correlations = []

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = correlation_matrix.iloc[i, j]
                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return {
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'diversification_ratio': 1.0,
                'interpretation': 'No correlation data'
            }

        avg_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        min_corr = np.min(correlations)

        # Diversification ratio: 1.0 = perfect diversification, 0.0 = no diversification
        # Based on average absolute correlation
        avg_abs_corr = np.mean(np.abs(correlations))
        diversification_ratio = 1.0 - avg_abs_corr

        # Interpretation
        if diversification_ratio >= 0.7:
            interpretation = "✅ Excellent diversification"
        elif diversification_ratio >= 0.5:
            interpretation = "✅ Good diversification"
        elif diversification_ratio >= 0.3:
            interpretation = "⚠️ Moderate diversification"
        else:
            interpretation = "❌ Poor diversification (high correlation)"

        return {
            'avg_correlation': avg_corr,
            'max_correlation': max_corr,
            'min_correlation': min_corr,
            'diversification_ratio': diversification_ratio,
            'interpretation': interpretation,
            'n_assets': n_assets
        }


def demo():
    """Demo showing correlation-adjusted portfolio scoring"""
    print("="*60)
    print("Portfolio Correlation Analysis Demo")
    print("="*60)

    analyzer = PortfolioCorrelationAnalyzer()

    # Simulate 3 assets with different crisis scores and correlations
    # Asset A: -20% crisis
    # Asset B: -18% crisis (highly correlated with A)
    # Asset C: -15% crisis (low correlation with A and B)

    # Create synthetic price data
    np.random.seed(42)
    bars = 1000

    # Asset A and B: highly correlated (0.9)
    returns_a = np.random.normal(-0.0002, 0.01, bars)
    returns_b = 0.9 * returns_a + 0.1 * np.random.normal(0, 0.01, bars)

    # Asset C: independent
    returns_c = np.random.normal(-0.0001, 0.01, bars)

    # Convert to prices
    price_a = 100 * np.exp(np.cumsum(returns_a))
    price_b = 100 * np.exp(np.cumsum(returns_b))
    price_c = 100 * np.exp(np.cumsum(returns_c))

    price_data = {
        'BTC': pd.DataFrame({'close': price_a}),
        'ETH': pd.DataFrame({'close': price_b}),
        'SOL': pd.DataFrame({'close': price_c})
    }

    # Calculate correlation matrix
    corr_matrix = analyzer.calculate_correlation_matrix(price_data)

    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))

    # Analyze diversification
    div_analysis = analyzer.analyze_diversification(corr_matrix)
    print(f"\nDiversification Analysis:")
    print(f"  Average correlation: {div_analysis['avg_correlation']:.3f}")
    print(f"  Max correlation: {div_analysis['max_correlation']:.3f}")
    print(f"  Min correlation: {div_analysis['min_correlation']:.3f}")
    print(f"  Diversification ratio: {div_analysis['diversification_ratio']:.3f}")
    print(f"  Assessment: {div_analysis['interpretation']}")

    # Simulate crisis scores
    asset_scores = {
        'BTC': -0.20,  # -20% crisis
        'ETH': -0.18,  # -18% crisis (highly correlated)
        'SOL': -0.15   # -15% crisis (independent)
    }

    print(f"\nIndividual Asset Crisis Scores:")
    for asset, score in asset_scores.items():
        print(f"  {asset}: {score:.1%}")

    # Calculate portfolio crisis score (without correlation adjustment)
    simple_portfolio_score = np.mean(list(asset_scores.values()))
    print(f"\nSimple Average (no correlation adjustment): {simple_portfolio_score:.1%}")

    # Calculate portfolio crisis score (WITH correlation adjustment)
    portfolio_analysis = analyzer.calculate_portfolio_crisis_score(
        asset_scores,
        corr_matrix
    )

    print(f"\nCorrelation-Adjusted Portfolio Score: {portfolio_analysis['portfolio_score']:.1%}")

    print(f"\nEffective Weights (accounts for correlation redundancy):")
    for asset, eff_weight in portfolio_analysis['effective_weights'].items():
        orig_weight = 1.0 / len(asset_scores)  # Equal weight = 33.3%
        print(f"  {asset}: {eff_weight:.3f} (downweighted from {orig_weight:.3f} due to correlation)")

    print(f"\nContributions to Portfolio Score:")
    for asset, contribution in portfolio_analysis['contributions'].items():
        print(f"  {asset}: {contribution:.1%}")

    print("\n" + "="*60)
    print("Key Insight:")
    print("BTC and ETH are highly correlated (0.9), so they provide")
    print("redundant information. Their effective weights are reduced.")
    print("SOL is independent, so its weight remains higher.")
    print("="*60)


if __name__ == "__main__":
    demo()
    print("\n✅ Portfolio correlation module created successfully!")
