"""
Governance Rules - Week 2 Day 14
=================================

Risk management and decision-making rules that govern the trading system.

Philosophy: "ML models are tools, not masters. Governance overrides everything."

5 Core Rules:
=============

Q11: When to HALT trading completely?
Q12: When to override ML models?
Q13: Position sizing based on opportunity score?
Q14: Risk management rules (max drawdown, correlation limits)?
Q15: Emergency stop-loss triggers?

Author: Scarlet Sails Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TradingState(Enum):
    """Trading system states."""
    ACTIVE = "active"              # Normal trading
    HALTED = "halted"              # All trading stopped
    REDUCED = "reduced"            # Reduced position sizes
    EMERGENCY = "emergency"        # Emergency liquidation mode


class HaltReason(Enum):
    """Reasons for halting trading."""
    CRASH_DETECTED = "crash_detected"                    # Crisis classifier: CRASH
    MAX_DRAWDOWN = "max_drawdown"                        # Daily loss limit hit
    CORRELATION_SPIKE = "correlation_spike"              # BTC correlation >0.85
    BOT_MANIPULATION = "bot_manipulation"                # Bot detector: BOT + MANIPULATION
    EXTREME_VOLATILITY = "extreme_volatility"            # Volatility >3x baseline
    NEGATIVE_CRITICAL_NEWS = "negative_critical_news"    # CRITICAL negative news
    EMERGENCY_STOP = "emergency_stop"                    # Manual emergency stop


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_equity: float               # Total portfolio value (USD)
    daily_pnl: float                  # Today's PnL (USD)
    daily_pnl_pct: float             # Today's PnL (%)
    max_drawdown_today: float        # Max drawdown today (%)
    open_positions: int              # Number of open positions
    total_position_size: float       # Total position size (USD)
    avg_btc_correlation: float       # Average BTC correlation of positions
    current_volatility: float        # Current market volatility
    baseline_volatility: float       # Baseline volatility (30d avg)


@dataclass
class MLSignals:
    """Signals from ML models."""
    crisis_type: str                      # CRASH/MANIPULATION/GLITCH/OPPORTUNITY
    crisis_confidence: float              # 0-1
    bot_detected: bool                    # True if bot detected
    bot_confidence: float                 # 0-1
    opportunity_score: float              # 0-1
    news_sentiment: float                 # 0-1 (0=very negative, 1=very positive)
    news_impact: str                      # LOW/MEDIUM/HIGH/CRITICAL
    btc_correlation: float                # 0-1


@dataclass
class GovernanceDecision:
    """Governance decision result."""
    trading_state: TradingState
    halt_reasons: List[HaltReason]
    max_position_size: float          # Maximum position size (USD)
    position_size_multiplier: float   # Multiplier for base position size (0-1)
    override_ml: bool                 # True if ML models overridden
    override_reason: Optional[str]    # Reason for override
    emergency_stop_loss: float        # Emergency stop-loss level (%)

    def __repr__(self):
        return (f"GovernanceDecision(state={self.trading_state.value}, "
                f"multiplier={self.position_size_multiplier:.2f}, "
                f"override={self.override_ml})")


class GovernanceRules:
    """
    Governance rules engine that makes final trading decisions.

    Rules override ML models to prevent catastrophic losses.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 5.0,           # Max 5% daily loss
        max_drawdown_pct: float = 10.0,            # Max 10% drawdown from peak
        max_position_size_pct: float = 20.0,       # Max 20% in single position
        correlation_halt_threshold: float = 0.85,  # Halt if BTC corr >0.85
        volatility_halt_multiplier: float = 3.0,   # Halt if vol >3x baseline
        bot_manipulation_threshold: float = 0.8,   # Confidence threshold
        crash_halt_threshold: float = 0.9,         # Confidence threshold for CRASH
    ):
        """Initialize governance rules."""
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.correlation_halt_threshold = correlation_halt_threshold
        self.volatility_halt_multiplier = volatility_halt_multiplier
        self.bot_manipulation_threshold = bot_manipulation_threshold
        self.crash_halt_threshold = crash_halt_threshold

        # Internal state
        self.halted_until: Optional[datetime] = None
        self.halt_reasons: List[HaltReason] = []

    def evaluate(
        self,
        portfolio: PortfolioState,
        signals: MLSignals,
        timestamp: datetime
    ) -> GovernanceDecision:
        """
        Main governance evaluation.

        Returns final trading decision with all rules applied.
        """
        # Start with normal state
        trading_state = TradingState.ACTIVE
        halt_reasons = []
        override_ml = False
        override_reason = None
        position_size_multiplier = 1.0

        # Check if still halted from previous decision
        if self.halted_until and timestamp < self.halted_until:
            trading_state = TradingState.HALTED
            halt_reasons = self.halt_reasons
            position_size_multiplier = 0.0
        else:
            self.halted_until = None
            self.halt_reasons = []

            # ============================================================
            # Q11: WHEN TO HALT TRADING COMPLETELY?
            # ============================================================

            # Rule 1: CRASH detected with high confidence
            if (signals.crisis_type == "CRASH" and
                signals.crisis_confidence >= self.crash_halt_threshold):
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.CRASH_DETECTED)
                self.halted_until = timestamp + timedelta(hours=24)
                override_ml = True
                override_reason = "CRASH detected - halt 24h"

            # Rule 2: Max daily loss exceeded
            if portfolio.daily_pnl_pct <= -self.max_daily_loss_pct:
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.MAX_DRAWDOWN)
                self.halted_until = timestamp + timedelta(hours=12)
                override_ml = True
                override_reason = f"Daily loss {portfolio.daily_pnl_pct:.1f}% > {self.max_daily_loss_pct}%"

            # Rule 3: Max drawdown exceeded
            if portfolio.max_drawdown_today >= self.max_drawdown_pct:
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.MAX_DRAWDOWN)
                self.halted_until = timestamp + timedelta(hours=12)
                override_ml = True
                override_reason = f"Drawdown {portfolio.max_drawdown_today:.1f}% > {self.max_drawdown_pct}%"

            # Rule 4: BTC correlation spike (market-wide crash)
            if signals.btc_correlation >= self.correlation_halt_threshold:
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.CORRELATION_SPIKE)
                self.halted_until = timestamp + timedelta(hours=6)
                override_ml = True
                override_reason = f"BTC correlation {signals.btc_correlation:.2f} > {self.correlation_halt_threshold}"

            # Rule 5: Bot manipulation detected
            if (signals.bot_detected and
                signals.bot_confidence >= self.bot_manipulation_threshold and
                signals.crisis_type == "MANIPULATION"):
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.BOT_MANIPULATION)
                self.halted_until = timestamp + timedelta(hours=2)
                override_ml = True
                override_reason = f"Bot manipulation detected ({signals.bot_confidence:.0%} confidence)"

            # Rule 6: Extreme volatility
            if portfolio.current_volatility >= portfolio.baseline_volatility * self.volatility_halt_multiplier:
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.EXTREME_VOLATILITY)
                self.halted_until = timestamp + timedelta(hours=4)
                override_ml = True
                override_reason = f"Volatility {portfolio.current_volatility/portfolio.baseline_volatility:.1f}x baseline"

            # Rule 7: CRITICAL negative news
            if signals.news_impact == "CRITICAL" and signals.news_sentiment < 0.3:
                trading_state = TradingState.HALTED
                halt_reasons.append(HaltReason.NEGATIVE_CRITICAL_NEWS)
                self.halted_until = timestamp + timedelta(hours=6)
                override_ml = True
                override_reason = "CRITICAL negative news"

        # Store halt reasons for next check
        if halt_reasons:
            self.halt_reasons = halt_reasons

        # ============================================================
        # Q12: WHEN TO OVERRIDE ML MODELS?
        # ============================================================

        # Already handled in Q11 - any halt condition overrides ML
        # Additional override cases:

        if not override_ml:
            # Override if opportunity score is excellent but bot detected
            if (signals.opportunity_score >= 0.8 and
                signals.bot_detected and
                signals.bot_confidence >= 0.7):
                override_ml = True
                override_reason = "Excellent opportunity but bot detected - reduce"
                trading_state = TradingState.REDUCED
                position_size_multiplier = 0.3

            # Override if opportunity score is low in crisis
            if (signals.crisis_type in ["CRASH", "MANIPULATION"] and
                signals.opportunity_score >= 0.6):
                override_ml = True
                override_reason = f"{signals.crisis_type} detected - ignore opportunity score"
                trading_state = TradingState.REDUCED
                position_size_multiplier = 0.2

        # ============================================================
        # Q13: POSITION SIZING BASED ON OPPORTUNITY SCORE
        # ============================================================

        if trading_state == TradingState.ACTIVE:
            # Base position sizing on opportunity score
            if signals.opportunity_score >= 0.8:
                # EXCELLENT: 80-100% of max position
                position_size_multiplier = 0.8 + (signals.opportunity_score - 0.8) * 1.0
            elif signals.opportunity_score >= 0.6:
                # GOOD: 50-80% of max position
                position_size_multiplier = 0.5 + (signals.opportunity_score - 0.6) * 1.5
            elif signals.opportunity_score >= 0.4:
                # MODERATE: 20-50% of max position
                position_size_multiplier = 0.2 + (signals.opportunity_score - 0.4) * 1.5
            else:
                # LOW: 0-20% of max position
                position_size_multiplier = max(0.0, signals.opportunity_score * 0.5)

            # Reduce size in high volatility
            if portfolio.current_volatility > portfolio.baseline_volatility * 2.0:
                position_size_multiplier *= 0.5

            # Reduce size if approaching daily loss limit
            loss_buffer = abs(portfolio.daily_pnl_pct) / self.max_daily_loss_pct
            if loss_buffer > 0.5:  # >50% of daily limit used
                position_size_multiplier *= (1.0 - loss_buffer)

        elif trading_state == TradingState.HALTED:
            position_size_multiplier = 0.0

        # ============================================================
        # Q14: RISK MANAGEMENT RULES
        # ============================================================

        # Calculate max position size
        max_position_size = portfolio.total_equity * (self.max_position_size_pct / 100.0)
        max_position_size *= position_size_multiplier

        # Correlation limit: reduce size if portfolio is too correlated with BTC
        if portfolio.avg_btc_correlation >= 0.7:
            correlation_penalty = (portfolio.avg_btc_correlation - 0.7) / 0.3  # 0-1 range
            max_position_size *= (1.0 - correlation_penalty * 0.5)  # Up to 50% reduction

        # Portfolio concentration limit: reduce size if too many open positions
        if portfolio.open_positions >= 5:
            max_position_size *= 0.7  # Reduce by 30%
        if portfolio.open_positions >= 8:
            max_position_size *= 0.5  # Additional 50% reduction

        # ============================================================
        # Q15: EMERGENCY STOP-LOSS TRIGGERS
        # ============================================================

        # Dynamic stop-loss based on volatility and crisis type
        if signals.crisis_type == "CRASH":
            emergency_stop_loss = 5.0  # Tight 5% stop in crash
        elif signals.crisis_type == "MANIPULATION":
            emergency_stop_loss = 8.0  # 8% stop for manipulation
        elif signals.crisis_type == "GLITCH":
            emergency_stop_loss = 15.0  # Wide 15% stop for glitch (allow recovery)
        elif signals.crisis_type == "OPPORTUNITY":
            emergency_stop_loss = 10.0  # 10% stop for normal opportunities
        else:
            emergency_stop_loss = 10.0  # Default 10%

        # Adjust for volatility
        volatility_ratio = portfolio.current_volatility / portfolio.baseline_volatility
        if volatility_ratio > 2.0:
            emergency_stop_loss *= 1.5  # Wider stop in high volatility
        elif volatility_ratio < 0.5:
            emergency_stop_loss *= 0.7  # Tighter stop in low volatility

        # Cap at reasonable limits
        emergency_stop_loss = max(3.0, min(20.0, emergency_stop_loss))

        # Return final decision
        return GovernanceDecision(
            trading_state=trading_state,
            halt_reasons=halt_reasons,
            max_position_size=max_position_size,
            position_size_multiplier=position_size_multiplier,
            override_ml=override_ml,
            override_reason=override_reason,
            emergency_stop_loss=emergency_stop_loss
        )

    def emergency_halt(self, reason: str, duration_hours: int = 24):
        """Manually trigger emergency halt."""
        self.halted_until = datetime.now() + timedelta(hours=duration_hours)
        self.halt_reasons = [HaltReason.EMERGENCY_STOP]
        print(f"⚠️ EMERGENCY HALT: {reason} (duration: {duration_hours}h)")

    def resume_trading(self):
        """Manually resume trading."""
        self.halted_until = None
        self.halt_reasons = []
        print("✅ Trading resumed manually")


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GOVERNANCE RULES DEMO - Day 14")
    print("=" * 60)
    print()

    # Initialize governance
    governance = GovernanceRules()

    # Test scenarios
    scenarios = [
        {
            "name": "Normal Market",
            "portfolio": PortfolioState(
                total_equity=10000,
                daily_pnl=100,
                daily_pnl_pct=1.0,
                max_drawdown_today=0.5,
                open_positions=3,
                total_position_size=3000,
                avg_btc_correlation=0.5,
                current_volatility=0.02,
                baseline_volatility=0.02
            ),
            "signals": MLSignals(
                crisis_type="OPPORTUNITY",
                crisis_confidence=0.85,
                bot_detected=False,
                bot_confidence=0.1,
                opportunity_score=0.75,
                news_sentiment=0.65,
                news_impact="LOW",
                btc_correlation=0.5
            )
        },
        {
            "name": "CRASH Detected",
            "portfolio": PortfolioState(
                total_equity=10000,
                daily_pnl=-200,
                daily_pnl_pct=-2.0,
                max_drawdown_today=3.0,
                open_positions=5,
                total_position_size=4000,
                avg_btc_correlation=0.8,
                current_volatility=0.06,
                baseline_volatility=0.02
            ),
            "signals": MLSignals(
                crisis_type="CRASH",
                crisis_confidence=0.95,
                bot_detected=False,
                bot_confidence=0.2,
                opportunity_score=0.2,
                news_sentiment=0.15,
                news_impact="CRITICAL",
                btc_correlation=0.85
            )
        },
        {
            "name": "Bot Manipulation",
            "portfolio": PortfolioState(
                total_equity=10000,
                daily_pnl=50,
                daily_pnl_pct=0.5,
                max_drawdown_today=1.0,
                open_positions=2,
                total_position_size=2000,
                avg_btc_correlation=0.3,
                current_volatility=0.04,
                baseline_volatility=0.02
            ),
            "signals": MLSignals(
                crisis_type="MANIPULATION",
                crisis_confidence=0.92,
                bot_detected=True,
                bot_confidence=0.88,
                opportunity_score=0.65,
                news_sentiment=0.5,
                news_impact="LOW",
                btc_correlation=0.25
            )
        },
        {
            "name": "Max Daily Loss",
            "portfolio": PortfolioState(
                total_equity=10000,
                daily_pnl=-520,
                daily_pnl_pct=-5.2,
                max_drawdown_today=6.5,
                open_positions=4,
                total_position_size=3500,
                avg_btc_correlation=0.6,
                current_volatility=0.03,
                baseline_volatility=0.02
            ),
            "signals": MLSignals(
                crisis_type="OPPORTUNITY",
                crisis_confidence=0.80,
                bot_detected=False,
                bot_confidence=0.15,
                opportunity_score=0.85,
                news_sentiment=0.7,
                news_impact="LOW",
                btc_correlation=0.6
            )
        },
        {
            "name": "Excellent Opportunity",
            "portfolio": PortfolioState(
                total_equity=10000,
                daily_pnl=150,
                daily_pnl_pct=1.5,
                max_drawdown_today=0.3,
                open_positions=2,
                total_position_size=2000,
                avg_btc_correlation=0.4,
                current_volatility=0.015,
                baseline_volatility=0.02
            ),
            "signals": MLSignals(
                crisis_type="OPPORTUNITY",
                crisis_confidence=0.92,
                bot_detected=False,
                bot_confidence=0.1,
                opportunity_score=0.92,
                news_sentiment=0.75,
                news_impact="LOW",
                btc_correlation=0.35
            )
        }
    ]

    # Evaluate each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 60}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'=' * 60}")

        # Print inputs
        print("\n📊 Portfolio State:")
        print(f"  Equity: ${scenario['portfolio'].total_equity:,.0f}")
        print(f"  Daily PnL: ${scenario['portfolio'].daily_pnl:,.0f} ({scenario['portfolio'].daily_pnl_pct:+.1f}%)")
        print(f"  Max Drawdown: {scenario['portfolio'].max_drawdown_today:.1f}%")
        print(f"  Open Positions: {scenario['portfolio'].open_positions}")
        print(f"  BTC Correlation: {scenario['portfolio'].avg_btc_correlation:.2f}")
        print(f"  Volatility: {scenario['portfolio'].current_volatility:.1%} (baseline: {scenario['portfolio'].baseline_volatility:.1%})")

        print("\n🤖 ML Signals:")
        print(f"  Crisis: {scenario['signals'].crisis_type} ({scenario['signals'].crisis_confidence:.0%} confidence)")
        print(f"  Bot Detected: {scenario['signals'].bot_detected} ({scenario['signals'].bot_confidence:.0%} confidence)")
        print(f"  Opportunity Score: {scenario['signals'].opportunity_score:.2f}")
        print(f"  News: {scenario['signals'].news_impact} impact, {scenario['signals'].news_sentiment:.0%} sentiment")

        # Evaluate
        decision = governance.evaluate(
            portfolio=scenario['portfolio'],
            signals=scenario['signals'],
            timestamp=datetime.now()
        )

        # Print decision
        print("\n⚖️ Governance Decision:")
        print(f"  Trading State: {decision.trading_state.value.upper()}")
        if decision.halt_reasons:
            print(f"  Halt Reasons: {', '.join([r.value for r in decision.halt_reasons])}")
        print(f"  Position Size Multiplier: {decision.position_size_multiplier:.2f}x")
        print(f"  Max Position Size: ${decision.max_position_size:,.0f}")
        print(f"  Emergency Stop-Loss: {decision.emergency_stop_loss:.1f}%")
        print(f"  Override ML: {decision.override_ml}")
        if decision.override_reason:
            print(f"  Override Reason: {decision.override_reason}")

        # Interpretation
        print("\n💡 Interpretation:")
        if decision.trading_state == TradingState.HALTED:
            print("  🛑 ALL TRADING HALTED - Do NOT enter new positions")
        elif decision.trading_state == TradingState.REDUCED:
            print(f"  ⚠️ REDUCED TRADING - Max {decision.position_size_multiplier:.0%} of normal position size")
        elif decision.position_size_multiplier >= 0.8:
            print(f"  ✅ FULL TRADING - Max {decision.position_size_multiplier:.0%} of normal position size")
        else:
            print(f"  ⚡ CAUTIOUS TRADING - Max {decision.position_size_multiplier:.0%} of normal position size")

    print("\n" + "=" * 60)
    print("✅ Governance rules module created successfully!")
    print("=" * 60)
    print("\nKey Features:")
    print("✅ Q11: 7 conditions for halting trading")
    print("✅ Q12: ML override logic for safety")
    print("✅ Q13: Dynamic position sizing (0-100% based on opportunity)")
    print("✅ Q14: Correlation limits, drawdown limits, concentration limits")
    print("✅ Q15: Dynamic stop-loss (3-20% based on volatility + crisis type)")
    print("\nReady for integration! ✅")
