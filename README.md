# Scarlet Sails - Algorithmic Trading System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Algorithmic cryptocurrency trading system with mathematical proof of strategy dispersion.**

## Overview

Scarlet Sails implements three trading strategies with mathematically rigorous analysis:

### Model 1: Rule-Based Strategy (P_rb)

**Formula:**
```
P_rb = W_opportunity * filters_product * (1 - costs) * (1 - risk_penalty)
```

Where:
- `W_opportunity` = OpportunityScorer output [0, 1]
- `filters_product` = RegimeDetector * VolatilityFilter * TrendFilter
- `costs` = commission + slippage estimate
- `risk_penalty` = AdvancedRiskPenalty (GARCH + CVaR)

### Model 2: XGBoost ML Strategy (P_ml)

**Formula:**
```
P_ml = XGBoost.predict_proba(features_74) * regime_weight * (1 - ood_penalty)
```

Where:
- `features_74` = 74 technical indicators (RSI, MACD, BB, ATR, etc.)
- `regime_weight` = RegimeDetector market state adjustment
- `ood_penalty` = Out-of-Distribution protection score

### Model 3: Hybrid Strategy (P_hyb)

**Formula:**
```
P_hyb = α·P_rb + β·P_ml + γ·V(S)
```

Where:
- `α = 0.45` - Rule-Based weight
- `β = 0.45` - ML weight  
- `γ = 0.10` - DQN Reinforcement Learning weight
- `V(S)` = DQN state-value function

## Core Components

| Component | Description | File |
|-----------|-------------|------|
| **OpportunityScorer** | Calculates entry quality [0,1] | `components/opportunity_scorer.py` |
| **AdvancedRiskPenalty** | GARCH volatility + CVaR risk | `components/advanced_risk_penalty.py` |
| **RegimeDetector** | Market regime classification | `lib/regime_detector.py` |
| **FeatureTransformer** | 74 technical indicators | `core/feature_engine.py` |
| **DQN Agent** | Deep Q-Network for V(S) | `rl/dqn.py` |

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Sharpe Ratio** | > 1.5 | Risk-adjusted return |
| **Profit Factor** | > 2.0 | Gross profit / Gross loss |
| **Max Drawdown** | < 15% | Maximum peak-to-trough decline |
| **Win Rate** | > 55% | Percentage of profitable trades |

## Quick Start

```bash
# Clone repository
git clone https://github.com/Antihrist-star/ScArlet-Sails.git
cd ScArlet-Sails

# Install dependencies
pip install -r requirements.txt

# Run backtest
python run_backtest.py --strategy hybrid --coin ENA --timeframe 15m
```

## Project Structure

```
scarlet-sails/
├── core/                    # Core modules
│   ├── backtest_engine.py   # Backtesting framework
│   ├── data_loader.py       # OHLCV data loader
│   ├── feature_loader.py    # 75-feature loader
│   └── metrics_calculator.py
├── strategies/              # Trading strategies
│   ├── rule_based_v2.py     # Model 1 (P_rb)
│   ├── xgboost_ml_v2.py     # Model 2 (P_ml)
│   └── hybrid_v2.py         # Model 3 (P_hyb)
├── components/              # Strategy components
│   ├── opportunity_scorer.py
│   └── advanced_risk_penalty.py
├── rl/                      # Reinforcement Learning
│   ├── dqn.py               # Deep Q-Network
│   └── trading_environment.py
└── data/
    ├── raw/                 # OHLCV data (via DVC)
    └── features/            # 75-feature datasets
```

## Supported Assets

14 cryptocurrency pairs on Binance:

```
ALGO, AVAX, BTC, DOT, ENA, ETH, HBAR
LDO, LINK, LTC, ONDO, SOL, SUI, UNI
```

Timeframes: `15m`, `1h`, `4h`, `1d`

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Core architecture |
| Phase 2 | ✅ Complete | Backtesting framework |
| Phase 3 | 🔄 In Progress | Feature integration + Model training |
| Phase 4 | ⏳ Planned | Production deployment |

## Documentation

- [Mathematical Framework](docs/MATHEMATICAL_FRAMEWORK.md)
- [System Architecture](docs/SYSTEM_ARCHITECTURE_DETAILED.md)
- [Model Formulas](docs/MODEL_FORMULAS.md)
- [Phase 3 Status](docs/PHASE3_STATUS.md)

## Team

- **STAR_ANT** - Project Lead, Strategy Development
- **EGOR 1** - Pattern Validation
- **EGOR 2** - ML Model Training

## License

MIT License - see [LICENSE](LICENSE) for details.
