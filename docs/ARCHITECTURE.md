# 🏭 Scarlet Sails - Architecture

## System Overview

```
┌─────────────────────────────────────────────┐
│         SCARLET SAILS TRADING SYSTEM        │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐  ┌──────────────┐         │
│  │ Rule-Based  │  │  XGBoost ML  │         │
│  │  Strategy   │  │   Strategy   │         │
│  └──────┴──────┘  └──────┴───────┘         │
│         │                │                  │
│         └────────┬───────┘                  │
│                  │                          │
│         ┌────────┴────────┐                 │
│         │ Hybrid Strategy │                 │
│         │   (Combiner)    │                 │
│         └────────┬────────┘                 │
│                  │                          │
│         ┌────────┴────────┐                 │
│         │   Backtester    │                 │
│         └────────┬────────┘                 │
│                  │                          │
│         ┌────────┴────────┐                 │
│         │ Live Trading    │                 │
│         └─────────────────┘                 │
│                                             │
└─────────────────────────────────────────────┘
```

## Component Details

### 1. Rule-Based Strategy
- Mathematical decision function P_rb(S)
- Technical indicators: RSI, MACD, Bollinger Bands
- Opportunity scoring system
- Crisis detection

### 2. XGBoost ML Strategy
- 74 advanced features from real market data
- Gradient boosting model
- OOD risk detection
- Adaptive cost calculation

### 3. Hybrid Strategy
- Combines Rule-Based + ML
- Adaptive weights α(t), β(t)
- RL component (Phase 4)
- Unified decision function P_hyb(S)

## Data Flow

1. **Data Ingestion**: Binance API → Parquet files
2. **Feature Engineering**: 74 advanced features
3. **Strategy Execution**: Generate signals
4. **Risk Management**: Adaptive penalties
5. **Backtesting**: Performance analysis
6. **Live Trading**: Binance Testnet

## Technology Stack

- **Language**: Python 3.11+
- **ML**: XGBoost, PyTorch (Phase 4)
- **Data**: Pandas, NumPy
- **Backtesting**: Custom framework
- **APIs**: Binance

## File Structure

See [FILE_ARCHITECTURE_COMPLETE.md](../FILE_ARCHITECTURE_COMPLETE.md) for detailed file tree.
