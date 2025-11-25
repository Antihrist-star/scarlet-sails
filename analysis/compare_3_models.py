"""
COMPARE 3 MODELS - Rule-Based vs XGBoost vs Hybrid
Shows what each model does independently
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("3 MODELS COMPARISON")
print("="*80)

project_root = Path(__file__).parent.parent
data_dir = project_root / "data" / "raw"

# Test on BTC 15m (common baseline)
df = pd.read_parquet(data_dir / "BTC_USDT_15m.parquet")

# Use last 6 months
test_start = '2025-05-01'
df_test = df[df.index >= test_start].copy()

print(f"\nTest period: {df_test.index[0]} → {df_test.index[-1]}")
print(f"Bars: {len(df_test):,}")

# ============================================================================
# MODEL 1: RULE-BASED (RSI < 30)
# ============================================================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rule_based_backtest(df):
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    
    capital = 100000
    position = None
    trades = []
    
    tp = 0.03
    sl = 0.012
    cooldown = 0
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        rsi = df['rsi'].iloc[i]
        
        # Check position
        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            
            if pnl_pct >= tp:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'tp'})
                position = None
                cooldown = 24
            elif pnl_pct <= -sl:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'sl'})
                position = None
                cooldown = 24
        
        # Cooldown
        if cooldown > 0:
            cooldown -= 1
        
        # New signal
        if rsi < 30 and position is None and cooldown == 0 and not np.isnan(rsi):
            position = {'entry': price}
    
    return capital, trades

print("\n[1/3] Testing Rule-Based...")
capital_rule, trades_rule = rule_based_backtest(df_test)

wins_rule = [t for t in trades_rule if t['pnl'] > 0]
losses_rule = [t for t in trades_rule if t['pnl'] < 0]

total_profit_rule = sum(t['pnl'] for t in wins_rule) if wins_rule else 0
total_loss_rule = abs(sum(t['pnl'] for t in losses_rule)) if losses_rule else 1
pf_rule = total_profit_rule / total_loss_rule if total_loss_rule > 0 else 0

print(f"✅ Return: {(capital_rule-100000)/1000:.2f}%")
print(f"✅ Trades: {len(trades_rule)}")
print(f"✅ WR: {len(wins_rule)/len(trades_rule)*100:.1f}%" if trades_rule else "✅ WR: 0%")
print(f"✅ PF: {pf_rule:.2f}")

# ============================================================================
# MODEL 2: XGBOOST
# ============================================================================

import xgboost as xgb
import joblib

models_dir = project_root / "models"

def calculate_features(df):
    df = df.copy()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # MAs
    for period in [7, 14, 30]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Returns
    for period in [1, 3, 7, 14]:
        df[f'returns_{period}'] = df['close'].pct_change(period)
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(14).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns_1'].rolling(14).std()
    
    # BB
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Price vs MA
    df['price_vs_sma7'] = df['close'] / df['sma_7']
    df['price_vs_sma14'] = df['close'] / df['sma_14']
    df['price_vs_sma30'] = df['close'] / df['sma_30']
    
    return df

def xgboost_backtest(df):
    # Load model
    model_path = models_dir / "xgboost_normalized_model.json"
    if not model_path.exists():
        model_path = models_dir / "xgboost_model.json"
    
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Load scaler
    scaler_path = models_dir / "xgboost_normalized_scaler.pkl"
    if not scaler_path.exists():
        scaler_path = models_dir / "xgboost_multi_tf_scaler.pkl"
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    
    # Calculate features
    df_feat = calculate_features(df)
    
    feature_cols = [col for col in df_feat.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    feature_cols = feature_cols[:31]
    
    X = df_feat[feature_cols].fillna(0).values
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Predict
    threshold = 0.46
    proba = model.predict_proba(X_scaled)[:, 1]
    signals = (proba >= threshold).astype(int)
    
    # Backtest
    capital = 100000
    position = None
    trades = []
    
    tp = 0.03
    sl = 0.012
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            
            if pnl_pct >= tp:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'tp'})
                position = None
            elif pnl_pct <= -sl:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'sl'})
                position = None
        
        if signals[i] == 1 and position is None:
            position = {'entry': price}
    
    return capital, trades

print("\n[2/3] Testing XGBoost...")
try:
    capital_xgb, trades_xgb = xgboost_backtest(df_test)
    
    wins_xgb = [t for t in trades_xgb if t['pnl'] > 0]
    losses_xgb = [t for t in trades_xgb if t['pnl'] < 0]
    
    total_profit_xgb = sum(t['pnl'] for t in wins_xgb) if wins_xgb else 0
    total_loss_xgb = abs(sum(t['pnl'] for t in losses_xgb)) if losses_xgb else 1
    pf_xgb = total_profit_xgb / total_loss_xgb if total_loss_xgb > 0 else 0
    
    print(f"✅ Return: {(capital_xgb-100000)/1000:.2f}%")
    print(f"✅ Trades: {len(trades_xgb)}")
    print(f"✅ WR: {len(wins_xgb)/len(trades_xgb)*100:.1f}%" if trades_xgb else "✅ WR: 0%")
    print(f"✅ PF: {pf_xgb:.2f}")
except Exception as e:
    print(f"❌ Error: {e}")
    capital_xgb = 100000
    trades_xgb = []
    pf_xgb = 0

# ============================================================================
# MODEL 3: HYBRID (Rule-Based AND XGBoost)
# ============================================================================

def hybrid_backtest(df):
    # Get Rule signals
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    rule_signals = (df['rsi'] < 30).astype(int).values
    
    # Get XGBoost signals
    try:
        model_path = models_dir / "xgboost_normalized_model.json"
        if not model_path.exists():
            model_path = models_dir / "xgboost_model.json"
        
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        
        scaler_path = models_dir / "xgboost_normalized_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = models_dir / "xgboost_multi_tf_scaler.pkl"
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        
        df_feat = calculate_features(df)
        feature_cols = [col for col in df_feat.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        feature_cols = feature_cols[:31]
        
        X = df_feat[feature_cols].fillna(0).values
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        proba = model.predict_proba(X_scaled)[:, 1]
        xgb_signals = (proba >= 0.46).astype(int)
    except:
        xgb_signals = np.zeros(len(df))
    
    # Hybrid: BOTH must agree
    hybrid_signals = (rule_signals & xgb_signals).astype(int)
    
    # Backtest
    capital = 100000
    position = None
    trades = []
    
    tp = 0.03
    sl = 0.012
    cooldown = 0
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            
            if pnl_pct >= tp:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'tp'})
                position = None
                cooldown = 24
            elif pnl_pct <= -sl:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'sl'})
                position = None
                cooldown = 24
        
        if cooldown > 0:
            cooldown -= 1
        
        if hybrid_signals[i] == 1 and position is None and cooldown == 0:
            position = {'entry': price}
    
    return capital, trades

print("\n[3/3] Testing Hybrid...")
try:
    capital_hybrid, trades_hybrid = hybrid_backtest(df_test)
    
    wins_hybrid = [t for t in trades_hybrid if t['pnl'] > 0]
    losses_hybrid = [t for t in trades_hybrid if t['pnl'] < 0]
    
    total_profit_hybrid = sum(t['pnl'] for t in wins_hybrid) if wins_hybrid else 0
    total_loss_hybrid = abs(sum(t['pnl'] for t in losses_hybrid)) if losses_hybrid else 1
    pf_hybrid = total_profit_hybrid / total_loss_hybrid if total_loss_hybrid > 0 else 0
    
    print(f"✅ Return: {(capital_hybrid-100000)/1000:.2f}%")
    print(f"✅ Trades: {len(trades_hybrid)}")
    print(f"✅ WR: {len(wins_hybrid)/len(trades_hybrid)*100:.1f}%" if trades_hybrid else "✅ WR: 0%")
    print(f"✅ PF: {pf_hybrid:.2f}")
except Exception as e:
    print(f"❌ Error: {e}")
    capital_hybrid = 100000
    trades_hybrid = []
    pf_hybrid = 0

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print("\nModel           Return      Trades    Win Rate    Profit Factor")
print("-"*80)

wr_rule = len(wins_rule)/len(trades_rule)*100 if trades_rule else 0
print(f"Rule-Based      {(capital_rule-100000)/1000:6.2f}%     {len(trades_rule):5d}     {wr_rule:5.1f}%        {pf_rule:5.2f}")

if trades_xgb:
    wr_xgb = len(wins_xgb)/len(trades_xgb)*100
    print(f"XGBoost         {(capital_xgb-100000)/1000:6.2f}%     {len(trades_xgb):5d}     {wr_xgb:5.1f}%        {pf_xgb:5.2f}")
else:
    print(f"XGBoost         ERROR       0         0.0%        0.00")

if trades_hybrid:
    wr_hybrid = len(wins_hybrid)/len(trades_hybrid)*100
    print(f"Hybrid          {(capital_hybrid-100000)/1000:6.2f}%     {len(trades_hybrid):5d}     {wr_hybrid:5.1f}%        {pf_hybrid:5.2f}")
else:
    print(f"Hybrid          ERROR       0         0.0%        0.00")

print("="*80)

print("\n✅ COMPARISON COMPLETE")
print(f"\nPeriod tested: {df_test.index[0]} → {df_test.index[-1]}")
print(f"Asset: BTC USDT 15m")
print(f"Bars: {len(df_test):,}")