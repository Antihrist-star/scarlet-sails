"""
INTEGRATED BACKTEST - Full System Test
Tests all components together: FeatureEngine + AdvancedModelManager + XGBoostStrategy + SignalValidator + BacktestEngine
"""
import yaml
import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from core.backtest_engine import BacktestEngine
from strategies.xgboost_ml import XGBoostStrategy
from ai_modules.advanced_model_manager import AdvancedModelManager
from ai_modules.enhanced_signal_validator import EnhancedSignalValidator

def main():
    logger.info("="*80)
    logger.info("INTEGRATED SYSTEM TEST")
    logger.info("="*80)
    
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("\n[1/6] Loading configuration...")
    logger.info(f"  Assets: {len(config['data']['assets'])}")
    logger.info(f"  Timeframes: {config['data']['timeframes']}")
    
    # 2. Initialize components
    logger.info("\n[2/6] Initializing components...")
    
    data_loader = DataLoader(config)
    feature_engine = FeatureEngine(config)
    
    # Advanced Model Manager
    model_path = Path(config['models']['xgboost']['model_path'])
    model_manager = AdvancedModelManager(config, str(model_path))
    
    # Signal Validator
    validator = EnhancedSignalValidator(feature_engine, model_manager)
    
    # Strategy
    strategy = XGBoostStrategy(config, feature_engine, model_manager)
    
    # Backtest Engine
    backtest = BacktestEngine(config)
    
    logger.info("  ✅ All components initialized")
    
    # 3. Load test data
    logger.info("\n[3/6] Loading test data...")
    
    test_asset = "BTC"
    test_tf = "15m"
    
    df = data_loader.load_ohlcv(test_asset, test_tf)
    
    if df is None:
        logger.error("  ❌ Failed to load data")
        return
    
    # Use last 30 days for quick test
    df_test = df.tail(30 * 24 * 4).copy()  # 30 days * 24 hours * 4 (15min bars)
    
    logger.info(f"  Loaded {len(df_test):,} bars")
    logger.info(f"  Period: {df_test.index[0]} → {df_test.index[-1]}")
    
    # 4. Generate signals
    logger.info("\n[4/6] Generating ML signals...")
    
    signals, ml_scores, pj_s_values = strategy.generate_signals_with_pj_s(df_test)
    
    logger.info(f"  Raw signals: {signals.sum()}")
    logger.info(f"  Average ML score: {ml_scores.mean():.3f}")
    logger.info(f"  Average P_j(S): {pj_s_values.mean():.3f}")
    
    # 5. Validate signals
    logger.info("\n[5/6] Validating signals...")
    
    validated_signals = validator.validate_batch(signals, df_test, pj_s_values)
    
    logger.info(f"  Validated signals: {validated_signals.sum()}")
    logger.info(f"  Filtered: {signals.sum() - validated_signals.sum()}")
    
    # 6. Run backtest
    logger.info("\n[6/6] Running backtest...")
    
    results = backtest.run(df_test, validated_signals)
    
    # Extract metrics
    metrics = results['metrics']
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)
    
    logger.info(f"\nCapital:")
    logger.info(f"  Initial:  ${config['trading']['initial_capital']:,.0f}")
    logger.info(f"  Final:    ${results['capital']:,.0f}")
    logger.info(f"  Return:   {metrics['total_pnl_pct']:.2f}%")
    
    logger.info(f"\nTrades:")
    logger.info(f"  Total:        {metrics['trades']}")
    logger.info(f"  Wins:         {metrics['wins']}")
    logger.info(f"  Losses:       {metrics['losses']}")
    logger.info(f"  Win Rate:     {metrics['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    logger.info(f"\nRisk:")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Performance assessment
    logger.info("\n" + "="*80)
    logger.info("ASSESSMENT")
    logger.info("="*80)
    
    if metrics['profit_factor'] > 2.0:
        logger.info("✅ EXCELLENT - PF > 2.0")
    elif metrics['profit_factor'] > 1.5:
        logger.info("✅ GOOD - PF > 1.5")
    elif metrics['profit_factor'] > 1.0:
        logger.info("⚠️  MARGINAL - PF > 1.0")
    else:
        logger.info("❌ LOSING - PF < 1.0")
    
    if metrics['max_drawdown'] < 0.10:
        logger.info("✅ LOW RISK - DD < 10%")
    elif metrics['max_drawdown'] < 0.15:
        logger.info("⚠️  MODERATE RISK - DD < 15%")
    else:
        logger.info("❌ HIGH RISK - DD > 15%")
    
    # Signal analysis
    logger.info("\n" + "="*80)
    logger.info("SIGNAL ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"ML Signals:        {signals.sum()}")
    logger.info(f"Validated:         {validated_signals.sum()}")
    if signals.sum() > 0:
        logger.info(f"Filter Rate:       {(1 - validated_signals.sum()/signals.sum())*100:.1f}%")
    logger.info(f"Executed Trades:   {metrics['trades']}")
    if validated_signals.sum() > 0:
        logger.info(f"Trade Rate:        {metrics['trades']/validated_signals.sum()*100:.1f}%")
    
    logger.info("\n" + "="*80)
    logger.info("✅ БЛОК 2 ЗАВЕРШЁН!")
    logger.info("="*80)
    
    return results

if __name__ == "__main__":
    results = main()