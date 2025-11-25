"""
DEBUG: Что возвращают стратегии?

Запустить в папке проекта:
python debug_strategies.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.getcwd())

print("="*80)
print("DEBUGGING STRATEGY SIGNALS")
print("="*80)

# Generate test data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=200, freq='1h')
data = pd.DataFrame({
    'timestamp': dates,
    'open': 30000 + np.random.randn(200) * 1000,
    'high': 31000 + np.random.randn(200) * 1000,
    'low': 29000 + np.random.randn(200) * 1000,
    'close': 30000 + np.random.randn(200) * 1000,
    'volume': 1000000 + np.random.randn(200) * 100000
})
data['close'] = data['close'].abs()
data['volume'] = data['volume'].abs()

print(f"\n✅ Test data created: {len(data)} bars")
print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

# Test each strategy
strategies_to_test = [
    ('Rule-Based', 'strategies.rule_based_v2', 'RuleBasedStrategy'),
    ('XGBoost', 'strategies.xgboost_ml_v2', 'XGBoostMLStrategy'),
    ('Hybrid', 'strategies.hybrid_v2', 'HybridStrategy'),
]

for name, module_path, class_name in strategies_to_test:
    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")
    
    try:
        # Import strategy
        module = __import__(module_path, fromlist=[class_name])
        StrategyClass = getattr(module, class_name)
        
        # Create instance
        if 'XGBoost' in name:
            strategy = StrategyClass(model_path='models/xgboost_trained_v2.json')
        else:
            strategy = StrategyClass()
        
        print(f"✅ {name} initialized")
        
        # Check what methods it has
        methods = [m for m in dir(strategy) if not m.startswith('_') and callable(getattr(strategy, m))]
        print(f"   Methods: {methods}")
        
        # Check for signal methods
        signal_methods = []
        if hasattr(strategy, 'generate_signals'):
            signal_methods.append('generate_signals')
        if hasattr(strategy, 'generate_signal'):
            signal_methods.append('generate_signal')
        if hasattr(strategy, 'get_signal'):
            signal_methods.append('get_signal')
        
        print(f"   Signal methods: {signal_methods}")
        
        if not signal_methods:
            print(f"   ❌ NO SIGNAL METHODS FOUND!")
            continue
        
        # Test signal generation
        window_data = data.iloc[:100]  # First 100 bars
        
        for method_name in signal_methods:
            print(f"\n   Testing method: {method_name}()")
            try:
                method = getattr(strategy, method_name)
                result = method(window_data)
                
                print(f"      Result type: {type(result)}")
                print(f"      Result value: {result}")
                
                # Try to extract signal
                if isinstance(result, dict):
                    signal = result.get('signal', None)
                    confidence = result.get('confidence', None)
                    print(f"      → signal={signal}, confidence={confidence}")
                elif isinstance(result, (int, float)):
                    print(f"      → numeric signal: {result}")
                elif result is None:
                    print(f"      → None (no signal)")
                else:
                    print(f"      → Unknown format: {result}")
                
            except Exception as e:
                print(f"      ❌ ERROR: {e}")
        
    except Exception as e:
        print(f"❌ Failed to test {name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("DEBUGGING COMPLETE")
print(f"{'='*80}")