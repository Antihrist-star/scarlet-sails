"""
CHECK STRATEGY METHODS
Find out REAL method names in RuleBasedStrategy

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import sys
sys.path.append('.')

from strategies.rule_based_v2 import RuleBasedStrategy

def main():
    """Check strategy methods"""
    print("="*80)
    print("CHECKING RULE-BASED STRATEGY METHODS")
    print("="*80)
    print()
    
    # Initialize strategy
    try:
        strategy = RuleBasedStrategy()
        print("✅ Strategy initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print()
    print("AVAILABLE METHODS:")
    print("-"*80)
    
    # Get all methods
    methods = [method for method in dir(strategy) if not method.startswith('_')]
    
    for i, method in enumerate(methods, 1):
        # Get method object
        method_obj = getattr(strategy, method)
        
        # Check if callable
        if callable(method_obj):
            print(f"{i:3d}. {method}()")
        else:
            print(f"{i:3d}. {method} [attribute]")
    
    print()
    print("="*80)
    print("LOOKING FOR SIGNAL GENERATION METHOD:")
    print("-"*80)
    
    # Look for signal-related methods
    signal_methods = []
    for method in methods:
        if any(keyword in method.lower() for keyword in ['signal', 'predict', 'generate', 'get', 'calculate']):
            method_obj = getattr(strategy, method)
            if callable(method_obj):
                signal_methods.append(method)
    
    if signal_methods:
        print("Found potential signal methods:")
        for method in signal_methods:
            print(f"   ✅ {method}()")
    else:
        print("⚠️ No obvious signal generation method found")
    
    print()
    print("="*80)
    print("TESTING LIKELY METHODS:")
    print("-"*80)
    
    # Test common method names
    test_names = [
        'generate_signal',
        'get_signal', 
        'predict',
        'generate',
        'signal',
        'calculate_signal'
    ]
    
    for name in test_names:
        if hasattr(strategy, name):
            method = getattr(strategy, name)
            if callable(method):
                print(f"   ✅ {name}() EXISTS!")
            else:
                print(f"   ⚠️ {name} exists but not callable")
        else:
            print(f"   ❌ {name}() NOT FOUND")
    
    print()
    print("="*80)
    print("SAVE THIS OUTPUT!")
    print("We need the correct method name for testing")
    print("="*80)


if __name__ == "__main__":
    main()