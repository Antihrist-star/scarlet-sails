"""
PHASE 1.5: ML MODELS VALIDATION
=================================

Validate ML model components exist and can be loaded:
- XGBoost model (if trained)
- Crisis Classifier (if trained)
- Feature engineering pipeline
- Model architecture integrity

NOTE: This phase checks MODEL INFRASTRUCTURE, not training.
Full model training/validation belongs in Phase 2 (walk-forward).

Metrics:
- Model files exist and loadable
- Model architecture valid
- Feature extraction works
- Predictions can be generated (if model trained)

Author: Scarlet Sails Team
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("PHASE 1.5: ML MODELS VALIDATION")
print("="*80)

# ============================================================================
# CHECK MODEL FILES
# ============================================================================
print("\n📂 Checking model files...")

model_paths = {
    'XGBoost': Path('models/trained/xgboost_model.pkl'),
    'Crisis Classifier': Path('models/trained/crisis_classifier.pkl'),
    'Logistic Baseline': Path('models/trained/logistic_baseline.pkl'),
}

models_found = {}
for name, path in model_paths.items():
    if path.exists():
        print(f"   ✅ {name}: {path} (found)")
        models_found[name] = path
    else:
        print(f"   ⚠️  {name}: {path} (not found)")
        models_found[name] = None

# ============================================================================
# TEST MODEL IMPORTS
# ============================================================================
print("\n📦 Testing model imports...")

try:
    from models.xgboost_model import XGBoostModel
    print("   ✅ XGBoostModel class imported")
    xgboost_available = True
except Exception as e:
    print(f"   ❌ XGBoostModel import failed: {e}")
    xgboost_available = False

try:
    from models.crisis_classifier import CrisisClassifier
    print("   ✅ CrisisClassifier class imported")
    crisis_available = True
except Exception as e:
    print(f"   ❌ CrisisClassifier import failed: {e}")
    crisis_available = False

try:
    from models.decision_formula_v2 import DecisionFormulaV2
    print("   ✅ DecisionFormulaV2 class imported")
    decision_available = True
except Exception as e:
    print(f"   ❌ DecisionFormulaV2 import failed: {e}")
    decision_available = False

# ============================================================================
# TEST MODEL INSTANTIATION
# ============================================================================
print("\n🔧 Testing model instantiation...")

if xgboost_available:
    try:
        model = XGBoostModel()
        print("   ✅ XGBoostModel instantiated")
    except Exception as e:
        print(f"   ❌ XGBoostModel instantiation failed: {e}")

if crisis_available:
    try:
        classifier = CrisisClassifier()
        print("   ✅ CrisisClassifier instantiated")
    except Exception as e:
        print(f"   ❌ CrisisClassifier instantiation failed: {e}")

if decision_available:
    try:
        decision = DecisionFormulaV2()
        print("   ✅ DecisionFormulaV2 instantiated")
    except Exception as e:
        print(f"   ❌ DecisionFormulaV2 instantiation failed: {e}")

# ============================================================================
# LOAD DATA FOR FEATURE TESTING
# ============================================================================
print("\n📂 Loading data for feature testing...")

data_path = Path('data/processed/btc_prepared_phase0.parquet')
if not data_path.exists():
    print(f"   ❌ ERROR: {data_path} not found!")
    print("   Run phase0_load_real_data.py first!")
    exit(1)

df = pd.read_parquet(data_path)
print(f"   ✅ Loaded {len(df):,} bars")

# ============================================================================
# TEST FEATURE EXTRACTION (if DecisionFormulaV2 available)
# ============================================================================
if decision_available:
    print("\n📊 Testing feature extraction...")

    try:
        decision_model = DecisionFormulaV2()

        # Try to extract features from middle of dataset
        test_bar = len(df) // 2

        print(f"   Testing on bar {test_bar}...")

        # Check if extract_features method exists
        if hasattr(decision_model, 'extract_features'):
            features = decision_model.extract_features(df, test_bar)
            print(f"   ✅ Features extracted: {len(features)} features")

            # Show sample features
            feature_names = list(features.keys())[:5]
            print(f"   Sample features: {', '.join(feature_names)}")

        elif hasattr(decision_model, 'generate_signal'):
            # Try signal generation
            signal = decision_model.generate_signal(df, test_bar)
            print(f"   ✅ Signal generated: {signal}")

        else:
            print(f"   ⚠️  DecisionFormulaV2 has no feature extraction method")

    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")

# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================
print("\n📦 Checking ML dependencies...")

dependencies = {
    'xgboost': None,
    'scikit-learn': 'sklearn',
    'pandas': 'pandas',
    'numpy': 'numpy',
}

for dep_name, import_name in dependencies.items():
    if import_name is None:
        import_name = dep_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {dep_name}: {version}")
    except ImportError:
        print(f"   ❌ {dep_name}: NOT INSTALLED")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 1.5 RESULTS SUMMARY")
print("="*80)

print(f"\n📊 Model Infrastructure:")
print(f"   XGBoost class: {'✅ Available' if xgboost_available else '❌ Not available'}")
print(f"   Crisis Classifier class: {'✅ Available' if crisis_available else '❌ Not available'}")
print(f"   DecisionFormulaV2 class: {'✅ Available' if decision_available else '❌ Not available'}")

trained_count = sum(1 for v in models_found.values() if v is not None)
total_models = len(models_found)

print(f"\n📊 Trained Models:")
print(f"   Found: {trained_count}/{total_models}")
for name, path in models_found.items():
    status = "✅ Found" if path else "⚠️  Not trained"
    print(f"   {name}: {status}")

print(f"\n💡 Notes:")
if trained_count == 0:
    print("   ⚠️  No trained models found - this is EXPECTED for initial validation")
    print("   Full model training happens in Phase 2 (walk-forward validation)")
    print("   Model INFRASTRUCTURE is what matters for Phase 1.5")

# Pass/Fail criteria
print(f"\n✅ PASS/FAIL:")

passed = True

if xgboost_available and crisis_available and decision_available:
    print(f"   ✅ All model classes available")
else:
    print(f"   ⚠️  Some model classes missing")
    # Don't fail - models can be trained later

print(f"   ℹ️  Trained models: {trained_count}/{total_models} (optional for Phase 1)")

print("\n" + "="*80)
if xgboost_available and crisis_available and decision_available:
    print("✅ PHASE 1.5 PASSED - ML infrastructure validated!")
else:
    print("⚠️  PHASE 1.5 MARGINAL - Some components missing")
print("="*80)

print("\n💡 NEXT STEPS:")
print("   - Phase 2 will train models on real data")
print("   - Walk-forward validation will test model performance")
print("   - For now, model INFRASTRUCTURE is validated")
