#!/usr/bin/env python3
"""
Verify that the timeframe fix is actually working

This script checks if:
1. extract_features_at_bar correctly uses target_tf (not hardcoded '15m')
2. Column names match between calculate_indicators and extract_features_at_bar
3. Features are normalized (values around 0-2, not thousands)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor

def verify_fix():
    """Verify the timeframe fix is working"""

    print("=" * 70)
    print("TIMEFRAME FIX VERIFICATION")
    print("=" * 70)

    extractor = MultiTimeframeFeatureExtractor()

    # Test on BTC with different timeframes
    test_cases = [
        ('BTC', '15m', 1000),
        ('BTC', '1h', 500),
        ('BTC', '4h', 200),
    ]

    results = {}

    for asset, target_tf, bar_idx in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {asset} {target_tf}")
        print(f"{'='*70}")

        try:
            # Load and prepare data
            print(f"Loading data...")
            all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, target_tf)

            # Check columns exist
            required_cols = [
                f'{target_tf}_RSI_14',
                f'{target_tf}_EMA_9',
                f'{target_tf}_EMA_21',
                f'{target_tf}_ATR_14',
                f'{target_tf}_BB_std',
            ]

            missing_cols = [col for col in required_cols if col not in primary_df.columns]

            if missing_cols:
                print(f"❌ FAILED: Missing columns for {target_tf}:")
                for col in missing_cols:
                    print(f"   - {col}")
                results[f"{asset}_{target_tf}"] = "MISSING_COLUMNS"
                continue

            print(f"✅ All required columns exist for {target_tf}")
            print(f"   Sample columns: {required_cols[:3]}")

            # Extract features
            features = extractor.extract_features_at_bar(all_tf, target_tf, bar_idx)

            if features is None:
                print(f"❌ FAILED: extract_features_at_bar returned None")
                results[f"{asset}_{target_tf}"] = "NONE_FEATURES"
                continue

            # Check feature count
            if len(features) != 31:
                print(f"❌ FAILED: Expected 31 features, got {len(features)}")
                results[f"{asset}_{target_tf}"] = "WRONG_FEATURE_COUNT"
                continue

            print(f"✅ Correct feature count: {len(features)}")

            # Check normalization (values should be reasonable)
            print(f"\nFeature value ranges:")
            print(f"   Min: {features.min():.4f}")
            print(f"   Max: {features.max():.4f}")
            print(f"   Mean: {features.mean():.4f}")
            print(f"   Sample first 5: {features[:5]}")

            # Check if features look normalized (not absolute prices)
            if features.max() > 1000:
                print(f"❌ FAILED: Features NOT normalized (max={features.max():.0f} >> 1000)")
                results[f"{asset}_{target_tf}"] = "NOT_NORMALIZED"
                continue

            if np.any(features > 50):
                print(f"⚠️  WARNING: Some features unusually large (max={features.max():.2f})")

            # Check for NaN/inf
            if np.isnan(features).any():
                print(f"❌ FAILED: Features contain NaN")
                results[f"{asset}_{target_tf}"] = "NAN_FEATURES"
                continue

            if np.isinf(features).any():
                print(f"❌ FAILED: Features contain Inf")
                results[f"{asset}_{target_tf}"] = "INF_FEATURES"
                continue

            print(f"✅ Features are normalized and valid")
            results[f"{asset}_{target_tf}"] = "PASS"

        except FileNotFoundError as e:
            print(f"⚠️  SKIPPED: Data file not found")
            print(f"   {e}")
            results[f"{asset}_{target_tf}"] = "NO_DATA"
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[f"{asset}_{target_tf}"] = f"ERROR: {str(e)[:50]}"

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for test_case, result in results.items():
        status = "✅" if result == "PASS" else ("⚠️" if result == "NO_DATA" else "❌")
        print(f"{status} {test_case:20s} → {result}")

    # Overall assessment
    passed = sum(1 for r in results.values() if r == "PASS")
    failed = sum(1 for r in results.values() if r not in ["PASS", "NO_DATA"])

    print(f"\n{'='*70}")
    if failed == 0 and passed > 0:
        print("✅ TIMEFRAME FIX IS WORKING!")
        print(f"   Passed: {passed}/{len(results)} tests")
        print("\n   You can proceed with retraining and auditing.")
    elif failed > 0:
        print("❌ TIMEFRAME FIX HAS ISSUES!")
        print(f"   Failed: {failed}/{len(results)} tests")
        print("\n   DO NOT retrain until this is fixed!")
    else:
        print("⚠️  NO DATA AVAILABLE FOR TESTING")
        print("   Tests look correct in code, but need data to verify.")
    print(f"{'='*70}")

if __name__ == "__main__":
    verify_fix()
