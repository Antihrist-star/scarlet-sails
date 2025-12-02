"""
Tests for train_xgboost_v3.py CLI interface.
Verifies that the script can be imported and run without ModuleNotFoundError.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_train_xgboost_v3_import():
    """Test that train_xgboost_v3.py can be imported without errors."""
    # This test verifies no ModuleNotFoundError on import
    try:
        import scripts.train_xgboost_v3 as train_script
        assert train_script is not None
        assert hasattr(train_script, 'main')
    except ModuleNotFoundError as e:
        pytest.fail(f"Failed to import train_xgboost_v3: {e}")


def test_train_xgboost_v3_help():
    """Test that train_xgboost_v3.py -h runs without errors."""
    script_path = Path(__file__).parent.parent / "scripts" / "train_xgboost_v3.py"
    
    if not script_path.exists():
        pytest.skip(f"Script not found: {script_path}")
    
    result = subprocess.run(
        [sys.executable, str(script_path), "-h"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Should exit with 0 and show help message
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "--coin" in result.stdout, "Missing --coin argument in help"
    assert "--tf" in result.stdout, "Missing --tf argument in help"
    assert "--no-backtest" in result.stdout, "Missing --no-backtest argument in help"


def test_simple_threshold_backtest_import():
    """Test that simple_threshold_backtest module can be imported."""
    try:
        from analysis.simple_threshold_backtest import (
            evaluate_thresholds,
            select_optimal_threshold,
            calculate_sharpe_ratio,
            calculate_max_drawdown,
        )
        assert evaluate_thresholds is not None
        assert select_optimal_threshold is not None
        assert calculate_sharpe_ratio is not None
        assert calculate_max_drawdown is not None
    except ModuleNotFoundError as e:
        pytest.fail(f"Failed to import simple_threshold_backtest: {e}")


if __name__ == "__main__":
    # Run tests
    print("Running tests for train_xgboost_v3.py CLI...")
    print("\n" + "="*60)
    
    try:
        test_train_xgboost_v3_import()
        print("✓ Import test passed")
    except Exception as e:
        print(f"✗ Import test failed: {e}")
    
    try:
        test_train_xgboost_v3_help()
        print("✓ Help command test passed")
    except Exception as e:
        print(f"✗ Help command test failed: {e}")
    
    try:
        test_simple_threshold_backtest_import()
        print("✓ Threshold backtest import test passed")
    except Exception as e:
        print(f"✗ Threshold backtest import test failed: {e}")
    
    print("="*60)
    print("\nAll tests completed!")
