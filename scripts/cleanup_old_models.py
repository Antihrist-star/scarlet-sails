"""
CLEANUP OLD MODELS - SCARLET SAILS (FIXED)
Remove old model files to free up disk space

Will delete:
- Old training datasets (.pt files): ~2.1 GB
- Old CNN models (.pth): ~1.5 MB
- Old DQN checkpoints: ~3 MB
- Old logistic models: ~6 KB

Total freed: ~2.2 GB

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 22, 2025
"""

import os
import shutil
from pathlib import Path

# Files to delete
OLD_FILES = [
    # Old training datasets (HUGE!)
    'models/X_train_swing_3d.pt',        # 1.69 GB
    'models/X_test_swing_3d.pt',         # 423 MB
    'models/X_train_enriched_v2.pt',     # 48 MB
    'models/X_train_clean.pt',           # 27 MB
    'models/X_test_enriched_v2.pt',      # 12 MB
    'models/X_test_clean.pt',            # 7 MB
    'models/y_train_swing_3d.pt',        # 1.8 MB
    'models/y_train_clean.pt',           # 1.8 MB
    'models/y_train_enriched_v2.pt',     # 1.8 MB
    'models/y_test_enriched_v2.pt',      # 456 KB
    'models/y_test_swing_3d.pt',         # 456 KB
    'models/y_test_clean.pt',            # 456 KB
    'models/X_test_rolling_3m.pt',       # 333 KB
    
    # Old CNN models
    'models/1d_cnn_model.pth',
    'models/daily_direction_v4.pth',
    'models/daily_direction_v4_experiment.pth',
    'models/profitable_cnn_v3.pth',
    'models/triple_barrier_v5.pth',
    'models/corrected_cnn_model.pth',
    
    # Old logistic models
    'models/logistic_enriched_v2.pth',
    'models/logistic_baseline_clean_2d.pth',
    'models/logistic_baseline_swing3d.pth',
    
    # Old DQN checkpoints (keep only dqn_best_pnl.pth)
    'models/dqn_best.pth',
    'models/dqn_final.pth',
    'models/dqn_episode_10.pth',
    'models/dqn_episode_20.pth',
    'models/dqn_episode_30.pth',
    'models/dqn_episode_40.pth',
    'models/dqn_episode_50.pth',
    'models/dqn_episode_60.pth',
    'models/dqn_episode_70.pth',
    'models/dqn_episode_80.pth',
    'models/dqn_episode_90.pth',
    'models/dqn_episode_100.pth',
    'models/dqn_best_reward.pth',
]

# Files to KEEP (important!)
KEEP_FILES = [
    'models/xgboost_trained.json',       # Old XGBoost (backup)
    'models/xgboost_trained_v2.json',    # New XGBoost (CURRENT)
    'models/dqn_best_pnl.pth',           # DQN model (CURRENT)
    'models/xgboost_model.json',         # Alternative XGBoost
    'models/training_history.csv',       # Training logs
]


def format_size(size_bytes):
    """
    Format file size in human-readable format
    
    FIXED: Handle large numbers without overflow
    """
    if size_bytes == 0:
        return "0 B"
    
    # Use float to avoid overflow
    size = float(size_bytes)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} PB"


def main():
    """
    Main cleanup function
    """
    print("="*80)
    print("SCARLET SAILS - MODEL CLEANUP (FIXED)")
    print("="*80)
    print()
    
    # Check current directory
    if not os.path.exists('models'):
        print("ERROR: models/ directory not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Calculate total size
    total_size = 0
    files_found = []
    files_missing = []
    
    print("SCANNING FILES...")
    print("-"*80)
    
    for filepath in OLD_FILES:
        if os.path.exists(filepath):
            try:
                size = os.path.getsize(filepath)
                total_size += size
                files_found.append((filepath, size))
                print(f"✓ Found: {filepath:50s} {format_size(size):>10s}")
            except Exception as e:
                print(f"✗ Error reading {filepath}: {e}")
                files_missing.append(filepath)
        else:
            files_missing.append(filepath)
            print(f"✗ Missing: {filepath}")
    
    print()
    print("-"*80)
    print(f"Total files to delete: {len(files_found)}")
    print(f"Total size to free: {format_size(total_size)}")
    print(f"Missing files: {len(files_missing)}")
    print()
    
    if not files_found:
        print("No files to delete!")
        return
    
    # Confirm deletion
    print("="*80)
    print("⚠️  WARNING: This will permanently delete the files listed above!")
    print("="*80)
    print()
    
    response = input("Continue with deletion? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Cleanup cancelled.")
        return
    
    print()
    print("DELETING FILES...")
    print("-"*80)
    
    deleted_count = 0
    deleted_size = 0
    errors = []
    
    for filepath, size in files_found:
        try:
            os.remove(filepath)
            deleted_count += 1
            deleted_size += size
            print(f"✓ Deleted: {filepath}")
        except Exception as e:
            errors.append((filepath, str(e)))
            print(f"✗ Error deleting {filepath}: {e}")
    
    print()
    print("="*80)
    print("CLEANUP COMPLETE!")
    print("="*80)
    print(f"Files deleted: {deleted_count}/{len(files_found)}")
    print(f"Space freed: {format_size(deleted_size)}")
    
    if errors:
        print(f"Errors: {len(errors)}")
        for filepath, error in errors:
            print(f"  - {filepath}: {error}")
    
    print()
    print("IMPORTANT FILES KEPT:")
    print("-"*80)
    for filepath in KEEP_FILES:
        if os.path.exists(filepath):
            try:
                size = os.path.getsize(filepath)
                print(f"✓ {filepath:50s} {format_size(size):>10s}")
            except:
                print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} (not found)")
    
    print()
    print("✅ CLEANUP SUCCESSFUL!")
    print()
    print("Next steps:")
    print("  1. Verify models still work:")
    print("     python run_dispersion_analysis.py")
    print("  2. Commit to git (tomorrow)")
    print()


if __name__ == "__main__":
    main()