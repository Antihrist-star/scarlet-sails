#!/bin/bash
#
# Clear Python cache to ensure fresh code is loaded
#
# WHY: Python caches bytecode in __pycache__ directories.
# If code was fixed but cache not cleared, Python may load OLD code!
#

echo "=========================================="
echo "CLEARING PYTHON CACHE"
echo "=========================================="

cd "$(dirname "$0")/.." || exit 1

# Find and remove all __pycache__ directories
echo ""
echo "Finding __pycache__ directories..."
find . -type d -name "__pycache__" | while read -r dir; do
    echo "  Removing: $dir"
    rm -rf "$dir"
done

# Find and remove all .pyc files
echo ""
echo "Finding .pyc files..."
pyc_count=$(find . -name "*.pyc" | wc -l)
if [ "$pyc_count" -gt 0 ]; then
    echo "  Removing $pyc_count .pyc files..."
    find . -name "*.pyc" -delete
else
    echo "  No .pyc files found"
fi

echo ""
echo "✅ Python cache cleared!"
echo ""
echo "Next steps:"
echo "  1. python scripts/verify_timeframe_fix.py  # Verify fix works"
echo "  2. python scripts/retrain_xgboost_normalized.py  # Retrain"
echo "  3. python scripts/comprehensive_model_audit.py  # Test"
echo ""
