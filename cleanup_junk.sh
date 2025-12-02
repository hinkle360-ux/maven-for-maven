#!/bin/bash
# Cleanup script for removing junk and temporary files from the Maven project.
#
# This script removes:
# - Python bytecode files (*.pyc)
# - __pycache__ directories
# - Log files (*.log)
# - Temporary test files (tmp_*.txt, *_tmp.txt)
# - Patch test files (patch_test*.txt)
# - Other temporary files
#
# Usage: ./cleanup_junk.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "============================================"
echo "Maven Project Cleanup Utility"
echo "============================================"
echo ""
echo "Cleaning up junk files in: $SCRIPT_DIR"
echo ""

COUNT=0

# Remove Python bytecode files
echo "[1/5] Removing Python bytecode files (*.pyc)..."
while IFS= read -r -d '' file; do
    rm -f "$file" && echo "  Deleted: $file" && ((COUNT++))
done < <(find "$SCRIPT_DIR" -type f -name "*.pyc" -print0 2>/dev/null)

# Remove __pycache__ directories
echo ""
echo "[2/5] Removing __pycache__ directories..."
while IFS= read -r -d '' dir; do
    rm -rf "$dir" && echo "  Deleted: $dir" && ((COUNT++))
done < <(find "$SCRIPT_DIR" -type d -name "__pycache__" -print0 2>/dev/null)

# Remove log files
echo ""
echo "[3/5] Removing log files (*.log)..."
while IFS= read -r -d '' file; do
    rm -f "$file" && echo "  Deleted: $file" && ((COUNT++))
done < <(find "$SCRIPT_DIR" -type f -name "*.log" -print0 2>/dev/null)

# Remove temporary test files
echo ""
echo "[4/5] Removing temporary test files (tmp_*.txt, *_tmp.txt)..."
while IFS= read -r -d '' file; do
    rm -f "$file" && echo "  Deleted: $file" && ((COUNT++))
done < <(find "$SCRIPT_DIR" -type f \( -name "tmp_*.txt" -o -name "*_tmp.txt" \) -print0 2>/dev/null)

# Remove patch test files
echo ""
echo "[5/5] Removing patch test files (patch_test*.txt)..."
while IFS= read -r -d '' file; do
    rm -f "$file" && echo "  Deleted: $file" && ((COUNT++))
done < <(find "$SCRIPT_DIR" -type f -name "patch_test*.txt" -print0 2>/dev/null)

echo ""
echo "============================================"
echo "Cleanup Complete! Removed $COUNT items."
echo "============================================"
echo ""
