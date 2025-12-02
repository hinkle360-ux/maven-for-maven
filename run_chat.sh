#!/bin/bash
# Simple launcher for the Maven chat interface on Linux/macOS.
#
# This script locates the project root directory, sets PYTHONPATH accordingly,
# clears stale cache, and launches the chat interface via Python's module system.

# Change to the directory containing this script (the Maven project root)
cd "$(dirname "$0")"

# Clear Python bytecode cache to ensure latest code runs
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Clear stale response cache from previous sessions
echo "Clearing stale response cache..."
rm -rf reports/system/*.json 2>/dev/null
rm -f reports/context_snapshot.json 2>/dev/null

# Set PYTHONPATH to the project root
export PYTHONPATH="$(pwd)"

echo "Starting Maven chat interface..."
echo ""

# Launch the chat interface as a module
python3 -m ui.maven_chat "$@"
