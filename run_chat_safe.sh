#!/bin/bash
# Launcher for Maven SAFE CHAT mode (no tools, pure conversation)
#
# This script launches Maven in SAFE_CHAT profile where:
# - No file access (read/write)
# - No shell/Python execution
# - No web access
# - No git operations
# - Pure conversation only

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

# Set the capability profile to SAFE_CHAT
export MAVEN_CAPABILITIES_PROFILE="SAFE_CHAT"

echo ""
echo "=========================================="
echo "  MAVEN CHAT - SAFE MODE"
echo "  No tools, pure conversation"
echo "=========================================="
echo ""

# Launch the chat interface as a module
python3 -m ui.maven_chat "$@"
