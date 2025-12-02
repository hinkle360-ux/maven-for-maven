#!/bin/bash
# Launcher for Maven FULL AGENCY mode (unrestricted tool access)
#
# This script launches Maven in FULL_AGENCY profile where:
# - Read/write files anywhere on disk (OS permissions apply)
# - Run shell commands
# - Run Python code
# - Browse the web
# - Full git operations (clone, push, etc.)
# - Run autonomous agents
#
# Only truly destructive commands (rm -rf /, mkfs, etc.) are blocked.

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

# Set the capability profile to FULL_AGENCY
export MAVEN_CAPABILITIES_PROFILE="FULL_AGENCY"

echo ""
echo "=========================================="
echo "  MAVEN CHAT - FULL AGENCY MODE"
echo "  Unrestricted access to all tools"
echo "=========================================="
echo ""
echo "WARNING: This mode allows Maven to execute shell commands,"
echo "access files anywhere, and run code. Use with care!"
echo ""

# Launch the chat interface as a module
python3 -m ui.maven_chat "$@"
