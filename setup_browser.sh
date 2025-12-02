#!/bin/bash
# Browser Runtime Setup Script
# =============================
#
# This script sets up the Maven Browser Runtime system.

set -e  # Exit on error

echo "============================================"
echo "Maven Browser Runtime Setup"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found. Please install Python 3.8+"; exit 1; }
echo "✓ Python found"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r browser_requirements.txt || { echo "Error: Failed to install dependencies"; exit 1; }
echo "✓ Dependencies installed"
echo ""

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install chromium || { echo "Error: Failed to install Playwright browsers"; exit 1; }
echo "✓ Playwright browsers installed"
echo ""

# Create .env.browser if it doesn't exist
if [ ! -f .env.browser ]; then
    echo "Creating .env.browser configuration file..."
    cp .env.browser.example .env.browser
    echo "✓ Configuration file created"
    echo ""
    echo "⚠️  Please review and edit .env.browser as needed"
else
    echo "✓ Configuration file already exists"
fi
echo ""

# Create log directories
echo "Creating log directories..."
mkdir -p logs/browser/tasks
mkdir -p logs/browser/screenshots
mkdir -p logs/browser/patterns
echo "✓ Log directories created"
echo ""

# Run smoke tests
echo "Running smoke tests..."
if pytest tests/test_browser_smoke.py -v -m smoke -x; then
    echo "✓ Smoke tests passed"
else
    echo "⚠️  Some smoke tests failed. This may be OK if server isn't running yet."
fi
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Review configuration: nano .env.browser"
echo "2. Start server:"
echo "   - Windows: start_browser_server.cmd"
echo "   - Linux/Mac: python -m optional.browser_runtime"
echo "3. Run examples: python docs/examples/browser_examples.py"
echo "4. Run tests: pytest tests/test_browser_*.py -v"
echo ""
echo "For more information, see:"
echo "  - README_BROWSER.md"
echo "  - docs/BROWSER_RUNTIME.md"
echo ""
