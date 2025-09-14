#!/bin/bash
"""
Start Isaac Sim with Cookbot Integration
Launches Isaac Sim and provides setup instructions
"""

set -e

echo "🤖 Cookbot - Isaac Sim + OpenVLA"
echo "================================"

# Check Isaac Sim installation
ISAAC_SIM_PATH="$HOME/isaacsim"
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "❌ Isaac Sim not found at $ISAAC_SIM_PATH"
    echo "Please install Isaac Sim first."
    exit 1
fi

# Check OpenVLA server
echo "🔍 Checking OpenVLA server..."
if curl -s http://0.0.0.0:8000/act > /dev/null 2>&1; then
    echo "✅ OpenVLA server is running"
else
    echo "⚠️  OpenVLA server not detected at http://0.0.0.0:8000/act"
    echo "   Start your OpenVLA server before running the demo"
fi

echo ""
echo "🚀 Starting Isaac Sim..."
echo ""
echo "After Isaac Sim loads, run in the Python console:"
echo "  exec(open('scripts/run_isaac_vla_demo.py').read())"
echo ""

# Start Isaac Sim
"$ISAAC_SIM_PATH/isaac-sim.sh"