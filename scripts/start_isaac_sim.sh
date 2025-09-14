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

# Set up Isaac Sim environment variables
export ISAAC_PATH="$ISAAC_SIM_PATH"
export CARB_APP_PATH="$ISAAC_SIM_PATH/kit"
export EXP_PATH="$ISAAC_SIM_PATH"

# Check OpenVLA server
echo "🔍 Checking OpenVLA server..."
if curl -s http://0.0.0.0:8000/act > /dev/null 2>&1; then
    echo "✅ OpenVLA server is running"
else
    echo "⚠️  OpenVLA server not detected at http://0.0.0.0:8000/act"
    echo "   Start your OpenVLA server before running the demo"
fi

echo ""
echo "🚀 Starting Isaac Sim with automatic demo execution..."
echo ""

# Create a temporary startup script that Isaac Sim will run
TEMP_SCRIPT="/tmp/cookbot_startup.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Use the minimal working test script
    cp "$SCRIPT_DIR/test_minimal_isaac.py" "$TEMP_SCRIPT"

# Start Isaac Sim with the startup script
"$ISAAC_SIM_PATH/isaac-sim.sh" --exec "$TEMP_SCRIPT"