#!/bin/bash
# Start Isaac Sim with Cookbot Integration
# Launches Isaac Sim and provides setup instructions

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
echo "🚀 Starting Isaac Sim with demo..."
echo ""

# Create script in Isaac Sim directory
ISAAC_SCRIPT="$ISAAC_SIM_PATH/cookbot_demo.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Choose demo script based on argument
DEMO_TYPE="${1:-vla}"
case "$DEMO_TYPE" in
    "vla")
        echo "🧠 Running Isaac Sim + VLA integration demo..."
        cp "$SCRIPT_DIR/run_isaac_vla_demo.py" "$ISAAC_SCRIPT"
        ;;
    "standalone")
        echo "� Running standalone OpenVLA test (no Isaac Sim)..."
        cd "$SCRIPT_DIR"
        python demo.py
        exit 0
        ;;
    *)
        echo "❌ Unknown demo type: $DEMO_TYPE"
        echo "Usage: $0 [vla|standalone]"
        exit 1
        ;;
esac

# Use Isaac Sim's Python environment instead of --exec
cd "$ISAAC_SIM_PATH"
echo "🔧 Using Isaac Sim's Python environment..."
./python.sh "$ISAAC_SCRIPT"

# Cleanup
rm -f "$ISAAC_SCRIPT"

echo ""
echo "✅ Demo completed!"
echo ""
echo "💡 Usage examples:"
echo "  ./scripts/start_isaac_sim.sh vla         # Isaac Sim + VLA integration demo"
echo "  ./scripts/start_isaac_sim.sh standalone  # Test OpenVLA without Isaac Sim"