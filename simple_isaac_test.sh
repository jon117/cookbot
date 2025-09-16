# scripts/simple_isaac_test.sh

#!/bin/bash
echo "🤖 Simple Isaac Sim + VLA Test"
echo "==============================="

ISAAC_SIM_PATH="$HOME/isaacsim"

# Check Isaac Sim
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "❌ Isaac Sim not found at $ISAAC_SIM_PATH"
    exit 1
fi

# Check VLA server
echo "🔍 Checking VLA server..."
if curl -s http://localhost:8000/act > /dev/null 2>&1; then
    echo "✅ VLA server is running"
else
    echo "⚠️ VLA server not detected at localhost:8000/act"
    echo "   Please start your OpenVLA server first"
fi

# Copy script to Isaac Sim directory
ISAAC_SCRIPT="$ISAAC_SIM_PATH/simple_vla_test.py"
cp "$(dirname "${BASH_SOURCE[0]}")/simple_isaac_vla_test.py" "$ISAAC_SCRIPT"

echo "🚀 Starting Isaac Sim with simple VLA test..."
cd "$ISAAC_SIM_PATH"
./python.sh "$ISAAC_SCRIPT"

# Cleanup
rm -f "$ISAAC_SCRIPT"
echo "✅ Test completed!"