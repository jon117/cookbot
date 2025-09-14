# Kitchen Robot Project 🤖👨‍🍳

A Jetsons-style kitchen automation system using foundation models for high-level reasoning and traditional robotics for precise control.

## 🎯 Project Vision

Building an integrated kitchen ecosystem that can autonomously prepare meals through:
- **Planning Layer**: LLM-powered task decomposition and recipe orchestration
- **Movement Layer**: Vision-Language-Action models for precise manipulation
- **Control Layer**: Traditional robotics for reliable motion execution

## 🚀 Quick Start

### Option 1: Standalone OpenVLA Test
```bash
# Test OpenVLA pipeline without Isaac Sim
./scripts/start_isaac_sim.sh standalone
```

### Option 2: Full Isaac Sim + VLA Demo
```bash
# Run complete Isaac Sim + VLA integration
./scripts/start_isaac_sim.sh vla
```

### Prerequisites
- OpenVLA server running at `http://0.0.0.0:8000/act`
- Isaac Sim installed at `~/isaacsim/` (for full demo)
- Python packages: `numpy`, `requests`, `json-numpy`

## 📋 Phase Roadmap

- **Phase 1**: Basic pickup and place (carrot → slicer) ✅ Current Focus
- **Phase 2**: Complete "steamed carrots" recipe execution
- **Phase 3**: Multiple recipes with appliance coordination
- **Phase 4**: Real hardware deployment

## 🏗️ Architecture

```
Planning Layer (LLM)    → Task decomposition & orchestration
     ↓
Movement Layer (VLA)    → Vision-grounded manipulation planning  
     ↓
Control Layer (Traditional) → Precise motion execution
```

## � Current Status

**Phase 1: Isaac Sim + OpenVLA Integration** ✅ **COMPLETE**

- ✅ Isaac Sim kitchen scene with robot, table, carrot, and slicer
- ✅ OpenVLA integration with proper numpy array handling
- ✅ VLA camera system (256x256 RGB preprocessing)
- ✅ Complete pipeline: Image → VLA → Robot Actions
- ✅ Fallback behavior when OpenVLA returns errors
- ✅ Clean demo and testing scripts

**Next: Phase 2 - Complete Task Execution**
- 🔄 Multi-step task execution
- 🔄 Recipe planning integration
- 🔄 Real robot hardware deployment

## ✅ SOLVED - Isaac Sim 5.0 Integration

**SOLUTION FOUND**: Use Isaac Sim's own Python environment with `python.sh` instead of `--exec`

**Root Cause**: Isaac Sim 5.0 requires its own Python environment and module loading order
- System Python with `--exec` flag causes segmentation faults (exit code 139)
- Isaac Sim's `python.sh` provides the correct environment and prevents crashes

**Working approach:**
```bash
# Copy script to Isaac Sim directory
ISAAC_SCRIPT="$ISAAC_SIM_PATH/cookbot_demo.py"
cp "$SCRIPT_DIR/test_minimal_isaac.py" "$ISAAC_SCRIPT"

# Use Isaac Sim's Python environment
cd "$ISAAC_SIM_PATH"
./python.sh "$ISAAC_SCRIPT"
```

**Status**: ✅ **RESOLVED** - Ready for Phase 2 development

## �📚 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Isaac Sim Integration](docs/isaac_sim_integration.md)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## 🧪 Key Technologies

- **AI Models**: Qwen (LLM), OpenVLA (Vision-Language-Action)
- **Simulation**: NVIDIA Isaac Sim
- **Robotics**: ROS2, MoveIt2
- **Hardware**: FR5 Robot Arms
- **Infrastructure**: Docker, OpenAI-compatible APIs

## 📊 Success Metrics

- ✅ Phase 1: Isaac Sim + OpenVLA integration working
- 🔄 Phase 2: Robot picks up carrot and places in slicer  
- 🔄 Phase 3: Complete steamed carrots preparation
- 🔄 Phase 4: 5+ different recipes automated
- 🔄 Phase 5: Real kitchen deployment

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

This project is open source.
