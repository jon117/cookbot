# Cookbot Scripts

This directory contains the main scripts for running the cookbot system.

## Main Scripts

### 🎮 `demo.py`
**The main entry point for testing the system**
```bash
python scripts/demo.py
```
- Tests the complete pipeline without requiring Isaac Sim
- Validates OpenVLA integration
- Shows expected robot actions
- **Run this first** to verify your setup

### 🚀 `start_isaac_sim.sh`
**Launches Isaac Sim with cookbot setup**
```bash
./scripts/start_isaac_sim.sh
```
- Checks OpenVLA server connection
- Starts Isaac Sim
- Provides instructions for running the demo

### 🏃 `run_isaac_vla_demo.py`
**Full Isaac Sim + OpenVLA demo (run inside Isaac Sim)**
```python
# In Isaac Sim Python console:
exec(open('scripts/run_isaac_vla_demo.py').read())
```
- Creates kitchen scene with robot and objects
- Connects to OpenVLA for action prediction
- Executes robot manipulation tasks

## Quick Start

1. **Test the system**:
   ```bash
   python scripts/demo.py
   ```

2. **Run with Isaac Sim**:
   ```bash
   ./scripts/start_isaac_sim.sh
   # Then in Isaac Sim console:
   exec(open('scripts/run_isaac_vla_demo.py').read())
   ```

## Requirements

- Isaac Sim installed at `~/isaacsim/`
- OpenVLA server running at `http://0.0.0.0:8000/act`
- Python packages: `numpy`, `requests`, `json-numpy`, `Pillow`

## Directory Structure

```
scripts/
├── demo.py                 # Main demo (no Isaac Sim required)
├── start_isaac_sim.sh      # Isaac Sim launcher
├── run_isaac_vla_demo.py   # Full Isaac Sim demo
├── deployment/             # Deployment scripts
├── development/            # Development utilities
├── setup/                  # Installation scripts
└── utilities/              # Helper scripts
```

## Workflow

The typical workflow is:

1. **`demo.py`** - Test that OpenVLA integration works
2. **`start_isaac_sim.sh`** - Launch Isaac Sim
3. **`run_isaac_vla_demo.py`** - Run the full demo in Isaac Sim

This gives you a complete pipeline from simulation to robot control with OpenVLA! 🤖