# Isaac Sim + OpenVLA Integration

This directory contains the integration between NVIDIA Isaac Sim and OpenVLA for robotic manipulation tasks.

## 🎯 Overview

The integration provides a complete pipeline:
1. **Scene Setup**: Kitchen environment with robot, table, carrot, and appliances
2. **Camera System**: RGB-D cameras for scene observation
3. **VLA Integration**: OpenVLA model at `0.0.0.0:8000` for action prediction
4. **Robot Control**: Simulated robot arm execution based on VLA predictions

## 🚀 Quick Start

### Prerequisites

1. **OpenVLA Model**: Ensure your OpenVLA model is running at `0.0.0.0:8000`
2. **Isaac Sim** (optional): For full 3D simulation
3. **Python Dependencies**: Install from `requirements/simulation.txt`

### Running the Demo

```bash
# Quick component test (no Isaac Sim required)
./scripts/run_demo.sh --quick-test

# Full Isaac Sim + VLA demo
./scripts/run_demo.sh
```

### Manual Execution

```bash
# Set up environment
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export VLA_API_KEY="demo-key"

# Run quick test
python scripts/run_isaac_vla_demo.py --quick-test

# Run full demo
python scripts/run_isaac_vla_demo.py
```

## 📁 File Structure

```
src/simulation/
├── kitchen_scene.py      # Isaac Sim scene setup
├── vla_camera.py        # Camera system for VLA data
├── robot_control.py     # Robot arm control interface
└── isaac_vla_bridge.py  # Main integration bridge

scripts/
├── run_isaac_vla_demo.py  # Demo script
└── run_demo.sh           # Convenience launcher
```

## 🧩 Components

### KitchenScene (`kitchen_scene.py`)
- Creates Isaac Sim environment with table, carrot, slicer
- Manages scene objects and physics
- Provides reset and step functionality

### VLACamera (`vla_camera.py`) 
- RGB-D camera setup for VLA input
- Multiple viewpoints (overhead, side, front)
- Data formatting for VLA model

### RobotController (`robot_control.py`)
- Simulated 6-DOF robot arm
- Inverse kinematics and motion planning
- Grasp execution based on VLA predictions

### IsaacVLABridge (`isaac_vla_bridge.py`)
- Main integration coordinator
- Scene state capture and VLA communication
- Action execution and result tracking

## 🔧 Configuration

### VLA Model Configuration

Edit `config/models/vla_config.yaml`:

```yaml
vla_models:
  primary_model:
    name: "openvla-7b"
    endpoint: "http://0.0.0.0:8000/v1/completions"
    # ... other settings
```

### Camera Settings

```python
# In vla_camera.py
image_width = 640
image_height = 480
focal_length = 600.0
```

### Robot Parameters

```python
# In robot_control.py
joint_limits = {
    "joint1": [-165, 165],  # degrees
    "joint2": [-135, 135],
    # ... other joints
}
```

## 🧪 Testing

### Component Tests

```bash
# Test VLA configuration
python -c "from src.movement_layer.grasp_planner import GraspPlanner; p = GraspPlanner()"

# Test data types
python -c "from src.common.data_types import CameraObservation; print('✅ Data types OK')"

# Test robot control
cd src/simulation && python robot_control.py
```

### Integration Test

```bash
# Run quick test without Isaac Sim
python scripts/run_isaac_vla_demo.py --quick-test
```

## 📊 Demo Scenarios

The demo includes several test scenarios:

1. **Carrot Pickup**: Basic grasp and lift of carrot object
2. **Carrot to Slicer**: Complete pick-and-place to appliance
3. **Object Inspection**: VLA-guided object examination

## 🔍 Monitoring and Debugging

### Logs

The system provides detailed logging:

```
📸 Capturing scene state...
✅ Scene captured: RGB (480, 640, 3)
🤖 Getting VLA prediction...
✅ VLA prediction: pos=(0.300, 0.100, 0.150), conf=0.750
🦾 Executing robot action...
✅ Robot action executed successfully!
```

### Statistics

Demo provides comprehensive statistics:
- VLA prediction confidence
- Robot execution success rate
- Total execution time
- Scene analysis results

## 🚨 Troubleshooting

### Common Issues

**OpenVLA Connection Error**
```
❌ VLA action prediction failed: Connection refused
```
→ Ensure OpenVLA model is running at `0.0.0.0:8000`

**Isaac Sim Import Error**
```
ImportError: No module named 'omni.isaac.kit'
```
→ Run with `--quick-test` flag or install Isaac Sim

**Scene Capture Failed**
```
❌ Failed to capture scene
```
→ Check camera setup and Isaac Sim initialization

### Performance Tuning

- **VLA Response Time**: Adjust `timeout` in VLA config
- **Robot Speed**: Modify `speed` parameter in motion commands
- **Camera Resolution**: Change `image_width/height` for faster processing

## 🛠️ Extending the System

### Adding New Objects

1. Add object to `kitchen_scene.py`:
```python
new_object = DynamicCuboid(
    prim_path="/World/NewObject",
    name="new_object",
    # ... parameters
)
```

2. Update VLA object configs in `vla_config.yaml`

### Adding New Actions

1. Extend `TaskAction` enum in `data_types.py`
2. Add action handling in `robot_control.py`
3. Update VLA instruction processing

### Multiple Robots

1. Create additional robot instances in `robot_control.py`
2. Coordinate actions in `isaac_vla_bridge.py`
3. Update scene layout for multiple arms

## 📚 References

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)
- [OpenVLA Model](https://github.com/openvla/openvla)
- [CookBot Architecture](../docs/architecture/overview.md)

## 🤝 Contributing

To contribute to the Isaac Sim + VLA integration:

1. Follow the existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Test with both quick-test and full demo modes