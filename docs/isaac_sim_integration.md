# Isaac Sim + OpenVLA Integration

This directory contains the complete Isaac Sim + OpenVLA integration for the cookbot project.

## Overview

The integration connects Isaac Sim's 3D physics simulation with OpenVLA for vision-language-action predictions, enabling the robot to understand natural language instructions and perform manipulation tasks in a simulated kitchen environment.

## Architecture

```
Isaac Sim Kitchen Scene
        ↓ (RGB-D Images)
    VLA Camera System
        ↓ (256x256 RGB)
    OpenVLA Model
        ↓ (Action Predictions)
    Robot Control Layer
        ↓ (Joint Commands)
    Isaac Sim Robot
```

## Quick Start

1. **Start OpenVLA Server** (if not already running):
   ```bash
   # Make sure your OpenVLA server is running at http://0.0.0.0:8000/act
   ```

2. **Test the Integration**:
   ```bash
   python scripts/test_full_integration.py
   ```

3. **Launch Isaac Sim**:
   ```bash
   ./scripts/start_isaac_sim.sh
   ```

4. **In Isaac Sim Python Console**:
   ```python
   # Load the kitchen scene
   exec(open('src/simulation/kitchen_scene.py').read())
   
   # Start the VLA integration
   exec(open('src/simulation/isaac_vla_bridge.py').read())
   ```

## Components

### Kitchen Scene (`src/simulation/kitchen_scene.py`)
- Creates a kitchen environment with table, carrot, slicer, and robot
- Handles physics simulation and object interactions
- Provides scene reset and object manipulation capabilities

### VLA Camera (`src/simulation/vla_camera.py`)
- Captures RGB-D data from Isaac Sim cameras
- Preprocesses images to 256x256 format for OpenVLA
- Provides camera calibration and intrinsics

### Isaac VLA Bridge (`src/simulation/isaac_vla_bridge.py`)
- Main coordinator between Isaac Sim and OpenVLA
- Handles task loop: capture → predict → execute
- Manages error handling and fallback behaviors

### OpenVLA Model (`src/movement_layer/models/openvla_model.py`)
- HTTP client for OpenVLA API at `/act` endpoint
- Handles numpy array serialization with json-numpy
- Provides fallback actions when OpenVLA returns errors
- Preprocesses images to 256x256 RGB format

### Robot Control (`src/simulation/robot_control.py`)
- Translates VLA predictions to robot joint commands
- Handles inverse kinematics and motion planning
- Provides gripper control and safety constraints

## Configuration

### VLA Configuration (`config/models/vla_config.yaml`)
```yaml
model:
  name: "openvla-7b"
  endpoint: "http://0.0.0.0:8000/act"
  image_size: [256, 256]
  timeout: 30
  max_retries: 2
```

### Camera Configuration
- **Format**: RGB (256x256x3, uint8)
- **Frequency**: 10 Hz
- **Field of View**: 60 degrees
- **Position**: Above table looking down

## API Format

### OpenVLA Request
```python
{
    "image": np.ndarray,  # (256, 256, 3) uint8
    "instruction": str    # Natural language command
}
```

### OpenVLA Response (Expected)
```python
{
    "action": [x, y, z, rx, ry, rz, gripper_width],
    "confidence": float
}
```

### Fallback Action (When OpenVLA Returns "error")
```python
ActionPrediction(
    position={"x": 0.3, "y": 0.1, "z": 0.15},
    orientation={"rx": 0.0, "ry": 0.0, "rz": 0.0},
    gripper_width=0.05,
    confidence=0.5,
    metadata={"source": "fallback", "reason": "openvla_error"}
)
```

## Dependencies

- **Isaac Sim**: Installed at `~/isaacsim`
- **OpenVLA Server**: Running at `http://0.0.0.0:8000/act`
- **Python Packages**:
  - `numpy`
  - `requests`
  - `json-numpy`
  - `Pillow`

## Testing

### Integration Test
```bash
python scripts/test_full_integration.py
```

### VLA Model Test
```bash
python scripts/test_vla_integration.py
```

### OpenVLA API Test
```bash
python scripts/test_openvla_exact.py
```

## Troubleshooting

### OpenVLA Returns "error"
- **Cause**: Model not properly loaded or incorrect input format
- **Solution**: Check OpenVLA server logs, verify image format
- **Fallback**: System uses predefined safe action

### Isaac Sim Connection Issues
- **Cause**: Isaac Sim not running or Python path issues
- **Solution**: Ensure Isaac Sim is started, check import paths

### Camera Data Issues
- **Cause**: Camera not properly configured in Isaac Sim
- **Solution**: Verify camera position and rendering settings

## Next Steps

1. **Real Robot Integration**: Connect to physical robot hardware
2. **Advanced Tasks**: Implement multi-step cooking procedures
3. **Error Recovery**: Improve failure detection and recovery
4. **Performance**: Optimize VLA inference speed
5. **Safety**: Add collision detection and safety constraints

## Status

✅ Isaac Sim kitchen scene created  
✅ VLA camera system implemented  
✅ OpenVLA API integration working  
✅ Fallback behavior for OpenVLA errors  
✅ Complete pipeline testing  
🔄 Ready for real Isaac Sim testing  

The system is ready for Isaac Sim + OpenVLA integration testing!