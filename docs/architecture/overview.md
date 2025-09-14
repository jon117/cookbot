# Kitchen Robot Architecture Overview

## System Design Philosophy

The Kitchen Robot employs a **three-layer control architecture** that separates concerns while enabling seamless integration:

1. **Planning Layer**: High-level reasoning using Large Language Models (LLMs)
2. **Movement Layer**: Vision-grounded manipulation using Vision-Language-Action models (VLAs) 
3. **Control Layer**: Precise motion execution using traditional robotics

## Architecture Diagram

```
┌──────────────────────────────────────────────┐
│ PLANNING LAYER (LLM)                         │
│ ┌────────────────────┐ ┌──────────────────┐  │
│ │ Recipe Planner     │ │ Task Orchestrator│  │
│ │ - Decomposition    │ │ - Appliance Coord│  │
│ │ - Sequencing       │ │ - Error Recovery │  │
│ └────────────────────┘ └──────────────────┘  │
└──────────────────────────────────────────────┘
                        │
┌──────────────────────────────────────────────┐
│ MOVEMENT LAYER (VLA)                         │
│ ┌────────────────────┐ ┌───────────────────┐ │
│ │ Vision Processor   │ │ Grasp Planner     │ │
│ │ - Object Detection │ │ - Pose Generation │ │
│ │ - Scene Analysis   │ │ - Spatial Reason  │ │
│ └────────────────────┘ └───────────────────┘ │
└──────────────────────────────────────────────┘
                        │
┌──────────────────────────────────────────────┐
│ CONTROL LAYER (Traditional Robotics)         │
│ ┌────────────────────┐ ┌───────────────────┐ │
│ │ Motion Planner     │ │ Safety Monitor    │ │
│ │ -Inverse Kinematics│ │- Collision Avoid  │ │
│ │ -Trajectory Plan   │ │ - Joint Limits    │ │
│ └────────────────────┘ └───────────────────┘ │
└──────────────────────────────────────────────┘
```

## Key Principles

### Model-Agnostic Design
- **Standardized APIs**: All AI models accessible via OpenAI-compatible endpoints
- **Hot-swappable Models**: Easy to switch between Qwen, Claude, OpenVLA, etc.
- **Interface Abstraction**: Models hidden behind consistent interfaces

### Modular Architecture  
- **Independent Layers**: Each layer can be developed and tested separately
- **Loose Coupling**: Layers communicate through well-defined APIs
- **Extensible**: Easy to add new appliances, recipes, or capabilities

### Simulation-First Development
- **Isaac Sim Integration**: Full physics simulation for safe development
- **Realistic Testing**: Photorealistic rendering and accurate dynamics
- **Seamless Transition**: Same code runs in simulation and on hardware

## Data Flow

1. **Input**: Natural language recipe instruction ("prepare steamed carrots")
2. **Planning**: LLM decomposes into structured tasks
3. **Perception**: Cameras capture current kitchen state
4. **Movement Planning**: VLA generates manipulation poses  
5. **Execution**: Traditional robotics executes precise motions
6. **Feedback**: Results feed back for error recovery and replanning

## Technology Stack Summary

| Layer | Primary Tech | Purpose |
|-------|-------------|---------|
| Planning | LLM | Task decomposition & orchestration |
| Movement | OpenVLA | Vision-grounded manipulation planning |
| Control | MoveIt2 + ROS2 | Motion planning & execution |
| Simulation | Isaac Sim | Development & testing environment |
| Infrastructure | Docker + APIs | Deployment & model serving |
