# Kitchen Robot Project 🤖👨‍🍳

A Jetsons-style kitchen automation system using foundation models for high-level reasoning and traditional robotics for precise control.

## 🎯 Project Vision

Building an integrated kitchen ecosystem that can autonomously prepare meals through:
- **Planning Layer**: LLM-powered task decomposition and recipe orchestration
- **Movement Layer**: Vision-Language-Action models for precise manipulation
- **Control Layer**: Traditional robotics for reliable motion execution

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/jon117/cookbot
cd kitchen-robot

# Set up development environment
make setup-dev

# Run simulation
make run-simulation

# Execute first task (carrot pickup)
make test-carrot-pickup
```

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

## 📚 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Development Setup](docs/setup/development-environment.md)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## 🧪 Key Technologies

- **AI Models**: Qwen (LLM), OpenVLA (Vision-Language-Action)
- **Simulation**: NVIDIA Isaac Sim
- **Robotics**: ROS2, MoveIt2
- **Hardware**: FR5 Robot Arms
- **Infrastructure**: Docker, OpenAI-compatible APIs

## 📊 Success Metrics

- Phase 1: Robot picks up carrot and places in slicer
- Phase 2: Complete steamed carrots preparation
- Phase 3: 5+ different recipes automated
- Phase 4: Real kitchen deployment

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

This project is open source.
