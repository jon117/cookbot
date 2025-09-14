#!/usr/bin/env python3
"""
Isaac Sim Kitchen Scene Demo (Minimal Version)
Minimal Isaac Sim demo without external package conflicts.
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ["EXP_PATH"] = os.path.expanduser("~/isaacsim")
os.environ["CARB_APP_PATH"] = os.path.expanduser("~/isaacsim") 
os.environ["ISAAC_PATH"] = os.path.expanduser("~/isaacsim")

# Add Isaac Sim path
isaac_sim_path = os.path.expanduser("~/isaacsim")
sys.path.insert(0, isaac_sim_path)

# CRITICAL: Import ONLY Isaac Sim modules first
from omni.isaac.kit import SimulationApp

# Start Isaac Sim with minimal config
config = {
    "width": 1280,
    "height": 720,
    "headless": False,
}

print("Starting Isaac Sim...")
simulation_app = SimulationApp(config)

# ONLY NOW import Isaac Sim specific modules
print("Importing Isaac Sim modules...")
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid

# Start minimal scene after Isaac Sim is fully loaded
print("Creating Isaac Sim world...")

def create_simple_scene():
    """Create a very simple Isaac Sim scene"""
    try:
        # Create world
        world = World(stage_units_in_meters=1.0)
        
        # Create ground plane
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=[0.0, 0.0, -0.05],
            scale=[10.0, 10.0, 0.1],
            color=[0.5, 0.5, 0.5]
        )
        world.scene.add(ground)
        
        # Create a simple object
        cube = DynamicCuboid(
            prim_path="/World/Cube",
            name="test_cube",
            position=[0.0, 0.0, 1.0],
            scale=[0.2, 0.2, 0.2],
            color=[1.0, 0.0, 0.0]
        )
        world.scene.add(cube)
        
        print("✓ Simple scene created successfully!")
        
        # Run simulation for a few steps
        print("Running simulation...")
        for i in range(100):
            world.step(render=True)
            if i % 20 == 0:
                print(f"  Step {i}")
        
        print("✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during scene creation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function"""
    try:
        print("🚀 Isaac Sim Kitchen Demo Starting...")
        create_simple_scene()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔄 Closing Isaac Sim...")
        if simulation_app:
            simulation_app.close()
        print("✅ Demo finished")


if __name__ == "__main__":
    main()