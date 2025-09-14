"""
Isaac Sim Kitchen Scene Setup
Creates a kitchen environment with robot arm, table, and objects for VLA testing.
"""

import numpy as np
from omni.isaac.kit import SimulationApp

# Start Isaac Sim with essential extensions
config = {
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,  # Set to True for headless mode
    "renderer": "RayTracedLighting",
}

simulation_app = SimulationApp(config)

import omni
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.transformations import tf_matrices_from_poses
import omni.isaac.core.utils.nucleus as nucleus_utils

class KitchenScene:
    """Kitchen scene setup with robot arm, table, and objects."""
    
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()
        
    def create_scene(self):
        """Create the complete kitchen scene."""
        print("Creating kitchen scene...")
        
        # Create table
        self._create_table()
        
        # Create carrot object
        self._create_carrot()
        
        # Create slicer appliance (placeholder)
        self._create_slicer()
        
        # Add lighting
        self._setup_lighting()
        
        # Create robot arm (placeholder - will use actual URDF later)
        self._create_robot_placeholder()
        
        print("Kitchen scene created successfully!")
        
    def _create_table(self):
        """Create kitchen table."""
        table = FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.5, 0.0, -0.05]),  # Table surface at z=0
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # No rotation
            size=np.array([1.2, 0.8, 0.1]),  # length, width, height
            color=np.array([0.8, 0.6, 0.4])  # Wood color
        )
        self.world.scene.add(table)
        
    def _create_carrot(self):
        """Create carrot object for manipulation."""
        # Create carrot as orange cylinder
        carrot = DynamicCuboid(
            prim_path="/World/Carrot",
            name="carrot",
            position=np.array([0.3, 0.1, 0.1]),  # On table surface
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.15, 0.03, 0.03]),  # Carrot-like dimensions
            color=np.array([1.0, 0.5, 0.0]),  # Orange color
            mass=0.1  # Light object
        )
        self.world.scene.add(carrot)
        
    def _create_slicer(self):
        """Create slicer appliance placeholder."""
        slicer_base = FixedCuboid(
            prim_path="/World/Slicer",
            name="slicer",
            position=np.array([0.6, 0.3, 0.1]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.25, 0.2, 0.2]),
            color=np.array([0.7, 0.7, 0.7])  # Metallic gray
        )
        self.world.scene.add(slicer_base)
        
        # Slicer input area
        slicer_input = FixedCuboid(
            prim_path="/World/SlicerInput",
            name="slicer_input",
            position=np.array([0.5, 0.3, 0.05]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.1, 0.15, 0.02]),
            color=np.array([0.9, 0.9, 0.9])  # Light gray input area
        )
        self.world.scene.add(slicer_input)
        
    def _create_robot_placeholder(self):
        """Create simple robot arm placeholder (will be replaced with actual URDF)."""
        # Robot base
        robot_base = FixedCuboid(
            prim_path="/World/RobotBase",
            name="robot_base",
            position=np.array([0.0, 0.0, 0.1]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.3, 0.3, 0.8])  # Blue robot color
        )
        self.world.scene.add(robot_base)
        
        # Robot arm segments (simplified)
        arm_segment1 = DynamicCuboid(
            prim_path="/World/RobotArm1",
            name="robot_arm1",
            position=np.array([0.15, 0.0, 0.25]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.3, 0.05, 0.05]),
            color=np.array([0.4, 0.4, 0.9]),
            mass=0.5
        )
        self.world.scene.add(arm_segment1)
        
        # End effector placeholder
        end_effector = DynamicCuboid(
            prim_path="/World/EndEffector",
            name="end_effector",
            position=np.array([0.3, 0.0, 0.25]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.array([0.05, 0.05, 0.08]),
            color=np.array([0.2, 0.8, 0.2]),  # Green gripper
            mass=0.1
        )
        self.world.scene.add(end_effector)
        
    def _setup_lighting(self):
        """Set up scene lighting."""
        # Create dome light for even illumination
        stage_utils.create_new_stage()
        prim_utils.create_prim(
            "/World/DomeLight",
            "DomeLight",
            attributes={
                "inputs:intensity": 1000,
                "inputs:texture:file": "",
            }
        )
        
    def reset(self):
        """Reset scene to initial state."""
        self.world.reset()
        
    def step(self):
        """Step simulation."""
        self.world.step(render=True)
        
    def close(self):
        """Close simulation."""
        simulation_app.close()

def main():
    """Main function to run the kitchen scene."""
    scene = KitchenScene()
    
    try:
        # Create the scene
        scene.create_scene()
        
        # Initialize world
        scene.world.reset()
        
        print("Kitchen scene is ready!")
        print("The scene includes:")
        print("- Table with carrot")
        print("- Slicer appliance")  
        print("- Robot arm placeholder")
        print("\nPress Enter to close...")
        
        # Keep simulation running
        input()
        
    finally:
        scene.close()

if __name__ == "__main__":
    main()