"""
Isaac Sim Kitchen Scene Setup
Creates a kitchen environment with robot arm, table, and objects for VLA testing.
"""

import os
import sys

# Set environment variables for Isaac Sim if not already set
def setup_isaac_env():
    """Set up Isaac Sim environment variables if not already configured."""
    isaac_sim_path = os.path.expanduser("~/isaacsim")
    
    if "EXP_PATH" not in os.environ:
        default_exp_path = os.path.join(isaac_sim_path, "apps")
        os.environ["EXP_PATH"] = default_exp_path
        print(f"EXP_PATH not set, defaulting to: {default_exp_path}")

    if "CARB_APP_PATH" not in os.environ:
        os.environ["CARB_APP_PATH"] = isaac_sim_path
        print(f"CARB_APP_PATH not set, defaulting to: {isaac_sim_path}")

    if "ISAAC_PATH" not in os.environ:
        os.environ["ISAAC_PATH"] = isaac_sim_path
        print(f"ISAAC_PATH not set, defaulting to: {isaac_sim_path}")

# Setup environment but don't create SimulationApp here
setup_isaac_env()

# Isaac Sim imports - these will be available when running in Isaac context
try:
    import numpy as np
    import omni
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
    from omni.isaac.core.prims import XFormPrimView
    from omni.isaac.core.utils.transformations import tf_matrices_from_poses
    import omni.isaac.core.utils.nucleus as nucleus_utils
except ImportError as e:
    print(f"Isaac Sim modules not available: {e}")
    # Define minimal fallback for development
    class World:
        def __init__(self, *args, **kwargs):
            pass
    
    class DynamicCuboid:
        def __init__(self, *args, **kwargs):
            pass
    
    class FixedCuboid:
        def __init__(self, *args, **kwargs):
            pass

class KitchenScene:
    """Kitchen scene setup with robot arm, table, and objects."""
    
    def __init__(self):
        try:
            self.world = World(stage_units_in_meters=1.0)
            # Try to get stage, handle fallback
            if hasattr(omni, 'usd'):
                self.stage = omni.usd.get_context().get_stage()
            else:
                print("Isaac Sim USD context not available - using fallback mode")
                self.stage = None
        except Exception as e:
            print(f"Isaac Sim World not available - using fallback mode: {e}")
            self.world = None
            self.stage = None
        
    def create_scene(self):
        """Create the complete kitchen scene."""
        print("Creating kitchen scene...")
        
        if self.world is None:
            print("Using fallback scene creation...")
            self._create_fallback_scene()
            return
        
        try:
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
            
        except Exception as e:
            print(f"Error creating scene, using fallback: {e}")
            self._create_fallback_scene()
    
    def _create_fallback_scene(self):
        """Create a fallback scene when Isaac Sim is not available."""
        print("Setting up fallback kitchen scene...")
        
        # Just store some basic scene information
        self.objects = {
            'table': {'position': [0.4, 0.0, 0.0], 'size': [0.8, 0.6, 0.02]},
            'carrot': {'position': [0.3, 0.1, 0.05], 'size': [0.02, 0.02, 0.08]},
            'slicer': {'position': [0.5, -0.2, 0.02], 'size': [0.15, 0.15, 0.1]}
        }
        
        print("Fallback scene created with simulated objects:")
        for obj_name, obj_info in self.objects.items():
            print(f"  - {obj_name}: {obj_info['position']}")
    
    def _create_table(self):
        """Create kitchen table."""
        table = FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.5, 0.0, -0.05]),  # Table surface at z=0
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # No rotation
            scale=np.array([1.2, 0.8, 0.1]),  # length, width, height
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
            scale=np.array([0.15, 0.03, 0.03]),  # Carrot-like dimensions
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
            scale=np.array([0.25, 0.2, 0.2]),
            color=np.array([0.7, 0.7, 0.7])  # Metallic gray
        )
        self.world.scene.add(slicer_base)
        
        # Slicer input area
        slicer_input = FixedCuboid(
            prim_path="/World/SlicerInput",
            name="slicer_input",
            position=np.array([0.5, 0.3, 0.05]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            scale=np.array([0.1, 0.15, 0.02]),
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
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0.3, 0.3, 0.8])  # Blue robot color
        )
        self.world.scene.add(robot_base)
        
        # Robot arm segments (simplified)
        arm_segment1 = DynamicCuboid(
            prim_path="/World/RobotArm1",
            name="robot_arm1",
            position=np.array([0.15, 0.0, 0.25]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            scale=np.array([0.3, 0.05, 0.05]),
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
            scale=np.array([0.05, 0.05, 0.08]),
            color=np.array([0.2, 0.8, 0.2]),  # Green gripper
            mass=0.1
        )
        self.world.scene.add(end_effector)
        
    def _setup_lighting(self):
        """Set up scene lighting."""
        # Create dome light for even illumination (don't create new stage!)
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
        try:
            if self.world:
                self.world.step(render=True)
            else:
                # Fallback: just simulate time passing
                import time
                time.sleep(0.016)  # ~60 FPS
        except Exception as e:
            print(f"Error stepping simulation: {e}")
            # Fallback: just simulate time passing
            import time
            time.sleep(0.016)
        
    def close(self):
        """Close simulation."""
        try:
            # Don't close simulation_app here - it should be managed externally
            print("Kitchen scene cleanup completed")
        except Exception as e:
            print(f"Error closing scene: {e}")

def main():
    """Main function to run the kitchen scene."""
    # Import here to avoid circular imports
    try:
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
        
    except ImportError:
        print("Isaac Sim not available for standalone testing")
        return
    
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
        try:
            simulation_app.close()
        except:
            pass
        scene.close()

if __name__ == "__main__":
    main()