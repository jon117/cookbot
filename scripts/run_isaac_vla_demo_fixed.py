#!/usr/bin/env python3
"""
Isaac Sim + OpenVLA Integration Demo (Fixed Import Order)
Runs the full integration test directly in Isaac Sim avoiding import conflicts.
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ["EXP_PATH"] = os.path.expanduser("~/isaacsim")
os.environ["CARB_APP_PATH"] = os.path.expanduser("~/isaacsim") 
os.environ["ISAAC_PATH"] = os.path.expanduser("~/isaacsim")

# Add paths
isaac_sim_path = os.path.expanduser("~/isaacsim")
sys.path.insert(0, isaac_sim_path)

# Add our project to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# CRITICAL: Import SimulationApp FIRST before any other packages
from omni.isaac.kit import SimulationApp

# Start Isaac Sim
config = {
    "width": 1280,
    "height": 720,
    "sync_loads": True,
    "headless": False,
    "renderer": "RayTracedLighting",
}

simulation_app = SimulationApp(config)

# ONLY NOW import other modules (after SimulationApp is started)
import numpy as np
import asyncio
import time
import requests
import json_numpy  # Our custom numpy JSON serializer
import logging
from typing import Dict, Any, Optional

# Isaac Sim modules
import omni
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.transformations import tf_matrices_from_poses
import omni.isaac.core.utils.nucleus as nucleus_utils
from omni.isaac.sensor import Camera
from omni.kit.viewport.utility import get_active_viewport_window
from pxr import Gf, UsdGeom

logger = logging.getLogger(__name__)


class IsaacSimKitchenScene:
    """
    Kitchen scene setup integrated directly into Isaac Sim script.
    """
    
    def __init__(self):
        self.world = None
        self.cameras = {}
        self.robot = None
        self.objects = []
        
    def setup(self):
        """Initialize the kitchen scene"""
        print("Setting up Isaac Sim kitchen scene...")
        
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        
        # Add basic lighting
        stage = omni.usd.get_context().get_stage()
        
        # Create a sphere light
        prim_utils.create_prim(
            "/World/SphereLight",
            "SphereLight",
            attributes={
                "radius": 1.5,
                "intensity": 30000,
                "xformOp:translate": (0, 0, 3)
            }
        )
        
        # Create ground plane
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.05]),
            scale=np.array([10.0, 10.0, 0.1]),
            color=np.array([0.5, 0.5, 0.5])
        )
        self.world.scene.add(ground)
        
        # Create kitchen table
        table = FixedCuboid(
            prim_path="/World/Table",
            name="kitchen_table",
            position=np.array([1.0, 0.0, 0.4]),
            scale=np.array([1.2, 0.8, 0.05]),
            color=np.array([0.6, 0.4, 0.2])
        )
        self.world.scene.add(table)
        
        # Create some objects on the table
        carrot = DynamicCuboid(
            prim_path="/World/Carrot",
            name="carrot",
            position=np.array([0.8, 0.2, 0.5]),
            scale=np.array([0.02, 0.15, 0.02]),
            color=np.array([1.0, 0.5, 0.0])
        )
        self.world.scene.add(carrot)
        self.objects.append(carrot)
        
        # Create camera positioned to see the scene
        camera_prim = prim_utils.create_prim(
            "/World/Camera",
            "Camera",
            attributes={
                "xformOp:translate": (0.5, -1.5, 1.2),
                "xformOp:rotateXYZ": (15, 0, 0)
            }
        )
        
        # Setup camera for image capture
        self.setup_camera()
        
        print("Kitchen scene setup complete!")
        
    def setup_camera(self):
        """Setup camera for capturing images"""
        try:
            # Create Isaac Sim camera
            self.cameras["main"] = Camera(
                prim_path="/World/Camera",
                name="main_camera",
                position=np.array([0.5, -1.5, 1.2]),
                orientation=np.array([0.9659, 0.2588, 0, 0])  # Quaternion for slight downward angle
            )
            
            # Initialize camera
            self.cameras["main"].initialize()
            print("Camera setup complete!")
            
        except Exception as e:
            print(f"Camera setup failed: {e}")
            
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture RGB image from main camera"""
        try:
            if "main" not in self.cameras:
                print("Main camera not available")
                return None
                
            # Get camera data
            camera = self.cameras["main"]
            rgb_data = camera.get_rgba()
            
            if rgb_data is not None:
                # Convert RGBA to RGB
                rgb_image = rgb_data[:, :, :3]
                return (rgb_image * 255).astype(np.uint8)
            
            return None
            
        except Exception as e:
            print(f"Failed to capture image: {e}")
            return None
            
    def step_simulation(self):
        """Step the simulation forward"""
        if self.world:
            self.world.step(render=True)


class VLAInterface:
    """
    Interface to communicate with OpenVLA model
    """
    
    def __init__(self, server_url: str = "http://0.0.0.0:8000"):
        self.server_url = server_url
        self.endpoint = f"{server_url}/act"
        
    def predict_action(self, image: np.ndarray, instruction: str) -> Dict[str, Any]:
        """
        Send image and instruction to VLA model for action prediction
        """
        try:
            # Prepare request data
            data = {
                "image": image,
                "instruction": instruction,
                "unnorm_key": "bridge_orig"
            }
            
            # Send request with json_numpy serialization
            response = requests.post(
                self.endpoint,
                data=json_numpy.dumps(data),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = json_numpy.loads(response.text)
                return {
                    "success": True,
                    "action": result.get("action", "unknown"),
                    "raw_response": result
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "action": "wait"  # Fallback action
                }
                
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {e}",
                "action": "wait"  # Fallback action
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {e}",
                "action": "wait"  # Fallback action
            }


async def run_integration_demo():
    """
    Main integration demo loop
    """
    print("Starting Isaac Sim + OpenVLA Integration Demo...")
    
    # Setup kitchen scene
    kitchen = IsaacSimKitchenScene()
    kitchen.setup()
    
    # Initialize VLA interface
    vla = VLAInterface()
    
    # Test instruction
    instruction = "pick up the orange carrot"
    
    print(f"Running demo with instruction: '{instruction}'")
    print("Demo will run for 30 seconds...")
    
    start_time = time.time()
    step_count = 0
    
    while time.time() - start_time < 30.0:  # Run for 30 seconds
        # Step simulation
        kitchen.step_simulation()
        
        # Capture image every 10 steps
        if step_count % 10 == 0:
            image = kitchen.capture_image()
            
            if image is not None:
                print(f"Step {step_count}: Captured image {image.shape}")
                
                # Get VLA prediction
                result = vla.predict_action(image, instruction)
                
                if result["success"]:
                    print(f"  VLA Action: {result['action']}")
                else:
                    print(f"  VLA Error: {result['error']}")
                    print(f"  Fallback Action: {result['action']}")
            else:
                print(f"Step {step_count}: Failed to capture image")
        
        step_count += 1
        
        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.1)
    
    print("Demo completed!")


def main():
    """Main entry point"""
    try:
        # Run the async demo
        asyncio.run(run_integration_demo())
        
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if simulation_app:
            simulation_app.close()


if __name__ == "__main__":
    main()