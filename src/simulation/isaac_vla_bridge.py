"""
Isaac Sim - OpenVLA Bridge
Connects Isaac Sim simulation with OpenVLA model for real-time action prediction.
"""

import asyncio
import time
import numpy as np
import logging
from typing import Dict, Any, Optional

# Import our components
from .kitchen_scene import KitchenScene
from .vla_camera import VLACamera, setup_scene_cameras
from ..movement_layer.grasp_planner import GraspPlanner
from ..common.data_types import CameraObservation, GraspPose

logger = logging.getLogger(__name__)


class IsaacVLABridge:
    """Bridge between Isaac Sim and OpenVLA model."""
    
    def __init__(self, vla_config_path: str = "config/models/vla_config.yaml"):
        """
        Initialize Isaac Sim - VLA bridge.
        
        Args:
            vla_config_path: Path to VLA configuration file
        """
        self.vla_config_path = vla_config_path
        
        # Initialize components
        self.scene = None
        self.camera_system = None
        self.primary_camera = None
        self.grasp_planner = None
        
        # State tracking
        self.current_task = None
        self.action_history = []
        self.scene_objects = {}
        
        self.setup_components()
        
    def setup_components(self):
        """Initialize all system components."""
        logger.info("Setting up Isaac Sim - VLA bridge components...")
        
        try:
            # Initialize kitchen scene
            self.scene = KitchenScene()
            
            # Set up camera system
            self.camera_system, self.primary_camera = setup_scene_cameras()
            
            # Initialize VLA grasp planner
            self.grasp_planner = GraspPlanner(self.vla_config_path)
            
            logger.info("✓ Bridge components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize bridge components: {e}")
            raise
    
    def start_simulation(self):
        """Start the Isaac Sim simulation."""
        try:
            logger.info("Starting Isaac Sim simulation...")
            
            # Create scene
            self.scene.create_scene()
            
            # Initialize simulation
            self.scene.world.reset()
            
            logger.info("✓ Isaac Sim simulation started")
            
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            raise
    
    def capture_scene_state(self) -> Optional[CameraObservation]:
        """
        Capture current scene state for VLA processing.
        
        Returns:
            CameraObservation with RGB-D data
        """
        try:
            # Step simulation to update scene
            self.scene.step()
            
            # Capture camera data
            camera_data = self.primary_camera.capture_for_vla()
            
            if camera_data is None:
                logger.warning("Failed to capture camera data")
                return None
                
            logger.debug(f"Captured scene: RGB {camera_data.rgb.shape}, "
                        f"Depth {camera_data.depth.shape if camera_data.depth is not None else 'None'}")
            
            return camera_data
            
        except Exception as e:
            logger.error(f"Failed to capture scene state: {e}")
            return None
    
    async def predict_action(self, 
                           camera_data: CameraObservation, 
                           instruction: str,
                           target_object: str = "carrot") -> Optional[GraspPose]:
        """
        Use VLA model to predict manipulation action (async version).
        
        Args:
            camera_data: Current scene observation
            instruction: Task instruction
            target_object: Object to manipulate
            
        Returns:
            Predicted grasp pose
        """
        return self.predict_action_sync(camera_data, instruction, target_object)
    
    def predict_action_sync(self, 
                           camera_data: CameraObservation, 
                           instruction: str,
                           target_object: str = "carrot") -> Optional[GraspPose]:
        """
        Use VLA model to predict manipulation action (synchronous version).
        
        Args:
            camera_data: Current scene observation
            instruction: Task instruction
            target_object: Object to manipulate
            
        Returns:
            Predicted grasp pose
        """
        try:
            logger.info(f"Predicting action for: {instruction}")
            
            # Use grasp planner with VLA model
            grasp_pose = self.grasp_planner.plan_grasp(
                camera_data=camera_data,
                target_object=target_object,
                context=instruction
            )
            
            if grasp_pose:
                logger.info(f"VLA predicted grasp: pos=({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f}), "
                           f"conf={grasp_pose.confidence:.3f}")
                
                # Record action in history
                self.action_history.append({
                    'timestamp': time.time(),
                    'instruction': instruction,
                    'target_object': target_object,
                    'predicted_pose': grasp_pose,
                    'confidence': grasp_pose.confidence
                })
                
            return grasp_pose
            
        except Exception as e:
            logger.error(f"VLA action prediction failed: {e}")
            return None
    
    def visualize_prediction(self, grasp_pose: GraspPose):
        """
        Visualize VLA prediction in Isaac Sim.
        
        Args:
            grasp_pose: Predicted grasp pose to visualize
        """
        try:
            # In a full implementation, this would create visual markers
            # For now, just log the prediction
            logger.info(f"Visualizing grasp prediction:")
            logger.info(f"  Position: ({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f})")
            logger.info(f"  Orientation: ({grasp_pose.rx:.3f}, {grasp_pose.ry:.3f}, {grasp_pose.rz:.3f})")
            logger.info(f"  Gripper width: {grasp_pose.gripper_width:.3f}")
            logger.info(f"  Confidence: {grasp_pose.confidence:.3f}")
            
            # TODO: Add visual markers in Isaac Sim scene
            # - Coordinate frame at grasp position
            # - Gripper width visualization
            # - Approach vector arrow
            
        except Exception as e:
            logger.error(f"Failed to visualize prediction: {e}")
    
    async def run_task_loop(self, instruction: str, target_object: str = "carrot"):
        """
        Run main task execution loop.
        
        Args:
            instruction: Task instruction (e.g., "pick up the carrot")
            target_object: Object to manipulate
        """
        logger.info(f"Starting task loop: {instruction}")
        
        try:
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                logger.info(f"Task iteration {iteration + 1}/{max_iterations}")
                
                # Capture current scene
                camera_data = self.capture_scene_state()
                if camera_data is None:
                    logger.error("Failed to capture scene, stopping task")
                    break
                
                # Predict action using VLA
                grasp_pose = await self.predict_action(
                    camera_data=camera_data,
                    instruction=instruction,
                    target_object=target_object
                )
                
                if grasp_pose is None:
                    logger.error("VLA failed to predict action, stopping task")
                    break
                
                # Visualize prediction
                self.visualize_prediction(grasp_pose)
                
                # TODO: Execute action in simulation
                # This would involve moving the robot arm to the predicted pose
                
                # Check if task is complete
                if self._is_task_complete(instruction, target_object):
                    logger.info("✓ Task completed successfully!")
                    break
                
                iteration += 1
                await asyncio.sleep(1.0)  # Brief pause between iterations
                
            if iteration >= max_iterations:
                logger.warning("Task loop reached maximum iterations")
                
        except Exception as e:
            logger.error(f"Task loop failed: {e}")
    
    def _is_task_complete(self, instruction: str, target_object: str) -> bool:
        """
        Check if the current task has been completed.
        
        Args:
            instruction: Task instruction
            target_object: Target object
            
        Returns:
            True if task appears complete
        """
        # Placeholder logic - in reality would check object positions
        # For demo purposes, consider task "complete" after 3 iterations
        return len(self.action_history) >= 3
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge execution statistics."""
        return {
            'total_actions': len(self.action_history),
            'average_confidence': np.mean([h['confidence'] for h in self.action_history]) if self.action_history else 0.0,
            'scene_objects': len(self.scene_objects),
            'current_task': self.current_task
        }
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        try:
            logger.info("Resetting simulation...")
            
            # Reset scene
            self.scene.reset()
            
            # Clear action history
            self.action_history.clear()
            self.current_task = None
            
            logger.info("✓ Simulation reset complete")
            
        except Exception as e:
            logger.error(f"Failed to reset simulation: {e}")
    
    def shutdown(self):
        """Shutdown bridge and cleanup resources."""
        try:
            logger.info("Shutting down Isaac Sim - VLA bridge...")
            
            # Close simulation
            if self.scene:
                self.scene.close()
            
            logger.info("✓ Bridge shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Convenience functions

async def run_simple_demo():
    """Run a simple demonstration of the Isaac Sim - VLA bridge."""
    bridge = IsaacVLABridge()
    
    try:
        # Start simulation
        bridge.start_simulation()
        
        # Run carrot pickup task
        await bridge.run_task_loop(
            instruction="pick up the carrot and place it in the slicer",
            target_object="carrot"
        )
        
        # Show statistics
        stats = bridge.get_statistics()
        logger.info(f"Demo statistics: {stats}")
        
    finally:
        bridge.shutdown()


def test_bridge_components():
    """Test bridge component initialization."""
    logger.info("Testing bridge components...")
    
    try:
        bridge = IsaacVLABridge()
        
        # Test scene capture
        camera_data = bridge.capture_scene_state()
        if camera_data:
            logger.info("✓ Scene capture working")
        else:
            logger.error("✗ Scene capture failed")
        
        # Show component status
        logger.info(f"Bridge statistics: {bridge.get_statistics()}")
        
        bridge.shutdown()
        
    except Exception as e:
        logger.error(f"Bridge component test failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    asyncio.run(run_simple_demo())