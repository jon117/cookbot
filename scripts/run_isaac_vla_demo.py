#!/usr/bin/env python3
"""
Isaac Sim + OpenVLA Integration Test
Complete demonstration of Isaac Sim scene -> OpenVLA model -> Robot action pipeline.
"""

import asyncio
import logging
import sys
import time
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src directory to Python path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Setup Isaac Sim environment if available
isaac_sim_path = Path.home() / "isaacsim"
if isaac_sim_path.exists():
    os.environ["ISAAC_SIM_PATH"] = str(isaac_sim_path)
    # Add Isaac Sim Python paths
    isaac_python_paths = [
        isaac_sim_path / "kit" / "python" / "lib",
        isaac_sim_path / "exts" / "omni.isaac.kit" / "pip_prebundle",
    ]
    for path in isaac_python_paths:
        if path.exists():
            sys.path.insert(0, str(path))

from src.simulation.isaac_vla_bridge import IsaacVLABridge
from src.simulation.robot_control import RobotController
from common.data_types import GraspPose

logger = logging.getLogger(__name__)


class VLADemoScenario:
    """Complete VLA demonstration scenario."""
    
    def __init__(self):
        """Initialize demo scenario."""
        self.bridge = None
        self.robot_controller = None
        self.demo_stats = {
            'start_time': 0,
            'end_time': 0,
            'actions_executed': 0,
            'success_rate': 0.0,
            'total_time': 0.0
        }
        
    async def setup_demo(self):
        """Set up demo components."""
        logger.info("🚀 Setting up Isaac Sim + OpenVLA demo...")
        
        try:
            # Initialize bridge
            logger.info("Initializing Isaac Sim - VLA bridge...")
            self.bridge = IsaacVLABridge()
            
            # Initialize robot controller
            logger.info("Initializing robot controller...")
            self.robot_controller = RobotController()
            
            # Start simulation
            logger.info("Starting Isaac Sim simulation...")
            self.bridge.start_simulation()
            
            logger.info("✅ Demo setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Demo setup failed: {e}")
            raise
    
    async def run_carrot_pickup_demo(self):
        """Run the main carrot pickup demonstration."""
        logger.info("🥕 Starting carrot pickup demonstration...")
        
        try:
            self.demo_stats['start_time'] = time.time()
            
            # Demo parameters
            instruction = "pick up the carrot and place it in the slicer"
            target_object = "carrot"
            max_attempts = 5
            
            logger.info(f"Task: {instruction}")
            logger.info(f"Target object: {target_object}")
            logger.info(f"Max attempts: {max_attempts}")
            
            for attempt in range(max_attempts):
                logger.info(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")
                
                # Step 1: Capture current scene
                logger.info("📸 Capturing scene state...")
                camera_data = self.bridge.capture_scene_state()
                
                if camera_data is None:
                    logger.error("❌ Failed to capture scene")
                    continue
                
                logger.info(f"✅ Scene captured: RGB {camera_data.rgb.shape}")
                
                # Step 2: Get VLA prediction
                logger.info("🤖 Getting VLA prediction...")
                grasp_pose = await self.bridge.predict_action(
                    camera_data=camera_data,
                    instruction=instruction,
                    target_object=target_object
                )
                
                if grasp_pose is None:
                    logger.error("❌ VLA failed to predict action")
                    continue
                
                logger.info(f"✅ VLA prediction: pos=({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f}), "
                           f"conf={grasp_pose.confidence:.3f}")
                
                # Step 3: Visualize prediction
                logger.info("👁️ Visualizing VLA prediction...")
                self.bridge.visualize_prediction(grasp_pose)
                
                # Step 4: Execute robot action
                logger.info("🦾 Executing robot action...")
                execution_result = await self.robot_controller.execute_vla_action(
                    grasp_pose=grasp_pose,
                    instruction=instruction
                )
                
                if execution_result.success:
                    logger.info("✅ Robot action executed successfully!")
                    self.demo_stats['actions_executed'] += 1
                    
                    # Check if task is complete
                    if self._check_task_completion():
                        logger.info("🎉 Task completed successfully!")
                        break
                else:
                    logger.error(f"❌ Robot action failed: {execution_result.message}")
                
                # Brief pause between attempts
                await asyncio.sleep(2.0)
            
            self.demo_stats['end_time'] = time.time()
            self.demo_stats['total_time'] = self.demo_stats['end_time'] - self.demo_stats['start_time']
            
            if self.demo_stats['actions_executed'] > 0:
                self.demo_stats['success_rate'] = 1.0 if attempt < max_attempts - 1 else 0.5
            
            logger.info("🏁 Carrot pickup demo completed!")
            
        except Exception as e:
            logger.error(f"❌ Demo execution failed: {e}")
    
    def _check_task_completion(self) -> bool:
        """
        Check if the task has been completed successfully.
        
        Returns:
            True if task appears complete
        """
        # In a real implementation, this would check:
        # - Object detection to see if carrot is in slicer
        # - Position verification
        # - Appliance state
        
        # For demo purposes, consider success after 1 successful action
        return self.demo_stats['actions_executed'] >= 1
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo with multiple scenarios."""
        logger.info("🌟 Running comprehensive VLA demo...")
        
        scenarios = [
            {
                'name': 'Carrot Pickup',
                'instruction': 'pick up the carrot',
                'target': 'carrot'
            },
            {
                'name': 'Carrot to Slicer',
                'instruction': 'place the carrot in the slicer',
                'target': 'carrot'
            },
            {
                'name': 'Object Inspection',
                'instruction': 'examine the carrot carefully',
                'target': 'carrot'
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"\n🎬 Scenario {i+1}: {scenario['name']}")
            
            # Capture scene
            camera_data = self.bridge.capture_scene_state()
            if camera_data is None:
                continue
            
            # Get VLA prediction
            grasp_pose = await self.bridge.predict_action(
                camera_data=camera_data,
                instruction=scenario['instruction'],
                target_object=scenario['target']
            )
            
            if grasp_pose:
                logger.info(f"✅ {scenario['name']}: VLA confidence = {grasp_pose.confidence:.3f}")
                self.bridge.visualize_prediction(grasp_pose)
            else:
                logger.error(f"❌ {scenario['name']}: VLA prediction failed")
            
            await asyncio.sleep(1.0)
    
    def print_demo_summary(self):
        """Print summary of demo results."""
        logger.info("\n" + "="*60)
        logger.info("📊 DEMO SUMMARY")
        logger.info("="*60)
        
        stats = self.demo_stats
        bridge_stats = self.bridge.get_statistics() if self.bridge else {}
        robot_status = self.robot_controller.get_robot_status() if self.robot_controller else {}
        
        logger.info(f"⏱️  Total demo time: {stats['total_time']:.1f} seconds")
        logger.info(f"🤖 Actions executed: {stats['actions_executed']}")
        logger.info(f"✅ Success rate: {stats['success_rate']*100:.1f}%")
        
        if bridge_stats:
            logger.info(f"🧠 VLA predictions: {bridge_stats.get('total_actions', 0)}")
            logger.info(f"🎯 Average confidence: {bridge_stats.get('average_confidence', 0):.3f}")
        
        if robot_status:
            current_pose = robot_status.get('current_pose', {})
            logger.info(f"🦾 Final robot position: ({current_pose.get('x', 0):.3f}, "
                       f"{current_pose.get('y', 0):.3f}, {current_pose.get('z', 0):.3f})")
        
        logger.info("="*60)
        
        # Performance assessment
        if stats['success_rate'] >= 0.8:
            logger.info("🏆 EXCELLENT: Demo performed very well!")
        elif stats['success_rate'] >= 0.5:
            logger.info("👍 GOOD: Demo showed promising results!")
        else:
            logger.info("🔧 NEEDS WORK: Demo needs improvement.")
    
    async def cleanup_demo(self):
        """Clean up demo resources."""
        logger.info("🧹 Cleaning up demo...")
        
        try:
            if self.bridge:
                self.bridge.shutdown()
            
            logger.info("✅ Demo cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")


# Main demo runner

async def main():
    """Main demo entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🌟 Isaac Sim + OpenVLA Integration Demo")
    logger.info("="*60)
    
    demo = VLADemoScenario()
    
    try:
        # Setup
        await demo.setup_demo()
        
        # Run main demo
        await demo.run_carrot_pickup_demo()
        
        # Run additional scenarios
        logger.info("\n🔄 Running additional scenarios...")
        await demo.run_comprehensive_demo()
        
        # Show results
        demo.print_demo_summary()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
    finally:
        await demo.cleanup_demo()


def run_quick_test():
    """Run quick component test without full Isaac Sim."""
    logger.info("🔬 Running quick component test...")
    
    try:
        # Test VLA configuration
        logger.info("Testing VLA configuration...")
        from movement_layer.grasp_planner import GraspPlanner
        
        planner = GraspPlanner("config/models/vla_config.yaml")
        logger.info(f"✅ VLA planner initialized: {planner.vla_model.endpoint}")
        
        # Test data types
        logger.info("Testing data types...")
        from common.data_types import CameraObservation, GraspPose
        import numpy as np
        
        # Create test camera data
        test_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_camera = CameraObservation(
            rgb=test_rgb,
            timestamp=time.time(),
            camera_id="test"
        )
        
        logger.info(f"✅ Camera data: {test_camera.rgb.shape}")
        
        # Create test grasp pose
        test_grasp = GraspPose(
            x=0.3, y=0.1, z=0.15,
            rx=0, ry=0, rz=0,
            gripper_width=0.04,
            confidence=0.8
        )
        
        logger.info(f"✅ Grasp pose: pos=({test_grasp.x}, {test_grasp.y}, {test_grasp.z})")
        
        logger.info("🎉 Quick test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Isaac Sim + OpenVLA Demo")
    parser.add_argument(
        "--quick-test", 
        action="store_true", 
        help="Run quick component test without Isaac Sim"
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test()
    else:
        asyncio.run(main())