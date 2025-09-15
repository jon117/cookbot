"""
Isaac Sim Robot Control Interface
Controls simulated robot arm based on VLA predictions.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

# Import Isaac Sim modules conditionally
try:
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.prims import RigidPrimView
    from omni.isaac.core.utils.types import ArticulationAction
    import omni.isaac.core.utils.prims as prim_utils
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Isaac Sim modules not available for robot control - running in fallback mode")

from ..common.data_types import GraspPose, Pose6D, ExecutionResult

logger = logging.getLogger(__name__)


class SimulatedRobotArm:
    """Simulated robot arm controller for Isaac Sim."""
    
    def __init__(self, robot_prim_path: str = "/World/Robot"):
        """
        Initialize simulated robot arm.
        
        Args:
            robot_prim_path: USD path to robot in scene
        """
        self.robot_prim_path = robot_prim_path
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.joint_limits = self._get_joint_limits()
        
        # Current state
        self.current_joint_positions = np.zeros(6)
        self.current_end_effector_pose = Pose6D(0.3, 0.0, 0.3, 0, 0, 0)
        self.gripper_state = "open"  # "open" or "closed"
        
        # Control parameters
        self.position_gain = 100.0
        self.damping_gain = 10.0
        self.max_velocity = 1.0
        
        self.setup_robot()
        
    def setup_robot(self):
        """Set up robot articulation in Isaac Sim."""
        try:
            logger.info("Setting up simulated robot arm...")
            
            if not ISAAC_AVAILABLE:
                logger.warning("Isaac Sim not available - using fallback robot simulation")
                self._setup_fallback_robot()
                return
            
            # For now, use placeholder robot (simple kinematic chain)
            logger.debug("About to create simple robot...")
            self._create_simple_robot()
            logger.debug("Simple robot created successfully")
            
            logger.info("✓ Robot arm setup complete")
            
        except Exception as e:
            logger.error(f"Failed to set up robot: {e}")
    
    def _setup_fallback_robot(self):
        """Set up fallback robot simulation when Isaac Sim is not available."""
        logger.info("Setting up fallback robot simulation...")
        
        # Use simple kinematic model
        self.robot_base = None
        self.arm_segments = []
        self.end_effector = None
        
        # Set reasonable initial state
        self.current_joint_positions = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0])
        self.current_end_effector_pose = Pose6D(0.3, 0.0, 0.3, 0, 0, 0)
        self.gripper_state = "open"
        
        logger.info("✓ Fallback robot simulation initialized")
        
    def _create_simple_robot(self):
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _create_simple_robot(self):
        """Create a simple robot arm for demonstration."""
        # This is a placeholder - in a full implementation, you would load
        # the actual FR5 robot URDF using Isaac Sim's URDF loader
        
        # Create robot base
        self.robot_base = DynamicCuboid(
            prim_path=f"{self.robot_prim_path}/Base",
            name="robot_base",
            position=np.array([0.0, 0.0, 0.1]),
            scale=np.array([0.2, 0.2, 0.2]),  # [x, y, z] scale factors
            color=np.array([0.3, 0.3, 0.8]),
            mass=5.0
        )
        
        # Create arm segments (simplified 6-DOF arm)
        self.arm_segments = []
        
        # Link 1
        link1 = DynamicCuboid(
            prim_path=f"{self.robot_prim_path}/Link1",
            name="link1",
            position=np.array([0.0, 0.0, 0.25]),
            scale=np.array([0.05, 0.05, 0.2]),
            color=np.array([0.4, 0.4, 0.9]),
            mass=1.0
        )
        self.arm_segments.append(link1)
        
        # Link 2 
        link2 = DynamicCuboid(
            prim_path=f"{self.robot_prim_path}/Link2",
            name="link2",
            position=np.array([0.15, 0.0, 0.35]),
            scale=np.array([0.3, 0.05, 0.05]),
            color=np.array([0.4, 0.4, 0.9]),
            mass=0.8
        )
        self.arm_segments.append(link2)
        
        # End effector
        self.end_effector = DynamicCuboid(
            prim_path=f"{self.robot_prim_path}/EndEffector",
            name="end_effector",
            position=np.array([0.3, 0.0, 0.35]),
            scale=np.array([0.05, 0.05, 0.1]),
            color=np.array([0.2, 0.8, 0.2]),
            mass=0.2
        )
        
    def _get_joint_limits(self) -> Dict[str, List[float]]:
        """Get joint limits for the robot."""
        return {
            "joint1": [-165, 165],
            "joint2": [-135, 135], 
            "joint3": [-135, 135],
            "joint4": [-180, 180],
            "joint5": [-180, 180],
            "joint6": [-180, 180]
        }
    
    def move_to_pose(self, target_pose: Pose6D, speed: float = 0.5) -> ExecutionResult:
        """
        Move robot to target pose.
        
        Args:
            target_pose: Target end-effector pose
            speed: Movement speed [0, 1]
            
        Returns:
            ExecutionResult with success status
        """
        try:
            logger.info(f"Moving robot to pose: ({target_pose.x:.3f}, {target_pose.y:.3f}, {target_pose.z:.3f})")
            
            # Solve inverse kinematics
            joint_angles = self._solve_ik(target_pose)
            if joint_angles is None:
                return ExecutionResult(
                    success=False,
                    message="Inverse kinematics failed",
                    error_code="IK_FAILED"
                )
            
            # Validate joint limits
            if not self._validate_joint_angles(joint_angles):
                return ExecutionResult(
                    success=False,
                    message="Joint limits exceeded",
                    error_code="JOINT_LIMITS"
                )
            
            # Execute motion
            success = self._execute_joint_motion(joint_angles, speed)
            
            if success:
                self.current_end_effector_pose = target_pose
                self.current_joint_positions = np.array(joint_angles)
                
                return ExecutionResult(
                    success=True,
                    message="Motion completed successfully",
                    final_pose=target_pose,
                    execution_time=2.0  # Simulated execution time
                )
            else:
                return ExecutionResult(
                    success=False,
                    message="Motion execution failed",
                    error_code="EXECUTION_FAILED"
                )
                
        except Exception as e:
            logger.error(f"Move to pose failed: {e}")
            return ExecutionResult(
                success=False,
                message=f"Motion error: {str(e)}",
                error_code="MOTION_ERROR"
            )
    
    def execute_grasp(self, grasp_pose: GraspPose) -> ExecutionResult:
        """
        Execute grasping motion based on VLA prediction.
        
        Args:
            grasp_pose: Predicted grasp pose from VLA
            
        Returns:
            ExecutionResult with success status
        """
        try:
            logger.info(f"Executing grasp at ({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f})")
            
            if not ISAAC_AVAILABLE:
                # Fallback simulation
                logger.info("Simulating grasp execution in fallback mode...")
                
                # Update simulated robot state
                self.current_end_effector_pose = Pose6D(
                    x=grasp_pose.x,
                    y=grasp_pose.y,
                    z=grasp_pose.z,
                    rx=grasp_pose.rx,
                    ry=grasp_pose.ry,
                    rz=grasp_pose.rz
                )
                self.gripper_state = "closed"
                
                # Simulate some execution time
                import time
                time.sleep(2.0)
                
                return ExecutionResult(
                    success=True,
                    message=f"Simulated grasp executed at ({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f})",
                    execution_time=2.0,
                    final_pose=self.current_end_effector_pose
                )
            
            # Step 1: Move to pre-grasp position (approach)
            pre_grasp_pose = Pose6D(
                x=grasp_pose.x,
                y=grasp_pose.y,
                z=grasp_pose.z + 0.1,  # 10cm above grasp
                rx=grasp_pose.rx,
                ry=grasp_pose.ry,
                rz=grasp_pose.rz
            )
            
            result = self.move_to_pose(pre_grasp_pose, speed=0.3)
            if not result.success:
                return result
            
            # Step 2: Open gripper
            self.set_gripper_state("open", grasp_pose.gripper_width + 0.02)
            
            # Step 3: Move to grasp position
            grasp_target = Pose6D(
                x=grasp_pose.x,
                y=grasp_pose.y,
                z=grasp_pose.z,
                rx=grasp_pose.rx,
                ry=grasp_pose.ry,
                rz=grasp_pose.rz
            )
            
            result = self.move_to_pose(grasp_target, speed=0.2)
            if not result.success:
                return result
            
            # Step 4: Close gripper
            self.set_gripper_state("closed", grasp_pose.gripper_width)
            
            # Step 5: Lift object
            lift_pose = Pose6D(
                x=grasp_pose.x,
                y=grasp_pose.y,
                z=grasp_pose.z + 0.05,  # Lift 5cm
                rx=grasp_pose.rx,
                ry=grasp_pose.ry,
                rz=grasp_pose.rz
            )
            
            result = self.move_to_pose(lift_pose, speed=0.2)
            
            if result.success:
                logger.info("✓ Grasp execution completed successfully")
                return ExecutionResult(
                    success=True,
                    message="Grasp completed successfully",
                    final_pose=lift_pose,
                    execution_time=8.0  # Total grasp sequence time
                )
            else:
                return result
                
        except Exception as e:
            logger.error(f"Grasp execution failed: {e}")
            return ExecutionResult(
                success=False,
                message=f"Grasp error: {str(e)}",
                error_code="GRASP_ERROR"
            )
    
    def set_gripper_state(self, state: str, width: float = 0.05):
        """
        Set gripper state.
        
        Args:
            state: "open" or "closed"
            width: Gripper width in meters
        """
        try:
            logger.debug(f"Setting gripper to {state}, width={width:.3f}")
            
            self.gripper_state = state
            
            # In a full implementation, this would control actual gripper
            # For now, just update visualization
            if hasattr(self, 'end_effector'):
                # Change color to indicate gripper state
                if state == "closed":
                    # Red for closed
                    gripper_color = np.array([0.8, 0.2, 0.2])
                else:
                    # Green for open  
                    gripper_color = np.array([0.2, 0.8, 0.2])
                    
                # Update end effector color (placeholder)
                logger.debug(f"Gripper visual state updated to {state}")
                
        except Exception as e:
            logger.error(f"Failed to set gripper state: {e}")
    
    def _solve_ik(self, target_pose: Pose6D) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_pose: Target end-effector pose
            
        Returns:
            Joint angles in radians, or None if no solution
        """
        # Simplified IK for demonstration
        # In reality, this would use proper IK solver
        
        # Check if pose is reachable
        distance = np.sqrt(target_pose.x**2 + target_pose.y**2 + target_pose.z**2)
        max_reach = 0.8  # Maximum reach of arm
        
        if distance > max_reach:
            logger.warning(f"Target pose out of reach: {distance:.3f}m > {max_reach}m")
            return None
        
        # Simple geometric IK (placeholder)
        joint_angles = np.array([
            np.arctan2(target_pose.y, target_pose.x),  # Base rotation
            np.arctan2(target_pose.z - 0.1, np.sqrt(target_pose.x**2 + target_pose.y**2)),  # Shoulder
            -np.pi/4,  # Elbow
            0.0,       # Wrist 1
            0.0,       # Wrist 2  
            target_pose.rz  # Wrist 3 (tool rotation)
        ])
        
        return joint_angles
    
    def _validate_joint_angles(self, joint_angles: np.ndarray) -> bool:
        """Validate joint angles against limits."""
        for i, (angle, joint_name) in enumerate(zip(joint_angles, self.joint_names)):
            limits = self.joint_limits.get(joint_name, [-180, 180])
            limit_rad = [np.radians(limits[0]), np.radians(limits[1])]
            
            if not (limit_rad[0] <= angle <= limit_rad[1]):
                logger.warning(f"Joint {joint_name} exceeds limits: {np.degrees(angle):.1f}°")
                return False
                
        return True
    
    def _execute_joint_motion(self, target_angles: np.ndarray, speed: float) -> bool:
        """
        Execute joint motion to target angles.
        
        Args:
            target_angles: Target joint angles in radians
            speed: Motion speed [0, 1]
            
        Returns:
            True if motion successful
        """
        try:
            # Simulate motion execution
            logger.debug(f"Executing joint motion with speed {speed}")
            
            # In real Isaac Sim, this would send commands to articulation
            # For now, just update current positions
            self.current_joint_positions = np.array(target_angles).copy()
            
            # Update end effector position based on forward kinematics (simplified)
            self._update_end_effector_pose(target_angles)
            
            return True
            
        except Exception as e:
            logger.error(f"Joint motion execution failed: {e}")
            return False
    
    def _update_end_effector_pose(self, joint_angles: np.ndarray):
        """Update end effector pose based on joint angles (forward kinematics)."""
        # Simplified forward kinematics
        base_rotation = joint_angles[0]
        shoulder_angle = joint_angles[1]
        
        # Calculate approximate end effector position
        arm_length = 0.7  # Total arm length
        x = arm_length * np.cos(shoulder_angle) * np.cos(base_rotation)
        y = arm_length * np.cos(shoulder_angle) * np.sin(base_rotation)
        z = 0.1 + arm_length * np.sin(shoulder_angle)
        
        # Update end effector visual position
        if hasattr(self, 'end_effector'):
            try:
                self.end_effector.set_world_pose(position=np.array([x, y, z]))
            except:
                pass  # Ignore errors in simulation
    
    def get_current_pose(self) -> Pose6D:
        """Get current end-effector pose."""
        return self.current_end_effector_pose
    
    def get_current_joint_state(self) -> np.ndarray:
        """Get current joint angles."""
        return self.current_joint_positions.copy()
    
    def reset_to_home(self) -> ExecutionResult:
        """Reset robot to home position."""
        home_pose = Pose6D(x=0.3, y=0.0, z=0.3, rx=0, ry=0, rz=0)
        return self.move_to_pose(home_pose, speed=0.5)


class RobotController:
    """High-level robot controller for VLA integration."""
    
    def __init__(self):
        """Initialize robot controller."""
        self.robot_arm = SimulatedRobotArm()
        self.task_queue = []
        self.current_task = None
        
    async def execute_vla_action(self, grasp_pose: GraspPose, instruction: str) -> ExecutionResult:
        """
        Execute action predicted by VLA model.
        
        Args:
            grasp_pose: VLA predicted grasp pose
            instruction: Original task instruction
            
        Returns:
            ExecutionResult with execution status
        """
        logger.info(f"Executing VLA action for: {instruction}")
        
        try:
            # Execute grasp
            result = self.robot_arm.execute_grasp(grasp_pose)
            
            if result.success:
                logger.info("✓ VLA action executed successfully")
            else:
                logger.error(f"✗ VLA action failed: {result.message}")
            
            return result
            
        except Exception as e:
            logger.error(f"VLA action execution error: {e}")
            return ExecutionResult(
                success=False,
                message=f"Execution error: {str(e)}",
                error_code="EXECUTION_ERROR"
            )
    
    def get_robot_status(self) -> Dict[str, Any]:
        """Get current robot status."""
        current_pose = self.robot_arm.get_current_pose()
        joint_state = self.robot_arm.get_current_joint_state()
        
        # Debug: Check types
        logger.debug(f"joint_state type: {type(joint_state)}, value: {joint_state}")
        
        # Convert to list safely
        if hasattr(joint_state, 'tolist'):
            joint_angles_list = joint_state.tolist()
        else:
            joint_angles_list = list(joint_state) if joint_state is not None else []
        
        return {
            'current_pose': {
                'x': current_pose.x,
                'y': current_pose.y, 
                'z': current_pose.z,
                'rx': current_pose.rx,
                'ry': current_pose.ry,
                'rz': current_pose.rz
            },
            'joint_angles': joint_angles_list,
            'gripper_state': self.robot_arm.gripper_state,
            'task_queue_length': len(self.task_queue)
        }


# Testing functions

def test_robot_control():
    """Test robot control functionality."""
    logger.info("Testing robot control...")
    
    try:
        controller = RobotController()
        
        # Test home position
        result = controller.robot_arm.reset_to_home()
        if result.success:
            logger.info("✓ Home position test passed")
        else:
            logger.error("✗ Home position test failed")
        
        # Test basic motion
        test_pose = Pose6D(x=0.4, y=0.2, z=0.2, rx=0, ry=0, rz=0)
        result = controller.robot_arm.move_to_pose(test_pose)
        
        if result.success:
            logger.info("✓ Basic motion test passed")
        else:
            logger.error("✗ Basic motion test failed")
        
        # Show status
        status = controller.get_robot_status()
        logger.info(f"Robot status: {status}")
        
    except Exception as e:
        logger.error(f"Robot control test failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_robot_control()