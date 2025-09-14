"""
Motion Planner - Traditional robotics motion execution
Handles inverse kinematics, trajectory planning, and robot control using MoveIt2.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import yaml
import os
import time
import math
import numpy as np

# Import our common types
from ..common.data_types import (
    Pose6D, Trajectory, Constraints, ExecutionResult, 
    WorkspaceBounds, ValidationResult
)

logger = logging.getLogger(__name__)

class MotionPlanner:
    """
    Traditional motion planning using MoveIt2 and inverse kinematics.
    
    Implements collision-free trajectory planning, joint-space motion,
    and real-time execution with safety monitoring.
    """
    
    def __init__(self, robot_config: str = "config/robots/fr5_left_arm.yaml"):
        """
        Initialize MoveIt2 planning scene and robot model.
        
        Args:
            robot_config: Path to robot configuration file
        """
        self.robot_config_path = robot_config
        self.config = self._load_robot_config()
        
        # Robot parameters
        self.robot_name = self.config.get("robot", {}).get("name", "fr5_left_arm")
        self.joint_names = self._get_joint_names()
        self.joint_limits = self._get_joint_limits()
        
        # Planning configuration
        self.planning_config = self.config.get("planning", {})
        self.planner_name = self.planning_config.get("planner", "RRTstar")
        self.planning_time = self.planning_config.get("planning_time", 5.0)
        
        # Safety parameters
        self.max_velocity = self.planning_config.get("max_velocity_scaling", 0.5)
        self.max_acceleration = self.planning_config.get("max_acceleration_scaling", 0.5)
        
        # Workspace bounds
        workspace_config = self.config.get("workspace_bounds", {})
        self.workspace_bounds = WorkspaceBounds(**workspace_config)
        
        # Initialize MoveIt2 interface (mock for now)
        self._initialize_moveit()
        
        # Current robot state
        self.current_joint_state = [0.0] * len(self.joint_names)
        self.current_pose = Pose6D(0.3, 0.0, 0.3, 0, 0, 0)
        
        logger.info(f"MotionPlanner initialized for robot: {self.robot_name}")
    
    def execute_pose(self, 
                     target_pose: Pose6D, 
                     constraints: Optional[Constraints] = None) -> ExecutionResult:
        """
        Execute motion to target pose with collision avoidance.
        
        Args:
            target_pose: Target position and orientation
            constraints: Motion constraints (velocity, acceleration, etc.)
            
        Returns:
            ExecutionResult with success status and execution details
        """
        start_time = time.time()
        
        try:
            # Validate target pose
            validation_result = self.validate_pose(target_pose)
            if not validation_result.is_valid:
                return ExecutionResult(
                    success=False,
                    message=f"Invalid target pose: {validation_result.errors}",
                    error_code="INVALID_POSE"
                )
            
            # Apply default constraints if none provided
            if constraints is None:
                constraints = self._get_default_constraints()
            
            # Plan trajectory to target pose
            trajectory = self.plan_trajectory([self.current_pose, target_pose], constraints)
            if not trajectory.waypoints:
                return ExecutionResult(
                    success=False,
                    message="Failed to plan trajectory",
                    error_code="PLANNING_FAILED"
                )
            
            # Execute trajectory with monitoring
            execution_success = self._execute_trajectory(trajectory, constraints)
            
            if execution_success:
                # Update current state
                self.current_pose = target_pose
                execution_time = time.time() - start_time
                
                logger.info(f"Motion executed successfully in {execution_time:.2f}s")
                
                return ExecutionResult(
                    success=True,
                    message="Motion completed successfully",
                    final_pose=target_pose,
                    trajectory_points=trajectory.waypoints,
                    execution_time=execution_time
                )
            else:
                return ExecutionResult(
                    success=False,
                    message="Trajectory execution failed",
                    error_code="EXECUTION_FAILED"
                )
                
        except Exception as e:
            logger.error(f"Motion execution failed: {e}")
            return ExecutionResult(
                success=False,
                message=f"Motion execution error: {str(e)}",
                error_code="EXECUTION_ERROR"
            )
    
    def plan_trajectory(self, 
                       waypoints: List[Pose6D], 
                       constraints: Optional[Constraints] = None) -> Trajectory:
        """
        Generate smooth, collision-free path through waypoints.
        
        Args:
            waypoints: List of poses to visit
            constraints: Motion constraints
            
        Returns:
            Trajectory with waypoints, joint angles, and timing
        """
        try:
            if len(waypoints) < 2:
                logger.error("Need at least 2 waypoints for trajectory planning")
                return Trajectory(waypoints=[])
            
            # Apply default constraints
            if constraints is None:
                constraints = self._get_default_constraints()
            
            # Plan path through waypoints
            planned_waypoints = []
            joint_trajectories = []
            timestamps = []
            
            current_time = 0.0
            
            for i in range(len(waypoints)):
                waypoint = waypoints[i]
                
                # Solve inverse kinematics for each waypoint
                joint_solution = self._solve_inverse_kinematics(waypoint)
                if joint_solution is None:
                    logger.warning(f"No IK solution for waypoint {i}")
                    continue
                
                planned_waypoints.append(waypoint)
                joint_trajectories.append(joint_solution)
                
                # Calculate timing based on distance and velocity constraints
                if i > 0:
                    distance = self._calculate_pose_distance(waypoints[i-1], waypoint)
                    segment_time = distance / constraints.max_velocity
                    current_time += max(segment_time, 0.1)  # Minimum 0.1s per segment
                
                timestamps.append(current_time)
            
            # Generate smooth trajectory with intermediate points
            if len(planned_waypoints) >= 2:
                smooth_trajectory = self._generate_smooth_trajectory(
                    planned_waypoints, joint_trajectories, timestamps, constraints
                )
                return smooth_trajectory
            else:
                logger.error("Insufficient valid waypoints after IK solving")
                return Trajectory(waypoints=[])
                
        except Exception as e:
            logger.error(f"Trajectory planning failed: {e}")
            return Trajectory(waypoints=[])
    
    def validate_pose(self, pose: Pose6D) -> ValidationResult:
        """
        Validate pose against workspace constraints and robot limits.
        
        Args:
            pose: Pose to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check workspace bounds
        if not self.workspace_bounds.contains(pose):
            errors.append("Pose outside workspace bounds")
        
        # Check reachability via IK
        joint_solution = self._solve_inverse_kinematics(pose)
        if joint_solution is None:
            errors.append("No inverse kinematics solution found")
        else:
            # Check joint limits
            for i, (joint_angle, joint_name) in enumerate(zip(joint_solution, self.joint_names)):
                limits = self.joint_limits.get(joint_name, [-180, 180])
                if not (math.radians(limits[0]) <= joint_angle <= math.radians(limits[1])):
                    errors.append(f"Joint {joint_name} exceeds limits: {math.degrees(joint_angle):.1f}°")
        
        # Check for singularities (simplified check)
        if abs(pose.z) < 0.05:
            warnings.append("Pose near workspace singularity")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop - halt all motion immediately.
        
        Returns:
            True if stop was successful
        """
        try:
            logger.warning("EMERGENCY STOP ACTIVATED")
            # In real implementation, this would stop all robot motion
            # and engage brakes/safety systems
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False
    
    def get_current_pose(self) -> Pose6D:
        """Get current robot end-effector pose."""
        return self.current_pose
    
    def get_current_joint_state(self) -> List[float]:
        """Get current joint angles in radians."""
        return self.current_joint_state.copy()
    
    # Private methods
    
    def _load_robot_config(self) -> Dict[str, Any]:
        """Load robot configuration from YAML file."""
        if not os.path.exists(self.robot_config_path):
            logger.warning(f"Robot config {self.robot_config_path} not found, using defaults")
            return self._get_default_robot_config()
        
        try:
            with open(self.robot_config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load robot config: {e}")
            return self._get_default_robot_config()
    
    def _get_default_robot_config(self) -> Dict[str, Any]:
        """Get default robot configuration."""
        return {
            "robot": {
                "name": "fr5_left_arm",
                "urdf_package": "fr5_description",
                "urdf_file": "fr5.urdf.xacro"
            },
            "joint_limits": {
                "joint1": [-165, 165],
                "joint2": [-135, 135],
                "joint3": [-135, 135],
                "joint4": [-180, 180],
                "joint5": [-180, 180],
                "joint6": [-180, 180]
            },
            "planning": {
                "group_name": "fr5_arm",
                "planner": "RRTstar",
                "planning_time": 5.0,
                "max_velocity_scaling": 0.5,
                "max_acceleration_scaling": 0.5
            },
            "workspace_bounds": {
                "x_min": -0.5, "x_max": 0.8,
                "y_min": -0.6, "y_max": 0.6,
                "z_min": 0.0, "z_max": 1.0
            }
        }
    
    def _get_joint_names(self) -> List[str]:
        """Get joint names from configuration."""
        joint_limits = self.config.get("joint_limits", {})
        return list(joint_limits.keys())
    
    def _get_joint_limits(self) -> Dict[str, List[float]]:
        """Get joint limits from configuration."""
        return self.config.get("joint_limits", {})
    
    def _initialize_moveit(self) -> None:
        """Initialize MoveIt2 interface (mock implementation)."""
        # In real implementation, this would:
        # 1. Initialize ROS2 node
        # 2. Create MoveIt2 MoveGroupInterface
        # 3. Load robot model and planning scene
        # 4. Set up collision objects
        logger.info("MoveIt2 interface initialized (mock)")
    
    def _get_default_constraints(self) -> Constraints:
        """Get default motion constraints."""
        return Constraints(
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
            position_tolerance=0.01,
            orientation_tolerance=0.1,
            avoid_collisions=True
        )
    
    def _solve_inverse_kinematics(self, pose: Pose6D) -> Optional[List[float]]:
        """
        Solve inverse kinematics for target pose.
        
        Args:
            pose: Target end-effector pose
            
        Returns:
            Joint angles in radians, or None if no solution
        """
        # Mock IK solver - in real implementation would use:
        # - KDL kinematics solver
        # - MoveIt2 IK solver
        # - Custom analytical IK for specific robot
        
        # Simple heuristic IK for demonstration
        if self.workspace_bounds.contains(pose):
            # Generate reasonable joint angles based on pose
            joint_angles = [
                math.atan2(pose.y, pose.x),  # Base rotation
                math.atan2(pose.z - 0.1, math.sqrt(pose.x**2 + pose.y**2)),  # Shoulder
                -math.pi/4,  # Elbow
                0.0,         # Wrist1
                0.0,         # Wrist2
                pose.rz      # Wrist3 (tool rotation)
            ]
            
            # Check joint limits
            for i, (angle, joint_name) in enumerate(zip(joint_angles, self.joint_names)):
                limits = self.joint_limits.get(joint_name, [-180, 180])
                limit_rad = [math.radians(limits[0]), math.radians(limits[1])]
                if not (limit_rad[0] <= angle <= limit_rad[1]):
                    return None
            
            return joint_angles
        
        return None
    
    def _calculate_pose_distance(self, pose1: Pose6D, pose2: Pose6D) -> float:
        """Calculate Euclidean distance between poses."""
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        dz = pose2.z - pose1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _generate_smooth_trajectory(self, 
                                  waypoints: List[Pose6D],
                                  joint_trajectories: List[List[float]],
                                  timestamps: List[float],
                                  constraints: Constraints) -> Trajectory:
        """Generate smooth trajectory with interpolated points."""
        
        # Create trajectory with original waypoints
        trajectory = Trajectory(
            waypoints=waypoints,
            joint_angles=joint_trajectories,
            timestamps=timestamps
        )
        
        # Add intermediate points for smoother motion
        if len(waypoints) > 1:
            interpolated_waypoints = []
            interpolated_joints = []
            interpolated_times = []
            
            for i in range(len(waypoints) - 1):
                # Add current waypoint
                interpolated_waypoints.append(waypoints[i])
                interpolated_joints.append(joint_trajectories[i])
                interpolated_times.append(timestamps[i])
                
                # Add interpolated points
                num_interp = 5  # Number of interpolation points
                for j in range(1, num_interp):
                    alpha = j / num_interp
                    
                    # Interpolate pose
                    interp_pose = Pose6D(
                        x=waypoints[i].x + alpha * (waypoints[i+1].x - waypoints[i].x),
                        y=waypoints[i].y + alpha * (waypoints[i+1].y - waypoints[i].y),
                        z=waypoints[i].z + alpha * (waypoints[i+1].z - waypoints[i].z),
                        rx=waypoints[i].rx + alpha * (waypoints[i+1].rx - waypoints[i].rx),
                        ry=waypoints[i].ry + alpha * (waypoints[i+1].ry - waypoints[i].ry),
                        rz=waypoints[i].rz + alpha * (waypoints[i+1].rz - waypoints[i].rz)
                    )
                    
                    # Interpolate joint angles
                    interp_joints = []
                    for k in range(len(joint_trajectories[i])):
                        joint_val = (joint_trajectories[i][k] + 
                                   alpha * (joint_trajectories[i+1][k] - joint_trajectories[i][k]))
                        interp_joints.append(joint_val)
                    
                    # Interpolate time
                    interp_time = timestamps[i] + alpha * (timestamps[i+1] - timestamps[i])
                    
                    interpolated_waypoints.append(interp_pose)
                    interpolated_joints.append(interp_joints)
                    interpolated_times.append(interp_time)
            
            # Add final waypoint
            interpolated_waypoints.append(waypoints[-1])
            interpolated_joints.append(joint_trajectories[-1])
            interpolated_times.append(timestamps[-1])
            
            trajectory.waypoints = interpolated_waypoints
            trajectory.joint_angles = interpolated_joints
            trajectory.timestamps = interpolated_times
        
        return trajectory
    
    def _execute_trajectory(self, trajectory: Trajectory, constraints: Constraints) -> bool:
        """
        Execute trajectory with real-time monitoring.
        
        Args:
            trajectory: Planned trajectory to execute
            constraints: Execution constraints
            
        Returns:
            True if execution successful
        """
        try:
            logger.info(f"Executing trajectory with {len(trajectory.waypoints)} waypoints")
            
            # In real implementation, this would:
            # 1. Send trajectory to robot controller
            # 2. Monitor execution progress
            # 3. Check for collisions and force limits
            # 4. Handle execution errors
            
            # Mock execution with timing
            if trajectory.timestamps:
                execution_time = trajectory.duration()
                logger.info(f"Mock trajectory execution: {execution_time:.2f}s")
                
                # Simulate execution delay
                time.sleep(min(execution_time * 0.1, 1.0))  # Quick mock execution
            
            # Update current joint state to final position
            if trajectory.joint_angles:
                self.current_joint_state = trajectory.joint_angles[-1].copy()
            
            return True
            
        except Exception as e:
            logger.error(f"Trajectory execution failed: {e}")
            return False
