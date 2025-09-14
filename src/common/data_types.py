"""
Common Data Types and Structures
Shared data structures used across all components of the kitchen robot system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import numpy as np

# Core geometric types

@dataclass
class Pose6D:
    """6-DOF pose representation."""
    x: float
    y: float  
    z: float
    rx: float  # rotation around x (roll)
    ry: float  # rotation around y (pitch)
    rz: float  # rotation around z (yaw)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        import math
        
        # Convert angles to radians if needed
        rx_rad = math.radians(self.rx) if abs(self.rx) > math.pi else self.rx
        ry_rad = math.radians(self.ry) if abs(self.ry) > math.pi else self.ry
        rz_rad = math.radians(self.rz) if abs(self.rz) > math.pi else self.rz
        
        # Create rotation matrices
        cos_rx, sin_rx = math.cos(rx_rad), math.sin(rx_rad)
        cos_ry, sin_ry = math.cos(ry_rad), math.sin(ry_rad)
        cos_rz, sin_rz = math.cos(rz_rad), math.sin(rz_rad)
        
        # Combined rotation matrix (ZYX order)
        R = np.array([
            [cos_ry*cos_rz, -cos_ry*sin_rz, sin_ry],
            [cos_rx*sin_rz + sin_rx*sin_ry*cos_rz, cos_rx*cos_rz - sin_rx*sin_ry*sin_rz, -sin_rx*cos_ry],
            [sin_rx*sin_rz - cos_rx*sin_ry*cos_rz, sin_rx*cos_rz + cos_rx*sin_ry*sin_rz, cos_rx*cos_ry]
        ])
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.x, self.y, self.z]
        
        return T
    
    def to_ros_pose(self) -> Dict[str, Any]:
        """Convert to ROS2 geometry_msgs/Pose format."""
        import math
        
        # Convert Euler angles to quaternion
        rx_rad = math.radians(self.rx) if abs(self.rx) > math.pi else self.rx
        ry_rad = math.radians(self.ry) if abs(self.ry) > math.pi else self.ry
        rz_rad = math.radians(self.rz) if abs(self.rz) > math.pi else self.rz
        
        # Quaternion from Euler angles (ZYX order)
        cy = math.cos(rz_rad * 0.5)
        sy = math.sin(rz_rad * 0.5)
        cp = math.cos(ry_rad * 0.5)
        sp = math.sin(ry_rad * 0.5)
        cr = math.cos(rx_rad * 0.5)
        sr = math.sin(rx_rad * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return {
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw}
        }

@dataclass
class GraspPose(Pose6D):
    """Extended pose for grasping with gripper information."""
    gripper_width: float = 0.05  # Gripper opening width in meters
    confidence: float = 1.0      # Confidence score [0, 1]
    approach_vector: Optional[Dict[str, float]] = None  # Pre-grasp approach direction
    quality_score: float = 0.0   # Grasp quality assessment
    
    def __post_init__(self):
        """Validate grasp pose parameters."""
        if not 0 <= self.confidence <= 1:
            self.confidence = max(0, min(1, self.confidence))
        if not 0 <= self.gripper_width <= 0.1:  # Reasonable gripper limits
            self.gripper_width = max(0, min(0.1, self.gripper_width))

# Planning and task types

class TaskAction(Enum):
    """Available task actions."""
    PICK_AND_PLACE = "pick_and_place"
    OPERATE_APPLIANCE = "operate_appliance"
    WAIT = "wait"
    CHECK_STATUS = "check_status"
    MOVE_TO_POSITION = "move_to_position"

@dataclass
class TaskStep:
    """Individual task step from planning layer."""
    id: str
    action: TaskAction
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_time: float = 0.0
    priority: int = 1
    status: str = "pending"  # pending, executing, completed, failed
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Ensure action is TaskAction enum."""
        if isinstance(self.action, str):
            self.action = TaskAction(self.action)

@dataclass
class ValidationResult:
    """Result of task plan validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)

# Camera and perception types

@dataclass
class CameraObservation:
    """Camera data with calibration info."""
    rgb: np.ndarray  # (H, W, 3) RGB image
    depth: Optional[np.ndarray] = None  # (H, W) depth map
    camera_matrix: Optional[np.ndarray] = None  # (3, 3) intrinsic matrix
    distortion: Optional[np.ndarray] = None     # Distortion coefficients
    timestamp: float = 0.0
    camera_id: str = "default"
    
    def __post_init__(self):
        """Validate camera data dimensions."""
        if self.rgb is not None and len(self.rgb.shape) != 3:
            raise ValueError("RGB image must be 3D array (H, W, 3)")
        if self.depth is not None and len(self.depth.shape) != 2:
            raise ValueError("Depth image must be 2D array (H, W)")

@dataclass
class DetectedObject:
    """Object detected in scene with pose and properties."""
    name: str
    class_id: int
    confidence: float
    bounding_box: Dict[str, float]  # {"x": float, "y": float, "width": float, "height": float}
    pose: Optional[Pose6D] = None
    dimensions: Optional[Dict[str, float]] = None  # {"length": float, "width": float, "height": float}
    properties: Dict[str, Any] = field(default_factory=dict)  # Material, color, etc.
    
    def __post_init__(self):
        """Validate detection parameters."""
        if not 0 <= self.confidence <= 1:
            self.confidence = max(0, min(1, self.confidence))

# Appliance and execution types

class ApplianceStatus(Enum):
    """Appliance status states."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

@dataclass
class OperationResult:
    """Result of appliance operation."""
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_code: Optional[str] = None

@dataclass
class ExecutionResult:
    """Result of motion execution."""
    success: bool
    message: str = ""
    final_pose: Optional[Pose6D] = None
    trajectory_points: List[Pose6D] = field(default_factory=list)
    execution_time: float = 0.0
    error_code: Optional[str] = None

@dataclass
class Trajectory:
    """Robot trajectory with timing."""
    waypoints: List[Pose6D]
    joint_angles: List[List[float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    velocities: List[List[float]] = field(default_factory=list)
    accelerations: List[List[float]] = field(default_factory=list)
    
    def duration(self) -> float:
        """Get total trajectory duration."""
        return max(self.timestamps) - min(self.timestamps) if self.timestamps else 0.0

@dataclass
class Constraints:
    """Motion constraints for trajectory planning."""
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.1
    avoid_collisions: bool = True
    keep_orientation: bool = False
    custom_constraints: Dict[str, Any] = field(default_factory=dict)

# Configuration types

@dataclass
class WorkspaceBounds:
    """Robot workspace boundaries."""
    x_min: float = -0.5
    x_max: float = 0.5
    y_min: float = -0.5
    y_max: float = 0.5
    z_min: float = 0.0
    z_max: float = 0.8
    
    def contains(self, pose: Pose6D) -> bool:
        """Check if pose is within workspace bounds."""
        return (self.x_min <= pose.x <= self.x_max and
                self.y_min <= pose.y <= self.y_max and
                self.z_min <= pose.z <= self.z_max)

# Response types for API interactions

@dataclass
class ChatResponse:
    """Response from chat completion API."""
    content: str
    model: str = ""
    tokens_used: int = 0
    finish_reason: str = ""
    cost: float = 0.0

@dataclass
class VisionResponse:
    """Response from vision-language API."""
    content: str
    model: str = ""
    tokens_used: int = 0
    finish_reason: str = ""
    cost: float = 0.0
    image_tokens: int = 0