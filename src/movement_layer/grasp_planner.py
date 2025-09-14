"""
Grasp Planner - VLA-powered manipulation planning
Generates grasping poses from visual input and task context.
"""

from typing import Dict, Any, List, Optional
import logging
import yaml
import os
import numpy as np

# Import our common types
from ..common.data_types import (
    CameraObservation, DetectedObject, GraspPose, 
    WorkspaceBounds, ValidationResult
)
from .models.base_vla import BaseVLA, ActionPrediction
from .models.openvla_model import OpenVLAModel

logger = logging.getLogger(__name__)

class GraspPlanner:
    """
    VLA-powered grasp planning with camera input processing.
    
    Implements Vision-Language-Action model integration for grasp planning 
    and spatial reasoning using camera data and natural language instructions.
    """
    
    def __init__(self, config_path: str = "config/models/vla_config.yaml"):
        """
        Initialize VLA model interface and preprocessing pipeline.
        
        Args:
            config_path: Path to VLA configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize VLA model
        self.vla_model = self._initialize_vla_model()
        
        # Set up workspace constraints
        self.workspace_bounds = WorkspaceBounds(**self.config.get("workspace_bounds", {}))
        
        # Camera calibration parameters
        self.camera_config = self.config.get("camera", {})
        
        # Detection tracking
        self.detected_objects = []
        self.scene_history = []
        
        logger.info(f"GraspPlanner initialized with model: {self.vla_model.model_name}")
    
    def plan_grasp(self, 
                   camera_data: CameraObservation, 
                   target_object: str, 
                   context: str = "") -> GraspPose:
        """
        Plan a grasp for the target object using VLA model.
        
        Args:
            camera_data: RGB-D camera observations with calibration
            target_object: Name of object to grasp
            context: Additional task context
            
        Returns:
            GraspPose with position, orientation, and gripper state
        """
        try:
            # Validate camera data
            if not self._validate_camera_data(camera_data):
                logger.error("Invalid camera data provided")
                return self._fallback_grasp(target_object)
            
            # Preprocess camera data
            processed_image = self._preprocess_camera_data(camera_data)
            
            # Generate instruction for VLA model
            instruction = self._format_grasp_instruction(target_object, context)
            
            # Get action prediction from VLA model
            action_prediction = self.vla_model.predict_action(
                image=processed_image,
                instruction=instruction,
                history=self._get_action_history()
            )
            
            # Convert to GraspPose
            grasp_pose = self._convert_to_grasp_pose(action_prediction)
            
            # Validate grasp within workspace
            validation_result = self.validate_grasp_pose(grasp_pose)
            if not validation_result.is_valid:
                logger.warning(f"Grasp validation failed: {validation_result.errors}")
                return self._adjust_or_fallback_grasp(grasp_pose, target_object)
            
            # Log successful planning
            logger.info(f"Planned grasp for {target_object}: "
                       f"pos=({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f}) "
                       f"conf={grasp_pose.confidence:.3f}")
            
            return grasp_pose
            
        except Exception as e:
            logger.error(f"Grasp planning failed: {e}")
            return self._fallback_grasp(target_object)
    
    def analyze_scene(self, camera_data: CameraObservation) -> List[DetectedObject]:
        """
        Analyze scene to detect and locate objects.
        
        Args:
            camera_data: RGB-D camera observations
            
        Returns:
            List of detected objects with poses and properties
        """
        try:
            # Update detected objects list
            self.detected_objects = []
            
            if not self._validate_camera_data(camera_data):
                return self.detected_objects
            
            # Simple object detection (placeholder for actual detection)
            # In a real implementation, this would use YOLO, DINO, or similar
            objects = self._detect_objects_simple(camera_data)
            
            # Estimate 3D poses from depth data if available
            if camera_data.depth is not None:
                objects = self._estimate_3d_poses(objects, camera_data)
            
            self.detected_objects = objects
            
            # Update scene history for temporal consistency
            self._update_scene_history(objects, camera_data.timestamp)
            
            logger.info(f"Detected {len(objects)} objects in scene")
            return objects
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return []
    
    def validate_grasp_pose(self, grasp_pose: GraspPose) -> ValidationResult:
        """
        Validate grasp pose against workspace constraints and safety rules.
        
        Args:
            grasp_pose: Proposed grasp pose
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check workspace bounds
        if not self.workspace_bounds.contains(grasp_pose):
            errors.append(f"Grasp pose outside workspace bounds")
        
        # Check reachability (simplified)
        if grasp_pose.z < 0.05:  # Too close to table
            errors.append("Grasp pose too close to surface")
        
        # Check gripper limits
        max_gripper_width = self.config.get("gripper", {}).get("max_width", 0.1)
        if grasp_pose.gripper_width > max_gripper_width:
            warnings.append(f"Gripper width {grasp_pose.gripper_width} exceeds maximum {max_gripper_width}")
        
        # Check confidence threshold
        min_confidence = self.config.get("min_confidence", 0.3)
        if grasp_pose.confidence < min_confidence:
            warnings.append(f"Low confidence grasp: {grasp_pose.confidence}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggested_fixes=[]
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load VLA configuration from file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default VLA configuration."""
        return {
            "model": {
                "name": "openvla-7b",
                "endpoint": "http://localhost:8002/v1/completions",
                "image_size": [224, 224],
                "max_tokens": 512,
                "temperature": 0.1,
                "timeout": 30
            },
            "workspace_bounds": {
                "x_min": -0.5, "x_max": 0.5,
                "y_min": -0.5, "y_max": 0.5,
                "z_min": 0.0, "z_max": 0.8
            },
            "camera": {
                "default_intrinsics": [600, 600, 320, 240]  # fx, fy, cx, cy
            },
            "gripper": {
                "max_width": 0.1,
                "default_width": 0.05
            },
            "min_confidence": 0.3
        }
    
    def _initialize_vla_model(self) -> BaseVLA:
        """Initialize VLA model based on configuration."""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "openvla-7b")
        
        if "openvla" in model_name.lower():
            return OpenVLAModel(model_config)
        else:
            logger.warning(f"Unknown model type: {model_name}, using OpenVLA")
            return OpenVLAModel(model_config)
    
    def _validate_camera_data(self, camera_data: CameraObservation) -> bool:
        """Validate camera data has required components."""
        if camera_data.rgb is None:
            return False
        if len(camera_data.rgb.shape) != 3:
            return False
        return True
    
    def _preprocess_camera_data(self, camera_data: CameraObservation) -> np.ndarray:
        """Preprocess camera data for VLA model input."""
        return self.vla_model.preprocess_image(camera_data.rgb)
    
    def _format_grasp_instruction(self, target_object: str, context: str) -> str:
        """Format instruction for VLA model."""
        return self.vla_model.format_instruction(target_object, context)
    
    def _get_action_history(self) -> List[Dict]:
        """Get recent action history for context."""
        # Placeholder - would maintain action history in real implementation
        return []
    
    def _convert_to_grasp_pose(self, action_prediction: ActionPrediction) -> GraspPose:
        """Convert VLA action prediction to GraspPose."""
        return GraspPose(
            x=action_prediction.position["x"],
            y=action_prediction.position["y"],
            z=action_prediction.position["z"],
            rx=action_prediction.orientation["rx"],
            ry=action_prediction.orientation["ry"],
            rz=action_prediction.orientation["rz"],
            gripper_width=action_prediction.gripper_width,
            confidence=action_prediction.confidence,
            approach_vector=action_prediction.approach_vector
        )
    
    def _fallback_grasp(self, target_object: str) -> GraspPose:
        """Generate fallback grasp when VLA fails."""
        # Object-specific heuristics
        fallback_poses = {
            "carrot": GraspPose(x=0.3, y=0.1, z=0.15, rx=0, ry=0, rz=0, 
                              gripper_width=0.04, confidence=0.3),
            "apple": GraspPose(x=0.25, y=0.0, z=0.12, rx=0, ry=0, rz=0,
                             gripper_width=0.06, confidence=0.3),
            "default": GraspPose(x=0.2, y=0.0, z=0.1, rx=0, ry=0, rz=0,
                               gripper_width=0.05, confidence=0.2)
        }
        
        return fallback_poses.get(target_object.lower(), fallback_poses["default"])
    
    def _adjust_or_fallback_grasp(self, grasp_pose: GraspPose, target_object: str) -> GraspPose:
        """Adjust invalid grasp or return fallback."""
        # Try to adjust pose to workspace bounds
        adjusted_pose = GraspPose(
            x=max(self.workspace_bounds.x_min, min(self.workspace_bounds.x_max, grasp_pose.x)),
            y=max(self.workspace_bounds.y_min, min(self.workspace_bounds.y_max, grasp_pose.y)),
            z=max(self.workspace_bounds.z_min, min(self.workspace_bounds.z_max, grasp_pose.z)),
            rx=grasp_pose.rx,
            ry=grasp_pose.ry,
            rz=grasp_pose.rz,
            gripper_width=grasp_pose.gripper_width,
            confidence=grasp_pose.confidence * 0.8  # Lower confidence for adjusted pose
        )
        
        # Validate adjusted pose
        if self.validate_grasp_pose(adjusted_pose).is_valid:
            return adjusted_pose
        else:
            return self._fallback_grasp(target_object)
    
    def _detect_objects_simple(self, camera_data: CameraObservation) -> List[DetectedObject]:
        """Simple object detection placeholder."""
        # This would be replaced with actual object detection model
        # For now, return mock detections based on common kitchen objects
        return [
            DetectedObject(
                name="carrot",
                class_id=1,
                confidence=0.8,
                bounding_box={"x": 100, "y": 150, "width": 80, "height": 200}
            )
        ]
    
    def _estimate_3d_poses(self, objects: List[DetectedObject], 
                          camera_data: CameraObservation) -> List[DetectedObject]:
        """Estimate 3D poses from depth data."""
        # Placeholder for 3D pose estimation
        # Would use camera intrinsics and depth data for real 3D localization
        for obj in objects:
            if camera_data.depth is not None:
                # Mock 3D pose estimation
                obj.pose = GraspPose(x=0.3, y=0.1, z=0.15, rx=0, ry=0, rz=0)
        
        return objects
    
    def _update_scene_history(self, objects: List[DetectedObject], timestamp: float = 0.0) -> None:
        """Update scene history for temporal consistency."""
        self.scene_history.append({
            "timestamp": timestamp,
            "objects": objects
        })
        
        # Keep only recent history
        max_history = 10
        if len(self.scene_history) > max_history:
            self.scene_history = self.scene_history[-max_history:]
