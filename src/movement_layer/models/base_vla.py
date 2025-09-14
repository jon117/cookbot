"""
Base Vision-Language-Action Model Interface
Abstract base class for all VLA model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class ActionPrediction:
    """Standardized action prediction from VLA models."""
    position: Dict[str, float]  # {"x": float, "y": float, "z": float}
    orientation: Dict[str, float]  # {"rx": float, "ry": float, "rz": float}
    gripper_width: float
    confidence: float
    approach_vector: Optional[Dict[str, float]] = None
    reasoning: Optional[str] = None

class BaseVLA(ABC):
    """Abstract base for Vision-Language-Action models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLA model with configuration.
        
        Args:
            config: Model configuration including endpoint, parameters, etc.
        """
        self.config = config
        self.model_name = config.get("name", "unknown")
        self.endpoint = config.get("endpoint", "")
        self.image_size = tuple(config.get("image_size", [224, 224]))
        
    @abstractmethod
    def predict_action(self, 
                      image: np.ndarray, 
                      instruction: str, 
                      history: Optional[List[Dict]] = None) -> ActionPrediction:
        """
        Predict action from visual input and instruction.
        
        Args:
            image: RGB image array (H, W, 3)
            instruction: Natural language instruction
            history: Optional action history for context
            
        Returns:
            ActionPrediction with pose and gripper state
        """
        raise NotImplementedError("Subclasses must implement predict_action")
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Raw RGB image
            
        Returns:
            Preprocessed image tensor
        """
        raise NotImplementedError("Subclasses must implement preprocess_image")
    
    def validate_prediction(self, prediction: ActionPrediction) -> bool:
        """
        Validate prediction is within reasonable bounds.
        
        Args:
            prediction: Action prediction to validate
            
        Returns:
            True if prediction is valid
        """
        # Basic validation - can be overridden by subclasses
        pos = prediction.position
        if not all(isinstance(pos.get(k), (int, float)) for k in ['x', 'y', 'z']):
            return False
            
        # Check workspace bounds (configurable)
        workspace = self.config.get("workspace_bounds", {
            "x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.8]
        })
        
        for axis, bounds in workspace.items():
            if not (bounds[0] <= pos.get(axis, 0) <= bounds[1]):
                return False
                
        # Check confidence threshold
        min_confidence = self.config.get("min_confidence", 0.3)
        if prediction.confidence < min_confidence:
            return False
            
        return True
    
    def format_instruction(self, target_object: str, context: str = "") -> str:
        """
        Format instruction for VLA model.
        
        Args:
            target_object: Object to manipulate
            context: Additional context
            
        Returns:
            Formatted instruction string
        """
        base_instruction = f"Grasp the {target_object}"
        if context:
            return f"{base_instruction}. {context}"
        return base_instruction
