"""
OpenVLA Model Implementation
Implements the BaseVLA interface for OpenVLA models.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import requests
from PIL import Image
import io
import base64
import json_numpy

from .base_vla import BaseVLA, ActionPrediction

# Patch json to handle numpy arrays
json_numpy.patch()

logger = logging.getLogger(__name__)

class OpenVLAModel(BaseVLA):
    """OpenVLA model implementation with HTTP API interface."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenVLA model with configuration."""
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
    def predict_action(self, 
                      image: np.ndarray, 
                      instruction: str, 
                      history: Optional[List[Dict]] = None) -> ActionPrediction:
        """
        Predict action using OpenVLA model.
        
        Args:
            image: RGB image array (H, W, 3)
            instruction: Natural language instruction
            history: Optional action history
            
        Returns:
            ActionPrediction with pose and gripper state
        """
        try:
            # Run async prediction in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self._async_predict_action(image, instruction, history))
                    )
                    return future.result(timeout=self.timeout)
            else:
                return loop.run_until_complete(
                    self._async_predict_action(image, instruction, history)
                )
        except Exception as e:
            logger.error(f"VLA prediction failed: {e}")
            return self._fallback_prediction(instruction)
    
    async def _async_predict_action(self, 
                                   image: np.ndarray, 
                                   instruction: str, 
                                   history: Optional[List[Dict]] = None) -> ActionPrediction:
        """Async implementation of action prediction."""
        
        # Preprocess image to match OpenVLA expected format
        processed_image = self.preprocess_image(image)
        
        # Format request payload for OpenVLA /act endpoint
        # OpenVLA expects: {'image': np.ndarray, 'instruction': str}
        payload = {
            "image": processed_image,  # Send numpy array directly - json_numpy handles serialization
            "instruction": instruction
        }
        
        if history:
            payload["history"] = history
            
        headers = {
            "Content-Type": "application/json"
        }
        
        # Note: OpenVLA /act endpoint doesn't use Bearer auth, so no Authorization header
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                # Use requests instead of aiohttp for json_numpy compatibility
                import requests
                
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle OpenVLA error responses
                    if result == "error" or (isinstance(result, str) and "error" in result.lower()):
                        logger.warning(f"OpenVLA returned error: {result}")
                        
                        # For development, return a fallback action when OpenVLA returns error
                        logger.info("Using fallback action due to OpenVLA error")
                        return ActionPrediction(
                            position={"x": 0.3, "y": 0.1, "z": 0.15},
                            orientation={"rx": 0.0, "ry": 0.0, "rz": 0.0},
                            gripper_width=0.05,
                            confidence=0.5,
                            metadata={"source": "fallback", "reason": "openvla_error"}
                        )
                        
                    return self._parse_response(result)
                else:
                    logger.warning(f"OpenVLA API returned status {response.status_code}, attempt {attempt + 1}")
                    if attempt == self.max_retries - 1:
                        raise Exception(f"API request failed with status {response.status_code}")
                            
            except requests.exceptions.Timeout:
                logger.warning(f"OpenVLA API timeout, attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise Exception("API request timed out")
            except Exception as e:
                logger.warning(f"OpenVLA API error: {e}, attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise e
                    
            # Exponential backoff
            import time
            time.sleep(2 ** attempt)
            
        raise Exception("All retry attempts failed")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OpenVLA model.
        
        Args:
            image: Raw RGB image (H, W, 3)
            
        Returns:
            Preprocessed image (256, 256, 3) as uint8
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Resize to OpenVLA expected size (256x256)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
        
        return np.array(pil_image, dtype=np.uint8)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return image_b64
    
    def _parse_response(self, response: Dict[str, Any]) -> ActionPrediction:
        """
        Parse OpenVLA model response into ActionPrediction.
        
        Args:
            response: Raw API response from OpenVLA /act endpoint
            
        Returns:
            Parsed ActionPrediction
        """
        try:
            # OpenVLA /act endpoint returns action directly
            # Expected format: {"action": [x, y, z, rx, ry, rz, gripper_action]}
            # or similar format depending on the actual OpenVLA implementation
            
            if "action" in response:
                action_data = response["action"]
                
                # Handle different possible formats
                if isinstance(action_data, list) and len(action_data) >= 6:
                    # Format: [x, y, z, rx, ry, rz, gripper_width]
                    position = {"x": float(action_data[0]), "y": float(action_data[1]), "z": float(action_data[2])}
                    orientation = {"rx": float(action_data[3]), "ry": float(action_data[4]), "rz": float(action_data[5])}
                    gripper_width = float(action_data[6]) if len(action_data) > 6 else 0.05
                    confidence = float(action_data[7]) if len(action_data) > 7 else 0.8
                    
                elif isinstance(action_data, dict):
                    # Format: {"position": {...}, "orientation": {...}, ...}
                    position = action_data.get("position", {"x": 0.3, "y": 0.1, "z": 0.15})
                    orientation = action_data.get("orientation", {"rx": 0.0, "ry": 0.0, "rz": 0.0})
                    gripper_width = action_data.get("gripper_width", 0.05)
                    confidence = action_data.get("confidence", 0.8)
                    
                else:
                    # Fallback: treat as raw action values
                    logger.warning(f"Unexpected action format: {action_data}")
                    return self._fallback_prediction("unknown format")
                    
            else:
                # Check for other possible response formats
                logger.warning(f"No 'action' key in response: {list(response.keys())}")
                return self._fallback_prediction("no action key")
            
            # Create prediction
            prediction = ActionPrediction(
                position=position,
                orientation=orientation,
                gripper_width=gripper_width,
                confidence=confidence,
                reasoning=response.get("reasoning", "OpenVLA prediction")
            )
            
            # Validate prediction
            if not self.validate_prediction(prediction):
                logger.warning("OpenVLA prediction failed validation, using fallback")
                return self._fallback_prediction("validation failed")
                
            return prediction
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse OpenVLA response: {e}")
            logger.debug(f"Response content: {response}")
            return self._fallback_prediction("parse error")
    
    def _fallback_prediction(self, instruction: str) -> ActionPrediction:
        """
        Generate fallback prediction when VLA fails.
        
        Args:
            instruction: Original instruction
            
        Returns:
            Safe fallback ActionPrediction
        """
        # Simple heuristic-based fallback
        if "carrot" in instruction.lower():
            return ActionPrediction(
                position={"x": 0.3, "y": 0.1, "z": 0.15},
                orientation={"rx": 0.0, "ry": 0.0, "rz": 0.0},
                gripper_width=0.04,
                confidence=0.3,
                reasoning="Fallback heuristic for carrot"
            )
        else:
            return ActionPrediction(
                position={"x": 0.2, "y": 0.0, "z": 0.1},
                orientation={"rx": 0.0, "ry": 0.0, "rz": 0.0},
                gripper_width=0.05,
                confidence=0.2,
                reasoning="Generic fallback prediction"
            )
