"""
OpenVLA Model Implementation
Implements the BaseVLA interface for OpenVLA models.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import aiohttp
from PIL import Image
import io
import base64

from .base_vla import BaseVLA, ActionPrediction

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
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        image_b64 = self._encode_image(processed_image)
        
        # Format request payload
        payload = {
            "model": self.model_name,
            "instruction": instruction,
            "image": image_b64,
            "max_tokens": self.config.get("max_tokens", 512),
            "temperature": self.config.get("temperature", 0.1)
        }
        
        if history:
            payload["history"] = history
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(self.endpoint, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            return self._parse_response(result)
                        else:
                            logger.warning(f"VLA API returned status {response.status}, attempt {attempt + 1}")
                            if attempt == self.max_retries - 1:
                                raise Exception(f"API request failed with status {response.status}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"VLA API timeout, attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise Exception("API request timed out")
            except Exception as e:
                logger.warning(f"VLA API error: {e}, attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise e
                    
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            
        raise Exception("All retry attempts failed")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OpenVLA model.
        
        Args:
            image: Raw RGB image (H, W, 3)
            
        Returns:
            Preprocessed image
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Resize to model input size
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return image_b64
    
    def _parse_response(self, response: Dict[str, Any]) -> ActionPrediction:
        """
        Parse VLA model response into ActionPrediction.
        
        Args:
            response: Raw API response
            
        Returns:
            Parsed ActionPrediction
        """
        try:
            # Extract action from response (format may vary by model)
            if "action" in response:
                action_data = response["action"]
            elif "choices" in response and len(response["choices"]) > 0:
                # OpenAI-style response format
                content = response["choices"][0].get("message", {}).get("content", "{}")
                action_data = json.loads(content)
            else:
                raise ValueError("Unexpected response format")
            
            # Parse pose data
            position = action_data.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
            orientation = action_data.get("orientation", {"rx": 0.0, "ry": 0.0, "rz": 0.0})
            gripper_width = action_data.get("gripper_width", 0.05)
            confidence = action_data.get("confidence", 0.5)
            approach_vector = action_data.get("approach_vector")
            reasoning = action_data.get("reasoning", "")
            
            prediction = ActionPrediction(
                position=position,
                orientation=orientation,
                gripper_width=gripper_width,
                confidence=confidence,
                approach_vector=approach_vector,
                reasoning=reasoning
            )
            
            # Validate prediction
            if not self.validate_prediction(prediction):
                logger.warning("VLA prediction failed validation, using fallback")
                return self._fallback_prediction("grasp object")
                
            return prediction
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse VLA response: {e}")
            return self._fallback_prediction("grasp object")
    
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
