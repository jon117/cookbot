"""
Universal API Client for OpenAI-compatible Model Endpoints
Unified client supporting multiple LLM and VLA model providers.
"""

import asyncio
import aiohttp
import json
import logging
import time
import base64
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import io

from .data_types import ChatResponse, VisionResponse

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics for API request tracking."""
    start_time: float
    end_time: float
    tokens_used: int = 0
    cost: float = 0.0
    model: str = ""
    provider: str = ""
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class ModelClient:
    """
    Universal client for OpenAI-compatible API endpoints.
    
    Supports multiple providers with automatic failover, rate limiting,
    request/response logging, and cost tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize client with configuration.
        
        Args:
            config: Configuration with endpoints, keys, and retry policies
        """
        self.config = config
        self.primary_endpoint = config.get("primary_model", {})
        self.fallback_endpoint = config.get("fallback_model", {})
        self.retry_policy = config.get("retry_policy", {})
        
        # Request tracking
        self.request_history: List[RequestMetrics] = []
        self.rate_limiter = self._create_rate_limiter()
        
        # HTTP session configuration
        self.timeout = aiohttp.ClientTimeout(total=config.get("timeout", 30))
        self.connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        
        logger.info(f"ModelClient initialized with primary: {self.primary_endpoint.get('name', 'unknown')}")
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]], 
                            model: Optional[str] = None,
                            **kwargs) -> ChatResponse:
        """
        Standard chat completion with automatic retry and fallback.
        
        Args:
            messages: List of chat messages in OpenAI format
            model: Override model name
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            ChatResponse with content and metadata
        """
        
        # Determine which endpoint to use
        endpoint_config = self.primary_endpoint
        if model and "fallback" in model.lower():
            endpoint_config = self.fallback_endpoint
        
        # Prepare request
        payload = self._prepare_chat_payload(messages, endpoint_config, **kwargs)
        headers = self._prepare_headers(endpoint_config)
        
        # Execute with retry logic
        try:
            response = await self._execute_with_retry(
                endpoint_config["endpoint"],
                payload,
                headers,
                endpoint_config.get("name", "unknown")
            )
            
            return self._parse_chat_response(response, endpoint_config.get("name", ""))
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            if endpoint_config == self.primary_endpoint and self.fallback_endpoint:
                logger.info("Attempting fallback endpoint")
                return await self.chat_completion(messages, model="fallback", **kwargs)
            else:
                # Return error response
                return ChatResponse(
                    content=f"Error: {str(e)}",
                    model=endpoint_config.get("name", ""),
                    finish_reason="error"
                )
    
    async def vision_completion(self, 
                              messages: List[Dict[str, Any]], 
                              images: List[np.ndarray],
                              **kwargs) -> VisionResponse:
        """
        Vision-language completion for VLA models.
        
        Args:
            messages: Chat messages with potential image references
            images: List of image arrays
            **kwargs: Additional parameters
            
        Returns:
            VisionResponse with content and metadata
        """
        
        # Use primary endpoint (assumes it supports vision)
        endpoint_config = self.primary_endpoint
        
        # Encode images
        encoded_images = [self._encode_image(img) for img in images]
        
        # Prepare payload with images
        payload = self._prepare_vision_payload(messages, encoded_images, endpoint_config, **kwargs)
        headers = self._prepare_headers(endpoint_config)
        
        try:
            response = await self._execute_with_retry(
                endpoint_config["endpoint"],
                payload,
                headers,
                endpoint_config.get("name", "vision")
            )
            
            return self._parse_vision_response(response, endpoint_config.get("name", ""))
            
        except Exception as e:
            logger.error(f"Vision completion failed: {e}")
            return VisionResponse(
                content=f"Vision error: {str(e)}",
                model=endpoint_config.get("name", ""),
                finish_reason="error"
            )
    
    async def stream_completion(self, 
                              messages: List[Dict[str, str]], 
                              callback=None) -> AsyncIterator[str]:
        """
        Streaming chat completion for real-time responses.
        
        Args:
            messages: Chat messages
            callback: Optional callback for each chunk
            
        Yields:
            String chunks as they arrive
        """
        endpoint_config = self.primary_endpoint
        payload = self._prepare_chat_payload(messages, endpoint_config, stream=True)
        headers = self._prepare_headers(endpoint_config)
        
        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=self.connector
        ) as session:
            async with session.post(
                endpoint_config["endpoint"], 
                json=payload, 
                headers=headers
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"Stream failed with status {response.status}")
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                            
                        try:
                            chunk = json.loads(data)
                            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                if callback:
                                    callback(content)
                                yield content
                        except json.JSONDecodeError:
                            continue
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        if not self.request_history:
            return {"total_requests": 0, "total_cost": 0.0, "total_tokens": 0}
        
        total_requests = len(self.request_history)
        total_cost = sum(req.cost for req in self.request_history)
        total_tokens = sum(req.tokens_used for req in self.request_history)
        avg_duration = sum(req.duration for req in self.request_history) / total_requests
        
        return {
            "total_requests": total_requests,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "average_duration": avg_duration,
            "requests_by_model": self._group_by_model()
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.request_history.clear()
    
    # Private methods
    
    async def _execute_with_retry(self, 
                                endpoint: str, 
                                payload: Dict, 
                                headers: Dict,
                                model_name: str) -> Dict[str, Any]:
        """Execute request with retry logic."""
        
        max_attempts = self.retry_policy.get("max_attempts", 3)
        backoff_factor = self.retry_policy.get("backoff_factor", 2)
        
        for attempt in range(max_attempts):
            start_time = time.time()
            
            try:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                async with aiohttp.ClientSession(
                    timeout=self.timeout,
                    connector=self.connector
                ) as session:
                    async with session.post(endpoint, json=payload, headers=headers) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Track metrics
                            end_time = time.time()
                            self._record_request(start_time, end_time, result, model_name)
                            
                            return result
                        
                        elif response.status == 429:  # Rate limited
                            logger.warning(f"Rate limited, attempt {attempt + 1}")
                            await asyncio.sleep(backoff_factor ** attempt)
                            
                        else:
                            error_text = await response.text()
                            logger.error(f"API error {response.status}: {error_text}")
                            if attempt == max_attempts - 1:
                                raise Exception(f"API request failed: {response.status}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, attempt {attempt + 1}")
                if attempt == max_attempts - 1:
                    raise Exception("Request timed out after all retries")
                    
            except Exception as e:
                logger.error(f"Request error: {e}, attempt {attempt + 1}")
                if attempt == max_attempts - 1:
                    raise e
            
            # Exponential backoff
            await asyncio.sleep(backoff_factor ** attempt)
        
        raise Exception("All retry attempts exhausted")
    
    def _prepare_chat_payload(self, 
                             messages: List[Dict[str, str]], 
                             endpoint_config: Dict[str, Any],
                             **kwargs) -> Dict[str, Any]:
        """Prepare chat completion payload."""
        payload = {
            "model": endpoint_config.get("name", "gpt-3.5-turbo"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", endpoint_config.get("max_tokens", 1000)),
            "temperature": kwargs.get("temperature", endpoint_config.get("temperature", 0.7)),
        }
        
        # Add optional parameters
        if "stream" in kwargs:
            payload["stream"] = kwargs["stream"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        
        return payload
    
    def _prepare_vision_payload(self, 
                               messages: List[Dict[str, Any]], 
                               encoded_images: List[str],
                               endpoint_config: Dict[str, Any],
                               **kwargs) -> Dict[str, Any]:
        """Prepare vision completion payload."""
        
        # Add images to the last message
        if encoded_images and messages:
            last_message = messages[-1].copy()
            if "content" in last_message and isinstance(last_message["content"], str):
                # Convert to multi-modal content
                last_message["content"] = [
                    {"type": "text", "text": last_message["content"]}
                ]
                
                # Add images
                for img_b64 in encoded_images:
                    last_message["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
                
                messages = messages[:-1] + [last_message]
        
        return self._prepare_chat_payload(messages, endpoint_config, **kwargs)
    
    def _prepare_headers(self, endpoint_config: Dict[str, Any]) -> Dict[str, str]:
        """Prepare request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CookBot/1.0"
        }
        
        # Add authentication
        api_key = endpoint_config.get("api_key", "")
        if api_key:
            if "anthropic" in endpoint_config.get("endpoint", "").lower():
                headers["x-api-key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        
        return headers
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string."""
        from PIL import Image
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _parse_chat_response(self, response: Dict[str, Any], model: str) -> ChatResponse:
        """Parse chat completion response."""
        try:
            content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason", "stop")
            tokens_used = response.get("usage", {}).get("total_tokens", 0)
            
            return ChatResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                finish_reason=finish_reason
            )
            
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse chat response: {e}")
            return ChatResponse(
                content="Error parsing response",
                model=model,
                finish_reason="error"
            )
    
    def _parse_vision_response(self, response: Dict[str, Any], model: str) -> VisionResponse:
        """Parse vision completion response."""
        try:
            content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason", "stop")
            usage = response.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            image_tokens = usage.get("prompt_tokens", 0) - usage.get("completion_tokens", 0)
            
            return VisionResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                image_tokens=max(0, image_tokens)
            )
            
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse vision response: {e}")
            return VisionResponse(
                content="Error parsing vision response",
                model=model,
                finish_reason="error"
            )
    
    def _record_request(self, start_time: float, end_time: float, 
                       response: Dict[str, Any], model: str) -> None:
        """Record request metrics."""
        tokens_used = response.get("usage", {}).get("total_tokens", 0)
        
        # Simple cost estimation (would be replaced with actual pricing)
        cost = tokens_used * 0.001  # $0.001 per 1K tokens estimate
        
        metrics = RequestMetrics(
            start_time=start_time,
            end_time=end_time,
            tokens_used=tokens_used,
            cost=cost,
            model=model,
            provider=self._get_provider_name(model)
        )
        
        self.request_history.append(metrics)
        
        # Keep only recent history
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-500:]
    
    def _get_provider_name(self, model: str) -> str:
        """Get provider name from model."""
        model_lower = model.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            return "OpenAI"
        elif "claude" in model_lower:
            return "Anthropic"
        elif "qwen" in model_lower:
            return "Qwen"
        elif "llama" in model_lower:
            return "Meta"
        else:
            return "Unknown"
    
    def _group_by_model(self) -> Dict[str, int]:
        """Group request count by model."""
        counts = {}
        for req in self.request_history:
            counts[req.model] = counts.get(req.model, 0) + 1
        return counts
    
    def _create_rate_limiter(self) -> 'RateLimiter':
        """Create rate limiter for API requests."""
        return RateLimiter(
            requests_per_minute=self.config.get("rate_limit", 60),
            burst_size=self.config.get("burst_size", 10)
        )

class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a rate limit token."""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Add tokens based on time passed
            self.tokens = min(
                self.burst_size,
                self.tokens + time_passed * (self.requests_per_minute / 60.0)
            )
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
            else:
                # Wait until we have a token
                wait_time = (1.0 - self.tokens) / (self.requests_per_minute / 60.0)
                await asyncio.sleep(wait_time)
                self.tokens = 0.0