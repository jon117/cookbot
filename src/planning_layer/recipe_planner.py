"""
LLM-powered recipe planner that converts natural language instructions 
into structured task sequences.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from ..common.data_types import TaskStep, TaskAction, ValidationResult, ChatResponse
from ..common.api_client import ModelClient


logger = logging.getLogger(__name__)


class RecipePlanner:
    """
    LLM-powered recipe planner that decomposes cooking instructions.
    
    Converts natural language cooking instructions into structured task sequences
    using OpenAI-compatible LLM APIs with robust error handling and validation.
    """
    
    def __init__(self, config_path: str = "config/models/llm_config.yaml"):
        """
        Initialize the recipe planner.
        
        Args:
            config_path: Path to LLM configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.client = ModelClient(self.config)
        self.system_prompt = self._load_system_prompt()
        
        # Available appliances and their capabilities
        self.available_appliances = {
            "slicer": {
                "capabilities": ["slice", "dice", "julienne"],
                "modes": ["thin", "medium", "thick"],
                "objects": ["carrot", "onion", "potato", "cucumber", "tomato"]
            },
            "griddle": {
                "capabilities": ["fry", "sear", "grill"],
                "modes": ["low", "medium", "high"],
                "temperature_range": [100, 250]  # Celsius
            },
            "air_fryer": {
                "capabilities": ["fry", "bake", "roast"],
                "modes": ["air_fry", "bake", "roast", "reheat"],
                "temperature_range": [80, 200]  # Celsius
            },
            "crock_pot": {
                "capabilities": ["slow_cook", "steam", "warm"],
                "modes": ["low", "high", "warm"],
                "max_time": 28800  # 8 hours in seconds
            },
            "rice_cooker": {
                "capabilities": ["cook_rice", "steam", "warm"],
                "modes": ["white_rice", "brown_rice", "steam", "warm"],
                "water_ratios": {"white_rice": 1.5, "brown_rice": 2.0}
            }
        }
        
        # Available actions and their parameters
        self.available_actions = {
            TaskAction.PICK_AND_PLACE: {
                "required_params": ["object", "source", "target"],
                "optional_params": ["approach", "force_limit", "precision"]
            },
            TaskAction.OPERATE_APPLIANCE: {
                "required_params": ["appliance"],
                "optional_params": ["mode", "temperature", "time", "count", "water_ratio"]
            },
            TaskAction.WAIT: {
                "required_params": ["duration"],
                "optional_params": ["reason", "condition"]
            },
            TaskAction.CHECK_STATUS: {
                "required_params": ["target"],
                "optional_params": ["expected_status", "timeout"]
            }
        }
        
        logger.info("RecipePlanner initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LLM configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "primary_model": {
                "name": "gpt-3.5-turbo",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "max_tokens": 1000,
                "temperature": 0.3
            },
            "fallback_model": {
                "name": "gpt-3.5-turbo",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "api_key": os.getenv("OPENAI_API_KEY", "")
            },
            "retry_policy": {
                "max_attempts": 3,
                "backoff_factor": 2,
                "timeout": 30
            }
        }
    
    def _load_system_prompt(self) -> str:
        """Load system prompt template."""
        prompt_path = Path(__file__).parent / "prompts" / "base_planner.txt"
        try:
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"System prompt not found: {prompt_path}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a kitchen robot planner. Convert cooking instructions into structured task sequences.
        
Output Format (JSON):
{
  "tasks": [
    {
      "id": "unique_id", 
      "action": "pick_and_place|operate_appliance|wait|check_status",
      "parameters": {...},
      "dependencies": ["task_id1"],
      "estimated_time": 30
    }
  ]
}"""
    
    async def plan_task(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> List[TaskStep]:
        """
        Convert natural language instruction into structured task sequence.
        
        Args:
            instruction: Natural language cooking instruction
            context: Optional context including available objects, appliance states, etc.
            
        Returns:
            List of TaskStep objects representing the planned sequence
        """
        try:
            logger.info(f"Planning task for instruction: '{instruction}'")
            
            # Build prompt with context
            prompt = self._build_prompt(instruction, context)
            
            # Get LLM response
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat_completion(
                messages=messages,
                max_tokens=self.config["primary_model"].get("max_tokens", 1000),
                temperature=self.config["primary_model"].get("temperature", 0.3)
            )
            
            # Parse response into task sequence
            task_sequence = self._parse_response(response.content)
            
            # Validate the planned sequence
            validation = self.validate_plan(task_sequence)
            if not validation.is_valid:
                logger.warning(f"Plan validation failed: {validation.errors}")
                # Try to fix common issues or return empty list
                return []
            
            logger.info(f"Successfully planned {len(task_sequence)} tasks")
            return task_sequence
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            return []
    
    def _build_prompt(self, instruction: str, context: Optional[Dict[str, Any]]) -> str:
        """Build the user prompt with instruction and context."""
        prompt_parts = [f"Instruction: {instruction}"]
        
        if context:
            # Add available objects
            if "available_objects" in context:
                objects = ", ".join(context["available_objects"])
                prompt_parts.append(f"Available objects: {objects}")
            
            # Add appliance states
            if "appliance_states" in context:
                states = []
                for appliance, state in context["appliance_states"].items():
                    states.append(f"{appliance}: {state}")
                prompt_parts.append(f"Appliance states: {', '.join(states)}")
            
            # Add workspace layout
            if "workspace_layout" in context:
                layout = context["workspace_layout"]
                prompt_parts.append(f"Workspace layout: {layout}")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, response_content: str) -> List[TaskStep]:
        """Parse LLM response into TaskStep objects."""
        try:
            # Try to extract JSON from response
            content = response_content.strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```"):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])  # Remove first and last lines
            
            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    data = json.loads(content[start_idx:end_idx])
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Convert to TaskStep objects
            task_steps = []
            tasks_data = data.get("tasks", [])
            
            for task_data in tasks_data:
                try:
                    # Convert action string to TaskAction enum
                    action_str = task_data.get("action", "")
                    action = TaskAction(action_str)
                    
                    task_step = TaskStep(
                        id=task_data.get("id", f"task_{len(task_steps)}"),
                        action=action,
                        parameters=task_data.get("parameters", {}),
                        dependencies=task_data.get("dependencies", []),
                        estimated_time=task_data.get("estimated_time", 0.0),
                        priority=task_data.get("priority", 1)
                    )
                    task_steps.append(task_step)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid task: {task_data}, error: {e}")
                    continue
            
            return task_steps
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response content: {response_content}")
            return []
    
    def validate_plan(self, plan: List[TaskStep]) -> ValidationResult:
        """
        Validate the planned task sequence.
        
        Args:
            plan: List of TaskStep objects to validate
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors = []
        warnings = []
        
        if not plan:
            return ValidationResult(
                is_valid=False,
                errors=["Empty plan generated"],
                confidence=0.0
            )
        
        # Check for unique task IDs
        task_ids = [task.id for task in plan]
        if len(set(task_ids)) != len(task_ids):
            errors.append("Duplicate task IDs found")
        
        # Validate each task
        for task in plan:
            # Validate action parameters
            action_validation = self._validate_task_action(task)
            if not action_validation:
                errors.append(f"Invalid action parameters for task {task.id}")
            
            # Check dependencies exist
            invalid_deps = [dep for dep in task.dependencies if dep not in task_ids]
            if invalid_deps:
                errors.append(f"Task {task.id} has invalid dependencies: {invalid_deps}")
            
            # Check for circular dependencies
            if self._has_circular_dependency(task, plan):
                errors.append(f"Circular dependency detected involving task {task.id}")
            
            # Validate appliance availability
            if task.action == TaskAction.OPERATE_APPLIANCE:
                appliance = task.parameters.get("appliance")
                if appliance not in self.available_appliances:
                    errors.append(f"Unknown appliance: {appliance}")
        
        # Check task ordering makes sense
        ordering_issues = self._validate_task_ordering(plan)
        warnings.extend(ordering_issues)
        
        # Calculate confidence based on validation results
        confidence = 1.0
        if errors:
            confidence *= 0.3  # Major issues
        if warnings:
            confidence *= 0.8  # Minor issues
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def _validate_task_action(self, task: TaskStep) -> bool:
        """Validate that a task has correct parameters for its action."""
        action_spec = self.available_actions.get(task.action)
        if not action_spec:
            return False
        
        # Check required parameters
        required_params = action_spec.get("required_params", [])
        missing_params = [p for p in required_params if p not in task.parameters]
        
        return len(missing_params) == 0
    
    def _has_circular_dependency(self, task: TaskStep, plan: List[TaskStep]) -> bool:
        """Check if a task has circular dependencies."""
        def check_deps(task_id: str, visited: set) -> bool:
            if task_id in visited:
                return True
            
            visited.add(task_id)
            
            # Find the task
            current_task = next((t for t in plan if t.id == task_id), None)
            if not current_task:
                return False
            
            # Check all dependencies
            for dep_id in current_task.dependencies:
                if check_deps(dep_id, visited.copy()):
                    return True
            
            return False
        
        return check_deps(task.id, set())
    
    def _validate_task_ordering(self, plan: List[TaskStep]) -> List[str]:
        """Validate logical ordering of tasks and return warnings."""
        warnings = []
        
        # Check for pick_and_place before operate_appliance patterns
        for i, task in enumerate(plan):
            if task.action == TaskAction.OPERATE_APPLIANCE:
                appliance = task.parameters.get("appliance")
                
                # Look for corresponding pick_and_place tasks
                placement_tasks = [
                    t for t in plan[:i] 
                    if t.action == TaskAction.PICK_AND_PLACE 
                    and t.parameters.get("target", "").startswith(appliance)
                ]
                
                if not placement_tasks and not task.dependencies:
                    warnings.append(
                        f"Task {task.id} operates {appliance} but no object placement found"
                    )
        
        return warnings
    
    async def replan_task(self, 
                         original_plan: List[TaskStep], 
                         failure_point: str, 
                         error_context: str) -> List[TaskStep]:
        """
        Replan a task sequence after a failure.
        
        Args:
            original_plan: The original plan that failed
            failure_point: ID of the task that failed
            error_context: Description of what went wrong
            
        Returns:
            New task sequence that attempts to handle the failure
        """
        try:
            # Build replanning prompt
            prompt = self._build_replan_prompt(original_plan, failure_point, error_context)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat_completion(messages=messages)
            new_plan = self._parse_response(response.content)
            
            # Validate the new plan
            validation = self.validate_plan(new_plan)
            if validation.is_valid:
                logger.info(f"Successfully replanned with {len(new_plan)} tasks")
                return new_plan
            else:
                logger.warning(f"Replanning validation failed: {validation.errors}")
                return []
                
        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            return []
    
    def _build_replan_prompt(self, 
                           original_plan: List[TaskStep], 
                           failure_point: str, 
                           error_context: str) -> str:
        """Build prompt for replanning after failure."""
        # Convert original plan to readable format
        plan_str = []
        for task in original_plan:
            plan_str.append(f"- {task.id}: {task.action.value} {task.parameters}")
        
        return f"""
The following plan failed at task '{failure_point}':
{chr(10).join(plan_str)}

Error: {error_context}

Please provide a revised plan that addresses this failure. You may:
1. Modify the failed task parameters
2. Add error recovery tasks
3. Reorder tasks if needed
4. Add safety checks

Provide the revised plan in the same JSON format.
"""
    
    async def close(self):
        """Close the planner and cleanup resources."""
        await self.client.close()
        logger.info("RecipePlanner closed")

from typing import List, Dict, Any
import yaml
from pathlib import Path

class RecipePlanner:
    """Plans recipe execution using LLM-based task decomposition."""
    
    def __init__(self, config_path: str = "config/models/llm_config.yaml"):
        self.config = self._load_config(config_path)
        # TODO: Initialize LLM client
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load planner configuration."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {"model": "qwen-2.5", "endpoint": "http://localhost:8001/v1"}
    
    def plan_task(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Plan a cooking task from natural language instruction.
        
        Args:
            instruction: Natural language cooking instruction
            
        Returns:
            List of structured task steps
        """
        # TODO: Implement LLM-based planning
        # For now, return a placeholder
        if "carrot" in instruction.lower() and "slicer" in instruction.lower():
            return [
                {
                    "action": "pick_and_place",
                    "object": "carrot", 
                    "source": "counter",
                    "target": "slicer",
                    "priority": 1
                }
            ]
        
        return []
    
    def load_recipe(self, recipe_name: str) -> Dict[str, Any]:
        """Load recipe configuration from file."""
        recipe_path = Path(f"config/recipes/{recipe_name}.yaml")
        with open(recipe_path, 'r') as f:
            return yaml.safe_load(f)
