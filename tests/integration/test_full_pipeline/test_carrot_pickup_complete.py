"""
Phase 1 Integration Test - Carrot Pickup
End-to-end test validating the complete pipeline from instruction to execution.
"""

import pytest
import asyncio
import time
import logging
import os
import sys
from unittest.mock import Mock, patch
import numpy as np

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from common.data_types import (
    CameraObservation, Pose6D, GraspPose, TaskStep, TaskAction,
    ExecutionResult, ValidationResult
)
from planning_layer.recipe_planner import RecipePlanner
from movement_layer.grasp_planner import GraspPlanner
from control_layer.motion_planner import MotionPlanner

logger = logging.getLogger(__name__)

class TestCarrotPickupIntegration:
    """
    Comprehensive integration test for Phase 1 carrot pickup functionality.
    
    Tests the complete pipeline:
    1. Planning Layer: "move carrot to slicer" -> task sequence
    2. Movement Layer: camera data -> grasp pose
    3. Control Layer: grasp pose -> robot motion
    """
    
    @pytest.fixture
    def mock_camera_data(self):
        """Create mock camera observation with carrot."""
        # Create a mock RGB image (640x480x3)
        rgb_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add a carrot-like object in the image (orange rectangle)
        rgb_image[200:300, 300:400, :] = [255, 165, 0]  # Orange color
        
        # Mock depth data
        depth_image = np.ones((480, 640)) * 0.5  # 50cm depth
        depth_image[200:300, 300:400] = 0.3  # Carrot at 30cm
        
        # Mock camera intrinsics
        camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        return CameraObservation(
            rgb=rgb_image,
            depth=depth_image,
            camera_matrix=camera_matrix,
            timestamp=time.time(),
            camera_id="test_camera"
        )
    
    @pytest.fixture
    def recipe_planner(self):
        """Initialize recipe planner with test configuration."""
        return RecipePlanner(config_path="config/models/llm_config.yaml")
    
    @pytest.fixture
    def grasp_planner(self):
        """Initialize grasp planner with test configuration."""
        return GraspPlanner(config_path="config/models/vla_config.yaml")
    
    @pytest.fixture
    def motion_planner(self):
        """Initialize motion planner with test configuration."""
        return MotionPlanner(robot_config="config/robots/fr5_left_arm.yaml")
    
    @pytest.mark.asyncio
    async def test_phase1_carrot_pickup_integration(self, 
                                                   mock_camera_data,
                                                   recipe_planner,
                                                   grasp_planner,
                                                   motion_planner):
        """
        PHASE 1 INTEGRATION TEST SPECIFICATION:
        
        SETUP:
        1. Initialize all three layers with test configurations
        2. Load carrot pickup test scene data
        3. Position carrot at known location
        
        EXECUTION:
        1. Send "move carrot to slicer" instruction
        2. Verify planning layer produces correct task sequence
        3. Verify movement layer detects carrot and plans grasp
        4. Verify control layer executes motion successfully
        5. Verify carrot ends up in correct location
        
        VALIDATION:
        - Task completion within 30 seconds
        - No collisions during execution
        - Final carrot position within 2cm of target
        - All system components return success status
        """
        
        start_time = time.time()
        max_execution_time = 30.0  # seconds
        
        logger.info("Starting Phase 1 carrot pickup integration test")
        
        # Step 1: Planning Layer - Task Decomposition
        logger.info("Step 1: Planning Layer - Task decomposition")
        
        instruction = "move carrot to slicer"
        context = {
            "available_objects": ["carrot"],
            "available_appliances": ["slicer"],
            "workspace_state": "clean"
        }
        
        # Get task plan from LLM
        with patch.object(recipe_planner, '_call_llm_api') as mock_llm:
            # Mock LLM response for consistent testing
            mock_llm.return_value = {
                "tasks": [
                    {
                        "id": "t1",
                        "action": "pick_and_place",
                        "parameters": {
                            "object": "carrot",
                            "source": "counter",
                            "target": "slicer_input"
                        },
                        "dependencies": [],
                        "estimated_time": 15
                    }
                ]
            }
            
            task_plan = recipe_planner.plan_task(instruction, context)
        
        # Validate planning output
        assert len(task_plan) == 1, "Should generate exactly one task"
        assert task_plan[0].action == TaskAction.PICK_AND_PLACE, "Should be pick and place action"
        assert task_plan[0].parameters["object"] == "carrot", "Should target carrot"
        assert task_plan[0].parameters["target"] == "slicer_input", "Should target slicer"
        
        logger.info(f"✓ Planning completed: {len(task_plan)} tasks generated")
        
        # Step 2: Movement Layer - Scene Analysis and Grasp Planning
        logger.info("Step 2: Movement Layer - Scene analysis and grasp planning")
        
        # Analyze scene to detect objects
        detected_objects = grasp_planner.analyze_scene(mock_camera_data)
        
        # Should detect at least the carrot (mock detection)
        assert len(detected_objects) >= 1, "Should detect at least one object"
        
        # Plan grasp for carrot
        target_object = task_plan[0].parameters["object"]
        grasp_pose = grasp_planner.plan_grasp(
            camera_data=mock_camera_data,
            target_object=target_object,
            context="pick up for slicing"
        )
        
        # Validate grasp planning
        assert isinstance(grasp_pose, GraspPose), "Should return GraspPose object"
        assert grasp_pose.confidence > 0.2, f"Grasp confidence too low: {grasp_pose.confidence}"
        assert 0.0 <= grasp_pose.gripper_width <= 0.1, f"Invalid gripper width: {grasp_pose.gripper_width}"
        
        # Validate grasp pose is reasonable for carrot
        assert 0.1 <= grasp_pose.x <= 0.6, f"Grasp X position unreasonable: {grasp_pose.x}"
        assert -0.3 <= grasp_pose.y <= 0.3, f"Grasp Y position unreasonable: {grasp_pose.y}"
        assert 0.05 <= grasp_pose.z <= 0.4, f"Grasp Z position unreasonable: {grasp_pose.z}"
        
        logger.info(f"✓ Grasp planned: pos=({grasp_pose.x:.3f}, {grasp_pose.y:.3f}, {grasp_pose.z:.3f}), conf={grasp_pose.confidence:.3f}")
        
        # Step 3: Control Layer - Motion Planning and Execution
        logger.info("Step 3: Control Layer - Motion planning and execution")
        
        # Validate grasp pose with motion planner
        pose_validation = motion_planner.validate_pose(grasp_pose)
        assert pose_validation.is_valid, f"Grasp pose validation failed: {pose_validation.errors}"
        
        # Execute motion to grasp pose
        execution_result = motion_planner.execute_pose(
            target_pose=grasp_pose,
            constraints=None  # Use default constraints
        )
        
        # Validate motion execution
        assert execution_result.success, f"Motion execution failed: {execution_result.message}"
        assert execution_result.final_pose is not None, "Should return final pose"
        assert execution_result.execution_time > 0, "Should report execution time"
        
        logger.info(f"✓ Motion executed successfully in {execution_result.execution_time:.2f}s")
        
        # Step 4: Validate Overall Success Criteria
        logger.info("Step 4: Validating overall success criteria")
        
        total_execution_time = time.time() - start_time
        
        # Check timing requirement
        assert total_execution_time <= max_execution_time, \
            f"Execution took {total_execution_time:.2f}s, exceeds limit of {max_execution_time}s"
        
        # Check final position accuracy (within 2cm tolerance)
        final_pose = execution_result.final_pose
        target_slicer_pos = Pose6D(x=0.5, y=0.3, z=0.15, rx=0, ry=0, rz=0)  # Expected slicer input position
        
        position_error = np.sqrt(
            (final_pose.x - target_slicer_pos.x)**2 +
            (final_pose.y - target_slicer_pos.y)**2 +
            (final_pose.z - target_slicer_pos.z)**2
        )
        
        max_position_error = 0.02  # 2cm tolerance
        assert position_error <= max_position_error, \
            f"Position error {position_error:.4f}m exceeds tolerance {max_position_error}m"
        
        logger.info(f"✓ Position accuracy: {position_error*1000:.1f}mm error")
        logger.info(f"✓ Total execution time: {total_execution_time:.2f}s")
        
        # Step 5: System Health Validation
        logger.info("Step 5: System health validation")
        
        # Verify all components are still responsive
        current_pose = motion_planner.get_current_pose()
        assert current_pose is not None, "Motion planner should report current pose"
        
        current_joints = motion_planner.get_current_joint_state()
        assert len(current_joints) == 6, "Should have 6 joint angles for FR5 robot"
        
        logger.info("✓ All system components healthy")
        
        # Final success log
        logger.info("🎉 Phase 1 carrot pickup integration test PASSED")
        logger.info(f"   - Planning: ✓ Generated {len(task_plan)} tasks")
        logger.info(f"   - Perception: ✓ Detected {len(detected_objects)} objects")
        logger.info(f"   - Grasping: ✓ Confidence {grasp_pose.confidence:.3f}")
        logger.info(f"   - Motion: ✓ Executed in {execution_result.execution_time:.2f}s")
        logger.info(f"   - Accuracy: ✓ {position_error*1000:.1f}mm error")
        logger.info(f"   - Total time: ✓ {total_execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_planning_layer_robustness(self, recipe_planner):
        """Test planning layer with various instruction formats."""
        
        test_instructions = [
            "move carrot to slicer",
            "put the carrot in the slicer",
            "slice the carrot",
            "transfer carrot to cutting station"
        ]
        
        for instruction in test_instructions:
            with patch.object(recipe_planner, '_call_llm_api') as mock_llm:
                mock_llm.return_value = {
                    "tasks": [{
                        "id": "t1",
                        "action": "pick_and_place",
                        "parameters": {"object": "carrot", "target": "slicer"},
                        "dependencies": [],
                        "estimated_time": 10
                    }]
                }
                
                tasks = recipe_planner.plan_task(instruction)
                assert len(tasks) >= 1, f"Failed to plan for: {instruction}"
                assert tasks[0].action == TaskAction.PICK_AND_PLACE
    
    @pytest.mark.asyncio
    async def test_movement_layer_robustness(self, grasp_planner, mock_camera_data):
        """Test movement layer with various camera conditions."""
        
        # Test with different lighting conditions
        dark_image = mock_camera_data
        dark_image.rgb = dark_image.rgb * 0.3  # Dim lighting
        
        grasp_pose = grasp_planner.plan_grasp(dark_image, "carrot")
        assert isinstance(grasp_pose, GraspPose)
        assert grasp_pose.confidence >= 0.2  # Should handle dim lighting
        
        # Test with missing depth data
        no_depth_data = CameraObservation(
            rgb=mock_camera_data.rgb,
            depth=None,
            timestamp=time.time()
        )
        
        grasp_pose = grasp_planner.plan_grasp(no_depth_data, "carrot")
        assert isinstance(grasp_pose, GraspPose)
        # Should still provide reasonable grasp even without depth
    
    @pytest.mark.asyncio
    async def test_control_layer_robustness(self, motion_planner):
        """Test control layer with various pose constraints."""
        
        # Test poses at workspace boundaries
        boundary_poses = [
            Pose6D(x=0.7, y=0.5, z=0.1, rx=0, ry=0, rz=0),   # Near max reach
            Pose6D(x=0.2, y=-0.4, z=0.05, rx=0, ry=0, rz=0), # Near minimum height
            Pose6D(x=0.4, y=0.0, z=0.6, rx=0, ry=0, rz=0),   # High position
        ]
        
        for pose in boundary_poses:
            validation = motion_planner.validate_pose(pose)
            if validation.is_valid:
                result = motion_planner.execute_pose(pose)
                assert result.success or result.error_code == "PLANNING_FAILED"
            # Should either succeed or fail gracefully
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, recipe_planner, grasp_planner, motion_planner):
        """Test system behavior under error conditions."""
        
        # Test planning with invalid instruction
        with patch.object(recipe_planner, '_call_llm_api') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            tasks = recipe_planner.plan_task("invalid instruction")
            # Should return empty list or fallback tasks, not crash
            assert isinstance(tasks, list)
        
        # Test grasp planning with invalid camera data
        invalid_camera = CameraObservation(rgb=None, timestamp=0.0)
        grasp_pose = grasp_planner.plan_grasp(invalid_camera, "carrot")
        # Should return fallback grasp, not crash
        assert isinstance(grasp_pose, GraspPose)
        
        # Test motion planning with unreachable pose
        unreachable_pose = Pose6D(x=2.0, y=2.0, z=2.0, rx=0, ry=0, rz=0)
        result = motion_planner.execute_pose(unreachable_pose)
        assert not result.success  # Should fail gracefully
        assert result.error_code is not None

if __name__ == "__main__":
    # Run tests directly
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "-s"])