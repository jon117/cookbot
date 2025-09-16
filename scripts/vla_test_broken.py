# scripts/isaac_sim_5_vla_proper.py

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import requests
import json
import time
import os
from datetime import datetime

# Isaac Sim 5.0 API imports - NEW STRUCTURE
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.nucleus import get_assets_root_path
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.types import ArticulationAction

# NEW: Isaac Sim 5.0 Robot Manipulator API  
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.core.api.tasks import FollowTarget
# RMPFlow controller - will try to import, may need different approach

# Robot workspace bounds
ROBOT_WORKSPACE = {
    "x": [-0.5, 0.5],
    "y": [-0.5, 0.5],
    "z": [0.0, 0.5]
}

def main():
    print("🚀 Isaac Sim 5.0 + VLA Integration (Proper Motion Control)")
    print("Using Isaac Sim's built-in Lula IK solver and motion generation!")
    print("Press Ctrl+C to stop")
    
    # Create output directory
    output_dir = "isaac_sim_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Images will be saved to: {output_dir}/")
    
    # 1. Create World
    my_world = World(stage_units_in_meters=1.0)
    
    # 2. Add room and objects
    try:
        room_path = get_assets_root_path() + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_path, prim_path="/World/Room")
        print("✅ Room and table added")
    except Exception as e:
        print(f"⚠️ Room loading failed: {e}")
    
    # Add red cube to manipulate
    try:
        red_block = DynamicCuboid(
            prim_path="/World/RedBlock",
            name="red_block",
            position=np.array([0.0, -0.5, 0.4]),
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.1
        )
        my_world.scene.add(red_block)
        print("✅ Red block added on table")
    except Exception as e:
        print(f"⚠️ Red block failed: {e}")
    
    # 3. Add Franka robot using NEW Isaac Sim 5.0 API
    robot = None
    kinematics_solver = None
    
    try:
        # Load Franka robot USD
        robot_path = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Franka")
        
        # Create SingleManipulator (Isaac Sim 5.0 high-level interface)
        robot = SingleManipulator(
            prim_path="/World/Franka",
            name="franka_robot",
            end_effector_prim_path="panda_hand"  # Franka end-effector
        )
        my_world.scene.add(robot)

        print("🔧 Stepping simulation to initialize robot physics...")
        for _ in range(5):
            my_world.step(render=False) # Use render=False for fast initialization
        
        # NEW: Initialize Lula Kinematics Solver (Isaac Sim 5.0 way)
        try:
            kinematics_solver = LulaKinematicsSolver(
                robot_description_path=get_assets_root_path() + "/Isaac/Robots/Franka/lula_franka_gen.yaml",
                urdf_path=get_assets_root_path() + "/Isaac/Robots/Franka/franka.urdf"
            )
            print("✅ Lula Kinematics Solver initialized")
        except Exception as e:
            print(f"⚠️ Lula Kinematics Solver failed: {e}")
            kinematics_solver = None
        
        # Set robot to initial pose
        robot.set_joint_positions([0, -1.157, 0, -2.0, 0, 1.571, 0.785, 0.04, 0.04])
        
        print("✅ Franka robot loaded with Isaac Sim 5.0 SingleManipulator API")
        
    except Exception as e:
        print(f"⚠️ Robot loading failed: {e}")
        import traceback
        traceback.print_exc()
        robot = None
    
    # 4. Create camera
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([0.0, -2.0, 1.6]),
        name="camera",
        frequency=20,
        resolution=(512, 512)
    )
    my_world.scene.add(camera)
    camera.set_local_pose(
        orientation=rot_utils.euler_angles_to_quats(np.array([53, 0, 0]), degrees=True),
        camera_axes="usd"
    )
    
    try:
        camera.set_focal_length(2.4)
        print("✅ Camera positioned with focal length 24mm")
    except Exception as e:
        print(f"⚠️ Could not set focal length: {e}")
    
    # 5. Initialize world
    my_world.reset()
    print("✅ World initialized")
    
    # 6. VLA Integration Loop with Isaac Sim 5.0 Motion Control
    print("\n🔄 Starting VLA integration with Isaac Sim 5.0...")
    print("   🤖 VLA commands → Lula IK → Smooth robot motion")
    print("   🦾 Using SingleManipulator for controlled motion")
    print("   ⏹️  Press Ctrl+C to stop\n")
    
    capture_count = 0
    target_fps = 2
    capture_interval = 1.0 / target_fps
    last_capture_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Always step the simulation
            my_world.step(render=True)
            
            # Only capture images at target FPS
            if current_time - last_capture_time >= capture_interval:
                capture_count += 1
                last_capture_time = current_time
                
                print(f"--- Capture {capture_count} (@ {target_fps} FPS) ---")
                
                # Get camera data
                try:
                    rgba_data = camera.get_rgba()
                    
                    if rgba_data is not None and rgba_data.size > 0:
                        rgb_image = rgba_data[:, :, :3]
                        print(f"📸 Image captured: {rgb_image.shape}")
                        
                        save_image(rgb_image, output_dir, capture_count)
                        
                        # VLA Integration with Proper Motion Control
                        print(f"🤖 VLA + Isaac Sim 5.0 Motion Control")
                        
                        current_prompt = "pick up the red cube"
                        print(f"📝 VLA Prompt: '{current_prompt}'")
                        
                        vla_action = send_to_vla(rgb_image, current_prompt)
                        if vla_action is not None and robot is not None:
                            apply_vla_action_isaac_sim_5(robot, kinematics_solver, vla_action)
                        else:
                            print("❌ VLA or robot failed - skipping this frame")
                        
                    else:
                        print("⚠️ No camera data")
                        
                except Exception as e:
                    print(f"❌ Camera capture failed: {e}")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped by user after {capture_count} captures")
    
    except Exception as e:
        print(f"\n❌ Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🎉 Cleaning up...")
        try:
            simulation_app.close()
            print("✅ Simulation closed cleanly")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

def save_image(image, output_dir, capture_count):
    """Save image with VLA preprocessing."""
    try:
        from PIL import Image as PILImage
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"isaac_sim_{capture_count:04d}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        pil_image = PILImage.fromarray(image)
        pil_image.save(filepath)
        
        # Save VLA-sized version
        vla_filename = f"isaac_sim_{capture_count:04d}_{timestamp}_vla_256x256.png"
        vla_filepath = os.path.join(output_dir, vla_filename)
        vla_image = pil_image.resize((256, 256))
        vla_image.save(vla_filepath)
        
        print(f"💾 Saved: {filename} and VLA version")
        
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

def send_to_vla(image, instruction):
    """Send to VLA with proper error handling."""
    try:
        from PIL import Image as PILImage
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = PILImage.fromarray(image)
        resized_image = pil_image.resize((256, 256))
        vla_image = np.array(resized_image, dtype=np.uint8)
        
        print(f"🤖 → VLA: {instruction}")
        
        import json_numpy
        json_numpy.patch()
        
        payload = {
            "image": vla_image,
            "instruction": instruction,
            "unnorm_key": "berkeley_autolab_ur5"
        }
        
        response = requests.post("http://localhost:8000/act", json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            if isinstance(result, str):
                print(f"   VLA returned: {result}")
                return None
            elif hasattr(result, '__len__') and len(result) >= 7:
                x, y, z, rx, ry, rz, gripper = result[:7]
                print(f"   🎯 VLA Action: pos({x:.3f},{y:.3f},{z:.3f}) rot({rx:.3f},{ry:.3f},{rz:.3f}) grip={gripper:.2f}")
                return {
                    "position": [x, y, z],
                    "rotation": [rx, ry, rz], 
                    "gripper": gripper
                }
            else:
                print(f"   Unexpected format: {result}")
                return None
        else:
            print(f"❌ HTTP {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ VLA server unreachable (localhost:8000/act)")
        return None
    except Exception as e:
        print(f"❌ VLA error: {e}")
        return None

def apply_vla_action_isaac_sim_5(robot, kinematics_solver, vla_action):
    """
    ISAAC SIM 5.0 APPROACH: Use SingleManipulator + Lula IK for smooth motion.
    
    This replaces the problematic custom IK with Isaac Sim's proper motion control:
    1. Get current end-effector pose
    2. Apply VLA delta
    3. Use Lula Kinematics Solver for IK if available
    4. Use SingleManipulator smooth joint control
    """
    try:
        # Step 1: Get current end-effector pose using Isaac Sim 5.0 API
        current_ee_position, current_ee_orientation = robot.end_effector.get_world_pose()
        
        print(f"🦾 Current EE Pose: ({current_ee_position[0]:.3f}, {current_ee_position[1]:.3f}, {current_ee_position[2]:.3f})")
        
        # Step 2: Apply VLA action delta with proper scaling
        ACTION_SCALE = 0.05  # Smaller scale for more controlled motion
        norm_pos_delta = vla_action["position"]
        
        # Coordinate transformation for VLA → Isaac Sim
        isaac_pos_delta = np.array([
            norm_pos_delta[1] * ACTION_SCALE,    # VLA Y → Isaac X
            -norm_pos_delta[0] * ACTION_SCALE,   # VLA X → Isaac -Y  
            norm_pos_delta[2] * ACTION_SCALE     # VLA Z → Isaac Z
        ])
        
        # Calculate new target position
        new_target_pos = current_ee_position + isaac_pos_delta
        
        # Clamp to workspace bounds
        clamped_target_pos = np.array([
            np.clip(new_target_pos[0], ROBOT_WORKSPACE["x"][0], ROBOT_WORKSPACE["x"][1]),
            np.clip(new_target_pos[1], ROBOT_WORKSPACE["y"][0], ROBOT_WORKSPACE["y"][1]),
            np.clip(new_target_pos[2], ROBOT_WORKSPACE["z"][0], ROBOT_WORKSPACE["z"][1])
        ])
        
        print(f"🤖 VLA Delta: ({norm_pos_delta[0]:.3f}, {norm_pos_delta[1]:.3f}, {norm_pos_delta[2]:.3f})")
        print(f"🌍 Target Position: ({clamped_target_pos[0]:.3f}, {clamped_target_pos[1]:.3f}, {clamped_target_pos[2]:.3f})")
        
        # Step 3: Use Lula Kinematics Solver if available
        if kinematics_solver:
            try:
                # Solve inverse kinematics using Lula
                target_joint_positions = kinematics_solver.compute_inverse_kinematics(
                    target_position=clamped_target_pos,
                    target_orientation=current_ee_orientation,  # Keep current orientation
                    warm_start=robot.get_joint_positions()  # Use current joints as starting point
                )
                
                if target_joint_positions is not None:
                    print(f"✅ Lula IK solution found")
                    
                    # Step 4: Apply smooth joint control using SingleManipulator
                    current_joints = robot.get_joint_positions()
                    
                    # Apply conservative smoothing to prevent jumps
                    alpha = 0.1  # Very conservative smoothing
                    smooth_joints = []
                    for i in range(len(current_joints)):
                        if i < len(target_joint_positions):
                            smooth_joint = current_joints[i] + alpha * (target_joint_positions[i] - current_joints[i])
                            smooth_joints.append(smooth_joint)
                        else:
                            smooth_joints.append(current_joints[i])
                    
                    # Apply smoothed joint positions using SingleManipulator
                    action = ArticulationAction(joint_positions=np.array(smooth_joints))
                    robot.apply_action(action)
                    print(f"🎯 Smooth Lula IK action applied")
                else:
                    print(f"❌ Lula IK solver failed - no solution found")
                    
            except Exception as e:
                print(f"⚠️ Lula IK failed: {e}")
                # Fallback to simple position-based control
                _apply_simple_position_control(robot, clamped_target_pos)
        else:
            print(f"⚠️ No Lula solver available, using simple control")
            # Fallback to simple position-based control  
            _apply_simple_position_control(robot, clamped_target_pos)
            
    except Exception as e:
        print(f"❌ Motion control error: {e}")
        import traceback
        traceback.print_exc()

def _apply_simple_position_control(robot, target_position):
    """Fallback: Simple position-based control without IK."""
    try:
        # Get current joint positions
        current_joints = robot.get_joint_positions()
        
        # Simple heuristic: move joints towards target (very basic approach)
        # This is just to keep the robot moving when Lula IK is not available
        target_joints = current_joints.copy()
        
        # Small random movements to simulate IK (placeholder)
        for i in range(3):  # Only move first 3 joints
            if i < len(target_joints):
                small_delta = (target_position[i] - 0.3) * 0.01  # Very small movement
                target_joints[i] += small_delta
        
        # Apply very conservative smoothing
        alpha = 0.05
        smooth_joints = []
        for i in range(len(current_joints)):
            smooth_joint = current_joints[i] + alpha * (target_joints[i] - current_joints[i])
            smooth_joints.append(smooth_joint)
        
        # Apply action
        action = ArticulationAction(joint_positions=np.array(smooth_joints))
        robot.apply_action(action)
        print(f"🎯 Simple position control applied (fallback)")
        
    except Exception as e:
        print(f"❌ Simple position control failed: {e}")

if __name__ == "__main__":
    main()