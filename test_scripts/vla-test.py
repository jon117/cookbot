#!/usr/bin/env python3
"""
VLA Robot Test with Isaac Sim Core APIs
Uses velocity control for smooth robot motion.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import time
import json
from datetime import datetime
import numpy as np
import requests
import json_numpy
from PIL import Image as PILImage

# Core API imports
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path

# Camera imports
from omni.isaac.sensor import Camera
from omni.kit.viewport.utility import get_viewport_from_window_name
import omni.isaac.core.utils.rotations as rot_utils

# Dynamic control interface for velocity control
from omni.isaac.dynamic_control import _dynamic_control

# For end-effector pose tracking
from omni.isaac.core.prims import XFormPrim

# Robot workspace constraints
ROBOT_WORKSPACE = {
    "x": [-0.5, 0.5],  # Forward/back
    "y": [-0.5, 0.5], # Left/right
    "z": [0.0, 0.5]  # Down/up
}

def save_image(image, output_dir, capture_count):
    """Save image to local directory with timestamp."""
    try:
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"isaac_sim_{capture_count:04d}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        pil_image = PILImage.fromarray(image)
        pil_image.save(filepath)
        
        print(f"💾 Saved: {filename}")
        
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

def send_to_vla(image, instruction):
    """Send image and instruction to VLA at localhost:8000/act"""
    try:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = PILImage.fromarray(image)
        resized_image = pil_image.resize((224, 224))
        vla_image = np.array(resized_image, dtype=np.uint8)
        
        print(f"🤖 → VLA: {instruction}")
        
        json_numpy.patch()
        
        payload = {
            "image": vla_image,
            "instruction": instruction,
            "unnorm_key": "berkeley_autolab_ur5"
        }
        
        response = requests.post("http://localhost:8000/act", json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ VLA responded: {type(result)}")
            
            if isinstance(result, str):
                print(f"   Response: {result}")
                return None
            elif hasattr(result, '__len__') and len(result) >= 7:
                x, y, z, rx, ry, rz, gripper = result[:7]
                print(f"   🎯 Raw VLA: pos({x:.3f},{y:.3f},{z:.3f}) rot({rx:.3f},{ry:.3f},{rz:.3f}) grip={gripper:.2f}")
                return {
                    "position": [x, y, z],
                    "rotation": [rx, ry, rz], 
                    "gripper": gripper
                }
            else:
                print(f"   Data: {result}")
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

def apply_vla_action_to_robot_velocity(robot, vla_action, dc, articulation):
    """
    Apply VLA action using velocity control for smooth motion.
    """
    try:
        if not robot or not dc or not articulation:
            print("❌ Robot or dynamic control not available")
            return

        # --- STEP 1: Get the robot's current end-effector pose ---
        ee_prim_path = "/World/UR5/wrist_3_link" 
        ee_prim = XFormPrim(ee_prim_path)
        positions, orientations = ee_prim.get_world_pose()
        
        if positions is None:
            print("⚠️ Could not get current end-effector pose. Aborting action.")
            return
            
        current_pos = positions
        
        # --- STEP 2: Get the normalized action delta from VLA ---
        norm_pos_delta = vla_action["position"]
        
        # --- STEP 3: Coordinate System Transformation & Scaling ---
        ACTION_SCALE = 1.0  # Increased from 0.05 to make movements more visible
        
        isaac_pos_delta = np.array([
            norm_pos_delta[1] * ACTION_SCALE,    # VLA Y -> Isaac X
            -norm_pos_delta[0] * ACTION_SCALE,   # VLA X -> Isaac -Y
            norm_pos_delta[2] * ACTION_SCALE     # VLA Z -> Isaac Z
        ])

        # --- STEP 4: Calculate the new target position ---
        new_target_pos = current_pos + isaac_pos_delta

        # --- STEP 5: Clamp the target to the defined workspace ---
        clamped_target_pos = np.array([
            np.clip(new_target_pos[0], ROBOT_WORKSPACE["x"][0], ROBOT_WORKSPACE["x"][1]),
            np.clip(new_target_pos[1], ROBOT_WORKSPACE["y"][0], ROBOT_WORKSPACE["y"][1]),
            np.clip(new_target_pos[2], ROBOT_WORKSPACE["z"][0], ROBOT_WORKSPACE["z"][1])
        ])

        print(f"🦾 Current EE Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
        print(f"🤖 VLA Delta (X,Y,Z): ({norm_pos_delta[0]:.2f}, {norm_pos_delta[1]:.2f}, {norm_pos_delta[2]:.2f})")
        print(f"🌍 World Target: ({clamped_target_pos[0]:.3f}, {clamped_target_pos[1]:.3f}, {clamped_target_pos[2]:.3f})")

        # --- STEP 6: Use velocity control for smooth motion ---
        velocity_gain = 5.0  # Increased gain for more responsive movement
        
        # Simple velocity control approach
        num_dofs = dc.get_articulation_dof_count(articulation)
        joint_vels = np.zeros(num_dofs)
        
        # Calculate position error
        pos_error = clamped_target_pos - current_pos
        error_magnitude = np.linalg.norm(pos_error)
        
        print(f"🎯 Position error magnitude: {error_magnitude:.4f}")
        
        # Only move if error is significant enough
        if error_magnitude > 0.001:  # 1mm threshold
            # Base rotation towards target
            base_angle_target = np.arctan2(clamped_target_pos[1], clamped_target_pos[0])
            base_dof = dc.find_articulation_dof(articulation, "shoulder_pan_joint")
            if base_dof:
                base_angle_current = dc.get_dof_state(base_dof, _dynamic_control.STATE_POS).pos
                base_error = base_angle_target - base_angle_current
                joint_vels[0] = base_error * velocity_gain * 0.5  # Reduced gain for base
            
            # Improved joint mapping for Cartesian movement
            if num_dofs >= 6:  # Standard 6-DOF robot
                # Shoulder lift (joint 1) - primarily for Z movement
                joint_vels[1] = pos_error[2] * velocity_gain * 3.0
                
                # Elbow (joint 2) - help with reaching distance
                horizontal_error = np.linalg.norm(pos_error[:2])
                joint_vels[2] = -horizontal_error * velocity_gain * 2.0
                
                # Wrist 1 (joint 3) - fine adjustment
                joint_vels[3] = pos_error[0] * velocity_gain * 1.0
                
                # Keep wrist 2 and 3 stable
                joint_vels[4] = 0
                joint_vels[5] = 0
        else:
            print("🎯 Target reached, no movement needed")
        
        # Limit velocities to reasonable ranges
        max_vel = 2.0  # Increased max velocity
        joint_vels = np.clip(joint_vels, -max_vel, max_vel)
        
        # Convert to float32 for dynamic control interface
        joint_vels = joint_vels.astype(np.float32)
        
        dc.set_articulation_dof_velocity_targets(articulation, joint_vels)
        print(f"🎯 Applied joint velocities: {[f'{v:.2f}' for v in joint_vels[:6]]}")

    except Exception as e:
        print(f"❌ Robot velocity control error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function - creates scene and runs VLA loop."""
    print("🚀 VLA-Isaac Sim Integration with Velocity Control")
    print("   Using Core Isaac Sim APIs for smooth robot motion")
    
    # 1. Create the world
    my_world = World(stage_units_in_meters=1.0)
    print("✅ World created")
    
    # Create output directory for images
    output_dir = "isaac_sim_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Images will be saved to: {output_dir}/")

    # 2. Add room and table
    try:
        room_path = get_assets_root_path() + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_path, prim_path="/World/Room")
        print("✅ Room and table added")
    except Exception as e:
        print(f"⚠️ Room loading failed (continuing without): {e}")

    # 3. Add robot
    robot_asset_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
    add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/UR5")
    robot = Robot(prim_path="/World/UR5", name="ur5_robot")
    my_world.scene.add(robot)
    print("✅ UR5 Robot added")

    # 4. Add red block
    red_block = DynamicCuboid(
        prim_path="/World/RedBlock",
        name="red_block",
        position=np.array([0.0, -0.5, 0.4]),
        scale=np.array([0.05, 0.05, 0.05]),
        color=np.array([1.0, 0.0, 0.0]),
        mass=0.1
    )
    my_world.scene.add(red_block)
    print("✅ Red block added")

    # 5. Create camera
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([0.0, -2.0, 1.6]),
        name="camera",
        frequency=20,
        resolution=(512, 512)
    )
    my_world.scene.add(camera)
    
    # Position camera to look at the robot
    camera.set_local_pose(
        orientation=rot_utils.euler_angles_to_quat(np.array([53,0,0]), degrees=True),
        camera_axes="usd"
    )
    
    # Set focal length
    try:
        camera.set_focal_length(2.4)
        print("✅ Camera positioned, focal length set to 2.4")
    except Exception as e:
        print(f"⚠️ Could not set focal length: {e}")
        print("✅ Camera positioned (default focal length)")

    # Set default camera in viewport
    viewport_api = get_viewport_from_window_name("Viewport")
    viewport_api.set_active_camera("/World/Camera")
    
    # 6. Initialize world
    my_world.reset()
    print("✅ World initialized")
    
    # 7. Start simulation for dynamic control
    import omni
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("✅ Simulation started")
    
    # 8. VLA Integration Loop (2 FPS = 0.5 seconds between captures)
    print("\n🔄 Starting VLA integration at 2 FPS...")
    print("   🤖 VLA will control the robot based on camera input!")
    print("   🦾 Robot should respond to VLA predictions with smooth motion")
    print("   ⏹️  Press Ctrl+C to stop\n")
    
    capture_count = 0
    target_fps = 2
    capture_interval = 1.0 / target_fps
    last_capture_time = 0
    
    # Initialize dynamic control variables
    dc = None
    articulation = None
    
    # Initialize robot to home position
    if robot:
        try:
            # Set robot to position control mode (important!)
            robot.set_solver_position_iteration_count(64)
            robot.set_solver_velocity_iteration_count(64)
            
            home_joints = [0, -1.57, -1.57, -1.57, 1.57, 0]  # UR5 home position
            robot.set_joint_positions(home_joints)
            
            print(f"🏠 Robot initialized to home position: {[f'{j:.2f}' for j in home_joints]}")
        except Exception as e:
            print(f"⚠️ Could not set home position: {e}")
    
    try:
        while True:
            current_time = time.time()
            
            # Always step the simulation
            my_world.step(render=True)
            
            # Initialize dynamic control on first few steps
            if dc is None and timeline.is_playing():
                try:
                    print("🔧 Attempting to initialize dynamic control interface...")
                    dc = _dynamic_control.acquire_dynamic_control_interface()
                    print(f"   Dynamic control interface acquired: {dc is not None}")
                    
                    if dc:
                        print("🔧 Getting UR5 articulation at /World/UR5/base_link...")
                        articulation = dc.get_articulation("/World/UR5/base_link")
                        
                        if articulation:
                            dc.wake_up_articulation(articulation)
                            num_dofs = dc.get_articulation_dof_count(articulation)
                            print(f"✅ Dynamic control interface initialized with {num_dofs} DOFs")
                            
                            # CRITICAL: Switch to velocity control mode
                            print("🔧 Switching robot to velocity control mode...")
                            for i in range(num_dofs):
                                # Set drive mode to velocity (not position)
                                dc.set_dof_velocity_target(dc.get_articulation_dof(articulation, i), 0.0)
                            
                            # Set joint stiffness to 0 for velocity control
                            from pxr import UsdPhysics
                            import omni.usd
                            stage = omni.usd.get_context().get_stage()
                            
                            for prim in stage.TraverseAll():
                                if prim.GetPath().pathString.startswith("/World/UR5") and prim.GetTypeName() == "PhysicsRevoluteJoint":
                                    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                    if drive:
                                        drive.GetStiffnessAttr().Set(0.0)  # Zero stiffness for velocity control
                                        drive.GetDampingAttr().Set(1000.0)  # High damping for stability
                            
                            print("✅ Robot switched to velocity control mode")
                        else:
                            print("❌ Could not get articulation handle")
                except Exception as e:
                    print(f"⚠️ Dynamic control init failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Only capture images at our target FPS
            if current_time - last_capture_time >= capture_interval:
                capture_count += 1
                last_capture_time = current_time
                
                print(f"--- Capture {capture_count} (@ {target_fps} FPS) ---")
                
                # Get camera data
                try:
                    rgba_data = camera.get_rgba()
                    
                    if rgba_data is not None and rgba_data.size > 0:
                        rgb_image = rgba_data[:, :, :3]  # Remove alpha
                        print(f"📸 Image captured: {rgb_image.shape}")
                        
                        save_image(rgb_image, output_dir, capture_count)
                        
                        # VLA testing
                        current_prompt = "pick up the red cube"
                        
                        vla_action = send_to_vla(rgb_image, current_prompt)
                        if vla_action is not None and robot is not None and dc is not None and articulation is not None:
                            apply_vla_action_to_robot_velocity(robot, vla_action, dc, articulation)
                        else:
                            if vla_action is None:
                                print("❌ VLA failed - skipping this frame")
                            else:
                                print(f"⚠️ Dynamic control not ready - robot:{robot is not None}, dc:{dc is not None}, articulation:{articulation is not None}")
                        
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
    
    finally:
        print("\n🎉 Cleaning up...")
        try:
            simulation_app.close()
            print("✅ Simulation closed cleanly")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

if __name__ == "__main__":
    main()