# scripts/simple_isaac_vla_test.py

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import requests
import json
import time
import os
from datetime import datetime

# Isaac Sim 5.0 API imports
from isaacsim.core.api.objects import DynamicCuboid
from omni.kit.viewport.utility import get_viewport_from_window_name
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
from isaacsim.core.prims import XFormPrim
from isaacsim.core.api.robots import Robot
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.numpy.rotations as rot_utils

# These are the [X_min, X_max], [Y_min, Y_max], [Z_min, Z_max] coordinates
# in Isaac Sim's world frame where you want the robot to operate.
# Adjust these values to fit your table setup.
ROBOT_WORKSPACE = {
    "x": [-0.5, 0.5],  # Forward/back
    "y": [-0.5, 0.5], # Left/right
    "z": [0.0, 0.5]  # Down/up (0.45 is just above the table)
}

def main():
    print("🚀 Isaac Sim 5.0 + VLA Integration (IK Verified)")
    print("Robot IK is working - now testing VLA integration!")
    print("Press Ctrl+C to stop")
    
    # Create output directory for images
    output_dir = "isaac_sim_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Images will be saved to: {output_dir}/")
    
    # 1. Create World
    my_world = World(stage_units_in_meters=1.0)
    
    # 3. Add robot
    robot = None
    try:
        robot_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/UR5")
        
        # Create robot object for control
        robot = Robot(prim_path="/World/UR5", name="ur5_robot")
        my_world.scene.add(robot)
        robot.set_local_pose(orientation=rot_utils.euler_angles_to_quats(np.array([0,0,90]), degrees=True))
        print("✅ UR5 robot loaded and added to scene")
    except Exception as e:
        print(f"⚠️ Robot loading failed: {e}")
        robot = None
    
    # 4. Add a simple room with table and red block
    try:
        room_path = get_assets_root_path() + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_path, prim_path="/World/Room")
        print("✅ Room and table added")
    except Exception as e:
        print(f"⚠️ Room loading failed: {e}")
    
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
    
    # 5. Create camera
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([0.0, -2.0, 1.6]),
        name="camera",
        frequency=20,
        resolution=(512, 512)
    )
    
    my_world.scene.add(camera)

    # Set camera orientation
    camera.set_local_pose(
        orientation=rot_utils.euler_angles_to_quats(np.array([53,0,0]), degrees=True),
        camera_axes="usd"
    )
    
    # Set focal length
    try:
        camera.set_focal_length(2.4)
        print("✅ Camera positioned, focal length set to 1.5cm (15mm)")
    except Exception as e:
        print(f"⚠️ Could not set focal length: {e}")
        print("✅ Camera positioned (default focal length)")

    # Set default camera in viewport
    viewport_api = get_viewport_from_window_name("Viewport")
    viewport_api.set_active_camera("/World/Camera")
    
    # 6. Initialize world
    my_world.reset()
    print("✅ World initialized")
    
    # 7. VLA Integration Loop (2 FPS = 0.5 seconds between captures)
    print("\n🔄 Starting VLA integration at 2 FPS...")
    print("   🤖 VLA will control the robot based on camera input!")
    print("   🦾 Robot should respond to VLA predictions")
    print("   🧪 First 10 captures = manual testing, then VLA takes over")
    print("   ⏹️  Press Ctrl+C to stop\n")
    
    capture_count = 0
    target_fps = 2
    capture_interval = 1.0 / target_fps
    last_capture_time = 0
    
    # Initialize robot to home position
    if robot:
        try:
            # Set robot to position control mode (important!)
            robot.set_solver_position_iteration_count(64)
            robot.set_solver_velocity_iteration_count(64)
            
            home_joints = [0, -1.57, -1.57, -1.57, 1.57, 0]  # UR5 home position
            robot.set_joint_positions(home_joints)
            
            # Set joint stiffness to fight gravity
            try:
                stiffness = [1000] * 6  # High stiffness to hold position
                robot.set_joint_position_targets(home_joints)
                print(f"🏠 Robot initialized with position control")
            except:
                print(f"⚠️ Could not set joint targets, using basic position control")
            
            print(f"🏠 Robot initialized to home position: {[f'{j:.2f}' for j in home_joints]}")
        except Exception as e:
            print(f"⚠️ Could not set home position: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        while True:
            current_time = time.time()
            
            # Always step the simulation
            my_world.step(render=True)
            
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
                        
                        # VLA testing - STICK with one prompt for multiple frames
                        print(f"🤖 VLA TEST - Capture {capture_count}")
                        
                        # Use same prompt for 20 frames, then switch
                        frames_per_prompt = 20
                        current_prompt = "pick up the red cube"
                        
                        frames_into_current_prompt = (capture_count - 11) % frames_per_prompt + 1
                        
                        print(f"📝 VLA Prompt: '{current_prompt}' (frame {frames_into_current_prompt}/{frames_per_prompt})")
                        
                        vla_action = send_to_vla(rgb_image, current_prompt)
                        if vla_action is not None and robot is not None:
                            apply_vla_action_to_robot(robot, vla_action)
                        else:
                            print("❌ VLA failed - skipping this frame")
                        
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

def save_image(image, output_dir, capture_count):
    """Save image to local directory with timestamp."""
    try:
        from PIL import Image as PILImage
        
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
        from PIL import Image as PILImage
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = PILImage.fromarray(image)
        resized_image = pil_image.resize((224, 224))
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

def simple_ur5_ik(target_pos, current_joints=None):
    """
    Improved inverse kinematics for UR5 robot.
    UR5 has 6 joints: base, shoulder, elbow, wrist1, wrist2, wrist3
    """
    if current_joints is None:
        current_joints = [0, -1.57, 1.57, -1.57, -1.57, 0]  # Home position
    
    x, y, z = target_pos
    
    print(f"🧮 IK INPUT: target=({x:.3f}, {y:.3f}, {z:.3f})")
    
    # Start with current joint positions
    joint_angles = current_joints.copy()
    
    # Joint 0: Base rotation (simple - just point towards target)
    joint_angles[0] = np.arctan2(y, x)
    print(f"🧮 Base rotation: {np.degrees(joint_angles[0]):.1f}°")
    
    # Calculate horizontal distance and height from base
    horizontal_dist = np.sqrt(x*x + y*y)
    height_from_base = z - 0.1  # UR5 base is ~10cm high
    
    print(f"🧮 Horizontal distance: {horizontal_dist:.3f}m")
    print(f"🧮 Height from base: {height_from_base:.3f}m")
    
    # UR5 link lengths (approximate)
    l1 = 0.425  # Upper arm length
    l2 = 0.392  # Forearm length
    
    # Calculate reach distance for 2D arm in the vertical plane
    reach_2d = np.sqrt(horizontal_dist*horizontal_dist + height_from_base*height_from_base)
    print(f"🧮 2D reach needed: {reach_2d:.3f}m")
    print(f"🧮 Max possible reach: {l1 + l2:.3f}m")
    
    if reach_2d > (l1 + l2 - 0.05):  # Leave 5cm margin
        print("⚠️ Target may be out of reach!")
        reach_2d = l1 + l2 - 0.05
    
    # Cosine law for elbow angle
    cos_elbow = (reach_2d*reach_2d - l1*l1 - l2*l2) / (2 * l1 * l2)
    cos_elbow = np.clip(cos_elbow, -1, 1)  # Ensure valid range
    
    # Joint 2: Elbow angle (negative because UR5 elbow bends "backwards")
    joint_angles[2] = -np.arccos(cos_elbow)
    print(f"🧮 Elbow angle: {np.degrees(joint_angles[2]):.1f}°")
    
    # Joint 1: Shoulder angle
    angle_to_target = np.arctan2(height_from_base, horizontal_dist)
    angle_from_elbow = np.arctan2(l2 * np.sin(-joint_angles[2]), l1 + l2 * np.cos(-joint_angles[2]))
    joint_angles[1] = angle_to_target - angle_from_elbow
    print(f"🧮 Shoulder angle: {np.degrees(joint_angles[1]):.1f}°")
    
    # Wrist joints: Keep end effector pointing down
    # Joint 3: Wrist 1 - compensate for shoulder and elbow to keep wrist level
    joint_angles[3] = -(joint_angles[1] + joint_angles[2])
    
    # Joint 4: Wrist 2 - keep it at -90 degrees (pointing down)
    joint_angles[4] = -np.pi/2
    
    # Joint 5: Wrist 3 - rotation around end effector (keep at 0)
    joint_angles[5] = 0
    
    print(f"🧮 Wrist angles: w1={np.degrees(joint_angles[3]):.1f}°, w2={np.degrees(joint_angles[4]):.1f}°, w3={np.degrees(joint_angles[5]):.1f}°")
    print(f"🧮 IK OUTPUT: {[f'{np.degrees(j):.1f}°' for j in joint_angles]}")
    
    return joint_angles

def apply_vla_action_to_robot(robot, vla_action):
    """
    Get current EE pose, apply VLA delta, un-normalize to workspace, and command robot.
    """
    try:
        if not robot:
            print("❌ No robot available")
            return

        # --- STEP 1: Get the robot's current end-effector pose ---
        # Using the CORRECT path you discovered!
        ee_prim_path = "/World/UR5/wrist_3_link" 
        
        # Correctly initialize the XFormPrim
        ee_prim = XFormPrim(ee_prim_path)
        
        # CORRECT API CALL: Use get_world_poses() and take the first element
        positions, orientations = ee_prim.get_world_poses()
        
        # Check if the returned arrays are valid and not empty
        if positions is None or len(positions) == 0:
            print("⚠️ Could not get current end-effector pose. Aborting action.")
            return
            
        current_pos = positions[0]
        current_rot_quat = orientations[0]
        
        # --- STEP 2: Get the normalized action delta from VLA ---
        norm_pos_delta = vla_action["position"]
        
        # --- STEP 3: Coordinate System Transformation & Scaling ---
        ACTION_SCALE = 0.1 
        
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

        # --- STEP 6: Compute IK and command the robot ---
        current_joints = robot.get_joint_positions()
        if current_joints is None:
            current_joints = [0, -1.57, 1.57, -1.57, -1.57, 0]

        target_joints = simple_ur5_ik(clamped_target_pos, current_joints)
        
        joint_limits = [
            [-2*np.pi, 2*np.pi], [-np.pi, np.pi], [-np.pi, np.pi],
            [-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]
        ]
        for i in range(len(target_joints)):
            target_joints[i] = np.clip(target_joints[i], joint_limits[i][0], joint_limits[i][1])
        
        alpha = 0.3
        smooth_joints = []
        for i in range(len(current_joints)):
            smooth_joint = current_joints[i] + alpha * (target_joints[i] - current_joints[i])
            smooth_joints.append(smooth_joint)
        
        robot.set_joint_positions(smooth_joints)
        print(f"🎯 Applied joints: {[f'{j:.2f}' for j in smooth_joints]}")

    except Exception as e:
        print(f"❌ Robot control error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()