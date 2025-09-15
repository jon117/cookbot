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
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.numpy.rotations as rot_utils

def main():
    print("🚀 Isaac Sim 5.0 + VLA test (Timer-based capture)")
    print("Press Ctrl+C to stop")
    
    # Create output directory for images
    output_dir = "isaac_sim_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Images will be saved to: {output_dir}/")
    
    # 1. Create World
    my_world = World(stage_units_in_meters=1.0)
    
    # 2. Add ground plane
    try:
        usd_path = get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path="/ground_plane")
        print("✅ Ground plane added")
    except Exception as e:
        print(f"⚠️ Ground plane failed: {e}")
    
    # 3. Add robot
    try:
        robot_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/UR5")
        print("✅ UR5 robot loaded")
    except Exception as e:
        print(f"⚠️ Robot loading failed: {e}")
    
    # 4. NO MORE GIANT BLOCKS! Clean scene with just robot
    print("✅ Clean scene (no blocks)")
    
    # 5. Create camera with exact position/orientation from Isaac Sim GUI
    # NOTE: Isaac Sim expects quaternion in [X, Y, Z, W] format, NOT [W, X, Y, Z]!
    # Your GUI values: W:0.01346 X:0.00707 Y:0.46516 Z:0.88509
    # So we reorder to: [X, Y, Z, W] = [0.00707, 0.46516, 0.88509, 0.01346]
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([0.0, -2.0, 2.0]),  # From GUI
        #orientation=np.array([0.38268, 0.0, 0.0, 0.92388]),  # Reordered to X,Y,Z,W - but this is absolute trash isaac sim DGAF about this. random shit everywhere.
        name="camera",
        frequency=20,
        resolution=(512, 512)
    )
    
    my_world.scene.add(camera)

    # use this instead to set local pose to the ACTUAL values you get in the GUI. jesus.
    camera.set_local_pose(
        orientation=rot_utils.euler_angles_to_quats(np.array([45,0,0]), degrees=True),
        camera_axes="usd"
    )
    
    # Set focal length after camera creation (Isaac Sim uses cm!)
    try:
        # Focal length in cm: 1.5cm = 15mm
        camera.set_focal_length(1.5)
        print("✅ Camera positioned, focal length set to 1.5cm (15mm)")
    except Exception as e:
        print(f"⚠️ Could not set focal length: {e}")
        print("✅ Camera positioned (default focal length)")
    
    # 6. Initialize world
    my_world.reset()
    print("✅ World initialized")
    
    # 7. Timer-based capture loop (5 FPS = 0.2 seconds between captures)
    print("\n🔄 Starting timer-based capture at 5 FPS...")
    print("   💡 Tip: Move the robot around in Isaac Sim to see different images!")
    print("   ⏹️  Press Ctrl+C to stop\n")
    
    capture_count = 0
    target_fps = 2
    capture_interval = 1.0 / target_fps  # 0.2 seconds
    last_capture_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Always step the simulation (let it run at full speed)
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
                        
                        # Send to VLA
                        send_to_vla(rgb_image, "extend the robot arm upwards")
                        
                    else:
                        print("⚠️ No camera data")
                        
                except Exception as e:
                    print(f"❌ Camera capture failed: {e}")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)  # 10ms sleep, allows sim to run ~100fps while capturing at 5fps
            
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
        filename = f"isaac_sim_{capture_count:04d}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        pil_image = PILImage.fromarray(image)
        pil_image.save(filepath)
        
        print(f"💾 Saved: {filename}")
        
        # Also save the VLA-sized version
        vla_filename = f"isaac_sim_{capture_count:04d}_{timestamp}_vla_256x256.png"
        vla_filepath = os.path.join(output_dir, vla_filename)
        
        vla_image = pil_image.resize((256, 256))
        vla_image.save(vla_filepath)
        
        print(f"💾 Saved VLA version: {vla_filename}")
        
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

def send_to_vla(image, instruction):
    """Send image and instruction to VLA at localhost:8000/act"""
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
        
        response = requests.post("http://localhost:8000/act", json=payload, timeout=5)  # Shorter timeout
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ VLA responded: {result}")
            
            if isinstance(result, str):
                print(f"   Response: {result}")
            elif hasattr(result, '__len__') and len(result) >= 7:
                x, y, z, rx, ry, rz, gripper = result[:7]
                print(f"   🎯 Action: pos({x:.2f},{y:.2f},{z:.2f}) rot({rx:.2f},{ry:.2f},{rz:.2f}) grip={gripper:.2f}")
            else:
                print(f"   Data: {result}")
                
        else:
            print(f"❌ HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ VLA server unreachable (localhost:8000/act)")
    except Exception as e:
        print(f"❌ VLA error: {e}")

if __name__ == "__main__":
    main()