#!/usr/bin/env python3
"""
trying to figure out how to load a robot with a gripper and control it
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np


# Core API imports
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path

# Camera imports
from omni.isaac.sensor import Camera
from omni.kit.viewport.utility import get_viewport_from_window_name
import omni.isaac.core.utils.rotations as rot_utils
from omni.isaac.core.prims import XFormPrim

# Dynamic control interface for velocity control
from omni.isaac.dynamic_control import _dynamic_control

def main():
    """Main function - creates scene and runs loop."""

    # 1. Create the world
    my_world = World(stage_units_in_meters=1.0)
    print("✅ World created")
    
    # 2. Add room and table
    try:
        room_path = get_assets_root_path() + "/Isaac/Environments/Simple_Room/simple_room.usd"
        add_reference_to_stage(usd_path=room_path, prim_path="/World/Room")
        print("✅ Room and table added")
    except Exception as e:
        print(f"⚠️ Room loading failed (continuing without): {e}")

    # 3. Add robot with gripper
    robot_asset_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
    add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/UR5")
    robot = Robot(prim_path="/World/UR5", name="ur5_robot")
    my_world.scene.add(robot)
    print("✅ UR5 Robot added")

    # 4. Add gripper to the robot
    try:
        print("🔧 Loading Robotiq 2F-140 gripper...")
        
        #https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd
        gripper_path = get_assets_root_path() + "/Isaac/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd"
        #gripper_prim_path = "/World/UR5/tool0/gripper"  # original path
        gripper_prim_path = "/World/UR5/wrist_3_link/gripper"
        
        add_reference_to_stage(usd_path=gripper_path, prim_path=gripper_prim_path)

        # Rotate gripper 90 degrees around X-axis
        gripper_prim = XFormPrim(gripper_prim_path)
        
        # 90 degrees around X-axis in quaternion (w, x, y, z)
        rotation_quat = rot_utils.euler_angles_to_quat(np.array([0, -90, 0]), degrees=True)
        gripper_prim.set_local_pose(orientation=rotation_quat)
        
        print("✅ Robotiq 2F-140 gripper loaded and rotated 90° around X-axis!")
        
    except Exception as e:
        print(f"⚠️ Gripper loading failed: {e}")
        print("   Continuing without gripper...")


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
    
    # 6.5. Debug: Check scene structure
    print("\n🔍 Scene structure analysis:")
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage:
        print("   Available robot links:")
        for prim in stage.TraverseAll():
            path_str = prim.GetPath().pathString
            if path_str.startswith("/World/UR5") and ("link" in path_str or "tool" in path_str):
                print(f"     {path_str}: {prim.GetTypeName()}")
    
    # 7. Start simulation for dynamic control
    import omni
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("✅ Simulation started")
    
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
                            
                            # List all DOF names to see what we have
                            print("🔍 Available DOFs:")
                            for i in range(num_dofs):
                                dof = dc.get_articulation_dof(articulation, i)
                                dof_name = dc.get_dof_name(dof)
                                print(f"   DOF {i}: {dof_name}")
                            
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
                                        
                                # Also handle prismatic joints (for gripper)
                                elif prim.GetPath().pathString.startswith("/World/UR5") and prim.GetTypeName() == "PhysicsPrismaticJoint":
                                    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                                    if drive:
                                        drive.GetStiffnessAttr().Set(0.0)  # Zero stiffness for velocity control
                                        drive.GetDampingAttr().Set(100.0)  # Lower damping for gripper
                            
                            print("✅ Robot switched to velocity control mode")
                        else:
                            print("❌ Could not get articulation handle")
                except Exception as e:
                    print(f"⚠️ Dynamic control init failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # send simple velocity commands every tick
            if dc and articulation:
                try:
                    num_dofs = dc.get_articulation_dof_count(articulation)
                    
                    # Create velocity commands for all DOFs
                    velocity_commands = [0.0] * num_dofs
                    
                    # Robot arm movements (first 6 DOFs)
                    if num_dofs >= 6:
                        velocity_commands[0] = 0.1  # Base rotation
                        velocity_commands[1] = 0.05  # Shoulder lift
                        velocity_commands[2] = -0.05  # Elbow
                        # Keep other arm joints stable
                        velocity_commands[3] = 0.0  # Wrist 1
                        velocity_commands[4] = 0.0  # Wrist 2  
                        velocity_commands[5] = 0.0  # Wrist 3
                    
                    # Gripper control (DOFs beyond 6)
                    gripper_open_close_cycle = (hash(str(int(timeline.get_current_time() * 1000))) % 4000) / 2000.0  # 2 second cycle
                    gripper_velocity = 0.1 if gripper_open_close_cycle < 1.0 else -0.1  # Open then close
                    
                    for i in range(6, num_dofs):
                        velocity_commands[i] = gripper_velocity
                    
                    # Apply velocity commands
                    for i in range(num_dofs):
                        dc.set_dof_velocity_target(dc.get_articulation_dof(articulation, i), velocity_commands[i])
                    
                    print(f"🎮 Arm velocities: {[f'{v:.2f}' for v in velocity_commands[:6]]}")
                    if num_dofs > 6:
                        print(f"🤏 Gripper velocities: {[f'{v:.2f}' for v in velocity_commands[6:]]}")
                        
                except Exception as e:
                    print(f"⚠️ Could not send velocity commands: {e}")


    except Exception as e:
        print(f"\n⏹️ Stopping simulation: {e}")

if __name__ == "__main__":
    main()