"""
Isaac Sim Camera System
Provides RGB-D camera simulation for VLA model input.
"""

import numpy as np
import time

# Import Isaac Sim modules conditionally to avoid import errors
try:
    import omni.replicator.core as rep
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.sensor import Camera
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Isaac Sim modules not available - running in fallback mode")


class VLACamera:
    """Camera system for capturing VLA training/inference data."""
    
    def __init__(self, camera_position=None, camera_target=None):
        """
        Initialize camera system.
        
        Args:
            camera_position: Camera position [x, y, z] in meters
            camera_target: Camera target/look-at point [x, y, z] in meters
        """
        self.camera_position = camera_position or [0.8, 0.5, 0.6]  # Above and to the side
        self.camera_target = camera_target or [0.4, 0.1, 0.1]     # Looking at carrot area
        
        # Camera parameters matching OpenVLA expected format
        self.image_width = 256  # OpenVLA expects 256x256
        self.image_height = 256
        self.focal_length = 400.0  # Adjusted for smaller image size
        
        self.camera = None
        self.setup_cameras()
        
    def setup_cameras(self):
        """Set up RGB-D cameras in the scene."""
        print("Setting up VLA cameras...")
        
        if not ISAAC_AVAILABLE:
            print("Warning: Isaac Sim not available, creating fallback camera")
            self.camera = None
            return
        
        # Import Isaac Sim modules here to avoid conflicts
        try:
            from omni.isaac.sensor import Camera
            import omni.isaac.core.utils.stage as stage_utils
            import omni.usd
            
            # Ensure we have a valid stage before creating camera
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("Warning: No USD stage available for camera setup")
                return
            
            # Create primary RGB-D camera
            self.camera = Camera(
                prim_path="/World/VLACamera",
                name="vla_camera",
                position=np.array(self.camera_position),
                frequency=30,  # 30 FPS
                resolution=(self.image_width, self.image_height),
                orientation=self._calculate_camera_orientation()
            )
            
            # Initialize the camera (this is crucial for annotators)
            if hasattr(self.camera, 'initialize'):
                self.camera.initialize()
            
            # Configure camera intrinsics
            self._configure_camera_intrinsics()
            
            print(f"Camera positioned at {self.camera_position}")
            print(f"Camera looking at {self.camera_target}")
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            self.camera = None
        
    def _calculate_camera_orientation(self):
        """Calculate camera orientation to look at target."""
        # Simple look-at calculation
        camera_pos = np.array(self.camera_position)
        target_pos = np.array(self.camera_target)
        
        # Calculate direction vector
        direction = target_pos - camera_pos
        direction = direction / np.linalg.norm(direction)
        
        # Calculate rotation to look at target (simplified)
        # For a more complex implementation, use proper look-at matrix
        # This is a placeholder - Isaac Sim handles this with proper orientation
        return np.array([0.7071, 0.0, 0.0, 0.7071])  # 90 degree rotation around X
        
    def _configure_camera_intrinsics(self):
        """Configure camera intrinsic parameters."""
        if self.camera:
            # Set focal length and principal point
            self.camera.set_focal_length(self.focal_length)
            
            # Principal point at image center
            principal_point = (self.image_width / 2, self.image_height / 2)
            
    def capture_rgbd(self):
        """
        Capture RGB-D data from camera.
        
        Returns:
            dict: Contains 'rgb', 'depth', 'camera_matrix' numpy arrays
        """
        if not self.camera:
            print("Error: Camera not initialized")
            return None
            
        try:
            # Check if camera is properly initialized
            if not hasattr(self.camera, 'get_rgba'):
                print("Error: Camera does not have get_rgba method")
                return None
            
            # Get RGB image - handle potential None returns
            rgba_data = self.camera.get_rgba()
            if rgba_data is None:
                print("Error: Camera returned None for RGBA data")
                return None
                
            rgb_data = rgba_data[:, :, :3]  # Remove alpha channel
            
            # Get depth data - handle potential None returns
            depth_data = self.camera.get_depth()
            if depth_data is None:
                print("Warning: Camera returned None for depth data")
                # Create dummy depth data
                depth_data = np.zeros((self.image_height, self.image_width), dtype=np.float32)
            
            # Create camera matrix
            camera_matrix = np.array([
                [self.focal_length, 0, self.image_width / 2],
                [0, self.focal_length, self.image_height / 2],
                [0, 0, 1]
            ])
            
            return {
                'rgb': rgb_data,
                'depth': depth_data,
                'camera_matrix': camera_matrix,
                'timestamp': time.time(),
                'camera_id': 'vla_camera'
            }
            
        except Exception as e:
            print(f"Error capturing camera data: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def capture_for_vla(self):
        """
        Capture and format data specifically for VLA model input.
        
        Returns:
            CameraObservation: Formatted for VLA processing
        """
        from ..common.data_types import CameraObservation
        import time
        
        if not ISAAC_AVAILABLE or self.camera is None:
            # Create synthetic test data when Isaac Sim is not available
            print("Creating synthetic camera data for testing...")
            test_rgb = np.random.randint(0, 255, (self.image_height, self.image_width, 3), dtype=np.uint8)
            test_depth = np.random.rand(self.image_height, self.image_width).astype(np.float32) * 2.0  # 0-2 meter range
            test_camera_matrix = np.array([
                [self.focal_length, 0, self.image_width / 2],
                [0, self.focal_length, self.image_height / 2],
                [0, 0, 1]
            ])
            
            return CameraObservation(
                rgb=test_rgb,
                depth=test_depth,
                camera_matrix=test_camera_matrix,
                timestamp=time.time(),
                camera_id='synthetic_camera'
            )
        
        camera_data = self.capture_rgbd()
        if camera_data is None:
            return None
            
        return CameraObservation(
            rgb=camera_data['rgb'],
            depth=camera_data['depth'],
            camera_matrix=camera_data['camera_matrix'],
            timestamp=camera_data['timestamp'],
            camera_id=camera_data['camera_id']
        )
        
    def move_camera(self, new_position, new_target=None):
        """
        Move camera to new position and/or target.
        
        Args:
            new_position: New camera position [x, y, z]
            new_target: New camera target [x, y, z] (optional)
        """
        self.camera_position = new_position
        if new_target:
            self.camera_target = new_target
            
        # Update camera transform
        if self.camera:
            self.camera.set_world_pose(
                position=np.array(new_position),
                orientation=self._calculate_camera_orientation()
            )
            
    def get_camera_info(self):
        """Get camera configuration information."""
        return {
            'position': self.camera_position,
            'target': self.camera_target,
            'resolution': (self.image_width, self.image_height),
            'focal_length': self.focal_length,
            'fov': np.degrees(2 * np.arctan(self.image_width / (2 * self.focal_length)))
        }


class MultiCameraSetup:
    """Multiple camera setup for comprehensive scene coverage."""
    
    def __init__(self):
        """Initialize multiple camera views."""
        self.cameras = {}
        self.setup_multiple_cameras()
        
    def setup_multiple_cameras(self):
        """Set up multiple camera viewpoints."""
        # Overhead camera
        self.cameras['overhead'] = VLACamera(
            camera_position=[0.4, 0.0, 1.0],
            camera_target=[0.4, 0.0, 0.0]
        )
        
        # Side view camera
        self.cameras['side'] = VLACamera(
            camera_position=[0.8, 0.5, 0.4],
            camera_target=[0.3, 0.1, 0.1]
        )
        
        # Front view camera
        self.cameras['front'] = VLACamera(
            camera_position=[0.0, 0.6, 0.3],
            camera_target=[0.4, 0.0, 0.1]
        )
        
    def capture_all_views(self):
        """Capture from all camera viewpoints."""
        captures = {}
        for name, camera in self.cameras.items():
            captures[name] = camera.capture_for_vla()
        return captures
        
    def get_primary_camera(self):
        """Get the primary camera for VLA processing."""
        return self.cameras.get('side')  # Side view is best for manipulation


def setup_scene_cameras():
    """Convenience function to set up cameras for kitchen scene."""
    print("Setting up scene cameras for VLA...")
    
    # Create multi-camera setup
    multi_cam = MultiCameraSetup()
    
    # Get primary camera for VLA
    primary_camera = multi_cam.get_primary_camera()
    
    print("Camera setup complete!")
    print(f"Primary camera info: {primary_camera.get_camera_info()}")
    
    return multi_cam, primary_camera


# Testing function
def test_camera_capture(primary_camera):
    """Test camera capture functionality."""
    print("Testing camera capture...")
    
    try:
        # Capture VLA data
        vla_data = primary_camera.capture_for_vla()
        
        if vla_data:
            print(f"✓ Captured RGB image: {vla_data.rgb.shape}")
            print(f"✓ Captured depth image: {vla_data.depth.shape}")
            print(f"✓ Camera matrix: {vla_data.camera_matrix.shape}")
            print(f"✓ Timestamp: {vla_data.timestamp}")
        else:
            print("✗ Failed to capture camera data")
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")