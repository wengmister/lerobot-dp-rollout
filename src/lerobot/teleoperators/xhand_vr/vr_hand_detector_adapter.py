"""
VR Hand Detector Adapter - Drop-in replacement for VRHandDetector using VRMessageRouter

This adapter wraps the new VRMessageRouter C++ implementation to provide the same interface
as the original VRHandDetector, enabling seamless integration with existing retargeting code.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np

import vr_message_router


logger = logging.getLogger(__name__)


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def adaptive_retargeting_xhand(landmarks):
    """
    Apply adaptive pinky retargeting specifically for XHand robot.
    Compensates for human-to-robot finger length differences by using
    adaptive scaling based on finger extension state.
    
    Args:
        landmarks: np.ndarray of shape (21, 3) with hand landmarks
        
    Returns:
        np.ndarray of shape (21, 3) with retargeted landmarks
    """
    landmarks = landmarks.copy()
    
    pinky_mcp = 17   # PINKY_MCP (base)
    pinky_pip = 18   # PINKY_PIP
    pinky_dip = 19   # PINKY_DIP  
    pinky_tip = 20   # PINKY_TIP
    
    # Adaptive scaling based on finger curl state
    # Calculate finger extension (distance from MCP to TIP)
    pinky_extension = np.linalg.norm(landmarks[pinky_tip] - landmarks[pinky_mcp])
    
    # Scale more when extended (for reaching), less when curled (for fist-making)
    # Tuned specifically for XHand robot kinematics
    max_extension = 0.10
    min_extension = 0.03
    
    # Normalize extension ratio (0.0 = fully curled, 1.0 = fully extended)
    extension_ratio = np.clip((pinky_extension - min_extension) / (max_extension - min_extension), 0.0, 1.0)
    
    # Adaptive scaling: more scaling when extended, less when curled
    base_scale = 1.2   # Minimum scaling for curled positions
    max_scale = 2.2    # Maximum scaling for extended positions
    
    adaptive_scale = base_scale + (max_scale - base_scale) * extension_ratio
    
    # Apply same adaptive scaling to all segments
    mcp_to_pip_scale = adaptive_scale
    pip_to_dip_scale = adaptive_scale  
    dip_to_tip_scale = adaptive_scale
    
    # Apply progressive scaling along kinematic chain
    # Start from MCP (base remains unchanged) and extend each segment
    
    # Extend MCP->PIP segment
    mcp_to_pip_vector = landmarks[pinky_pip] - landmarks[pinky_mcp]
    landmarks[pinky_pip] = landmarks[pinky_mcp] + mcp_to_pip_vector * mcp_to_pip_scale
    
    # Extend PIP->DIP segment (using new PIP position)
    pip_to_dip_vector = landmarks[pinky_dip] - landmarks[pinky_pip]  
    landmarks[pinky_dip] = landmarks[pinky_pip] + pip_to_dip_vector * pip_to_dip_scale
    
    # Extend DIP->TIP segment (using new DIP position)
    dip_to_tip_vector = landmarks[pinky_tip] - landmarks[pinky_dip]
    landmarks[pinky_tip] = landmarks[pinky_dip] + dip_to_tip_vector * dip_to_tip_scale
    
    return landmarks


class VRHandDetectorAdapter:
    """
    Adapter that wraps VRMessageRouter to provide VRHandDetector-compatible interface.
    
    This allows existing hand retargeting code to work with the new VR message routing
    pipeline without modification.
    """
    
    def __init__(self, hand_type: str = "Right", robot_name: str = "xhand", 
                 use_tcp: bool = True, tcp_port: int = 8000, verbose: bool = False, router: vr_message_router.VRMessageRouter = None):
        """
        Initialize the VR hand detector adapter.
        
        Args:
            hand_type: Hand type ("Right" or "Left") - currently only Right supported
            robot_name: Robot name (for compatibility, not used)
            use_tcp: Whether to use TCP (for compatibility, always True)
            tcp_port: TCP port for VR message router
            verbose: Enable verbose logging
        """
        self.hand_type = hand_type
        self.robot_name = robot_name
        self.tcp_port = tcp_port
        self.verbose = verbose
        self.operator2mano = OPERATOR2MANO_RIGHT
        self.router = router
        
        # Import and initialize VR message router
        try:           
            # Start TCP server
            if not self.router.start_tcp_server():
                logger.error("Failed to start VR message router TCP server")
                self.router_available = False
            else:
                logger.info(f"VR message router started on port {tcp_port}")
                self.router_available = True
                
        except ImportError as e:
            logger.error(f"VRMessageRouter not available: {e}. Please build the C++ module.")
            self.router = None
            self.router_available = False
        except Exception as e:
            logger.error(f"Failed to initialize VR message router: {e}")
            self.router = None
            self.router_available = False
    
    def detect(self) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        Detect hand pose from VR input.
        
        Returns:
            tuple: (None, joint_pos, keypoint_2d, None) matching VRHandDetector interface
                - joint_pos: numpy array of shape (21, 3) with 3D landmark positions
                - keypoint_2d: None (not used by retargeting)
        """
        if not self.router_available or self.router is None:
            return None, None, None, None
        
        try:
            # Get messages from router
            messages = self.router.get_messages()
            
            # Check if we have valid landmarks data
            if not messages.landmarks_valid or not messages.landmarks_data.valid:
                if self.verbose:
                    logger.debug("No valid landmarks data available")
                return None, None, None, None
            
            # Convert landmarks to numpy array
            landmarks = messages.landmarks_data.landmarks
            if len(landmarks) == 0:
                if self.verbose:
                    logger.debug("Empty landmarks data")
                return 0, None, None, None
            
            # Convert to numpy array (N, 3) where N should be 21 for hand landmarks
            joint_pos = np.array(landmarks, dtype=np.float32)
            
            if joint_pos.shape[0] != 21:
                logger.warning(f"Expected 21 landmarks, got {joint_pos.shape[0]}")
                return None, None, None, None
            
            if joint_pos.shape[1] != 3:
                logger.warning(f"Expected 3D landmarks, got shape {joint_pos.shape}")
                return None, None, None, None
            
            if self.verbose:
                logger.debug(f"Detected hand landmarks: {joint_pos.shape}")

            # Apply same transformations as original VRHandDetector
            keypoint_3d_array = joint_pos.copy()  # Fix: use joint_pos not landmarks
            
            # Scale to match MediaPipe coordinate range (from original)
            keypoint_3d_array *= 1.05
            
            # Convert coordinate system for right hand (from original)
            if self.hand_type == "Right":
                keypoint_3d_array[:, 0] = -keypoint_3d_array[:, 0]
            
            # Make wrist the origin (same as MediaPipe processing)
            keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

            # Estimate hand orientation using the shifted array
            wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)  # Fix: use self and correct input

            # Transform to MANO coordinate system
            joint_pos = keypoint_3d_array @ wrist_rot @ self.operator2mano
            
            # Apply robot-specific retargeting if specified
            if self.robot_name and "xhand" in self.robot_name:
                joint_pos = adaptive_retargeting_xhand(joint_pos)
             
            # Return in VRHandDetector format: (None, joint_pos, keypoint_2d, None)
            # keypoint_2d is set to None since retargeting only uses joint_pos
            return None, joint_pos, None, None
            
        except Exception as e:
            logger.error(f"Error in VR hand detection: {e}")
            return None, None, None, None
    
    def get_wrist_data(self) -> Optional[dict]:
        """
        Get wrist pose data (bonus feature not in original VRHandDetector).
        
        Returns:
            dict with position, quaternion, and fist_state, or None if not available
        """
        if not self.router_available or self.router is None:
            return None
        
        try:
            messages = self.router.get_messages()
            
            if not messages.wrist_valid or not messages.wrist_data.valid:
                return None
            
            wrist_data = messages.wrist_data
            return {
                'position': np.array(wrist_data.position),
                'quaternion': np.array(wrist_data.quaternion),  # [x, y, z, w]
                'fist_state': wrist_data.fist_state
            }
            
        except Exception as e:
            logger.error(f"Error getting wrist data: {e}")
            return None
    
    def get_status(self) -> dict:
        """
        Get router status information.
        
        Returns:
            dict with connection and data validity status
        """
        if not self.router_available or self.router is None:
            return {
                'router_available': False,
                'tcp_connected': False,
                'running': False,
                'wrist_valid': False,
                'landmarks_valid': False
            }
        
        try:
            status = self.router.get_status()
            status['router_available'] = True
            return status
        except Exception as e:
            logger.error(f"Error getting router status: {e}")
            return {'router_available': False}
        
    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'router') and self.router is not None:
            try:
                self.router.stop()
            except:
                pass