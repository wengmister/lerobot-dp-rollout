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

logger = logging.getLogger(__name__)


class VRHandDetectorAdapter:
    """
    Adapter that wraps VRMessageRouter to provide VRHandDetector-compatible interface.
    
    This allows existing hand retargeting code to work with the new VR message routing
    pipeline without modification.
    """
    
    def __init__(self, hand_type: str = "Right", robot_name: str = "xhand", 
                 use_tcp: bool = True, tcp_port: int = 8000, verbose: bool = False):
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
        
        # Import and initialize VR message router
        try:
            # Add the franka_xhand_teleoperator build path to sys.path
            build_path = Path(__file__).parent.parent.parent.parent.parent / "franka_xhand_teleoperator" / "build"
            if build_path.exists():
                sys.path.insert(0, str(build_path))
            
            import vr_message_router
            
            # Create router config
            config = vr_message_router.VRRouterConfig()
            config.tcp_port = tcp_port
            config.verbose = verbose
            config.message_timeout_ms = 100.0  # 100ms timeout
            
            # Initialize router
            self.router = vr_message_router.VRMessageRouter(config)
            self.router_available = True
            
            # Start TCP server
            if not self.router.start_tcp_server():
                logger.error("Failed to start VR message router TCP server")
                self.router_available = False
            else:
                logger.info(f"VR message router started on port {tcp_port}")
                
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
                return None, None, None, None
            
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
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'router') and self.router is not None:
            try:
                self.router.stop()
            except:
                pass