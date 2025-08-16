import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from dex_retargeting.constants import get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_xhand_vr import XHandVRTeleoperatorConfig
from .vr_hand_detector_adapter import VRHandDetectorAdapter
import vr_message_router

logger = logging.getLogger(__name__)


class XHandVRTeleoperator(Teleoperator):
    """
    Teleoperator that bridges VR hand tracking with XHand robot control.
    
    This teleoperator receives hand tracking data from a VR application via TCP,
    processes it through the dex-retargeting pipeline with coordinate transformations,
    and outputs robot joint commands compatible with the XHand robot.
    
    The data flow is:
    1. VR app sends 21 hand landmarks via TCP to VRMessageRouter
    2. VRHandDetectorAdapter receives and transforms landmarks to MANO coordinate system
    3. Dex-retargeting converts MANO poses to robot joint positions
    4. Joint positions are mapped to XHand joint order and sent as actions
    
    Requires:
    - VR message router C++ module built and available
    - VR application sending hand landmarks to configured TCP port
    - ADB reverse port forwarding set up for Meta Quest devices
    
    Attributes:
        config_class: Configuration class for this teleoperator
        name: Unique identifier for this teleoperator type
    """
    
    config_class = XHandVRTeleoperatorConfig
    name = "xhand_vr"
    
    def __init__(self, config: XHandVRTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        
        # Set default robot directory if not provided
        if config.robot_dir is None:
            robot_dir = Path(__file__).parent.parent.parent.parent.parent / "dex_retargeting" / "assets" / "robots" / "hands"
        else:
            robot_dir = Path(config.robot_dir)
        
        # Initialize retargeting system
        config_path = get_default_config_path(config.robot_name, config.retargeting_type, config.hand_type)
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        self.retargeting = RetargetingConfig.load_from_file(config_path).build()
        
        # Set up joint mapping from retargeting output to XHand joint order
        self._setup_joint_mapping()
        
        # Initialize VR hand detector adapter
        try:
            # Create router config
            router_config = vr_message_router.VRRouterConfig()
            router_config.tcp_port = config.vr_tcp_port
            router_config.verbose = config.vr_verbose
            router_config.message_timeout_ms = 100.0  # 100ms timeout
            
            # Initialize router
            self.router = vr_message_router.VRMessageRouter(router_config)
            self.router_available = True

            hand_type_str = "Right" if config.hand_type == config.hand_type.right else "Left"
            self.detector = VRHandDetectorAdapter(
                hand_type=hand_type_str, 
                robot_name=str(config.robot_name), 
                tcp_port=config.vr_tcp_port,
                verbose=config.vr_verbose,
                router=self.router
            )
        except ImportError as e:
            logger.error(f"VRHandDetectorAdapter not available: {e}. Please build the VR message router.")
            self.detector = None
            raise ImportError("VR message router C++ module not found. Please build franka_xhand_teleoperator.")
        except (RuntimeError, OSError) as e:
            logger.error(f"Failed to start VR message router: {e}")
            self.detector = None
            raise RuntimeError(f"VR message router startup failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error setting up VR teleoperator: {e}")
            self.detector = None
            raise RuntimeError(f"VR teleoperator setup failed: {e}")
        
        # Control settings
        self.control_frequency = config.control_frequency
        self.smoothing_alpha = config.smoothing_alpha
        self.last_joint_positions = None
        self._is_connected = False
        
        logger.info(f"XHand VR teleoperator initialized for {config.hand_type} hand")
    
    @property
    def action_features(self) -> dict:
        """Action features for XHand robot (12 joint positions)."""
        return {f"joint_{i}.pos": float for i in range(12)}
    
    @property
    def feedback_features(self) -> dict:
        """Feedback features (empty for VR teleoperator)."""
        return {}
    
    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self._is_connected
    
    @property 
    def is_calibrated(self) -> bool:
        """VR teleoperator doesn't require calibration."""
        return True
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect the teleoperator."""
        if self.detector is None:
            raise RuntimeError("VR hand detector not available")
        
        self._is_connected = True
        logger.info("XHand VR teleoperator connected")
    
    def calibrate(self) -> None:
        """Calibrate the teleoperator (no-op for VR)."""
        pass
    
    def configure(self) -> None:
        """Configure the teleoperator (no-op for VR).""" 
        pass
    
    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to teleoperator (no-op for VR)."""
        pass
    
    def disconnect(self) -> None:
        """Disconnect the teleoperator."""
        self._is_connected = False
        logger.info("XHand VR teleoperator disconnected")
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get action from VR hand tracking.
        
        Returns:
            dict: XHand action format {"joint_0.pos": float, ...}
        """
        if not self._is_connected:
            raise RuntimeError("Teleoperator not connected")
        
        if self.detector is None:
            raise RuntimeError("VR hand detector not available")
        
        try:
            # Detect hand pose
            _, joint_pos, _, _ = self.detector.detect()
            
            if joint_pos is None:
                # Return previous action or home position if no hand detected
                if self.last_joint_positions is not None:
                    return self._convert_to_xhand_action(self.last_joint_positions)
                else:
                    # Return home position
                    return {f"joint_{i}.pos": 0.0 for i in range(12)}
            
            # Retarget to robot joint positions
            retargeting_type = self.retargeting.optimizer.retargeting_type
            indices = self.retargeting.optimizer.target_link_human_indices
            
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # Get retargeted joint positions
            qpos = self.retargeting.retarget(ref_value)
            
            # Apply smoothing if we have previous positions
            if self.last_joint_positions is not None:
                qpos = self._apply_smoothing(qpos, self.last_joint_positions)
            
            self.last_joint_positions = qpos.copy()
            
            # Map retargeting output to correct XHand joint order
            xhand_joint_positions = self._map_to_xhand_order(qpos)
            
            # Convert to XHand action format
            return self._convert_to_xhand_action(xhand_joint_positions)
            
        except ValueError as e:
            logger.warning(f"Invalid VR data format: {e}")
            # Return safe default action
            return {f"joint_{i}.pos": 0.0 for i in range(12)}
        except np.linalg.LinAlgError as e:
            logger.warning(f"Hand pose calculation error: {e}")
            # Return safe default action  
            return {f"joint_{i}.pos": 0.0 for i in range(12)}
        except Exception as e:
            logger.warning(f"Unexpected error getting VR action: {e}")
            # Return safe default action
            return {f"joint_{i}.pos": 0.0 for i in range(12)}
    
    def _setup_joint_mapping(self) -> None:
        """Set up mapping from retargeting output to XHand joint order."""
        # Get retargeting joint names (URDF order)
        retargeting_joint_names = self.retargeting.joint_names
        
        # Desired XHand order
        desired_xhand_joint_names = [
            'right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2',
            'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
            'right_hand_mid_joint1', 'right_hand_mid_joint2',
            'right_hand_ring_joint1', 'right_hand_ring_joint2',
            'right_hand_pinky_joint1', 'right_hand_pinky_joint2'
        ]
        
        # Create mapping from retargeting output indices to desired XHand indices
        self.retargeting_to_xhand = []
        for desired_joint in desired_xhand_joint_names:
            if desired_joint in retargeting_joint_names:
                self.retargeting_to_xhand.append(retargeting_joint_names.index(desired_joint))
            else:
                logger.error(f"Desired joint {desired_joint} not found in retargeting joints!")
                self.retargeting_to_xhand.append(0)  # fallback
        
        self.retargeting_to_xhand = np.array(self.retargeting_to_xhand)
        logger.info(f"Joint mapping setup: {len(self.retargeting_to_xhand)} joints mapped")
    
    def _map_to_xhand_order(self, qpos: np.ndarray) -> np.ndarray:
        """Map retargeting output to correct XHand joint order."""
        # Map retargeting output to correct XHand joint order
        xhand_joint_positions = qpos[self.retargeting_to_xhand]
        
        # Apply specific joint inversions as in reference implementation
        if len(xhand_joint_positions) > 3:
            xhand_joint_positions[3] = -xhand_joint_positions[3]  # Invert index bend for XHand
        
        return xhand_joint_positions
    
    def _convert_to_xhand_action(self, joint_positions: np.ndarray) -> Dict[str, float]:
        """Convert joint positions to XHand action format."""
        if len(joint_positions) != 12:
            raise ValueError(f"Expected 12 joint positions, got {len(joint_positions)}")
        
        # Create XHand action dictionary
        action = {}
        for i in range(12):
            action[f"joint_{i}.pos"] = float(joint_positions[i])
        
        return action
    
    def _apply_smoothing(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to joint positions."""
        return self.smoothing_alpha * current + (1 - self.smoothing_alpha) * previous