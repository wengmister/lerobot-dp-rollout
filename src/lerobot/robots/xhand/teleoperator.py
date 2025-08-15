import logging
import multiprocessing
import time
import sys
from pathlib import Path
from queue import Empty
from typing import Optional, Dict, Any
import numpy as np

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

logger = logging.getLogger(__name__)


class XHandTeleoperator:
    """
    Teleoperator that bridges VR hand tracking with XHand robot control.
    
    Uses dex-retargeting to convert human hand poses to robot joint commands
    and sends them to the XHand robot via the lerobot interface.
    """
    
    def __init__(
        self, 
        xhand_robot,
        robot_name: RobotName = RobotName.xhand,
        retargeting_type: RetargetingType = RetargetingType.dexpilot,
        hand_type: HandType = HandType.right,
        robot_dir: Optional[str] = None,
    ):
        """
        Initialize the XHand teleoperator.
        
        Args:
            xhand_robot: Connected XHand robot instance
            robot_name: Target robot name for retargeting config
            retargeting_type: Type of retargeting algorithm
            hand_type: Which hand to track (left/right)
            robot_dir: Directory containing robot URDF files
        """
        self.xhand_robot = xhand_robot
        self.robot_name = robot_name
        self.retargeting_type = retargeting_type
        self.hand_type = hand_type
        
        # Set default robot directory if not provided
        if robot_dir is None:
            robot_dir = Path(__file__).parent.parent.parent.parent.parent / "dex_retargeting" / "assets" / "robots" / "hands"
        
        # Initialize retargeting system
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        self.retargeting = RetargetingConfig.load_from_file(config_path).build()
        
        # Set up joint mapping from retargeting output to XHand joint order
        self._setup_joint_mapping()
        
        # Initialize VR hand detector
        try:
            # Add dex_retargeting example path to sys.path
            dex_example_path = Path(__file__).parent.parent.parent.parent.parent / "dex_retargeting" / "example" / "vector_retargeting"
            sys.path.insert(0, str(dex_example_path))
            
            from vr_hand_detector import VRHandDetector
            hand_type_str = "Right" if hand_type == HandType.right else "Left"
            self.detector = VRHandDetector(hand_type=hand_type_str, robot_name=str(robot_name), use_tcp=True)
        except ImportError as e:
            logger.error(f"VRHandDetector not available: {e}. Please install required VR dependencies.")
            self.detector = None
        
        # Control settings
        self.control_frequency = 30.0  # Hz
        self.smoothing_alpha = 0.3  # Exponential smoothing factor
        self.last_joint_positions = None
        self.is_running = False
        
        logger.info(f"XHand teleoperator initialized for {hand_type} hand")
    
    def start_teleoperation(self) -> None:
        """Start real-time teleoperation loop."""
        if not self.xhand_robot.is_connected:
            raise RuntimeError("XHand robot must be connected before starting teleoperation")
        
        if self.detector is None:
            raise RuntimeError("VR hand detector not available")
        
        logger.info("Starting XHand teleoperation...")
        self.is_running = True
        
        try:
            while self.is_running:
                start_time = time.perf_counter()
                
                # Detect hand pose and retarget to robot
                success = self._process_hand_frame()
                
                if not success:
                    logger.debug("Hand detection failed, skipping frame")
                
                # Maintain control frequency
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, 1.0 / self.control_frequency - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Teleoperation interrupted by user")
        except Exception as e:
            logger.error(f"Teleoperation error: {e}")
        finally:
            self.stop_teleoperation()
    
    def _process_hand_frame(self) -> bool:
        """
        Process one frame of hand tracking and send commands to robot.
        
        Returns:
            bool: True if successful, False if hand detection failed
        """
        try:
            # Detect hand pose
            _, joint_pos, keypoint_2d, _ = self.detector.detect()
            
            if joint_pos is None:
                return False
            
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
            
            # Convert to XHand action format and send
            action = self._convert_to_xhand_action(xhand_joint_positions)
            self.xhand_robot.send_action(action)
            
            # Print joint angles for visualization in stub mode
            if self.xhand_robot._device is None:  # Stub mode
                self._print_joint_visualization(xhand_joint_positions)
            
            return True
            
        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            return False
    
    def _convert_to_xhand_action(self, joint_positions: np.ndarray) -> Dict[str, float]:
        """
        Convert retargeted joint positions to XHand action format.
        
        Args:
            joint_positions: Array of 12 joint positions in radians
            
        Returns:
            dict: XHand action format {"joint_0.pos": float, ...}
        """
        if len(joint_positions) != 12:
            raise ValueError(f"Expected 12 joint positions, got {len(joint_positions)}")
        
        # Create XHand action dictionary
        action = {}
        for i in range(12):
            action[f"joint_{i}.pos"] = float(joint_positions[i])
        
        return action
    
    def _apply_smoothing(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to joint positions.
        
        Args:
            current: Current joint positions
            previous: Previous joint positions
            
        Returns:
            np.ndarray: Smoothed joint positions
        """
        return self.smoothing_alpha * current + (1 - self.smoothing_alpha) * previous
    
    def stop_teleoperation(self) -> None:
        """Stop the teleoperation loop."""
        logger.info("Stopping XHand teleoperation...")
        self.is_running = False
    
    def set_control_frequency(self, frequency: float) -> None:
        """Set the control loop frequency in Hz."""
        self.control_frequency = max(1.0, min(100.0, frequency))
        logger.info(f"Control frequency set to {self.control_frequency} Hz")
    
    def set_smoothing(self, alpha: float) -> None:
        """
        Set smoothing factor for joint position commands.
        
        Args:
            alpha: Smoothing factor (0.0 = max smoothing, 1.0 = no smoothing)
        """
        self.smoothing_alpha = max(0.0, min(1.0, alpha))
        logger.info(f"Smoothing factor set to {self.smoothing_alpha}")
    
    def reset_robot_to_home(self) -> bool:
        """Reset the robot to home position."""
        return self.xhand_robot.reset_to_home()
    
    def emergency_stop(self) -> bool:
        """Emergency stop the robot."""
        self.stop_teleoperation()
        return self.xhand_robot.stop()
    
    def _print_joint_visualization(self, joint_positions: np.ndarray) -> None:
        """
        Print a simple visualization of joint angles in stub mode.
        
        Args:
            joint_positions: Array of 12 joint positions in radians
        """
        # Convert to degrees for easier reading
        joint_degrees = joint_positions * 180.0 / np.pi
        
        # Create simple bar visualization
        print("\n" + "="*80)
        print("XHand Joint Angles (degrees):")
        print("-" * 80)
        
        for i in range(12):
            angle_deg = joint_degrees[i]
            # Create a simple bar chart (scale: -90 to +90 degrees)
            bar_length = int(abs(angle_deg) / 5)  # 5 degrees per character
            bar_length = min(bar_length, 18)  # Max 18 characters
            
            if angle_deg >= 0:
                bar = "+" * bar_length
                spaces = " " * (18 - bar_length)
                bar_display = f"|{spaces}{bar}|"
            else:
                bar = "-" * bar_length
                spaces = " " * (18 - bar_length)
                bar_display = f"|{bar}{spaces}|"
            
            print(f"Joint {i:2d}: {angle_deg:6.1f}° {bar_display}")
        
        print("="*80)
        
        # Also print raw values on one line for compact view
        angles_str = " ".join([f"{a:5.1f}" for a in joint_degrees])
        print(f"Raw angles: [{angles_str}]")
        print()
    
    def _setup_joint_mapping(self) -> None:
        """
        Set up mapping from retargeting output to XHand joint order.
        Based on xhand_vr_motion_shadowing.py implementation.
        """
        # Get retargeting joint names (URDF order)
        retargeting_joint_names = self.retargeting.joint_names
        
        # Desired XHand order: [thumb_bend, thumb_rota1, thumb_rota2, index_bend, index_joint1, index_joint2, 
        #                       mid_joint1, mid_joint2, ring_joint1, ring_joint2, pinky_joint1, pinky_joint2]
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
        """
        Map retargeting output to correct XHand joint order.
        
        Args:
            qpos: Joint positions from retargeting (URDF order)
            
        Returns:
            np.ndarray: Joint positions in XHand order
        """
        # Map retargeting output to correct XHand joint order
        xhand_joint_positions = qpos[self.retargeting_to_xhand]
        
        # Apply specific joint inversions as in reference implementation
        if len(xhand_joint_positions) > 3:
            xhand_joint_positions[3] = -xhand_joint_positions[3]  # Invert index bend for XHand
        
        return xhand_joint_positions