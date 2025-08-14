#!/usr/bin/env python3
"""
VR Teleoperator for LeRobot using C++ IK bridge.
Integrates VR hand tracking with weighted IK solver for real-time robot control.
"""

import time
import logging
import subprocess
from typing import Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

logger = logging.getLogger(__name__)

@dataclass
class VRTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for VR teleoperator"""
    
    # TCP connection settings
    tcp_port: int = 8000
    auto_setup_adb: bool = True  # Automatically setup adb reverse port forwarding
    
    # VR processing settings
    smoothing_factor: float = 0.7  # Higher = more smoothing (0-1)
    position_deadzone: float = 0.001  # 1mm deadzone to prevent drift
    orientation_deadzone: float = 0.03  # ~1.7 degrees deadzone
    max_position_offset: float = 0.75  # 75cm max workspace
    
    # IK solver settings
    manipulability_weight: float = 1.0
    neutral_distance_weight: float = 2.0
    current_distance_weight: float = 2.0
    joint_weights: list[float] = None  # Will default to [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Q7 limits (can be configured for different end effectors)
    q7_min: float = -2.89  # Full Franka range
    q7_max: float = 2.89
    use_bidexhand_limits: bool = False  # If True, use [-0.2, 1.9] for BiDexHand
    
    # Debug settings
    verbose: bool = False
    
    def __post_init__(self):
        # Set default joint weights if not provided
        if self.joint_weights is None:
            # Higher weights for base joints for stability
            self.joint_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Apply BiDexHand limits if requested
        if self.use_bidexhand_limits:
            self.q7_min = -0.2
            self.q7_max = 1.9


class VRTeleoperator(Teleoperator):
    """
    VR Teleoperator for real-time robot control using hand tracking.
    
    This teleoperator receives VR hand pose data via TCP and uses a C++ weighted IK solver
    to compute target joint positions for the robot. It's designed to work seamlessly
    with LeRobot's data collection and policy deployment pipeline.
    
    Features:
    - Real-time VR hand tracking via TCP
    - Weighted IK solver optimized for Franka robots
    - Configurable workspace limits and smoothing
    - Support for BiDexHand and other end effectors
    - Automatic adb port forwarding setup for Android VR apps
    
    Usage:
        config = VRTeleoperatorConfig(tcp_port=8000, verbose=True)
        teleop = VRTeleoperator(config)
        
        # In your data collection loop:
        action = teleop.get_action()
    """
    
    def __init__(self, config: VRTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        
        # Import the C++ bridge
        try:
            import vr_ik_bridge
            self.bridge_module = vr_ik_bridge
        except ImportError as e:
            raise ImportError(
                "Failed to import vr_ik_bridge module. "
                "Please build the C++ extension first using: "
                "cd franka_teleoperator && python setup.py build_ext --inplace"
            ) from e
        
        # Initialize VR bridge
        bridge_config = vr_ik_bridge.VRTeleopConfig()
        bridge_config.tcp_port = config.tcp_port
        bridge_config.smoothing_factor = config.smoothing_factor
        bridge_config.position_deadzone = config.position_deadzone
        bridge_config.orientation_deadzone = config.orientation_deadzone
        bridge_config.max_position_offset = config.max_position_offset
        bridge_config.verbose = config.verbose
        
        self.vr_bridge = vr_ik_bridge.VRIKBridge(bridge_config)
        
        # State tracking
        self._robot_reference = None
        self._initialized = False
        self._is_connected = False
        self._last_action = None
        
        logger.info(f"VRTeleoperator initialized with TCP port {config.tcp_port}")
    
    def connect(self, robot=None):
        """Connect to VR input and setup IK solver"""
        if self._is_connected:
            raise RuntimeError("VRTeleoperator is already connected")
        
        # Store robot reference for getting current state
        self._robot_reference = robot
        
        if robot is None:
            raise ValueError("Robot reference is required for VR teleoperator")
        
        # Setup adb port forwarding if requested
        if self.config.auto_setup_adb:
            self._setup_adb_reverse()
        
        # Start TCP server for VR data
        if not self.vr_bridge.start_tcp_server():
            raise ConnectionError(f"Failed to start VR TCP server on port {self.config.tcp_port}")
        
        # Wait for robot to be connected to get initial state
        if hasattr(robot, 'is_connected') and not robot.is_connected:
            logger.warning("Robot is not connected yet. IK solver will be initialized when robot connects.")
        else:
            self._initialize_ik_solver()
        
        self._is_connected = True
        logger.info("VRTeleoperator connected successfully")
    
    def disconnect(self):
        """Disconnect from VR input"""
        if not self._is_connected:
            return
        
        self.vr_bridge.stop()
        
        # Cleanup adb reverse if we set it up
        if self.config.auto_setup_adb:
            self._cleanup_adb_reverse()
        
        self._is_connected = False
        self._initialized = False
        self._robot_reference = None
        
        logger.info("VRTeleoperator disconnected")
    
    def get_action(self) -> Dict[str, Any]:
        """
        Get the current action from VR input.
        
        Returns:
            Dictionary with joint position targets: {"joint_0.pos": float, ...}
        """
        if not self._is_connected:
            raise RuntimeError("VRTeleoperator is not connected")
        
        # Initialize IK solver if not done yet (delayed initialization)
        if not self._initialized:
            if not self._initialize_ik_solver():
                # Return current robot position if initialization fails
                if self._last_action is not None:
                    return self._last_action
                else:
                    # Emergency fallback
                    return {f"joint_{i}.pos": 0.0 for i in range(7)}
        
        # Get current robot joint positions
        try:
            current_obs = self._robot_reference.get_observation()
            current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
        except Exception as e:
            logger.error(f"Failed to get robot observation: {e}")
            # Return last known action or zero position
            if self._last_action is not None:
                return self._last_action
            else:
                return {f"joint_{i}.pos": 0.0 for i in range(7)}
        
        # Get target joint positions from VR IK solver
        try:
            target_joints = self.vr_bridge.get_joint_targets(current_joints)
            
            # Convert to action dictionary
            action = {f"joint_{i}.pos": float(target_joints[i]) for i in range(7)}
            
            # Store as last known good action
            self._last_action = action
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to get VR action: {e}")
            # Return current position to hold in place
            return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of VR teleoperator"""
        status = {
            "connected": self._is_connected,
            "initialized": self._initialized,
            "vr_connected": False,
            "vr_ready": False
        }
        
        if self._is_connected:
            try:
                vr_status = self.vr_bridge.get_vr_status()
                status.update(vr_status)
                status["vr_ready"] = self.vr_bridge.is_ready()
            except Exception as e:
                logger.error(f"Failed to get VR status: {e}")
        
        return status
    
    def _initialize_ik_solver(self) -> bool:
        """Initialize the IK solver with robot's current state"""
        if self._robot_reference is None:
            logger.error("No robot reference available for IK initialization")
            return False
        
        try:
            # Get current robot state
            current_obs = self._robot_reference.get_observation()
            current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
            
            # Setup IK solver with current joint positions as neutral pose
            success = self.vr_bridge.setup_ik_solver(
                neutral_pose=current_joints,
                manipulability_weight=self.config.manipulability_weight,
                neutral_distance_weight=self.config.neutral_distance_weight,
                current_distance_weight=self.config.current_distance_weight,
                joint_weights=self.config.joint_weights
            )
            
            if not success:
                logger.error("Failed to setup IK solver")
                return False
            
            # Set Q7 limits
            self.vr_bridge.set_q7_limits(self.config.q7_min, self.config.q7_max)
            
            # Get initial robot pose (end-effector transformation matrix)
            # Note: This assumes the robot observation includes end-effector pose
            # You might need to compute this from forward kinematics if not available
            if "ee_pose" in current_obs:
                ee_pose = current_obs["ee_pose"]
                if len(ee_pose) == 16:  # 4x4 transformation matrix
                    self.vr_bridge.set_initial_robot_pose(ee_pose)
                else:
                    logger.warning("End-effector pose format not recognized, using identity")
                    # Use identity transformation as fallback
                    identity = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
                    self.vr_bridge.set_initial_robot_pose(identity)
            else:
                logger.warning("No end-effector pose in observation, using identity")
                identity = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
                self.vr_bridge.set_initial_robot_pose(identity)
            
            self._initialized = True
            logger.info("IK solver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IK solver: {e}")
            return False
    
    def _setup_adb_reverse(self):
        """Setup adb reverse port forwarding for Android VR apps"""
        try:
            # Check if adb is available
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.warning("adb command not found. Please install Android SDK platform-tools.")
                return False
            
            # Check if device is connected
            if "device" not in result.stdout:
                logger.warning("No Android device connected via adb.")
                return False
            
            # Setup reverse port forwarding
            cmd = ['adb', 'reverse', f'tcp:{self.config.tcp_port}', f'tcp:{self.config.tcp_port}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"Successfully setup adb reverse tcp:{self.config.tcp_port}")
                return True
            else:
                logger.warning(f"Failed to setup adb reverse: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Error setting up adb reverse: {e}")
            return False
    
    def _cleanup_adb_reverse(self):
        """Remove adb reverse port forwarding"""
        try:
            cmd = ['adb', 'reverse', '--remove', f'tcp:{self.config.tcp_port}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"Successfully removed adb reverse tcp:{self.config.tcp_port}")
            else:
                logger.warning(f"Failed to remove adb reverse: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Error cleaning up adb reverse: {e}")
    
    # Abstract method implementations
    @property
    def action_features(self) -> dict:
        """Return action features for 7-DOF Franka robot"""
        return {f"joint_{i}.pos": float for i in range(7)}
    
    @property
    def feedback_features(self) -> dict:
        """Return feedback features - VR teleoperator doesn't use feedback"""
        return {}
    
    @property
    def is_connected(self) -> bool:
        """Whether the VR teleoperator is connected"""
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        """VR teleoperator doesn't require calibration"""
        return True
    
    def calibrate(self) -> None:
        """VR teleoperator doesn't require calibration"""
        pass
    
    def configure(self) -> None:
        """Apply configuration to VR teleoperator"""
        # Configuration is handled in connect()
        pass
    
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """VR teleoperator doesn't support feedback"""
        pass


# Set class attributes required by Teleoperator
VRTeleoperator.config_class = VRTeleoperatorConfig
VRTeleoperator.name = "vr"

# Register the teleoperator
TeleoperatorConfig.register_subclass("vr", VRTeleoperatorConfig)