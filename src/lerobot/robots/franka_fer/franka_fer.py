import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from ...motors.franka_fer.franky_client import FrankyClient
from .franka_fer_config import FrankaFERConfig

logger = logging.getLogger(__name__)


class FrankaFER(Robot):
    """
    Franka FER robot controlled via Franky client/server architecture.
    
    This robot implementation communicates with a Franky server running on a real-time
    computer to control a Franka Emika robot.
    """
    
    config_class = FrankaFERConfig
    name = "franka_fer"
    
    def __init__(self, config: FrankaFERConfig):
        super().__init__(config)
        self.config = config
        self.client = FrankyClient(config.server_ip, config.server_port)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation feature structure"""
        features = {}
        
        # Joint positions (7 joints)
        for i in range(7):
            features[f"joint_{i}.pos"] = float
            
        # Add camera features if any
        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
            
        return features
    
    @cached_property 
    def action_features(self) -> dict[str, type]:
        """Define action feature structure"""
        return {f"joint_{i}.pos": float for i in range(7)}
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        return self._is_connected and self.client.is_connected
    
    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Check server health first
        if not self.client.health_check():
            raise ConnectionError(f"Cannot reach franky server at {self.client.base_url}")
        
        # Connect to robot
        if not self.client.connect(self.config.dynamics_factor):
            raise ConnectionError("Failed to connect to Franka robot")
        
        self._is_connected = True
        
        # Connect cameras if any
        for cam in self.cameras.values():
            cam.connect()
        
        # Configure robot
        self.configure()
        
        logger.info(f"{self} connected with dynamics factor {self.config.dynamics_factor}")
    
    @property
    def is_calibrated(self) -> bool:
        """Franka robots don't require calibration in this implementation"""
        return True
    
    def calibrate(self) -> None:
        """No-op for Franka robots - they are pre-calibrated"""
        pass
    
    def configure(self) -> None:
        """Configure robot with current settings"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Apply dynamics factor and any other configuration
        self.client.configure(dynamics_factor=self.config.dynamics_factor)
    
    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        obs_dict = {}
        
        # Get joint positions
        start = time.perf_counter()
        positions = self.client.get_joint_positions()
        if positions is not None:
            for i, pos in enumerate(positions):
                obs_dict[f"joint_{i}.pos"] = float(pos)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joint positions: {dt_ms:.1f}ms")
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Extract joint positions from action dict
        joint_positions = []
        for i in range(7):
            key = f"joint_{i}.pos"
            if key not in action:
                raise ValueError(f"Missing joint position for {key}")
            joint_positions.append(action[key])
        
        target_positions = np.array(joint_positions)
        
        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            current_positions = self.client.get_joint_positions()
            if current_positions is not None:
                # Create goal_present_pos dict for safety function
                goal_present_pos = {}
                for i in range(7):
                    goal_present_pos[f"joint_{i}"] = (target_positions[i], current_positions[i])
                
                # Apply safety limits
                safe_positions = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
                target_positions = np.array([safe_positions[f"joint_{i}"] for i in range(7)])
        
        # Send command to robot
        success = self.client.move_joints(target_positions)
        if not success:
            logger.warning("Failed to send action to robot")
        
        # Return the actual action sent
        return {f"joint_{i}.pos": float(target_positions[i]) for i in range(7)}
    
    def disconnect(self) -> None:
        """Disconnect from robot"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        
        # Disconnect from robot
        self.client.disconnect()
        self._is_connected = False
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected")
    
    def reset_to_home(self) -> bool:
        """Reset robot to home position"""
        if not self.is_connected:
            return False
        
        home_position = np.array(self.config.home_position)
        
        # Slow down for reset motion
        original_factor = self.config.dynamics_factor
        self.client.configure(dynamics_factor=0.2)
        
        success = self.client.move_joints(home_position)
        
        # Restore original dynamics
        self.client.configure(dynamics_factor=original_factor)
        
        return success
    
    def stop(self) -> bool:
        """Emergency stop"""
        if not self.is_connected:
            return False
        return self.client.stop()
    
    def recover_from_errors(self) -> bool:
        """Recover from robot errors"""
        if not self.is_connected:
            return False
        return self.client.recover_from_errors()