import numpy as np
from typing import Dict
from src.lerobot.motors.franka_fer.franky_client import FrankyClient

class FrankaRobotLeRobot:
    """Robot abstraction for LeRobot framework"""
    
    def __init__(self, server_ip: str, server_port: int = 5000, dynamics_factor: float = 0.3):
        """
        Initialize Franka robot for LeRobot
        
        Args:
            server_ip: IP address of the RT PC
            server_port: Port of the franky server
            dynamics_factor: Speed factor (0.0-1.0)
        """
        self.client = FrankyClient(server_ip, server_port)
        self.dynamics_factor = dynamics_factor
        self.motors = []
        self.is_connected = False
        
    def connect(self) -> bool:
        """Initialize connection to robot"""
        # Check server health first
        if not self.client.health_check():
            print(f"Cannot reach franky server at {self.client.base_url}")
            return False
        
        self.is_connected = self.client.connect(self.dynamics_factor)
        
        if self.is_connected:
            # Create motor abstractions for each joint
            self.motors = [
                FrankaMotorLeRobot(self.client, i) for i in range(7)
            ]
            print(f"Connected to Franka robot with dynamics factor {self.dynamics_factor}")
        
        return self.is_connected
    
    def disconnect(self):
        """Disconnect from robot"""
        self.client.disconnect()
        self.is_connected = False
        self.motors = []
    
    def get_observation(self) -> Dict:
        """Get current robot observation for LeRobot"""
        state = self.client.get_state()
        
        if state:
            return {
                "joint_positions": np.array(state["q"]),
                "joint_velocities": np.array(state["dq"]),
                "joint_torques": np.array(state["tau_J"]),
                "ee_pose": np.array(state["O_T_EE"]).reshape(4, 4),
                "timestamp": state["time"]
            }
        
        return {}
    
    def send_action(self, action: np.ndarray, action_type: str = "position", **kwargs):
        """
        Send action to robot
        
        Args:
            action: Joint values (7 elements)
            action_type: "position" or "velocity"
            kwargs: Additional parameters (e.g., duration_ms for velocity)
        """
        if action_type == "position":
            return self.client.move_joints(action)
        elif action_type == "velocity":
            duration_ms = kwargs.get("duration_ms", 100)
            return self.client.move_joint_velocity(action, duration_ms)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def reset(self, home_position: np.ndarray = None) -> bool:
        """Reset robot to home position"""
        if home_position is None:
            # Default Franka home position
            home_position = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        # Slow down for reset motion
        original_factor = self.dynamics_factor
        self.client.configure(dynamics_factor=0.2)
        
        success = self.client.move_joints(home_position)
        
        # Restore original dynamics
        self.client.configure(dynamics_factor=original_factor)
        
        return success
    
    def stop(self) -> bool:
        """Emergency stop"""
        return self.client.stop()
    
    def configure(self, dynamics_factor: float = None, **kwargs):
        """Update robot configuration"""
        if dynamics_factor is not None:
            self.dynamics_factor = dynamics_factor
        return self.client.configure(dynamics_factor=dynamics_factor, **kwargs)
    
# LeRobot Integration Classes
class FrankaMotorLeRobot:
    """Motor abstraction for LeRobot framework"""
    
    def __init__(self, client: FrankyClient, joint_index: int):
        self.client = client
        self.joint_index = joint_index
    
    @property
    def position(self) -> float:
        """Get current position of this joint"""
        positions = self.client.get_joint_positions()
        if positions is not None:
            return positions[self.joint_index]
        return 0.0
    
    @property
    def velocity(self) -> float:
        """Get current velocity of this joint"""
        velocities = self.client.get_joint_velocities()
        if velocities is not None:
            return velocities[self.joint_index]
        return 0.0
    
    def set_position(self, position: float):
        """Set target position for this joint"""
        current_positions = self.client.get_joint_positions()
        if current_positions is not None:
            current_positions[self.joint_index] = position
            self.client.move_joints(current_positions)
    
    def set_velocity(self, velocity: float, duration_ms: int = 100):
        """Set velocity for this joint"""
        velocities = np.zeros(7)
        velocities[self.joint_index] = velocity
        self.client.move_joint_velocity(velocities, duration_ms)