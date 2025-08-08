# franky_client.py - Runs on your workstation
import requests
import numpy as np
from typing import List, Optional, Dict
import time

class FrankyClient:
    """Client to communicate with Franky server running on RT PC"""
    
    def __init__(self, server_ip: str, server_port: int = 5000):
        """
        Initialize Franky client
        
        Args:
            server_ip: IP address of the RT PC running franky server
            server_port: Port of the franky server
        """
        self.base_url = f"http://{server_ip}:{server_port}"
        self.is_connected = False
        
    def _request(self, method: str, endpoint: str, json_data: Dict = None) -> Dict:
        """Make HTTP request to server"""
        try:
            url = f"{self.base_url}/{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=json_data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def connect(self, dynamics_factor: float = 0.2) -> bool:
        """Connect to franky server and initialize robot"""
        response = self._request("POST", "connect", {
            "dynamics_factor": dynamics_factor
        })
        self.is_connected = response.get("connected", False)
        return self.is_connected
    
    def disconnect(self):
        """Disconnect from franky server"""
        self._request("POST", "disconnect")
        self.is_connected = False
    
    def get_state(self) -> Optional[Dict]:
        """Get complete robot state"""
        response = self._request("GET", "state")
        if response.get("success"):
            return response.get("state")
        return None
    
    def get_joint_positions(self) -> Optional[np.ndarray]:
        """Get current joint positions"""
        response = self._request("GET", "joint_positions")
        if response.get("success"):
            return np.array(response.get("positions"))
        return None
    
    def get_joint_velocities(self) -> Optional[np.ndarray]:
        """Get current joint velocities"""
        response = self._request("GET", "joint_velocities")
        if response.get("success"):
            return np.array(response.get("velocities"))
        return None
    
    def move_joints(self, positions: np.ndarray) -> bool:
        """Move to target joint positions"""
        response = self._request("POST", "move_joints", {
            "positions": positions.tolist()
        })
        return response.get("success", False)
    
    def move_joint_velocity(self, velocities: np.ndarray, duration_ms: int = 100) -> bool:
        """
        Command joint velocities
        
        Args:
            velocities: Target joint velocities (7 values)
            duration_ms: How long to maintain velocities in milliseconds
        """
        response = self._request("POST", "move_joint_velocity", {
            "velocities": velocities.tolist(),
            "duration_ms": duration_ms
        })
        return response.get("success", False)
    
    def stop(self) -> bool:
        """Stop current motion"""
        response = self._request("POST", "stop")
        return response.get("success", False)
    
    def configure(self, dynamics_factor: float = None,
                  joint_impedance: Dict = None) -> bool:
        """Configure robot parameters"""
        params = {}
        if dynamics_factor is not None:
            params["dynamics_factor"] = dynamics_factor
        if joint_impedance is not None:
            params["joint_impedance"] = joint_impedance
            
        response = self._request("POST", "configure", params)
        return response.get("success", False)
    
    def recover_from_errors(self) -> bool:
        """Recover from robot errors"""
        response = self._request("POST", "recover")
        return response.get("success", False)
    
    def health_check(self) -> bool:
        """Check if server is running"""
        try:
            response = self._request("GET", "health")
            return response.get("status") == "ok"
        except:
            return False

# Example usage
if __name__ == "__main__":
    # Basic client usage
    print("Testing basic client...")
    client = FrankyClient("192.168.18.1")  # RT PC IP
    
    if client.connect(dynamics_factor=0.2):
        print("Connected to Franky server!")
        
        # Get current state
        positions = client.get_joint_positions()
        print(f"Current positions: {positions}")
        
        # Test joint position motion
        target = positions.copy()
        target[0] += 0.1  # Move first joint by 0.1 rad
        print("Moving joints...")
        client.move_joints(target)
        
        full_state = client.get_state()
        print(f"Full state: {full_state}")
        # Test joint velocity motion
        print("Testing velocity control...")
        velocities = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        client.move_joint_velocity(velocities, duration_ms=1000)  # 1 second

        robot_velocities = client.get_joint_velocities()
        print("Current velocities:", robot_velocities)

        client.disconnect()
    
    print("\n" + "="*50 + "\n")