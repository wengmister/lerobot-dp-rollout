#!/usr/bin/env python3
"""
Simple test to verify basic robot movement works
"""

import time
import numpy as np
from lerobot.robots.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig

def test_simple_movement():
    config = FrankaFERConfig(
        id="simple_test",
        server_ip="192.168.18.1",
        server_port=5000
    )
    
    robot = FrankaFER(config)
    
    try:
        print("Connecting...")
        robot.connect(calibrate=False)
        
        # Get current position
        obs = robot.get_observation()
        current = [obs[f"joint_{i}.pos"] for i in range(7)]
        print(f"Current: {[f'{p:.3f}' for p in current]}")
        
        # Create small movement (0.1 rad on joint 6)
        target = current.copy()
        target[6] += 0.1  # Small movement
        
        print(f"Target:  {[f'{p:.3f}' for p in target]}")
        
        # Send action
        action = {f"joint_{i}.pos": target[i] for i in range(7)}
        robot.send_action(action)
        
        print("Waiting 3 seconds for movement...")
        time.sleep(3)
        
        # Check new position
        new_obs = robot.get_observation()
        new_pos = [new_obs[f"joint_{i}.pos"] for i in range(7)]
        print(f"New:     {[f'{p:.3f}' for p in new_pos]}")
        
        # Check if it moved
        moved = abs(new_pos[6] - current[6]) > 0.01
        print(f"Movement detected: {moved}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot.is_connected:
            robot.disconnect()

if __name__ == "__main__":
    test_simple_movement()