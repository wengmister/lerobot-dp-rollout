#!/usr/bin/env python3
"""
Test script for Franka FER robot with RealSense cameras.
Verifies camera connectivity and basic observation collection.
"""

import time
import numpy as np
import cv2
from pathlib import Path

from src.lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from src.lerobot.robots.franka_fer.franka_fer import FrankaFER

def test_cameras_only():
    """Test cameras without connecting to robot"""
    print("=== Testing Cameras Only ===")
    
    config = FrankaFERConfig()
    robot = FrankaFER(config)
    
    # Test camera connections
    try:
        for cam_name, cam in robot.cameras.items():
            print(f"Connecting to {cam_name} camera...")
            cam.connect()
            print(f"✓ {cam_name} camera connected")
            
            # Test reading a frame
            print(f"Reading frame from {cam_name}...")
            frame = cam.read()
            print(f"✓ Frame captured: {frame.shape}")
            
            # Save test image
            output_dir = Path("outputs/camera_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / f"{cam_name}_test.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"✓ Test image saved to {output_dir}/{cam_name}_test.jpg")
            
            cam.disconnect()
            print(f"✓ {cam_name} camera disconnected")
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False
    
    return True

def test_robot_with_cameras():
    """Test full robot integration with cameras"""
    print("\n=== Testing Robot with Cameras ===")
    
    config = FrankaFERConfig()
    robot = FrankaFER(config)
    
    try:
        # Connect to robot (includes cameras)
        print("Connecting to robot...")
        robot.connect()
        print("✓ Robot connected with cameras")
        
        # Get observation
        print("Getting robot observation...")
        obs = robot.get_observation()
        
        # Print observation info
        print("Observation keys:", list(obs.keys()))
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # Save camera images
        output_dir = Path("outputs/robot_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for cam_name in ["wrist", "third_person"]:
            if cam_name in obs:
                img = obs[cam_name]
                cv2.imwrite(str(output_dir / f"{cam_name}_observation.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"✓ Saved {cam_name} observation image")
        
        # Test multiple observations with delays
        print("Testing multiple observations...")
        start_time = time.time()
        for i in range(3):
            obs = robot.get_observation()
            print(f"  Observation {i+1}: {len(obs)} keys")
            time.sleep(0.1)  # Small delay to avoid overwhelming the connection
        
        elapsed = time.time() - start_time
        print(f"✓ 3 observations took {elapsed:.2f}s ({elapsed/3:.3f}s per obs)")
        
        # Test action sending
        print("Testing action sending...")
        current_positions = {f"joint_{i}.pos": obs[f"joint_{i}.pos"] for i in range(7)}
        result = robot.send_action(current_positions)
        print("✓ Action sent successfully")
        
        # Disconnect
        robot.disconnect()
        print("✓ Robot disconnected")
        
    except Exception as e:
        print(f"✗ Robot test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            robot.disconnect()
        except:
            pass
        return False
    
    return True

def main():
    print("Franka FER Camera Integration Test")
    print("=" * 50)
    
    # Test cameras independently first
    if not test_cameras_only():
        print("Camera-only test failed. Check camera connections.")
        return
    
    # Test full robot integration
    if not test_robot_with_cameras():
        print("Robot integration test failed.")
        return
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Cameras are integrated successfully.")
    print("Check the outputs/ directory for test images.")

if __name__ == "__main__":
    main()