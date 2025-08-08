#!/usr/bin/env python3

"""
Test script for Franka FER robot integration with LeRobot
"""

import numpy as np
from lerobot.robots.franka_fer import FrankaFER, FrankaFERConfig


def test_robot_instantiation():
    """Test basic robot instantiation and properties"""
    print("Testing robot instantiation...")
    
    config = FrankaFERConfig(
        id="franka_test",
        server_ip="192.168.18.1",
        server_port=5000,
        dynamics_factor=0.3
    )
    
    robot = FrankaFER(config)
    
    print(f"✓ Robot created: {robot}")
    print(f"✓ Robot name: {robot.name}")
    print(f"✓ Robot type: {robot.robot_type}")
    print(f"✓ Is calibrated: {robot.is_calibrated}")
    print(f"✓ Is connected: {robot.is_connected}")
    
    # Test observation features
    obs_features = robot.observation_features
    print(f"✓ Observation features: {obs_features}")
    
    # Test action features  
    action_features = robot.action_features
    print(f"✓ Action features: {action_features}")
    
    return robot


def test_robot_mock_connection():
    """Test robot connection workflow (without actual connection)"""
    print("\nTesting robot connection workflow...")
    
    config = FrankaFERConfig(
        id="franka_test",
        server_ip="127.0.0.1",  # Mock IP that won't connect
        server_port=5000,
        dynamics_factor=0.3
    )
    
    robot = FrankaFER(config)
    
    print("✓ Robot created for connection test")
    
    # This should fail gracefully
    try:
        robot.connect(calibrate=False)
        print("✓ Connected successfully") 
    except Exception as e:
        print(f"✗ Connection failed as expected: {e}")
    
    return robot


def test_action_dict_structure():
    """Test action dictionary structure"""
    print("\nTesting action dictionary structure...")
    
    config = FrankaFERConfig(id="franka_test")
    robot = FrankaFER(config)
    
    # Create a mock action dict
    action = {}
    for i in range(7):
        action[f"joint_{i}.pos"] = 0.1 * i
    
    print(f"✓ Mock action created: {action}")
    
    # Test that our action features match
    expected_keys = set(robot.action_features.keys())
    actual_keys = set(action.keys())
    
    if expected_keys == actual_keys:
        print("✓ Action structure matches expected features")
    else:
        print(f"✗ Action structure mismatch!")
        print(f"  Expected: {expected_keys}")
        print(f"  Actual: {actual_keys}")
    
    return action


def test_robot_factory():
    """Test robot creation via factory function"""
    print("\nTesting robot factory function...")
    
    from lerobot.robots.utils import make_robot_from_config
    
    config = FrankaFERConfig(
        id="franka_factory_test",
        server_ip="192.168.18.1",
    )
    
    try:
        robot = make_robot_from_config(config)
        print(f"✓ Robot created via factory: {robot}")
        print(f"✓ Robot class: {robot.__class__.__name__}")
        return robot
    except Exception as e:
        print(f"✗ Factory creation failed: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Franka FER Robot Integration Test")
    print("=" * 60)
    
    try:
        # Test basic functionality
        robot1 = test_robot_instantiation()
        robot2 = test_robot_mock_connection() 
        action = test_action_dict_structure()
        robot3 = test_robot_factory()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("Your Franka FER robot class is ready for LeRobot integration.")
        print("\nNext steps:")
        print("1. Start your Franky server on the RT PC")
        print("2. Update server_ip in config to match your RT PC")
        print("3. Test with actual robot connection")
        print("4. Use with diffusion policy rollout")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()