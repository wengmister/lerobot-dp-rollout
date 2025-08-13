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


def test_home_pose_with_real_connection():
    """Test connecting to real robot and going to home pose"""
    print("\nTesting real robot connection and home pose...")
    print("⚠️  Make sure your C++ position server is running!")
    
    config = FrankaFERConfig(
        id="franka_home_test",
        server_ip="192.168.18.1",  # Update this to your actual server IP
        server_port=5000,
    )
    
    robot = FrankaFER(config)
    
    try:
        print("🔄 Attempting to connect to robot server...")
        robot.connect(calibrate=False)
        print("✓ Successfully connected to robot!")
        
        # Get current position
        print("📍 Getting current robot state...")
        current_obs = robot.get_observation()
        current_positions = [current_obs[f"joint_{i}.pos"] for i in range(7)]
        print(f"✓ Current joint positions: {[f'{p:.3f}' for p in current_positions]}")
        
        # Show home position
        home_positions = config.home_position
        print(f"🏠 Target home positions: {[f'{p:.3f}' for p in home_positions]}")
        
        # Ask user confirmation
        user_input = input("\n🤖 Do you want to move the robot to home position? (y/N): ")
        
        if user_input.lower() in ['y', 'yes']:
            print("🏠 Moving robot to home position...")
            success = robot.reset_to_home()
            
            if success:
                print("✓ Successfully moved to home position!")
                
                # Verify we reached home
                import time
                time.sleep(2)  # Wait for movement to complete
                
                new_obs = robot.get_observation()
                new_positions = [new_obs[f"joint_{i}.pos"] for i in range(7)]
                print(f"✓ New joint positions: {[f'{p:.3f}' for p in new_positions]}")
                
                # Check if we're close to home
                errors = [abs(new - target) for new, target in zip(new_positions, home_positions)]
                max_error = max(errors)
                
                if max_error < 0.05:  # 0.05 radian tolerance
                    print(f"✓ Robot successfully reached home pose (max error: {max_error:.4f} rad)")
                else:
                    print(f"⚠️  Robot moved but not exactly at home (max error: {max_error:.4f} rad)")
                    
            else:
                print("✗ Failed to move to home position")
        else:
            print("⏭️  Skipping home pose movement")
            
        # Test stop command
        print("\n🛑 Testing stop command...")
        stop_success = robot.stop()
        print(f"✓ Stop command: {'SUCCESS' if stop_success else 'FAILED'}")
        
        return robot
        
    except ConnectionError as e:
        print(f"✗ Connection failed: {e}")
        print("💡 Make sure your C++ position server is running and reachable")
        return None
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Always try to disconnect
        try:
            if robot.is_connected:
                print("\n🔌 Disconnecting from robot...")
                robot.disconnect()
                print("✓ Disconnected successfully")
        except Exception as e:
            print(f"⚠️  Disconnect error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Franka FER Robot Integration Test")
    print("=" * 60)
    
    try:
        # Test basic functionality
        robot1 = test_robot_instantiation()
        action = test_action_dict_structure()
        robot3 = test_robot_factory()
        
        # Test with real robot if user wants to
        print("\n" + "=" * 60)
        test_real = input("Do you want to test with the real robot? (y/N): ")
        if test_real.lower() in ['y', 'yes']:
            robot_real = test_home_pose_with_real_connection()
        else:
            print("⏭️  Skipping real robot test")
        
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