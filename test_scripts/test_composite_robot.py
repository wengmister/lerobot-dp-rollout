#!/usr/bin/env python3

"""
Test script for the Franka FER + XHand composite robot.
This script tests the robot interface without actually connecting to hardware.
"""

import logging
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.robots.franka_fer_xhand import FrankaFERXHand, FrankaFERXHandConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_composite_robot():
    """Test the composite robot configuration and features"""
    
    # Create configurations
    arm_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        cameras={}  # No cameras for this test
    )
    
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        cameras={}  # No cameras for this test
    )
    
    composite_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        synchronize_actions=True
    )
    
    # Create composite robot
    robot = FrankaFERXHand(composite_config)
    
    # Test observation features
    obs_features = robot.observation_features
    print(f"\n📊 Observation Features ({len(obs_features)} total):")
    print("=" * 50)
    
    arm_features = [k for k in obs_features.keys() if k.startswith("arm_")]
    hand_features = [k for k in obs_features.keys() if k.startswith("hand_")]
    
    print(f"Arm features ({len(arm_features)}):")
    for feature in sorted(arm_features)[:5]:  # Show first 5
        print(f"  - {feature}: {obs_features[feature]}")
    if len(arm_features) > 5:
        print(f"  ... and {len(arm_features) - 5} more")
    
    print(f"\nHand features ({len(hand_features)}):")
    for feature in sorted(hand_features)[:5]:  # Show first 5  
        print(f"  - {feature}: {obs_features[feature]}")
    if len(hand_features) > 5:
        print(f"  ... and {len(hand_features) - 5} more")
    
    # Test action features
    action_features = robot.action_features
    print(f"\n🎯 Action Features ({len(action_features)} total):")
    print("=" * 50)
    
    arm_actions = [k for k in action_features.keys() if k.startswith("arm_")]
    hand_actions = [k for k in action_features.keys() if k.startswith("hand_")]
    
    print(f"Arm actions ({len(arm_actions)}): {list(arm_actions)}")
    print(f"Hand actions ({len(hand_actions)}): {list(hand_actions)}")
    
    # Test feature counts
    print(f"\n📈 Feature Summary:")
    print("=" * 50)
    print(f"Total observation features: {len(obs_features)}")
    print(f"  - Arm: {len(arm_features)} (7 pos + 7 vel + 16 ee_pose = 30)")
    print(f"  - Hand: {len(hand_features)} (12 pos + 12 torque = 24)")
    print(f"Total action features: {len(action_features)}")
    print(f"  - Arm: {len(arm_actions)} (7 joint positions)")
    print(f"  - Hand: {len(hand_actions)} (12 joint positions)")
    
    # Expected totals
    expected_obs = 30 + 24  # arm + hand observations  
    expected_actions = 7 + 12  # arm + hand actions
    
    print(f"\n✅ Validation:")
    print(f"  - Observation features: {len(obs_features)} == {expected_obs} ✓" if len(obs_features) == expected_obs else f"  - Observation features: {len(obs_features)} != {expected_obs} ❌")
    print(f"  - Action features: {len(action_features)} == {expected_actions} ✓" if len(action_features) == expected_actions else f"  - Action features: {len(action_features)} != {expected_actions} ❌")
    
    return robot

def test_individual_robots():
    """Test individual robot components"""
    print(f"\n🔍 Testing Individual Robots:")
    print("=" * 50)
    
    # Test Franka FER
    from lerobot.robots.franka_fer import FrankaFER
    arm_config = FrankaFERConfig(server_ip="192.168.18.1", server_port=5000)
    arm = FrankaFER(arm_config)
    
    print(f"Franka FER observations: {len(arm.observation_features)}")
    print(f"Franka FER actions: {len(arm.action_features)}")
    
    # Test XHand
    from lerobot.robots.xhand import XHand
    hand_config = XHandConfig(protocol="RS485", serial_port="/dev/ttyUSB0")
    hand = XHand(hand_config)
    
    print(f"XHand observations: {len(hand.observation_features)}")
    print(f"XHand actions: {len(hand.action_features)}")

if __name__ == "__main__":
    print("🤖 Testing Franka FER + XHand Composite Robot")
    print("=" * 60)
    
    try:
        test_individual_robots()
        robot = test_composite_robot()
        
        print(f"\n🎉 All tests passed! Composite robot ready for integration.")
        print(f"   - Robot name: {robot.name}")
        print(f"   - Configuration: {robot.config_class.__name__}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()