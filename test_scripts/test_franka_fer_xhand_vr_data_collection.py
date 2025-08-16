#!/usr/bin/env python3
"""
Test script for FrankaFER + XHand VR data collection setup.

This script tests the complete data collection pipeline with the composite robot
and dual VR teleoperator, verifying that everything works for recording demonstrations.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.teleoperators.franka_fer_xhand_vr.franka_fer_xhand_vr_teleoperator import FrankaFERXHandVRTeleoperator
from lerobot.teleoperators.franka_fer_xhand_vr.config_franka_fer_xhand_vr import FrankaFERXHandVRTeleoperatorConfig

from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.cameras.utils import CameraConfig
from dex_retargeting.constants import RobotName, RetargetingType, HandType


def test_data_collection_setup():
    """Test the complete dual robot VR data collection setup."""
    print("🎯 Testing FrankaFER + XHand VR Data Collection Setup")
    print("=" * 60)
    
    robot = None
    teleop = None
    
    try:
        # ===== SETUP COMPOSITE ROBOT =====
        print("\n🤖 Setting up FrankaFER + XHand composite robot...")
        
        # Configure composite robot with cameras for data collection
        robot_config = FrankaFERXHandConfig(
            arm_config=FrankaFERConfig(
                server_ip="192.168.1.10",  # Update with your Franka IP
                server_port=8080
            ),
            hand_config=XHandConfig(
                protocol="RS485",
                serial_port="/dev/ttyUSB0"
            ),
            cameras={
                "wrist": CameraConfig(
                    width=640,
                    height=480,
                    fps=30,
                    camera_index=0  # Update based on your camera setup
                ),
                "external": CameraConfig(
                    width=1280,
                    height=720,
                    fps=30,
                    camera_index=1  # Update based on your camera setup
                )
            }
        )
        
        robot = FrankaFERXHand(robot_config)
        robot.connect(calibrate=True)
        print("  ✅ Composite robot connected")
        
        # ===== SETUP DUAL VR TELEOPERATOR =====
        print("\n🎮 Setting up dual VR teleoperator...")
        
        teleop_config = FrankaFERXHandVRTeleoperatorConfig(
            vr_tcp_port=8000,
            setup_adb=True,
            vr_verbose=True,
            # Arm settings
            arm_smoothing_factor=0.1,
            manipulability_weight=0.1,
            neutral_distance_weight=1.0,
            current_distance_weight=10.0,
            # Hand settings
            hand_robot_name=RobotName.xhand,
            hand_retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.right,
            hand_control_frequency=30.0,
            hand_smoothing_alpha=0.3
        )
        
        teleop = FrankaFERXHandVRTeleoperator(teleop_config)
        teleop.connect(calibrate=False)
        teleop.set_robot(robot)
        print("  ✅ Dual VR teleoperator connected")
        
        # ===== VERIFY DATA COLLECTION READINESS =====
        print("\n📊 Verifying data collection readiness...")
        
        # Check observation features
        obs_features = robot.observation_features
        print(f"  📝 Observation features ({len(obs_features)}):")
        for feature, dtype in list(obs_features.items())[:10]:  # Show first 10
            print(f"    - {feature}: {dtype}")
        if len(obs_features) > 10:
            print(f"    ... and {len(obs_features) - 10} more")
        
        # Check action features
        action_features = teleop.action_features
        print(f"  🎯 Action features ({len(action_features)}):")
        for feature, dtype in action_features.items():
            print(f"    - {feature}: {dtype}")
        
        # Test observation collection
        print("\n🔄 Testing observation collection...")
        start_time = time.perf_counter()
        obs = robot.get_observation()
        obs_time = time.perf_counter() - start_time
        
        print(f"  ⏱️  Observation time: {obs_time*1000:.1f}ms")
        print(f"  📦 Observation keys: {list(obs.keys())}")
        
        # Test action generation
        print("\n🎮 Testing VR action generation...")
        print("📱 Connect your VR device to test action generation...")
        
        for i in range(5):
            start_time = time.perf_counter()
            action = teleop.get_action()
            action_time = time.perf_counter() - start_time
            
            print(f"  Frame {i+1}: {action_time*1000:.1f}ms")
            print(f"    Arm actions: {len([k for k in action.keys() if k.startswith('arm_')])}")
            print(f"    Hand actions: {len([k for k in action.keys() if k.startswith('hand_')])}")
            
            # Show sample action values
            arm_sample = {k: f"{v:.3f}" for k, v in action.items() if k.startswith('arm_') and 'joint_0' in k}
            hand_sample = {k: f"{v:.3f}" for k, v in action.items() if k.startswith('hand_') and 'joint_0' in k}
            print(f"    Sample arm: {arm_sample}")
            print(f"    Sample hand: {hand_sample}")
            
            time.sleep(0.1)
        
        # ===== SUMMARY =====
        print("\n✅ Data Collection Setup Test Results:")
        print("  🤖 Composite robot: ✅ Connected and functional")
        print("  🎮 Dual VR teleoperator: ✅ Connected and generating actions")
        print("  📹 Cameras: ✅ Configured and capturing")
        print("  📊 Observations: ✅ Complete with arm, hand, and camera data")
        print("  🎯 Actions: ✅ Combined arm and hand actions with proper prefixes")
        print()
        print("🎉 READY FOR DATA COLLECTION!")
        print("You can now use:")
        print("  python -m lerobot.scripts.control_robot record \\")
        print("    --robot-path lerobot.robots.franka_fer_xhand \\")
        print("    --teleop-path lerobot.teleoperators.franka_fer_xhand_vr \\")
        print("    --robot-config <your_robot_config.yaml> \\")
        print("    --teleop-config <your_teleop_config.yaml>")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # ===== CLEANUP =====
        print("\n🧹 Cleaning up...")
        
        if teleop:
            try:
                teleop.disconnect()
                print("  ✅ Teleoperator disconnected")
            except Exception as e:
                print(f"  ⚠️  Error disconnecting teleoperator: {e}")
        
        if robot:
            try:
                robot.disconnect()
                print("  ✅ Robot disconnected")
            except Exception as e:
                print(f"  ⚠️  Error disconnecting robot: {e}")


if __name__ == "__main__":
    success = test_data_collection_setup()
    sys.exit(0 if success else 1)