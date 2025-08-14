#!/usr/bin/env python3
"""
Automatic test script for VR integration with LeRobot.
Tests the VRTeleoperator with Franka FER robot without interactive prompts.
"""

import time
import sys
from pathlib import Path

# Add franka_teleoperator to path for importing the module
sys.path.insert(0, str(Path(__file__).parent / "franka_teleoperator"))

from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.teleoperators.vr_teleoperator import VRTeleoperator, VRTeleoperatorConfig

def test_vr_integration():
    """Test VR integration without robot connection"""
    print("🤖 Testing VR Integration (No Robot Connection)")
    print("=" * 50)
    
    # Test VR bridge import
    try:
        import vr_ik_bridge
        print("✓ vr_ik_bridge module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import vr_ik_bridge: {e}")
        return False
    
    # Test VR teleoperator creation
    vr_config = VRTeleoperatorConfig(
        tcp_port=8000,
        verbose=True,
        smoothing_factor=0.7,
        use_bidexhand_limits=False
    )
    
    try:
        vr_teleop = VRTeleoperator(vr_config)
        print("✓ VRTeleoperator created successfully")
    except Exception as e:
        print(f"✗ Failed to create VRTeleoperator: {e}")
        return False
    
    # Test connection without robot (should work for TCP server setup)
    try:
        print("🔄 Setting up VR teleoperator (no robot)...")
        vr_teleop.connect(robot=None)
        print("✓ VR TCP server started")
        print(f"✓ Listening on port {vr_config.tcp_port}")
    except Exception as e:
        print(f"✗ Error setting up VR teleoperator: {e}")
        return False
    
    # Test status and basic functionality for a few seconds
    print("\n📡 Testing VR bridge for 5 seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 5.0:
            status = vr_teleop.get_status()
            
            if status.get('connected', False):
                print(f"📱 VR connected! Position: {status.get('position', 'N/A')}")
                break
            
            time.sleep(0.5)
        
        # Test getting action (should return default values without robot)
        action = vr_teleop.get_action()
        print(f"✓ Got action keys: {list(action.keys())[:5]}...")  # Show first 5 keys
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            vr_teleop.disconnect()
            print("✓ VR teleoperator disconnected")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def test_robot_connection():
    """Test robot connection separately"""
    print("\n🔌 Testing Robot Connection")
    print("=" * 30)
    
    robot_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        cameras={}  # No cameras for this test
    )
    
    robot = FrankaFER(robot_config)
    
    try:
        print("🔄 Connecting to robot...")
        robot.connect(calibrate=False)
        print("✓ Robot connected successfully")
        
        # Test getting observation
        obs = robot.get_observation()
        joint_positions = [obs[f"joint_{i}.pos"] for i in range(7)]
        print(f"✓ Got joint positions: {[f'{j:.3f}' for j in joint_positions]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Robot connection failed: {e}")
        return False
    
    finally:
        try:
            if robot.is_connected:
                robot.disconnect()
            print("✓ Robot disconnected")
        except Exception as e:
            print(f"⚠️ Robot cleanup error: {e}")

def main():
    """Main test function"""
    print("VR Integration Automatic Test")
    print("=" * 35)
    
    # Test 1: VR integration
    vr_success = test_vr_integration()
    
    # Test 2: Robot connection (optional)
    robot_success = test_robot_connection()
    
    print("\n" + "=" * 35)
    print("📊 Test Results:")
    print(f"  VR Integration: {'✅ PASS' if vr_success else '❌ FAIL'}")
    print(f"  Robot Connection: {'✅ PASS' if robot_success else '❌ FAIL'}")
    
    if vr_success and robot_success:
        print("\n🎉 All tests passed! VR teleoperator is ready.")
        print("\n💡 Next steps:")
        print("1. Connect your VR app to localhost:8000")
        print("2. Run: python test_vr_teleoperation.py")
        print("3. Use with lerobot-record for data collection")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()