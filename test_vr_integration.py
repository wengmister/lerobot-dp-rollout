#!/usr/bin/env python3
"""
Test script for VR integration with LeRobot.
Tests the VRTeleoperator with Franka FER robot.
"""

import time
import sys
from pathlib import Path

# Add franka_teleoperator to path for importing the module
sys.path.insert(0, str(Path(__file__).parent / "franka_teleoperator"))

from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.teleoperators.vr_teleoperator import VRTeleoperator, VRTeleoperatorConfig

def test_vr_teleoperator():
    """Test VR teleoperator integration"""
    print("🤖 Testing VR Teleoperator Integration with Franka FER")
    print("=" * 60)
    
    # Create robot
    robot_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        cameras={}  # Disable cameras for this test
    )
    
    robot = FrankaFER(robot_config)
    
    # Create VR teleoperator
    vr_config = VRTeleoperatorConfig(
        tcp_port=8000,
        verbose=True,
        smoothing_factor=0.7,
        use_bidexhand_limits=False  # Use full Franka range
    )
    
    try:
        vr_teleop = VRTeleoperator(vr_config)
    except ImportError as e:
        print(f"✗ Failed to import VR bridge: {e}")
        print("\n💡 To fix this:")
        print("1. cd franka_teleoperator")
        print("2. ./build.sh")
        print("3. Run this test again")
        return False
    
    try:
        # Connect to robot
        print("🔄 Connecting to robot...")
        robot.connect(calibrate=False)
        print("✓ Robot connected")
        
        # Connect VR teleoperator
        print("🔄 Connecting VR teleoperator...")
        vr_teleop.connect(robot=robot)
        print("✓ VR teleoperator connected")
        print(f"✓ VR TCP server listening on port {vr_config.tcp_port}")
        
        # Wait for VR connection
        print("\n📱 Waiting for VR connection...")
        print("Make sure your VR app is:")
        print("1. Connected to this computer via adb")
        print("2. Sending data to localhost:8000")
        print("3. Or run: adb reverse tcp:8000 tcp:8000")
        print("\nPress Ctrl+C to exit")
        
        # Test loop
        last_status_time = time.time()
        iteration = 0
        
        while True:
            try:
                # Get VR status
                status = vr_teleop.get_status()
                
                # Print status every 2 seconds
                if time.time() - last_status_time > 2.0:
                    print(f"\n📊 Status (iteration {iteration}):")
                    print(f"  VR connected: {'✓' if status.get('connected', False) else '✗'}")
                    print(f"  VR ready: {'✓' if status.get('vr_ready', False) else '✗'}")
                    
                    if 'position' in status:
                        pos = status['position']
                        print(f"  VR position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    
                    if 'fist_state' in status:
                        print(f"  Fist state: {status['fist_state']}")
                    
                    last_status_time = time.time()
                    iteration += 1
                
                # Get action from VR
                action = vr_teleop.get_action()
                
                # Print joint targets if VR is working
                if status.get('vr_ready', False):
                    joint_positions = [action[f"joint_{i}.pos"] for i in range(7)]
                    print(f"🎯 Target joints: {[f'{j:.3f}' for j in joint_positions]}")
                    
                    # Send action to robot (optional - be careful!)
                    # robot.send_action(action)
                
                time.sleep(0.04)  # 25Hz loop
                
            except KeyboardInterrupt:
                print("\n🛑 Test interrupted by user")
                break
            except Exception as e:
                print(f"✗ Error in test loop: {e}")
                break
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            print("\n🔌 Disconnecting...")
            if 'vr_teleop' in locals():
                vr_teleop.disconnect()
            if robot.is_connected:
                robot.disconnect()
            print("✓ Cleanup complete")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

def test_module_import():
    """Test if VR bridge module can be imported"""
    print("🔍 Testing VR bridge module import...")
    
    try:
        import vr_ik_bridge
        print("✓ vr_ik_bridge module imported successfully")
        
        # Test basic functionality
        config = vr_ik_bridge.VRTeleopConfig()
        config.verbose = True
        
        bridge = vr_ik_bridge.VRIKBridge(config)
        print("✓ VRIKBridge created successfully")
        
        status = bridge.get_vr_status()
        print(f"✓ Got VR status: {status}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import vr_ik_bridge: {e}")
        print("\n💡 Build the module first:")
        print("cd franka_teleoperator && ./build.sh")
        return False
    except Exception as e:
        print(f"✗ Error testing module: {e}")
        return False

def main():
    """Main test function"""
    print("VR Integration Test for LeRobot")
    print("=" * 40)
    
    # Test 1: Module import
    if not test_module_import():
        print("\n❌ Module import test failed")
        return
    
    print("\n" + "=" * 40)
    
    # Test 2: Full integration (optional)
    user_input = input("Run full VR teleoperator test? (y/N): ")
    if user_input.lower() in ['y', 'yes']:
        success = test_vr_teleoperator()
        if success:
            print("\n✅ VR integration test completed!")
        else:
            print("\n❌ VR integration test failed!")
    else:
        print("Skipping full integration test")
    
    print("\n🎉 Testing complete!")
    print("\nNext steps:")
    print("1. Build the C++ module: cd franka_teleoperator && ./build.sh")
    print("2. Test with your VR app sending data to localhost:8000")
    print("3. Use with lerobot-record: --teleop.type=vr")

if __name__ == "__main__":
    main()