#!/usr/bin/env python3
"""
Basic test of VR IK Bridge module without robot connection.
Tests the VR TCP server and basic IK functionality.
"""

import time
import sys
import threading
import socket

def test_vr_bridge_basic():
    """Test basic VR bridge functionality"""
    print("🤖 Testing VR IK Bridge Module")
    print("=" * 40)
    
    try:
        import vr_ik_bridge
        print("✓ vr_ik_bridge module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import vr_ik_bridge: {e}")
        return False
    
    # Test configuration
    config = vr_ik_bridge.VRTeleopConfig()
    config.tcp_port = 8000  # Use standard VR port
    config.verbose = True
    config.smoothing_factor = 0.5
    
    print(f"✓ VRTeleopConfig created with port {config.tcp_port}")
    
    # Create VR bridge
    try:
        bridge = vr_ik_bridge.VRIKBridge(config)
        print("✓ VRIKBridge created successfully")
    except Exception as e:
        print(f"✗ Failed to create VRIKBridge: {e}")
        return False
    
    # Test TCP server
    try:
        success = bridge.start_tcp_server()
        if success:
            print(f"✓ VR TCP server started on port {config.tcp_port}")
        else:
            print("✗ Failed to start VR TCP server")
            return False
    except Exception as e:
        print(f"✗ Error starting TCP server: {e}")
        return False
    
    # Test status
    try:
        status = bridge.get_vr_status()
        print(f"✓ Got VR status: {status}")
    except Exception as e:
        print(f"✗ Error getting VR status: {e}")
        return False
    
    # Test IK solver setup
    try:
        neutral_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, -0.9]  # Home position
        success = bridge.setup_ik_solver(neutral_pose)
        if success:
            print("✓ IK solver setup successfully")
        else:
            print("✗ Failed to setup IK solver")
            return False
    except Exception as e:
        print(f"✗ Error setting up IK solver: {e}")
        return False
    
    # Test Q7 limits
    try:
        bridge.set_q7_limits(-2.89, 2.89)
        print("✓ Q7 limits set successfully")
    except Exception as e:
        print(f"✗ Error setting Q7 limits: {e}")
        return False
    
    # Test joint targets (without VR input)
    try:
        current_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, -0.9]
        target_joints = bridge.get_joint_targets(current_joints)
        print(f"✓ Got joint targets: {[f'{j:.3f}' for j in target_joints]}")
        print("  (Should return current joints since no VR input)")
    except Exception as e:
        print(f"✗ Error getting joint targets: {e}")
        return False
    
    print("\n📡 VR TCP server is now listening...")
    print(f"To test with VR data, send to localhost:{config.tcp_port}")
    print("Format: 'Right wrist:, x y z qx qy qz qw leftFist: open'")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep server running for a bit
        for i in range(10):
            time.sleep(1)
            status = bridge.get_vr_status()
            if status.get("connected", False):
                print(f"📱 VR connected! Position: {status.get('position', 'N/A')}")
                break
            elif i % 3 == 0:
                print(f"⏳ Waiting for VR connection... ({i+1}/10)")
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    
    # Cleanup
    try:
        bridge.stop()
        print("✓ VR bridge stopped successfully")
    except Exception as e:
        print(f"⚠️ Error stopping bridge: {e}")
    
    return True

def send_test_vr_data(port=8001):
    """Send some test VR data to the bridge"""
    print(f"\n🧪 Sending test VR data to port {port}")
    
    try:
        # Connect to the bridge
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        print("✓ Connected to VR bridge")
        
        # Send test VR data
        test_messages = [
            "Right wrist:, 0.5 0.2 0.3 0.0 0.0 0.0 1.0 leftFist: open",
            "Right wrist:, 0.5 0.25 0.3 0.1 0.0 0.0 0.995 leftFist: open", 
            "Right wrist:, 0.5 0.3 0.3 0.2 0.0 0.0 0.98 leftFist: closed",
        ]
        
        for i, msg in enumerate(test_messages):
            sock.send(msg.encode())
            print(f"📤 Sent message {i+1}: {msg[:50]}...")
            time.sleep(1)
        
        sock.close()
        print("✓ Test data sent successfully")
        
    except Exception as e:
        print(f"✗ Error sending test data: {e}")

def main():
    """Main test function"""
    print("VR IK Bridge Basic Test")
    print("=" * 30)
    
    # Test basic functionality
    success = test_vr_bridge_basic()
    
    if success:
        print("\n✅ Basic VR bridge test passed!")
        print("\n💡 Next steps:")
        print("1. Test with real VR data")
        print("2. Connect to robot for full teleoperation")
        print("3. Use with LeRobot data collection")
    else:
        print("\n❌ Basic VR bridge test failed!")

if __name__ == "__main__":
    main()