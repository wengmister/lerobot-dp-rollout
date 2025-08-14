#!/usr/bin/env python3
"""
Test VR message parsing with your exact message format.
"""

import sys
from pathlib import Path

# Add franka_teleoperator to path
sys.path.insert(0, str(Path(__file__).parent / "franka_teleoperator"))

def test_vr_parsing():
    """Test parsing with real VR message format"""
    print("🔍 Testing VR Message Parsing")
    print("=" * 35)
    
    # Real message from your VR app
    test_message = "Right wrist:, 0.123, 0.750, 0.263, -0.234, -0.435, -0.295, 0.818, leftFist: open"
    
    print(f"Test message: {test_message}")
    print()
    
    try:
        import vr_ik_bridge
        
        # Create VR bridge 
        config = vr_ik_bridge.VRTeleopConfig()
        config.verbose = True
        
        bridge = vr_ik_bridge.VRIKBridge(config)
        
        print("✅ VR bridge created")
        
        # Test the parsing by manually feeding the message
        # Since we can't directly call the parsing function, let's test via TCP
        
        bridge.start_tcp_server()
        print("✅ TCP server started")
        
        # Give it a moment to initialize
        import time
        time.sleep(0.5)
        
        # Send test message via socket
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 8000))
        
        print(f"📤 Sending test message...")
        sock.send(test_message.encode())
        
        time.sleep(0.5)  # Give it time to process
        
        # Check the parsed result
        status = bridge.get_vr_status()
        print(f"📊 Parsed VR status: {status}")
        
        if status.get('connected', False):
            pos = status.get('position', [0, 0, 0])
            quat = status.get('quaternion', [0, 0, 0, 1])
            fist = status.get('fist_state', '')
            
            print(f"✅ Position: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
            print(f"✅ Quaternion: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
            print(f"✅ Fist: '{fist}'")
            
            # Check if parsing worked correctly
            expected_pos = [0.123, 0.750, 0.263]
            expected_quat = [-0.234, -0.435, -0.295, 0.818]
            
            pos_correct = all(abs(p - e) < 0.01 for p, e in zip(pos, expected_pos))
            quat_correct = all(abs(q - e) < 0.01 for q, e in zip(quat, expected_quat))
            
            if pos_correct and quat_correct:
                print("🎉 Parsing is working correctly!")
            else:
                print("❌ Parsing issue detected:")
                print(f"   Expected pos: {expected_pos}")
                print(f"   Got pos: {pos}")
                print(f"   Expected quat: {expected_quat}")  
                print(f"   Got quat: {quat}")
        else:
            print("❌ VR not connected - parsing may have failed")
        
        sock.close()
        bridge.stop()
        
    except ImportError:
        print("❌ vr_ik_bridge module not found")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vr_parsing()