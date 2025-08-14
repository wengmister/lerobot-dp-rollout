#!/usr/bin/env python3
"""
Test script to receive and display real VR input from headset.
This will show the actual data streaming from your VR app.
"""

import time
import sys
import subprocess
from pathlib import Path

# Add franka_teleoperator to path for importing the module
sys.path.insert(0, str(Path(__file__).parent / "franka_teleoperator"))

def setup_adb_forwarding(port=8000):
    """Setup adb reverse port forwarding for VR headset"""
    print("🔧 Setting up adb port forwarding...")
    
    try:
        # Check if adb is available
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ adb command not found. Please install Android SDK platform-tools.")
            return False
        
        print("📱 adb devices output:")
        print(result.stdout)
        
        # Check if device is connected
        if "device" not in result.stdout or result.stdout.count("device") < 2:
            print("❌ No Android device connected via adb.")
            print("💡 Connect your VR headset via USB and enable USB debugging")
            return False
        
        # Setup reverse port forwarding
        cmd = ['adb', 'reverse', f'tcp:{port}', f'tcp:{port}']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ Successfully setup adb reverse tcp:{port}")
            return True
        else:
            print(f"❌ Failed to setup adb reverse: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up adb: {e}")
        return False

def test_real_vr_input():
    """Test receiving real VR input from headset"""
    print("🥽 Real VR Input Test")
    print("=" * 30)
    
    # Setup adb forwarding
    if not setup_adb_forwarding(8000):
        print("\n⚠️  adb setup failed, but you can still test if your VR app")
        print("   connects directly to this computer's IP address")
    
    try:
        import vr_ik_bridge
        
        # Create VR bridge with maximum verbosity
        config = vr_ik_bridge.VRTeleopConfig()
        config.tcp_port = 8000
        config.verbose = True
        config.smoothing_factor = 0.1  # Minimal smoothing for real-time response
        
        bridge = vr_ik_bridge.VRIKBridge(config)
        bridge.start_tcp_server()
        print(f"✅ VR TCP server started on port {config.tcp_port}")
        
        print("\n📡 Instructions:")
        print("1. Make sure your VR headset is connected via USB")
        print("2. Enable USB debugging on your VR headset")
        print("3. Start your VR hand tracking app")
        print("4. Configure your VR app to send data to localhost:8000")
        print("5. Move your hands in VR to see data below")
        print("\n" + "=" * 50)
        print("📊 Real-time VR Data Stream:")
        print("=" * 50)
        
        last_position = None
        last_quaternion = None
        connection_attempts = 0
        max_wait_time = 60  # Wait up to 60 seconds for connection
        
        for i in range(max_wait_time * 25):  # 25Hz for 60 seconds
            status = bridge.get_vr_status()
            
            if status.get('connected', False):
                position = status.get('position', [0, 0, 0])
                quaternion = status.get('quaternion', [0, 0, 0, 1])
                fist_state = status.get('fist_state', 'unknown')
                
                # Check for significant changes
                pos_changed = (last_position is None or 
                             any(abs(p - lp) > 0.001 for p, lp in zip(position, last_position)))
                quat_changed = (last_quaternion is None or 
                              any(abs(q - lq) > 0.01 for q, lq in zip(quaternion, last_quaternion)))
                
                # Print data every second OR when significant change occurs
                if i % 25 == 0 or pos_changed or quat_changed:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] 🟢 VR CONNECTED")
                    print(f"  Position:    [{position[0]:8.4f}, {position[1]:8.4f}, {position[2]:8.4f}]")
                    print(f"  Quaternion:  [{quaternion[0]:7.3f}, {quaternion[1]:7.3f}, {quaternion[2]:7.3f}, {quaternion[3]:7.3f}]")
                    print(f"  Fist:        {fist_state}")
                    
                    if pos_changed or quat_changed:
                        print("  📈 MOVEMENT DETECTED!")
                    print("-" * 50)
                
                last_position = position.copy()
                last_quaternion = quaternion.copy()
                connection_attempts = 0  # Reset counter
                
            else:
                # Print waiting message every 5 seconds
                if i % (25 * 5) == 0:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] ⏳ Waiting for VR connection... ({i//25}/60s)")
                    print("  Make sure your VR app is running and configured correctly")
                    
                connection_attempts += 1
                
                # Give up after max wait time
                if i >= max_wait_time * 25 - 25:
                    print(f"\n❌ No VR connection after {max_wait_time} seconds")
                    print("\n🔧 Troubleshooting:")
                    print("1. Check USB connection to VR headset")
                    print("2. Verify adb devices shows your headset")
                    print("3. Check VR app is sending to correct port (8000)")
                    print("4. Try: adb reverse tcp:8000 tcp:8000")
                    break
            
            time.sleep(0.04)  # 25Hz
        
        # Stop the bridge
        bridge.stop()
        print("\n✅ VR bridge stopped")
        
        if last_position is not None:
            print("\n🎉 SUCCESS! Real VR data was received:")
            print(f"  Final position: {last_position}")
            print(f"  Final quaternion: {last_quaternion}")
            print("\n💡 Your VR system is working correctly!")
            return True
        else:
            print("\n❌ No VR data was received")
            return False
        
    except ImportError:
        print("❌ vr_ik_bridge module not found")
        print("💡 Build it first: cd franka_teleoperator && ./build.sh")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_adb():
    """Clean up adb port forwarding"""
    try:
        cmd = ['adb', 'reverse', '--remove', 'tcp:8000']
        subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print("🧹 Cleaned up adb port forwarding")
    except:
        pass

def main():
    """Main test function"""
    print("Real VR Input Test Script")
    print("=" * 30)
    print("This script will receive and display actual VR data from your headset")
    print()
    
    try:
        success = test_real_vr_input()
        
        if success:
            print("\n✅ VR input test completed successfully!")
            print("🚀 Ready for robot teleoperation!")
        else:
            print("\n❌ VR input test failed")
            print("🔧 Check your VR setup and try again")
            
    finally:
        cleanup_adb()

if __name__ == "__main__":
    main()