#!/usr/bin/env python3
"""
Test VR joint target generation without actually moving the robot.
Shows that VR input produces different joint targets.
"""

import time
import sys
import threading
import socket
from pathlib import Path

# Add franka_teleoperator to path for importing the module
sys.path.insert(0, str(Path(__file__).parent / "franka_teleoperator"))

from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.teleoperators.vr_teleoperator import VRTeleoperator, VRTeleoperatorConfig

def send_vr_test_data(port=8000, duration=5.0):
    """Send animated VR test data to simulate hand movement"""
    print(f"📤 Sending animated VR data to port {port}")
    
    try:
        # Connect to the VR bridge
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', port))
        print("✓ Connected to VR bridge")
        
        # Send animated VR data to simulate hand movement
        start_time = time.time()
        frame = 0
        
        while time.time() - start_time < duration:
            # Simulate moving hand in a circle
            t = (time.time() - start_time) * 2  # 2 rad/s
            x = 0.5 + 0.1 * time.cos(t)  # X moves in circle
            y = 0.2 + 0.1 * time.sin(t)  # Y moves in circle  
            z = 0.3 + 0.05 * time.sin(t * 2)  # Z bobs up/down
            
            # Create quaternion (slight rotation)
            qx = 0.1 * time.sin(t * 0.5)
            qy = 0.0
            qz = 0.0
            qw = (1.0 - qx*qx)**0.5  # Normalize to unit quaternion
            
            # Alternate fist state
            fist_state = "closed" if frame % 40 < 20 else "open"
            
            message = f"Right wrist:, {x:.3f} {y:.3f} {z:.3f} {qx:.3f} {qy:.3f} {qz:.3f} {qw:.3f} leftFist: {fist_state}"
            
            sock.send(message.encode())
            frame += 1
            time.sleep(0.04)  # 25 Hz
        
        sock.close()
        print("✓ VR test data transmission complete")
        
    except Exception as e:
        print(f"✗ Error sending VR data: {e}")

def test_vr_joint_targets():
    """Test VR joint target generation"""
    print("🎯 Testing VR Joint Target Generation")
    print("=" * 45)
    
    # Create robot and VR teleoperator
    robot_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        cameras={}
    )
    
    vr_config = VRTeleoperatorConfig(
        tcp_port=8000,
        verbose=False,  # Reduce verbosity for cleaner output
        smoothing_factor=0.5,  # Less smoothing for more responsive demo
        use_bidexhand_limits=False
    )
    
    robot = None
    vr_teleop = None
    
    try:
        # Connect robot and VR
        print("🔄 Setting up robot and VR...")
        robot = FrankaFER(robot_config)
        robot.connect(calibrate=False)
        
        vr_teleop = VRTeleoperator(vr_config)
        vr_teleop.connect(robot=robot)
        print("✓ Robot and VR teleoperator connected")
        
        # Start VR data sender in background
        print("🔄 Starting VR data simulation...")
        vr_thread = threading.Thread(
            target=send_vr_test_data, 
            args=(vr_config.tcp_port, 8.0)
        )
        vr_thread.daemon = True
        vr_thread.start()
        
        # Wait a moment for VR connection
        time.sleep(1.0)
        
        # Monitor joint targets for changes
        print("\n📊 Monitoring joint target changes:")
        print("Current vs Target joint positions (should change with VR input)")
        print("-" * 60)
        
        last_targets = None
        changes_detected = 0
        
        for i in range(200):  # 8 seconds at 25Hz
            try:
                # Get current robot state
                obs = robot.get_observation()
                current_joints = [obs[f"joint_{i}.pos"] for i in range(7)]
                
                # Get VR action
                action = vr_teleop.get_action()
                target_joints = [action[f"joint_{i}.pos"] for i in range(7)]
                
                # Check for changes in targets
                if last_targets is not None:
                    max_change = max(abs(t - l) for t, l in zip(target_joints, last_targets))
                    if max_change > 0.001:  # 1mm threshold
                        changes_detected += 1
                
                # Print every 25 iterations (1 second)
                if i % 25 == 0:
                    status = vr_teleop.get_status()
                    vr_ready = status.get('vr_ready', False)
                    
                    print(f"\nTime: {i*0.04:.1f}s | VR Ready: {'✓' if vr_ready else '✗'} | Changes: {changes_detected}")
                    print("Current:  " + " ".join(f"{j:7.3f}" for j in current_joints))
                    print("Target:   " + " ".join(f"{j:7.3f}" for j in target_joints))
                    
                    # Show difference
                    diffs = [t - c for t, c in zip(target_joints, current_joints)]
                    print("Diff:     " + " ".join(f"{d:7.3f}" for d in diffs))
                    
                    if vr_ready and 'position' in status:
                        pos = status['position']
                        print(f"VR Pos:   [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")
                
                last_targets = target_joints.copy()
                time.sleep(0.04)
                
            except KeyboardInterrupt:
                print("\n🛑 Test interrupted")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                break
        
        # Results
        print(f"\n📈 Test Results:")
        print(f"  Total target changes detected: {changes_detected}")
        print(f"  VR system responsiveness: {'✅ GOOD' if changes_detected > 10 else '❌ POOR'}")
        
        if changes_detected > 10:
            print("\n✅ VR system is generating varying joint targets!")
            print("The robot would move if we enabled robot.send_action(action)")
        else:
            print("\n⚠️ VR system may not be receiving input properly")
        
        return changes_detected > 10
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if vr_teleop:
            vr_teleop.disconnect()
        if robot:
            robot.disconnect()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    test_vr_joint_targets()