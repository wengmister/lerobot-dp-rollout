#!/usr/bin/env python3
"""
Test script to measure robot state reading performance for VR teleop integration.
Tests if we can get robot state fast enough for real-time IK solving at 25Hz.
"""

import time
import numpy as np
import statistics
from threading import Thread, Event
import signal
import sys

from src.lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from src.lerobot.robots.franka_fer.franka_fer import FrankaFER

class VRTeleopPerformanceTest:
    def __init__(self):
        self.robot = None
        self.running = Event()
        self.running.set()
        
        # Performance tracking
        self.state_read_times = []
        self.loop_times = []
        self.max_samples = 1000
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\n🛑 Received interrupt signal, stopping test...")
        self.running.clear()
    
    def connect_robot(self):
        """Connect to the robot"""
        print("🔄 Connecting to Franka robot...")
        
        # Create config without cameras for performance testing
        config = FrankaFERConfig(
            server_ip="192.168.18.1",
            server_port=5000,
            cameras={}  # Disable cameras for pure robot state testing
        )
        
        self.robot = FrankaFER(config)
        
        try:
            self.robot.connect(calibrate=False)
            print("✓ Robot connected successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def test_single_state_read_speed(self):
        """Test speed of individual robot state reads"""
        print("\n📊 Testing single robot state read speed...")
        
        times = []
        for i in range(100):
            start = time.perf_counter()
            obs = self.robot.get_observation()
            end = time.perf_counter()
            
            read_time = (end - start) * 1000  # Convert to ms
            times.append(read_time)
            
            if i % 20 == 0:
                print(f"  Sample {i+1}: {read_time:.2f}ms")
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_time = statistics.stdev(times)
        
        print(f"\n📈 Single State Read Statistics (100 samples):")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Median:  {median_time:.2f}ms")
        print(f"  Min:     {min_time:.2f}ms")
        print(f"  Max:     {max_time:.2f}ms")
        print(f"  StdDev:  {std_time:.2f}ms")
        
        # Assess performance
        target_rate = 25  # Hz
        target_time = 1000 / target_rate  # ms
        
        print(f"\n🎯 VR Teleop Assessment (target: {target_rate}Hz = {target_time:.1f}ms per cycle):")
        if avg_time < target_time * 0.5:
            print(f"  ✅ EXCELLENT: State reads are {target_time/avg_time:.1f}x faster than needed")
        elif avg_time < target_time * 0.8:
            print(f"  ✅ GOOD: State reads leave {target_time-avg_time:.1f}ms for IK computation")
        else:
            print(f"  ⚠️  TIGHT: State reads use {avg_time/target_time*100:.1f}% of available time")
    
    def test_continuous_25hz_loop(self, duration_sec=10):
        """Test continuous 25Hz robot state reading loop"""
        print(f"\n🔄 Testing continuous 25Hz loop for {duration_sec} seconds...")
        print("Press Ctrl+C to stop early")
        
        target_hz = 25
        target_period = 1.0 / target_hz
        
        loop_count = 0
        start_time = time.perf_counter()
        last_loop_time = start_time
        
        self.state_read_times.clear()
        self.loop_times.clear()
        
        while self.running.is_set():
            loop_start = time.perf_counter()
            
            # Time the robot state read
            state_start = time.perf_counter()
            obs = self.robot.get_observation()
            state_end = time.perf_counter()
            
            state_read_time = (state_end - state_start) * 1000  # ms
            self.state_read_times.append(state_read_time)
            
            # Extract joint positions (what IK would need)
            joint_positions = [obs[f"joint_{i}.pos"] for i in range(7)]
            
            # Calculate loop timing
            current_time = time.perf_counter()
            actual_period = current_time - last_loop_time
            loop_time = (current_time - loop_start) * 1000  # ms
            self.loop_times.append(loop_time)
            
            loop_count += 1
            last_loop_time = current_time
            
            # Print periodic updates
            if loop_count % 50 == 0:
                elapsed = current_time - start_time
                actual_hz = loop_count / elapsed
                avg_state_time = statistics.mean(self.state_read_times[-50:])
                avg_loop_time = statistics.mean(self.loop_times[-50:])
                
                print(f"  Loop {loop_count}: {actual_hz:.1f}Hz, "
                      f"state: {avg_state_time:.1f}ms, "
                      f"total: {avg_loop_time:.1f}ms")
            
            # Check if we should stop
            elapsed = current_time - start_time
            if elapsed >= duration_sec:
                break
            
            # Sleep to maintain 25Hz
            sleep_time = target_period - (current_time - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Calculate final statistics
        total_elapsed = time.perf_counter() - start_time
        actual_hz = loop_count / total_elapsed
        
        print(f"\n📈 Continuous Loop Statistics ({loop_count} loops, {total_elapsed:.1f}s):")
        print(f"  Target rate: {target_hz:.1f}Hz")
        print(f"  Actual rate: {actual_hz:.1f}Hz")
        print(f"  Rate error:  {((actual_hz-target_hz)/target_hz)*100:+.1f}%")
        
        if self.state_read_times:
            avg_state = statistics.mean(self.state_read_times)
            max_state = max(self.state_read_times)
            print(f"  Avg state read: {avg_state:.2f}ms")
            print(f"  Max state read: {max_state:.2f}ms")
            
        if self.loop_times:
            avg_loop = statistics.mean(self.loop_times)
            max_loop = max(self.loop_times)
            remaining_time = (1000/target_hz) - avg_loop
            print(f"  Avg loop time:  {avg_loop:.2f}ms")
            print(f"  Max loop time:  {max_loop:.2f}ms")
            print(f"  Time for IK:    {remaining_time:.2f}ms")
        
        # Assessment for VR teleop
        print(f"\n🎯 VR Teleop Readiness:")
        if actual_hz >= target_hz * 0.95 and remaining_time > 10:
            print("  ✅ READY: Stable 25Hz with plenty of time for IK")
        elif actual_hz >= target_hz * 0.90:
            print("  ✅ GOOD: Nearly stable 25Hz, should work for VR")
        elif actual_hz >= target_hz * 0.80:
            print("  ⚠️  MARGINAL: Somewhat unstable rate, may work")
        else:
            print("  ❌ NOT READY: Rate too unstable for real-time VR")
    
    def test_state_plus_mock_ik(self, duration_sec=5):
        """Test robot state reading plus mock IK computation"""
        print(f"\n🧮 Testing state read + mock IK computation for {duration_sec} seconds...")
        
        def mock_ik_solve(current_joints, target_pose):
            """Simulate IK computation time - typically 1-5ms"""
            # Simulate some computation
            result = np.array(current_joints)
            for i in range(100):  # Adjust to simulate realistic IK time
                result = result + 0.001 * np.sin(result + i)
            return result
        
        target_hz = 25
        target_period = 1.0 / target_hz
        
        loop_count = 0
        start_time = time.perf_counter()
        
        total_times = []
        state_times = []
        ik_times = []
        
        while self.running.is_set() and (time.perf_counter() - start_time) < duration_sec:
            loop_start = time.perf_counter()
            
            # 1. Read robot state
            state_start = time.perf_counter()
            obs = self.robot.get_observation()
            joint_positions = [obs[f"joint_{i}.pos"] for i in range(7)]
            state_time = (time.perf_counter() - state_start) * 1000
            
            # 2. Mock IK computation
            ik_start = time.perf_counter()
            target_pose = [0.4, 0.0, 0.5, 0, 0, 0, 1]  # Mock target pose
            new_joints = mock_ik_solve(joint_positions, target_pose)
            ik_time = (time.perf_counter() - ik_start) * 1000
            
            total_time = (time.perf_counter() - loop_start) * 1000
            
            state_times.append(state_time)
            ik_times.append(ik_time)
            total_times.append(total_time)
            
            loop_count += 1
            
            if loop_count % 25 == 0:
                avg_state = statistics.mean(state_times[-25:])
                avg_ik = statistics.mean(ik_times[-25:])
                avg_total = statistics.mean(total_times[-25:])
                print(f"  Loop {loop_count}: state={avg_state:.1f}ms, ik={avg_ik:.1f}ms, total={avg_total:.1f}ms")
            
            # Sleep to maintain target rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Final statistics
        print(f"\n📈 State + IK Statistics ({loop_count} loops):")
        if state_times and ik_times and total_times:
            print(f"  Avg state read: {statistics.mean(state_times):.2f}ms")
            print(f"  Avg IK solve:   {statistics.mean(ik_times):.2f}ms")
            print(f"  Avg total:      {statistics.mean(total_times):.2f}ms")
            print(f"  Max total:      {max(total_times):.2f}ms")
            
            margin = (1000/target_hz) - statistics.mean(total_times)
            print(f"  Time margin:    {margin:.2f}ms")
            
            if margin > 5:
                print("  ✅ EXCELLENT: Plenty of margin for real VR teleop")
            elif margin > 0:
                print("  ✅ GOOD: Should work for VR teleop")
            else:
                print("  ❌ TOO SLOW: Need optimization for real-time VR")
    
    def run_all_tests(self):
        """Run all performance tests"""
        if not self.connect_robot():
            return
        
        try:
            print("\n" + "="*60)
            print("VR TELEOP PERFORMANCE TEST")
            print("="*60)
            
            # Test 1: Single read speed
            self.test_single_state_read_speed()
            
            # Test 2: Continuous 25Hz loop
            self.test_continuous_25hz_loop(duration_sec=10)
            
            # Test 3: State + mock IK
            self.test_state_plus_mock_ik(duration_sec=5)
            
            print(f"\n" + "="*60)
            print("FINAL ASSESSMENT FOR VR TELEOP INTEGRATION")
            print("="*60)
            
            if self.state_read_times:
                avg_state_time = statistics.mean(self.state_read_times)
                print(f"Average robot state read time: {avg_state_time:.2f}ms")
                
                if avg_state_time < 10:
                    print("✅ Robot state reading is fast enough for VR teleop")
                    print("✅ Ready to implement C++ IK bridge with LeRobot")
                    print("\nNext steps:")
                    print("1. Create pybind11 bridge for your VR IK code")
                    print("2. Implement VRTeleoperator class for LeRobot")
                    print("3. Test with actual VR input")
                else:
                    print("⚠️  Robot state reading might be too slow")
                    print("Consider optimizing TCP communication")
            
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.robot and self.robot.is_connected:
                print("\n🔌 Disconnecting from robot...")
                self.robot.disconnect()
                print("✓ Disconnected")

def main():
    """Main test function"""
    print("🤖 Franka FER VR Teleop Performance Test")
    print("This test will measure if robot state can be read fast enough for VR teleop at 25Hz")
    print("Make sure your velocity_server.cpp is running on the RTPC!")
    
    # Check if running in interactive mode
    import sys
    if sys.stdin.isatty():
        user_input = input("\n🤖 Ready to start the test? (y/N): ")
        if user_input.lower() not in ['y', 'yes']:
            print("Test cancelled")
            return
    else:
        print("\n🤖 Running in non-interactive mode, starting test automatically...")
    
    test = VRTeleopPerformanceTest()
    test.run_all_tests()

if __name__ == "__main__":
    main()