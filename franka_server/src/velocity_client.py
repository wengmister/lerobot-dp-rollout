#!/usr/bin/env python3
"""
Simple Python client for testing the Franka Velocity Server
"""
import socket
import time
import numpy as np
import argparse

class FrankaVelocityClient:
    def __init__(self, server_ip="192.168.18.1", server_port=5000):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect to velocity server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.connected = True
            print(f"Connected to velocity server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.connected:
            try:
                self._send_command("DISCONNECT")
                self.socket.close()
            except:
                pass
            self.connected = False
            print("Disconnected from server")
    
    def _send_command(self, command):
        """Send command and get response"""
        if not self.connected:
            raise RuntimeError("Not connected to server")
        
        try:
            self.socket.send((command + "\n").encode())
            response = self.socket.recv(1024).decode().strip()
            return response
        except Exception as e:
            print(f"Communication error: {e}")
            self.connected = False
            return None
    
    def get_state(self):
        """Get current robot state"""
        response = self._send_command("GET_STATE")
        if response and response.startswith("STATE"):
            # Parse: STATE pos0 pos1 ... pos6 vel0 vel1 ... vel6
            parts = response.split()[1:]  # Skip "STATE"
            if len(parts) >= 14:
                positions = [float(x) for x in parts[:7]]
                velocities = [float(x) for x in parts[7:14]]
                return {
                    'positions': np.array(positions),
                    'velocities': np.array(velocities)
                }
        return None
    
    def set_position(self, positions):
        """Send position command to robot"""
        if len(positions) != 7:
            raise ValueError("Position command must have 7 values")
        
        cmd = "SET_POSITION " + " ".join(f"{p:.6f}" for p in positions)
        response = self._send_command(cmd)
        return response == "OK"
    
    def move_to_start(self, start_positions):
        """Move robot to starting position safely"""
        if len(start_positions) != 7:
            raise ValueError("Start position must have 7 values")
        
        cmd = "MOVE_TO_START " + " ".join(f"{p:.6f}" for p in start_positions)
        response = self._send_command(cmd)
        return response == "OK"
    
    def stop(self):
        """Stop robot motion"""
        response = self._send_command("STOP")
        return response == "OK"

def test_basic_connection(client):
    """Test basic connection and state reading"""
    print("\n=== Testing Basic Connection ===")
    
    state = client.get_state()
    if state:
        print(f"Current positions: {state['positions']}")
        print(f"Current velocities: {state['velocities']}")
    else:
        print("Failed to get robot state")

def test_zero_velocity(client):
    """Test sending zero velocity commands"""
    print("\n=== Testing Zero Velocity ===")
    
    zero_vel = np.zeros(7)
    success = client.set_velocity(zero_vel)
    print(f"Zero velocity command: {'SUCCESS' if success else 'FAILED'}")
    
    time.sleep(1.0)
    state = client.get_state()
    if state:
        print(f"Robot velocities after zero command: {state['velocities']}")

def test_small_velocity(client):
    """Test small velocity commands"""
    print("\n=== Testing Small Velocity Commands ===")
    
    # Small velocity on joint 0
    small_vel = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("Sending small velocity command for 2 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 2.0:
        success = client.set_velocity(small_vel)
        if not success:
            print("Failed to send velocity command")
            break
        time.sleep(0.05)  # 20Hz command rate
    
    print("Motion stopped")

def replay_csv_trajectory(client, csv_file, position_scale=1.0):
    """Replay trajectory from CSV file using positions"""
    print(f"\n=== Replaying Trajectory from {csv_file} ===")
    
    import pandas as pd
    
    try:
        # Load CSV data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} trajectory points")
        
        # Extract recorded positions (joint0_pos to joint6_pos)
        pos_columns = [f'joint{i}_pos' for i in range(7)]
        positions = df[pos_columns].values
        
        if len(positions) == 0:
            print("No position data found in CSV")
            return
        
        print(f"Position scaling factor: {position_scale}")
        
        # Move to starting position first
        start_position = positions[0] * position_scale
        print(f"Sending start position: {start_position}")
        print("(Robot will move to start position when control begins)")
        
        success = client.move_to_start(start_position)
        if not success:
            print("Failed to send start position command")
            return
        
        print("Start position command sent successfully. Beginning trajectory replay...")
        print("Press Ctrl+C to stop")

        time.sleep(5)  # Give server time to move to start position

        start_time = time.time()
        
        for i, pos_cmd in enumerate(positions):
            try:
                scaled_pos = pos_cmd * position_scale
                success = client.set_position(scaled_pos)
                if not success:
                    print(f"Failed to send position command at step {i}")
                    break
                
                # Print progress every 50 steps
                if i % 50 == 0:
                    print(f"Step {i}/{len(positions)}")
                
                # Maintain ~25Hz rate (matching original recording)
                time.sleep(0.04)
                
            except KeyboardInterrupt:
                print("\nTrajectory stopped by user")
                break
    
        print("Trajectory replay completed")
        
    except Exception as e:
        print(f"Error during trajectory replay: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Franka Velocity Server')
    parser.add_argument('--server_ip', default='192.168.18.1',
                       help='IP address of velocity server')
    parser.add_argument('--test', choices=['basic', 'zero', 'small', 'replay'],
                       default='basic', help='Test to run')
    parser.add_argument('--csv_file', help='CSV file for trajectory replay')
    parser.add_argument('--position_scale', type=float, default=1.0,
                       help='Position scaling factor for replay')
    
    args = parser.parse_args()
    
    # Create client and connect
    client = FrankaVelocityClient(args.server_ip)
    
    if not client.connect():
        return
    
    try:
        if args.test == 'basic':
            test_basic_connection(client)
        elif args.test == 'zero':
            test_zero_velocity(client)
        elif args.test == 'small':
            test_small_velocity(client)
        elif args.test == 'replay':
            if not args.csv_file:
                print("Error: --csv_file required for replay test")
                return
            replay_csv_trajectory(client, args.csv_file, args.position_scale)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()