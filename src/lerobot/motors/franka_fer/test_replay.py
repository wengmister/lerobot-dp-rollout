import csv
import numpy as np
import time
import argparse
from typing import List, Tuple
from franky_client import FrankyClient


def load_trajectory_data(csv_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load trajectory data from CSV file
    
    Returns:
        timestamps: Array of timestamps
        positions: Array of joint positions (N x 7)
        velocities: Array of recorded joint velocities (N x 7)
    """
    timestamps = []
    positions = []
    velocities = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            
            # Extract joint positions (joint0_pos to joint6_pos)
            pos = [float(row[f'joint{i}_pos']) for i in range(7)]
            positions.append(pos)
            
            # Extract recorded joint velocities (joint0_vel to joint6_vel)  
            vel = [float(row[f'joint{i}_vel']) for i in range(7)]
            velocities.append(vel)
    
    return np.array(timestamps), np.array(positions), np.array(velocities)


def smooth_velocities(velocities: np.ndarray, smoothing_factor: float = 0.8) -> np.ndarray:
    """
    Apply smoothing filter to velocity data to reduce jitter
    
    Args:
        velocities: Raw velocity data (N x 7)
        smoothing_factor: Smoothing factor (0-1, higher = more smoothing)
    
    Returns:
        smoothed_velocities: Filtered velocity commands
    """
    smoothed = velocities.copy()
    for i in range(1, len(velocities)):
        smoothed[i] = smoothing_factor * smoothed[i-1] + (1 - smoothing_factor) * velocities[i]
    return smoothed


class TrajectoryReplayer:
    def __init__(self, server_ip: str = "192.168.18.1", server_port: int = 5000):
        self.client = FrankyClient(server_ip, server_port)
        self.is_connected = False
        
    def connect(self, dynamics_factor: float = 0.1) -> bool:
        """Connect to robot with conservative dynamics"""
        self.is_connected = self.client.connect(dynamics_factor)
        return self.is_connected
        
    def disconnect(self):
        """Disconnect from robot"""
        if self.is_connected:
            self.client.stop()
            self.client.disconnect()
            self.is_connected = False
            
    def replay_trajectory(self, csv_file: str, velocity_scale: float = 0.5, 
                         use_recorded_velocities: bool = False, smoothing: float = 0.7):
        """
        Replay trajectory from CSV file using velocity control
        
        Args:
            csv_file: Path to CSV file containing trajectory
            velocity_scale: Scale factor for velocity commands (0-1)
            use_recorded_velocities: Use recorded velocities instead of calculated
        """
        if not self.is_connected:
            print("Error: Not connected to robot")
            return False
            
        # Load trajectory data
        print(f"Loading trajectory from {csv_file}...")
        timestamps, positions, recorded_velocities = load_trajectory_data(csv_file)
        print(f"Loaded {len(positions)} trajectory points")

        velocities = recorded_velocities
        print("Using recorded velocities")
        
        # Apply smoothing to reduce jitter
        velocities = smooth_velocities(velocities, smoothing_factor=smoothing)
        print(f"Applied velocity smoothing (factor: {smoothing})")
    
        # Scale velocities for safety
        velocities *= velocity_scale
        print(f"Velocity scaling factor: {velocity_scale}")
        
        # Get initial robot position
        current_pos = self.client.get_joint_positions()
        if current_pos is None:
            print("Error: Could not get current robot position")
            return False
            
        print(f"Current robot position: {current_pos}")
        print(f"Target start position: {positions[0]}")
        
        # Move to trajectory starting position
        position_diff = np.abs(current_pos - positions[0])
        max_diff = np.max(position_diff)
        
        if max_diff > 0.01:  # 0.01 rad tolerance
            print(f"Moving to trajectory start position (diff: {max_diff:.3f} rad)")
            success = self.client.move_joints(positions[0])
            if not success:
                print("Error: Failed to move to starting position")
                return False
            
            # Wait for movement to complete and verify position
            time.sleep(2.0)
            current_pos = self.client.get_joint_positions()
            if current_pos is None:
                print("Error: Could not verify robot position after move")
                return False
                
            final_diff = np.max(np.abs(current_pos - positions[0]))
            print(f"Moved to start position (final diff: {final_diff:.3f} rad)")
            
            if final_diff > 0.05:  # 0.05 rad tolerance after movement
                print(f"Warning: Still not close to start position after move")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    return False
        else:
            print("Robot already at trajectory start position")
        
        print("Starting trajectory replay...")
        print("Press Ctrl+C to stop")
        
        try:
            start_time = time.time()
            
            for i in range(len(velocities)):
                # Calculate timing and duration
                target_time = timestamps[i]
                elapsed_time = time.time() - start_time
                
                # Calculate duration until next point
                if i < len(velocities) - 1:
                    duration_ms = int((timestamps[i+1] - timestamps[i]) * 1000)
                else:
                    duration_ms = 100  # Default for last point
                
                # Wait for proper timing
                sleep_time = target_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Send velocity command with proper duration
                vel_cmd = velocities[i]
                success = self.client.move_joint_velocity(vel_cmd, duration_ms=duration_ms)
                
                if not success:
                    print(f"Error sending velocity command at step {i}")
                    break
                    
                # Print progress every 10 steps
                if i % 10 == 0:
                    print(f"Step {i}/{len(velocities)}, Velocity: {vel_cmd}")
                    
        except KeyboardInterrupt:
            print("\nTrajectory stopped by user")
        except Exception as e:
            print(f"Error during replay: {e}")
        finally:
            # Stop robot motion
            self.client.stop()
            print("Robot stopped")
            
        return True


def main():
    parser = argparse.ArgumentParser(description='Replay recorded robot trajectory')
    parser.add_argument('csv_file', help='Path to CSV file containing trajectory')
    parser.add_argument('--server_ip', default='192.168.18.1', 
                       help='IP address of franky server')
    parser.add_argument('--velocity_scale', type=float, default=1.0,
                       help='Scale factor for velocities (0-1, default: 1.0)')
    parser.add_argument('--use_recorded_velocities', default=True,
                    #    action='store_true',
                       help='Use recorded velocities instead of calculated')
    parser.add_argument('--dynamics_factor', type=float, default=1.0,
                       help='Robot dynamics factor (0-1, default: 1.0)')
    parser.add_argument('--smoothing', type=float, default=0.7,
                       help='Velocity smoothing factor (0-1, default: 0.7)')
    
    args = parser.parse_args()
    
    # Create replayer and connect
    replayer = TrajectoryReplayer(args.server_ip)
    
    if not replayer.connect(args.dynamics_factor):
        print("Failed to connect to robot")
        return
        
    try:
        # Replay trajectory
        replayer.replay_trajectory(
            args.csv_file, 
            args.velocity_scale,
            args.use_recorded_velocities,
            args.smoothing
        )
    finally:
        replayer.disconnect()


if __name__ == "__main__":
    main()