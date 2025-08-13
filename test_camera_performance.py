#!/usr/bin/env python3
"""
Test different camera configurations for performance optimization.
"""

import time
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from src.lerobot.robots.franka_fer.franka_fer import FrankaFER
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode

def test_parallel_cameras():
    """Test parallel camera reading performance"""
    print("=== Testing Parallel Camera Reading ===")
    
    config = FrankaFERConfig()
    robot = FrankaFER(config)
    
    try:
        # Connect cameras
        for cam_name, cam in robot.cameras.items():
            cam.connect()
        
        def read_single_camera(cam_name_cam_tuple):
            cam_name, cam = cam_name_cam_tuple
            start = time.perf_counter()
            frame = cam.async_read()
            duration = time.perf_counter() - start
            return cam_name, frame, duration
        
        # Test sequential reading (current approach)
        print("Sequential reading:")
        start_total = time.perf_counter()
        for i in range(5):
            start_iter = time.perf_counter()
            for cam_name, cam in robot.cameras.items():
                frame = cam.async_read()
            iter_time = time.perf_counter() - start_iter
            print(f"  Iteration {i+1}: {iter_time*1000:.1f}ms")
        seq_total = time.perf_counter() - start_total
        print(f"Sequential total: {seq_total:.3f}s ({seq_total/5:.3f}s per obs)")
        
        # Test parallel reading
        print("\nParallel reading:")
        start_total = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            for i in range(5):
                start_iter = time.perf_counter()
                futures = [executor.submit(read_single_camera, (name, cam)) 
                          for name, cam in robot.cameras.items()]
                results = [future.result() for future in futures]
                iter_time = time.perf_counter() - start_iter
                print(f"  Iteration {i+1}: {iter_time*1000:.1f}ms")
        par_total = time.perf_counter() - start_total
        print(f"Parallel total: {par_total:.3f}s ({par_total/5:.3f}s per obs)")
        
        speedup = seq_total / par_total
        print(f"Speedup: {speedup:.2f}x")
        
        # Disconnect cameras
        for cam in robot.cameras.values():
            cam.disconnect()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_lower_resolution():
    """Test with lower resolution cameras"""
    print("\n=== Testing Lower Resolution (320x240) ===")
    
    # Create config with smaller resolution
    from dataclasses import dataclass, field
    
    @dataclass
    class FastFrankaConfig(FrankaFERConfig):
        cameras: dict = field(default_factory=lambda: {
            "overhead": RealSenseCameraConfig(
                serial_number_or_name="233522075872",
                fps=30,
                width=320,  # Half resolution
                height=240,
                color_mode=ColorMode.RGB
            ),
            "third_person": RealSenseCameraConfig(
                serial_number_or_name="938422076779", 
                fps=30,
                width=320,  # Half resolution
                height=240,
                color_mode=ColorMode.RGB
            )
        })
    
    config = FastFrankaConfig()
    robot = FrankaFER(config)
    
    try:
        # Connect cameras
        for cam_name, cam in robot.cameras.items():
            cam.connect()
        
        # Test reading speed
        print("Low resolution reading:")
        start_total = time.perf_counter()
        for i in range(5):
            start_iter = time.perf_counter()
            for cam_name, cam in robot.cameras.items():
                frame = cam.async_read()
                print(f"    {cam_name}: {frame.shape}")
            iter_time = time.perf_counter() - start_iter
            print(f"  Iteration {i+1}: {iter_time*1000:.1f}ms")
        total_time = time.perf_counter() - start_total
        print(f"Low res total: {total_time:.3f}s ({total_time/5:.3f}s per obs)")
        
        # Disconnect cameras
        for cam in robot.cameras.values():
            cam.disconnect()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_robot_observation_speed():
    """Test full robot observation with optimizations"""
    print("\n=== Testing Full Robot Observations ===")
    
    config = FrankaFERConfig()
    robot = FrankaFER(config)
    
    try:
        robot.connect()
        
        # Warm up
        robot.get_observation()
        
        print("Full robot observations:")
        start_total = time.perf_counter()
        for i in range(5):
            start_iter = time.perf_counter()
            obs = robot.get_observation()
            iter_time = time.perf_counter() - start_iter
            
            # Count data
            joint_count = sum(1 for k in obs.keys() if k.startswith('joint_'))
            camera_count = sum(1 for k, v in obs.items() if isinstance(v, np.ndarray) and len(v.shape) == 3)
            
            print(f"  Obs {i+1}: {iter_time*1000:.1f}ms ({joint_count} joints, {camera_count} cameras)")
        
        total_time = time.perf_counter() - start_total
        print(f"Robot obs total: {total_time:.3f}s ({total_time/5:.3f}s per obs)")
        hz = 5 / total_time
        print(f"Effective rate: {hz:.1f} Hz")
        
        robot.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Camera Performance Optimization Tests")
    print("=" * 50)
    
    test_parallel_cameras()
    test_lower_resolution() 
    test_robot_observation_speed()
    
    print("\n" + "=" * 50)
    print("Performance optimization recommendations:")
    print("1. Use parallel camera reading")
    print("2. Consider lower resolution (320x240)")
    print("3. For real-time control, target 10-20 Hz is often sufficient")
    print("4. Diffusion policies typically run at 10 Hz")

if __name__ == "__main__":
    main()