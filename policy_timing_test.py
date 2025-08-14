#!/usr/bin/env python3

import time
import numpy as np

print("Quick deployment timing test...")
print("This will run the actual policy deployment and measure real-world timing")

# Run the deployment and time it
start_time = time.time()

# Import subprocess to run the deployment
import subprocess
import sys

try:
    # Run deployment for 10 seconds and capture output
    result = subprocess.run([
        sys.executable, "-m", "lerobot.record",
        "--robot.type=franka_fer",
        "--robot.server_ip=192.168.18.1", 
        "--robot.server_port=5000",
        "--policy.path=./outputs/diffusion_policy_franka_wave/checkpoints/050000/pretrained_model",
        "--dataset.fps=30",
        "--dataset.episode_time_s=10",
        "--dataset.num_episodes=1", 
        "--dataset.root=./datasets/timing_test",
        "--dataset.repo_id=local/eval_timing_test",
        "--dataset.single_task=Timing test",
        "--dataset.push_to_hub=false"
    ], capture_output=True, text=True, timeout=30)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nDeployment Results:")
    print(f"Total time: {total_time:.1f}s")
    print(f"Target time: 10s + setup/teardown")
    
    # Look for timing info in output
    output_lines = result.stdout.split('\n') + result.stderr.split('\n')
    
    # Count log messages to estimate control frequency
    step_logs = [line for line in output_lines if 'step:' in line.lower()]
    if step_logs:
        print(f"Found {len(step_logs)} step logs")
        
    # Look for inference-related timing
    for line in output_lines:
        if 'inference' in line.lower() or 'policy' in line.lower():
            print(f"Policy info: {line}")
            
    # Check if successful
    if result.returncode == 0:
        print("✅ Deployment completed successfully")
    else:
        print("❌ Deployment failed")
        print("STDERR:", result.stderr[-500:])  # Last 500 chars
        
except subprocess.TimeoutExpired:
    print("⚠️ Deployment timed out after 30s")
except Exception as e:
    print(f"❌ Error running deployment: {e}")

print("\nFor detailed timing analysis:")
print("1. Check the robot movement smoothness visually")  
print("2. Count the number of pause/movement cycles")
print("3. Time between action chunks should be ~267ms (8 steps at 30Hz)")
print("4. Inference pauses should be visible as brief stops")

# Basic calculation
expected_steps = 10 * 30  # 10s at 30Hz  
expected_inferences = expected_steps // 8  # Every 8 steps
expected_inference_time = expected_inferences * 0.2  # Assume 200ms per inference

print(f"\nExpected Analysis:")
print(f"- Total control steps: {expected_steps}")  
print(f"- Number of inferences: {expected_inferences}")
print(f"- Time spent on inference: ~{expected_inference_time:.1f}s")
print(f"- Time spent moving: ~{10 - expected_inference_time:.1f}s") 
print(f"- Duty cycle: {((10 - expected_inference_time) / 10) * 100:.1f}%")