#!/usr/bin/env python3

import torch
import time
import threading
import queue
import numpy as np
from copy import copy
from contextlib import nullcontext

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.datasets.utils import build_dataset_frame

class FixedAsyncPolicyController:
    def __init__(self, policy_path, dataset_path, robot_config):
        # Load policy exactly like the working deployment
        self.dataset = LeRobotDataset(dataset_path, root='./datasets/franka_wave')
        config = PreTrainedConfig.from_pretrained(policy_path)
        self.policy = make_policy(config, ds_meta=self.dataset.meta)
        
        self.robot = FrankaFER(robot_config)
        
        # Threading components
        self.action_queue = queue.Queue(maxsize=50)  # Buffer many actions
        self.observation_queue = queue.Queue(maxsize=1)
        self.stop_inference = threading.Event()
        
        self.device = next(self.policy.parameters()).device
        
    def inference_worker(self):
        """Background thread using the EXACT same logic as predict_action"""
        while not self.stop_inference.is_set():
            try:
                # Get observation (blocking with timeout)
                observation_frame = self.observation_queue.get(timeout=0.1)
                
                start_time = time.perf_counter()
                
                # Use the exact same predict_action logic
                observation = copy(observation_frame)
                with torch.inference_mode():
                    # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
                    for name in observation:
                        observation[name] = torch.from_numpy(observation[name])
                        if "image" in name:
                            observation[name] = observation[name].type(torch.float32) / 255
                            observation[name] = observation[name].permute(2, 0, 1).contiguous()
                        observation[name] = observation[name].unsqueeze(0)
                        observation[name] = observation[name].to(self.device)
                    
                    observation["task"] = "Policy deployment test"
                    observation["robot_type"] = "franka_fer"
                    
                    # Get action from policy
                    action = self.policy.select_action(observation)
                    action = action.squeeze(0)  # Remove batch dimension
                    
                inference_time = (time.perf_counter() - start_time) * 1000
                
                # Convert to numpy and put in queue
                action_np = action.cpu().numpy()
                self.action_queue.put((action_np, inference_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
                break
    
    def run_control_loop(self, duration_s=30, fps=30):
        """Run control loop using LeRobot's exact approach"""
        print(f"Starting optimized parallel control for {duration_s}s at {fps} Hz")
        
        # Connect robot
        self.robot.connect()
        self.robot.reset_to_home()
        
        # Start inference thread
        inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        inference_thread.start()
        
        # Control loop
        total_steps = int(duration_s * fps)
        inference_count = 0
        action_times = []
        
        for step in range(total_steps):
            start_time = time.perf_counter()
            
            # Get robot observation
            observation = self.robot.get_observation()
            
            # Build observation frame using LeRobot's method
            observation_frame = build_dataset_frame(
                self.dataset.features, observation, prefix="observation"
            )
            
            # Submit for inference (non-blocking)
            try:
                self.observation_queue.put_nowait(observation_frame)
            except queue.Full:
                pass  # Skip if queue full
            
            # Get action (blocking with short timeout)
            try:
                action_np, inf_time = self.action_queue.get(timeout=0.02)
                action_times.append(inf_time)
                
                # Convert to robot format
                action_dict = {}
                joint_names = ['joint_0.pos', 'joint_1.pos', 'joint_2.pos', 
                              'joint_3.pos', 'joint_4.pos', 'joint_5.pos', 'joint_6.pos']
                for i, joint_name in enumerate(joint_names):
                    action_dict[joint_name] = float(action_np[i])
                
                # Send action
                self.robot.send_action(action_dict)
                inference_count += 1
                
            except queue.Empty:
                # No action available - this means truly parallel execution!
                pass
            
            # Maintain timing
            dt = time.perf_counter() - start_time
            busy_wait(1/fps - dt)
            
            if step % (fps * 5) == 0:  # Every 5 seconds
                queue_size = self.action_queue.qsize()
                avg_inf_time = np.mean(action_times[-10:]) if action_times else 0
                print(f"Step {step}/{total_steps}, queue: {queue_size}, avg_inf: {avg_inf_time:.1f}ms")
        
        # Cleanup
        self.stop_inference.set()
        inference_thread.join(timeout=1.0)
        self.robot.disconnect()
        
        print(f"\\nParallel control completed!")
        print(f"Total inferences: {inference_count}")
        print(f"Inference rate: {inference_count/duration_s:.1f} Hz")
        if action_times:
            print(f"Average inference time: {np.mean(action_times):.1f}ms")
            print(f"Max inference time: {np.max(action_times):.1f}ms")

def main():
    # Configuration
    policy_path = "./outputs/diffusion_policy_franka_wave/checkpoints/050000/pretrained_model"
    dataset_path = "local_dataset_franka_wave"
    
    robot_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000
    )
    
    # Run parallel control
    controller = FixedAsyncPolicyController(policy_path, dataset_path, robot_config)
    controller.run_control_loop(duration_s=15, fps=30)  # Shorter test

if __name__ == "__main__":
    main()