#!/usr/bin/env python3
"""
Direct deployment script for combo robot that bypasses draccus CLI parsing.
"""

import sys
from pathlib import Path
import time
import torch
import numpy as np
import rerun as rr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig

def main():
    print("=== Combo Robot Policy Deployment ===")
    
    # Tunable parameters
    ACTION_SCALE = 1.0  # Scale actions (< 1.0 for safer/slower movements)
    SMOOTHING_ALPHA = 0.3  # 0.0 = max smoothing, 1.0 = no smoothing
    QUERY_FREQUENCY = 3  # Query new action chunk every N steps (1-8)
    GRASP_ASSIST = True  # Help with grasping based on arm position
    GRASP_Z_THRESHOLD = 0.2  # Z height below which to attempt grasp (meters)
    
    # Create robot configuration
    arm_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.9],
        cameras={}
    )
    
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        control_frequency=30.0,
        max_torque=250.0,
        cameras={}
    )
    
    cameras = {
        "tpv": OpenCVCameraConfig(
            index_or_path="/dev/video11",
            fps=30,
            width=320,
            height=240,
            color_mode=ColorMode.RGB
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video6",
            fps=30,
            width=424,
            height=240,
            color_mode=ColorMode.RGB
        )
    }
    
    robot_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        cameras=cameras,
        synchronize_actions=True,
        action_timeout=0.2,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create robot
    robot = FrankaFERXHand(robot_config)
    
    # Load policy
    policy_path = Path("outputs/act_vision_orange_cube")
    print(f"Loading policy from {policy_path}")
    
    # Load the checkpoint
    checkpoint_path = policy_path / "checkpoint_10000.pth"
    if not checkpoint_path.exists():
        # Try other checkpoints
        checkpoints = sorted(list(policy_path.glob("checkpoint_*.pth")))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
            print(f"Using checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found!")
            return 1
    
    # Load checkpoint with weights_only=False (we trust our own trained model)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    config = checkpoint["config"]
    
    # Try to load actual stats from dataset if available
    stats_path = policy_path / "stats.json"
    if stats_path.exists():
        import json
        with open(stats_path) as f:
            raw_stats = json.load(f)
        # Convert to torch tensors
        stats = {}
        for key in ["observation.state", "observation.environment_state", "observation.images.tpv", "observation.images.wrist", "action"]:
            if key in raw_stats:
                stats[key] = {
                    "mean": torch.tensor(raw_stats[key]["mean"]),
                    "std": torch.tensor(raw_stats[key]["std"])
                }
        print("Loaded dataset statistics from stats.json")
    else:
        # Create policy with dummy stats (will be normalized by the model)
        print("Warning: No stats.json found, using dummy normalization")
        stats = {
            "observation.state": {
                "mean": torch.zeros(42),
                "std": torch.ones(42)
            },
            "observation.environment_state": {
                "mean": torch.zeros(42),
                "std": torch.ones(42)
            },
            "observation.images.tpv": {
                "mean": torch.zeros(3, 1, 1),
                "std": torch.ones(3, 1, 1)
            },
            "observation.images.wrist": {
                "mean": torch.zeros(3, 1, 1),
                "std": torch.ones(3, 1, 1)
            },
            "action": {
                "mean": torch.zeros(19),
                "std": torch.ones(19)
            }
        }
    
    policy = ACTPolicy(config, dataset_stats=stats)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    policy.to("cuda")
    
    print("Policy loaded successfully")
    
    # Connect robot
    print("Connecting to robot...")
    robot.connect(calibrate=False)
    
    if not robot.is_connected:
        print("Failed to connect to robot!")
        return 1
    
    print("Robot connected successfully")
    
    # Initialize Rerun
    rr.init("combo_robot_deployment", spawn=True)
    
    # Home robot
    print("Homing robot...")
    robot.reset_to_home()
    time.sleep(2)
    
    # Main control loop
    print("\n=== Starting control loop ===")
    print("Press Ctrl+C to stop")
    
    fps = 30
    dt = 1.0 / fps
    frame_idx = 0
    
    # Action smoothing and chunking
    action_smoothing_alpha = SMOOTHING_ALPHA
    prev_action = None
    
    # Action chunking - ACT outputs 8 actions at once
    action_chunk = None
    chunk_idx = 0
    chunk_size = 8
    query_frequency = QUERY_FREQUENCY
    
    try:
        while True:
            start_time = time.perf_counter()
            
            # Set rerun time
            rr.set_time_sequence("frame", frame_idx)
            rr.set_time_seconds("time", time.time())
            
            # Get observation
            obs = robot.get_observation()
            
            # Prepare observation for policy
            # Combine arm and hand states into environment_state
            env_state = []
            
            # Add arm joint positions (7)
            for i in range(7):
                env_state.append(obs[f"arm_joint_{i}.pos"])
            
            # Add arm joint velocities (7)
            for i in range(7):
                env_state.append(obs[f"arm_joint_{i}.vel"])
            
            # Add ee_pose (16)
            for i in range(16):
                env_state.append(obs[f"arm_ee_pose.{i:02d}"])
            
            # Add hand joint positions (12)
            for i in range(12):
                env_state.append(obs[f"hand_joint_{i}.pos"])
            
            env_state = np.array(env_state, dtype=np.float32)
            
            # Query policy for new action chunk when needed
            if action_chunk is None or chunk_idx % query_frequency == 0:
                # Create batch for policy - ACT needs both keys
                env_state_tensor = torch.FloatTensor(env_state).unsqueeze(0).cuda()
                
                # Process camera images
                tpv_image = torch.FloatTensor(obs["tpv"]).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0  # Normalize to [0,1]
                wrist_image = torch.FloatTensor(obs["wrist"]).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0  # Normalize to [0,1]
                
                # Resize images to match expected input size (64x64)
                import torch.nn.functional as F
                tpv_image = F.interpolate(tpv_image, size=(64, 64), mode='bilinear', align_corners=False)
                wrist_image = F.interpolate(wrist_image, size=(64, 64), mode='bilinear', align_corners=False)
                
                batch = {
                    "observation.state": env_state_tensor,  # Needed for device check
                    "observation.environment_state": env_state_tensor,  # Actual input
                    "observation.images.tpv": tpv_image,
                    "observation.images.wrist": wrist_image,
                    "action_is_pad": torch.zeros(1, chunk_size, dtype=torch.bool).cuda()
                }
                
                # Get action chunk from policy
                with torch.no_grad():
                    # Get full action chunk (8 steps)
                    action_chunk_pred = policy.predict_action_chunk(batch)
                    action_chunk = action_chunk_pred[0].cpu().numpy()  # Shape: (8, 19)
                chunk_idx = 0
            
            # Use current action from chunk
            action = action_chunk[min(chunk_idx, chunk_size-1)]  # Shape: (19,)
            
            # Store raw action for debugging
            raw_action = action.copy()
            
            # Apply exponential smoothing
            if prev_action is not None:
                action = action_smoothing_alpha * action + (1 - action_smoothing_alpha) * prev_action
            prev_action = action.copy()
            
            # Apply action scaling for safety
            action = action * ACTION_SCALE
            
            chunk_idx += 1
            
            # Split into arm and hand actions
            action_dict = {}
            
            # Arm actions (first 7)
            for i in range(7):
                action_dict[f"arm_joint_{i}.pos"] = float(action[i])
            
            # Hand actions (next 12)
            for i in range(12):
                action_dict[f"hand_joint_{i}.pos"] = float(action[7 + i])
            
            # Debug: Print hand actions periodically to see if they're changing
            if frame_idx % 30 == 0:  # Every second
                raw_hand = [raw_action[7+i] for i in range(12)]
                smooth_hand = [action[7+i] for i in range(12)]
                print(f"RAW hand actions: mean={np.mean(raw_hand):.3f}, std={np.std(raw_hand):.3f}, range=[{min(raw_hand):.3f}, {max(raw_hand):.3f}]")
                print(f"SMOOTH hand actions: mean={np.mean(smooth_hand):.3f}, std={np.std(smooth_hand):.3f}")
                # Print specific joints to see if they're trying to move
                for i in [0, 4, 6, 8]:
                    print(f"  Joint {i}: current={obs[f'hand_joint_{i}.pos']:.3f}, raw_target={raw_action[7+i]:.3f}, smooth_target={action[7+i]:.3f}")
            
            # Send action to robot
            robot.send_action(action_dict)
            
            # Log to Rerun
            # Log arm joint positions
            for i in range(7):
                rr.log(f"robot/arm/joint_{i}/position", rr.Scalar(obs[f"arm_joint_{i}.pos"]))
                rr.log(f"robot/arm/joint_{i}/velocity", rr.Scalar(obs[f"arm_joint_{i}.vel"]))
                rr.log(f"robot/arm/joint_{i}/action", rr.Scalar(action_dict[f"arm_joint_{i}.pos"]))
            
            # Log hand joint positions
            for i in range(12):
                rr.log(f"robot/hand/joint_{i}/position", rr.Scalar(obs[f"hand_joint_{i}.pos"]))
                rr.log(f"robot/hand/joint_{i}/action", rr.Scalar(action_dict[f"hand_joint_{i}.pos"]))
            
            # Log camera images if available
            if "tpv" in obs:
                rr.log("cameras/tpv", rr.Image(obs["tpv"]))
            if "wrist" in obs:
                rr.log("cameras/wrist", rr.Image(obs["wrist"]))
            
            # Log end-effector pose as 4x4 matrix
            ee_pose = np.array([obs[f"arm_ee_pose.{i:02d}"] for i in range(16)]).reshape(4, 4)
            rr.log("robot/arm/ee_pose", rr.Transform3D(mat3x3=ee_pose[:3, :3], translation=ee_pose[:3, 3]))
            
            # Maintain loop timing
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
            # Debug print every second
            if frame_idx % fps == 0:
                print(f"Control loop running... (frame {frame_idx}, loop time: {elapsed*1000:.1f}ms)")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n=== Stopping control loop ===")
    except Exception as e:
        print(f"Error in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Disconnecting robot...")
        if robot.is_connected:
            robot.stop()
            robot.disconnect()
        print("Done!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())