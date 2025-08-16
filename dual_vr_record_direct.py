#!/usr/bin/env python3
"""
Direct Python script for recording data with Franka FER + XHand using dual VR teleoperator.

This script bypasses the CLI argument parsing issues and directly instantiates 
the robot and teleoperator configurations.
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.franka_fer_xhand.franka_fer_xhand import FrankaFERXHand
from lerobot.robots.franka_fer_xhand.franka_fer_xhand_config import FrankaFERXHandConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.teleoperators.franka_fer_xhand_vr.franka_fer_xhand_vr_teleoperator import FrankaFERXHandVRTeleoperator
from lerobot.teleoperators.franka_fer_xhand_vr.config_franka_fer_xhand_vr import FrankaFERXHandVRTeleoperatorConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode

# Default recording parameters
DEFAULT_NUM_EPISODES = 2
DEFAULT_FPS = 30
DEFAULT_EPISODE_TIME_SEC = 30
DEFAULT_TASK_DESCRIPTION = "Teleoperate dual arm-hand system to pick and place objects"
DEFAULT_DATASET_NAME = "test_pick_and_place"

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Record data with dual robot VR teleoperation")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME,
                       help=f"Name of the dataset (default: {DEFAULT_DATASET_NAME})")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES,
                       help=f"Number of episodes to record (default: {DEFAULT_NUM_EPISODES})")
    parser.add_argument("--episode-time", type=float, default=DEFAULT_EPISODE_TIME_SEC,
                       help=f"Duration of each episode in seconds (default: {DEFAULT_EPISODE_TIME_SEC})")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_DESCRIPTION,
                       help=f"Task description (default: '{DEFAULT_TASK_DESCRIPTION}')")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                       help=f"Recording frame rate (default: {DEFAULT_FPS})")
    parser.add_argument("--resume", action="store_true",
                       help="Resume recording from existing dataset")
    return parser.parse_args()

def get_existing_episode_count(dataset_path):
    """Check how many episodes already exist in the dataset."""
    try:
        if Path(dataset_path).exists():
            # Try to load existing dataset to count episodes
            dataset = LeRobotDataset(dataset_path)
            return len(dataset.episode_indices)
        return 0
    except Exception as e:
        logger.warning(f"Could not check existing episodes: {e}")
        return 0

def main():
    """Main function to run the recording test."""
    args = parse_args()
    
    logger.info("Setting up dual VR recording test...")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Episodes: {args.num_episodes}")
    logger.info(f"Episode time: {args.episode_time}s")
    logger.info(f"Task: {args.task}")
    
    # Create arm configuration
    arm_config = FrankaFERConfig(
        server_ip="192.168.18.1",
        server_port=5000,
        home_position=[0, -0.785, 0, -2.356, 0, 1.571, -0.9],  # Modified for XHand
        max_relative_target=None,
        cameras={}  # Use composite robot cameras instead
    )
    
    # Create hand configuration
    hand_config = XHandConfig(
        protocol="RS485",
        serial_port="/dev/ttyUSB0",
        baud_rate=3000000,
        hand_id=0,
        control_frequency=200.0,
        max_torque=250.0,
        cameras={}  # Use composite robot cameras instead
    )
    
    # Create camera configurations
    cameras = {
        "tpv": OpenCVCameraConfig(
            index_or_path="/dev/video18",
            fps=30,
            width=320,
            height=240,
            color_mode=ColorMode.RGB
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video12",
            fps=30,
            width=424,
            height=240,
            color_mode=ColorMode.RGB
        )
    }
    
    # Note: will need to manufally change v4l2 camera config
    
    # Create composite robot configuration
    robot_config = FrankaFERXHandConfig(
        arm_config=arm_config,
        hand_config=hand_config,
        cameras=cameras,
        synchronize_actions=True,
        action_timeout=0.1,
        check_arm_hand_collision=True,
        emergency_stop_both=True
    )
    
    # Create the composite robot
    robot = FrankaFERXHand(robot_config)
    
    # Create teleoperator configuration
    teleop_config = FrankaFERXHandVRTeleoperatorConfig(
        vr_tcp_port=8000,
        setup_adb=True,
        vr_verbose=False,
        # Arm settings
        arm_smoothing_factor=0.1,
        manipulability_weight=1.0,
        neutral_distance_weight=2.0,
        current_distance_weight=2.0,
        arm_joint_weights=[3.0, 3.0, 1.5, 1.5, 1.0, 1.0, 1.0],
        q7_min=-2.8973,
        q7_max=2.8973,
        # Hand settings
        hand_robot_name="xhand_right",
        hand_retargeting_type="dexpilot",
        hand_type="right",
        hand_control_frequency=30.0,
        hand_smoothing_alpha=0.3
    )
    
    # Create the teleoperator
    teleop = FrankaFERXHandVRTeleoperator(teleop_config)
    
    # Configure dataset features
    logger.info("Setting up dataset...")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    logger.info(f"Robot action features: {list(robot.action_features.keys())}")
    logger.info(f"Robot observation features: {list(robot.observation_features.keys())}")
    
    # Set dataset path to save under lerobot/datasets/
    dataset_path = Path("/home/zkweng/lerobot") / args.dataset_name
    existing_episodes = 0
    
    if args.resume:
        # Check for existing dataset in lerobot/datasets/
        existing_episodes = get_existing_episode_count(dataset_path)
        if existing_episodes > 0:
            logger.info(f"Found {existing_episodes} existing episodes. Resuming from episode {existing_episodes + 1}")
        else:
            logger.info("No existing episodes found. Starting from episode 1")
    
    # Create or load the dataset
    if existing_episodes > 0:
        dataset = LeRobotDataset(dataset_path)
    else:
        dataset = LeRobotDataset.create(
            repo_id=str(dataset_path),
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,  # Enable videos for proper storage
            image_writer_threads=4,
        )
    
    try:
        # Connect robot first
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        
        # Initialize visualization early to avoid port conflicts with VR
        _init_rerun(session_name="dual_vr_record_test")
        
        # Initialize keyboard listener for control events
        listener, events = init_keyboard_listener()
        
        # Verify robot connection
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")
        
        # Note: Teleoperator will be connected in the episode loop
        
        logger.info("Starting recording session...")
        total_episodes_to_record = args.num_episodes
        log_say(f"Recording {total_episodes_to_record} episodes (starting from episode {existing_episodes + 1})")
        log_say("Press 's' to stop recording, 'r' to re-record current episode")
        
        # Recording loop
        recorded_episodes = existing_episodes
        while recorded_episodes < total_episodes_to_record and not events.get("stop_recording", False):
            current_episode = recorded_episodes + 1
            
            # Manual confirmation and homing before each episode
            print(f"\n=== EPISODE {current_episode} PREPARATION ===")
            
            # Check teleoperation status (should be disconnected from previous episode)
            print(f"1. Teleoperation status: {'connected' if teleop.is_connected else 'disconnected'}")
            if teleop.is_connected:
                print("   - Disconnecting teleoperation...")
                teleop.disconnect()
            
            print("2. Homing robot and hand...")
            # Home the robot arm
            if hasattr(robot.arm, 'reset_to_home'):
                print("   - Sending arm to home position...")
                success = robot.arm.reset_to_home()
                if success:
                    print("   - Arm homing initiated successfully")
                else:
                    print("   - WARNING: Arm homing failed!")
                print("   - Waiting for arm to reach home position...")
                time.sleep(2.0)  # Give arm time to reach home
            else:
                print("   - WARNING: Arm reset_to_home not available")
            
            # Home the hand
            if hasattr(robot.hand, 'reset_to_home'):
                print("   - Sending hand to home position...")
                success = robot.hand.reset_to_home()
                if success:
                    print("   - Hand homing initiated successfully")
                else:
                    print("   - WARNING: Hand homing failed!")
                time.sleep(1.0)  # Give hand time to reach home
            else:
                print("   - WARNING: Hand reset_to_home not available")
            
            print("   - Waiting for robot to stabilize at home position...")
            time.sleep(1.0)  # Extra safety pause
            
            print("3. Ready for episode")
            print("="*60)
            print(f">>> Press ENTER to start recording episode {current_episode} <<<")
            print(f">>> Or press Ctrl+C to stop recording <<<")
            print("="*60)
            import sys
            sys.stdout.flush()  # Ensure prompt is displayed
            input("Waiting for your confirmation: ")
            
            print("4. Connecting/reconnecting teleoperation...")
            if not teleop.is_connected:
                teleop.connect(calibrate=False)
                teleop.set_robot(robot)
                print("   - Waiting for VR stream to stabilize...")
                time.sleep(1.0)  # Give VR stream time to stabilize
                
                # Reset VR initial pose to current robot position
                print("   - Resetting VR reference frame to current robot position...")
                if hasattr(teleop, 'reset_initial_pose'):
                    success = teleop.reset_initial_pose()
                    if success:
                        print("   - VR reference frame reset successfully")
                    else:
                        print("   - Warning: VR reference frame reset may have failed")
                else:
                    print("   - Warning: VR reference frame reset not available")
                
                print("   - VR teleoperation ready. Move headset to comfortable position before starting.")
                time.sleep(1.0)  # Give operator time to position themselves
            
            log_say(f"Recording episode {current_episode} of {total_episodes_to_record}")
            
            # Record the episode
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                dataset=dataset,
                teleop=teleop,
                control_time_s=args.episode_time,
                single_task=args.task,
                display_data=True,
            )
            
            # Handle re-recording if requested
            if events.get("rerecord_episode", False):
                log_say("Re-recording episode - will reset and restart")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                
                # Disconnect teleop to prepare for reset
                if teleop.is_connected:
                    teleop.disconnect()
                    print("Teleoperator disconnected for re-record")
                
                # Don't increment episode count, just continue to redo this episode
                continue
            
            # Save the episode
            dataset.save_episode()
            recorded_episodes += 1
            logger.info(f"Episode {recorded_episodes} saved successfully")
            
            # Disconnect teleoperator after each episode for proper cleanup
            if teleop.is_connected:
                teleop.disconnect()
                print(f"\n{'='*60}")
                print(f"Episode {recorded_episodes} COMPLETE - teleoperator disconnected")
                print(f"{'='*60}\n")
            
            # Final completion message
            if recorded_episodes >= total_episodes_to_record:
                print(f"\n=== ALL {total_episodes_to_record} EPISODES COMPLETED! ===")
            
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    except Exception as e:
        logger.error(f"Error during recording: {e}")
        raise
    finally:
        # Clean up
        logger.info("Cleaning up...")
        
        # Save dataset locally
        logger.info(f"Dataset saved locally at: {dataset.root}")
        
        # Disconnect devices
        try:
            if teleop.is_connected:
                teleop.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting teleoperator: {e}")
            
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting robot: {e}")
        
        # Stop keyboard listener
        try:
            listener.stop()
        except Exception as e:
            logger.error(f"Error stopping keyboard listener: {e}")
        
        logger.info("Recording session complete!")
        try:
            logger.info(f"Recorded {recorded_episodes} episodes")
        except NameError:
            logger.info("Recording session ended before completion")
        logger.info(f"Dataset saved to: {dataset.root}")


if __name__ == "__main__":
    main()