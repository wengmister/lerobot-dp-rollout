#!/usr/bin/env python3
"""
Example script demonstrating XHand teleoperation using VR hand tracking.

This script shows how to:
1. Initialize and connect to XHand robot
2. Set up VR hand tracking teleoperator  
3. Run real-time teleoperation loop
4. Handle graceful shutdown

Usage:
    python examples/xhand_teleoperation.py --hand-type right --control-freq 30
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.robots.xhand.xhand import XHand
from lerobot.robots.xhand.xhand_config import XHandConfig
from lerobot.robots.xhand.teleoperator import XHandTeleoperator
from dex_retargeting.constants import RobotName, RetargetingType, HandType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="XHand VR Teleoperation")
    parser.add_argument(
        "--control-freq",
        type=float,
        default=30.0,
        help="Control frequency in Hz (default: 30.0)"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="Position smoothing factor 0-1 (default: 0.3)"
    )
    parser.add_argument(
        "--protocol",
        choices=["RS485", "EtherCAT"],
        default="RS485",
        help="Communication protocol (default: RS485)"
    )
    parser.add_argument(
        "--serial-port",
        default="/dev/ttyUSB0",
        help="Serial port for RS485 (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub mode (no hardware required)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting XHand teleoperation for right hand")
    
    try:
        # Initialize XHand robot
        config = XHandConfig(
            protocol=args.protocol,
            serial_port=args.serial_port,
        )
        robot = XHand(config)
        
        # Connect to robot
        logger.info("Connecting to XHand robot...")
        if args.stub:
            # Force stub mode for testing
            robot._connect_stub()
        else:
            robot.connect(calibrate=True)
        
        # Initialize teleoperator
        logger.info("Setting up VR teleoperator...")
        teleop = XHandTeleoperator(
            xhand_robot=robot,
            robot_name=RobotName.xhand,
            retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.right,
        )
        
        # Configure teleoperator settings
        teleop.set_control_frequency(args.control_freq)
        teleop.set_smoothing(args.smoothing)
        
        # Reset robot to home position
        logger.info("Resetting robot to home position...")
        teleop.reset_robot_to_home()
        
        # Start teleoperation
        logger.info("Starting teleoperation. Press Ctrl+C to stop.")
        logger.info("Move your hand in front of the camera to control the robot.")
        
        teleop.start_teleoperation()
        
    except KeyboardInterrupt:
        logger.info("Teleoperation stopped by user")
    except Exception as e:
        logger.error(f"Error during teleoperation: {e}")
        return 1
    finally:
        # Clean shutdown
        try:
            if 'teleop' in locals():
                teleop.emergency_stop()
            if 'robot' in locals() and robot.is_connected:
                logger.info("Disconnecting robot...")
                robot.disconnect()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
    
    logger.info("XHand teleoperation completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())