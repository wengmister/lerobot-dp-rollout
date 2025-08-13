#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("franka_fer")
@dataclass
class FrankaFERConfig(RobotConfig):
    # Server IP and port for Franka Robot Server
    server_ip: str = "192.168.18.1"
    server_port: int = 5000

    # Home position for robot reset
    home_position: list[float] = field(default_factory=lambda: [0, -0.785, 0, -2.356, 0, 1.571, -0.9]) # modified home pose for xhand

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    max_relative_target: float | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
