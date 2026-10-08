"""
object_detection - Visual target detection package for FruitFly-Drone.

Supports detecting white targets (e.g. sugar cube) from both MuJoCo simulation
and real-world USB/CSI cameras.
"""

from .target import (
    TargetConfig,
    get_target_position,
    set_target_position,
    get_relative_target_vector,
)
from .camera_interface import (
    CameraInterface,
    MuJoCoCamera,
    PhysicalCamera,
    MockCamera,
)
from .detector import (
    TargetDetection,
    WhiteSugarDetector,
)

__all__ = [
    "TargetConfig",
    "get_target_position",
    "set_target_position",
    "get_relative_target_vector",
    "CameraInterface",
    "MuJoCoCamera",
    "PhysicalCamera",
    "MockCamera",
    "TargetDetection",
    "WhiteSugarDetector",
]
