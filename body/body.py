"""
body.py
-------
Top-level Body interface wrapping the MuJoCo Skydio X2 physical system.
Provides the clean embodiment abstraction for the project.
"""

from pathlib import Path
from typing import Union, Sequence, Dict, Optional
import numpy as np

from .drone.drone import Drone


class Body:
    """
    Top-level embodiment abstraction.
    Wraps the Skydio X2 quadrotor and provides the standard interface:
        body = Body()
        state = body.read_sensors()
        body.set_motor_commands(...)
        body.step()
    """

    def __init__(self, xml_path: Optional[Union[str, Path]] = None):
        self.drone = Drone(xml_path=xml_path)

    @property
    def model(self):
        """Direct access to MjModel."""
        return self.drone.model

    @property
    def data(self):
        """Direct access to MjData."""
        return self.drone.data

    @property
    def timestep(self) -> float:
        """Physics timestep (seconds)."""
        return self.drone.timestep

    @property
    def time(self) -> float:
        """Current simulation time (seconds)."""
        return self.drone.time

    def read_sensors(self) -> Dict[str, np.ndarray]:
        """
        Query current sensor telemetry from the drone.
        Returns dict containing position, velocity, gyro, accelerometer, etc.
        """
        return self.drone.read_sensors()

    def set_motor_commands(
        self,
        commands: Union[Sequence[float], np.ndarray, Dict[str, float]],
    ) -> np.ndarray:
        """
        Set actuation commands for the 4 drone rotors.
        Accepts direct motor thrusts or flight axis dict.
        """
        return self.drone.set_motor_commands(commands)

    def step(self) -> None:
        """Advance the physics simulation by one timestep."""
        self.drone.step()

    def reset(self, keyframe_id: int = 0) -> None:
        """Reset physical state to default hover keyframe."""
        self.drone.reset(keyframe_id=keyframe_id)
