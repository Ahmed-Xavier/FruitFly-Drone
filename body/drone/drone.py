"""
drone.py
--------
Complete drone abstraction for the MuJoCo Skydio X2 quadrotor.
Encapsulates physics loading, state management, sensor queries, and actuator control.
"""

from pathlib import Path
from typing import Union, Sequence, Dict, Any, Optional
import numpy as np
import mujoco

from .sensors import DroneSensors
from .motors import MotorActuators


class Drone:
    """
    MuJoCo Skydio X2 Drone.

    Public interface:
        drone = Drone()
        state = drone.read_sensors()
        drone.set_motor_commands(...)
        drone.step()
    """

    def __init__(self, xml_path: Optional[Union[str, Path]] = None):
        if xml_path is None:
            xml_path = Path(__file__).resolve().parent / "scene.xml"
        self.xml_path = Path(xml_path).resolve()

        if not self.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo model file not found at: {self.xml_path}")

        # Load MuJoCo model & data
        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.d = mujoco.MjData(self.m)

        # Subsystems
        self.sensors = DroneSensors(self.m, self.d)
        self.motors = MotorActuators(self.m, self.d)

        # Initialize to default hover state
        self.reset()

    @property
    def model(self) -> mujoco.MjModel:
        """Direct access to MjModel."""
        return self.m

    @property
    def data(self) -> mujoco.MjData:
        """Direct access to MjData."""
        return self.d

    @property
    def timestep(self) -> float:
        """Simulation physics timestep in seconds."""
        return float(self.m.opt.timestep)

    @property
    def time(self) -> float:
        """Current elapsed simulation time in seconds."""
        return float(self.d.time)

    def reset(self, keyframe_id: int = 0) -> None:
        """
        Resets drone kinematics and state to the specified keyframe.
        Keyframe 0 is defined as the 'hover' state in x2.xml.
        """
        if self.m.nkey > keyframe_id:
            mujoco.mj_resetDataKeyframe(self.m, self.d, keyframe_id)
        else:
            mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)

    def read_sensors(self) -> Dict[str, np.ndarray]:
        """
        Query current onboard telemetry.

        Returns:
            dict containing 'position', 'quaternion', 'velocity',
            'angular_velocity', 'gyro', 'accelerometer', 'orientation'
        """
        return self.sensors.read()

    def set_motor_commands(
        self,
        commands: Union[Sequence[float], np.ndarray, Dict[str, float]],
    ) -> np.ndarray:
        """
        Assign motor commands to the drone.

        Can accept:
          - 4-element sequence of direct motor thrusts: [m1, m2, m3, m4]
          - dictionary of flight axes: {'thrust': T, 'roll': R, 'pitch': P, 'yaw': Y}
        """
        if isinstance(commands, dict):
            thrust = float(commands.get("thrust", self.motors.HOVER_THRUST))
            roll = float(commands.get("roll", 0.0))
            pitch = float(commands.get("pitch", 0.0))
            yaw = float(commands.get("yaw", 0.0))
            return self.motors.mix_flight_commands(thrust=thrust, roll=roll, pitch=pitch, yaw=yaw)
        else:
            return self.motors.set_thrusts(commands)

    def step(self) -> None:
        """Advance the MuJoCo physics simulation by one step."""
        mujoco.mj_step(self.m, self.d)
