"""
motors.py
---------
Actuator and motor mixing abstraction for the Skydio X2 quadrotor.
Handles 4-motor thrust allocations, aerodynamic mixer matrices, and safety limits.
"""

from typing import Union, Sequence
import numpy as np
import mujoco


class MotorActuators:
    """
    Manages the 4 thruster actuators of the Skydio X2.
    Actuator order in XML:
      0: thrust1 (pos: -.14 -.18 .05, gear:  0 0 1 0 0  .0201)
      1: thrust2 (pos: -.14  .18 .05, gear:  0 0 1 0 0 -.0201)
      2: thrust3 (pos:  .14  .18 .08, gear:  0 0 1 0 0  .0201)
      3: thrust4 (pos:  .14 -.18 .08, gear:  0 0 1 0 0 -.0201)
    """

    # Model defaults
    HOVER_THRUST = 3.2495625  # Nominal hover per motor (N)
    CTRL_MIN = 0.0            # Min rotor thrust (N)
    CTRL_MAX = 13.0           # Max rotor thrust (N)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def set_thrusts(self, thrusts: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """
        Directly assign individual thrust values to the 4 rotors.
        Values are automatically clamped to [CTRL_MIN, CTRL_MAX].
        """
        arr = np.asarray(thrusts, dtype=np.float64)
        if arr.shape[0] != 4:
            raise ValueError(f"Expected 4 rotor thrust values, received {arr.shape[0]}")
        clipped = np.clip(arr, self.CTRL_MIN, self.CTRL_MAX)
        self.data.ctrl[:4] = clipped
        return clipped

    def mix_flight_commands(
        self,
        thrust: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> np.ndarray:
        """
        Converts 4-axis flight commands (thrust, roll, pitch, yaw) into
        individual motor commands using standard X-configuration quadrotor mixing:

          motor[0] = thrust + roll + pitch - yaw
          motor[1] = thrust - roll + pitch + yaw
          motor[2] = thrust - roll - pitch - yaw
          motor[3] = thrust + roll - pitch + yaw
        """
        motors = np.array([
            thrust + roll + pitch - yaw,
            thrust - roll + pitch + yaw,
            thrust - roll - pitch - yaw,
            thrust + roll - pitch + yaw,
        ], dtype=np.float64)
        return self.set_thrusts(motors)

    def get_thrusts(self) -> np.ndarray:
        """Returns the current 4 rotor commands."""
        return self.data.ctrl[:4].copy()

    def stop(self) -> np.ndarray:
        """Set all motor commands to zero."""
        return self.set_thrusts([0.0, 0.0, 0.0, 0.0])

    def set_hover(self) -> np.ndarray:
        """Set all motors to the equilibrium hover thrust."""
        return self.set_thrusts([self.HOVER_THRUST] * 4)
