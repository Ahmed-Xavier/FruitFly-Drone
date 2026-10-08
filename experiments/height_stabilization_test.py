"""
height_stabilization_test.py
----------------------------
Simple vertical PD height stabilization test for Skydio X2.
Migrated from scripts/test.py to use the modular body package.
"""

import time
import sys
from pathlib import Path
import mujoco
import mujoco.viewer

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from body import Body


class PDController:
    def __init__(self, kp: float, kd: float, setpoint: float):
        self.kp = kp
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0

    def compute(self, measured_value: float) -> float:
        error = self.setpoint - measured_value
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.kd * derivative)
        self.prev_error = error
        return output


def run_test(duration: float = 5.0, headless: bool = True):
    body = Body()
    pd_controller = PDController(kp=0.5, kd=0.1, setpoint=0.0)

    # Initial thrust
    body.set_motor_commands([4.0, 4.0, 4.0, 4.0])

    print("Running height stabilization test...")
    step_count = 0
    while step_count * body.timestep < duration:
        sensors = body.read_sensors()
        measured_speed = sensors["velocity"][2]
        control_signal = pd_controller.compute(measured_speed)

        current_thrusts = body.data.ctrl[:4].copy()
        body.set_motor_commands(current_thrusts + control_signal)
        body.step()
        step_count += 1

    final_sensors = body.read_sensors()
    print(f"Test completed. Final altitude: {final_sensors['position'][2]:.3f} m, "
          f"vertical speed: {final_sensors['velocity'][2]:.3f} m/s")


if __name__ == "__main__":
    run_test(duration=5.0, headless=True)
