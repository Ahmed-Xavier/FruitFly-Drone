"""
drone_pid_flight.py
-------------------
Autonomous waypoint flight demonstration for the MuJoCo Skydio X2 quadrotor.
Uses cascaded PID controllers (outer velocity loop + inner attitude/rate loop)
to navigate through a sequence of 3D spatial waypoints.
"""

import time
import argparse
import numpy as np
from simple_pid import PID
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from body import Body, Drone


class WaypointPlanner:
    """Waypoint planner that computes desired velocity vectors to spatial setpoints."""

    def __init__(self, target: np.ndarray, vel_limit: float = 2.0):
        self.target = np.array(target, dtype=np.float64)
        self.vel_limit = vel_limit
        self.pid_x = PID(2.0, 0.15, 1.5, setpoint=self.target[0], output_limits=(-vel_limit, vel_limit))
        self.pid_y = PID(2.0, 0.15, 1.5, setpoint=self.target[1], output_limits=(-vel_limit, vel_limit))

    def __call__(self, loc: np.ndarray) -> np.ndarray:
        vels = np.zeros(3)
        vels[0] = self.pid_x(loc[0])
        vels[1] = self.pid_y(loc[1])
        return vels

    def get_alt_setpoint(self, loc: np.ndarray) -> float:
        distance = self.target[2] - loc[2]
        if distance > 0.5:
            time_sample = 0.25
            time_to_target = distance / self.vel_limit
            number_steps = max(1, int(time_to_target / time_sample))
            delta_alt = distance / number_steps
            return loc[2] + 2.0 * delta_alt
        return float(self.target[2])

    def update_target(self, target: np.ndarray):
        self.target = np.array(target, dtype=np.float64)
        self.pid_x.setpoint = self.target[0]
        self.pid_y.setpoint = self.target[1]


class AutopilotDrone:
    """Wraps Body/Drone with cascaded flight controllers for waypoint tracking."""

    def __init__(self, target: np.ndarray = np.array([0.0, 0.0, 1.0])):
        self.body = Body()
        self.planner = WaypointPlanner(target=target)

        # Inner control loop (attitude & altitude stabilization)
        self.pid_alt = PID(5.50844, 0.57871, 1.2, setpoint=0.0)
        self.pid_roll = PID(2.6785, 0.56871, 1.2508, setpoint=0.0, output_limits=(-1.0, 1.0))
        self.pid_pitch = PID(2.6785, 0.56871, 1.2508, setpoint=0.0, output_limits=(-1.0, 1.0))
        self.pid_yaw = PID(0.54, 0.0, 5.358333, setpoint=1.0, output_limits=(-3.0, 3.0))

        # Outer control loop (horizontal velocity tracking)
        self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint=0.0, output_limits=(-0.1, 0.1))
        self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint=0.0, output_limits=(-0.1, 0.1))

    def update_outer_control(self):
        sensors = self.body.read_sensors()
        v = sensors["velocity"]
        location = sensors["position"]

        cmd_vels = self.planner(loc=location)
        self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
        self.pid_v_x.setpoint = cmd_vels[0]
        self.pid_v_y.setpoint = cmd_vels[1]

        angle_pitch = self.pid_v_x(v[0])
        angle_roll = -self.pid_v_y(v[1])

        self.pid_pitch.setpoint = angle_pitch
        self.pid_roll.setpoint = angle_roll

    def update_inner_control(self):
        sensors = self.body.read_sensors()
        alt = sensors["position"][2]
        angles = sensors["quaternion"]  # [qw, qx, qy, qz]

        cmd_thrust = self.pid_alt(alt) + 3.2495
        cmd_roll = -self.pid_roll(angles[1])
        cmd_pitch = self.pid_pitch(angles[2])
        cmd_yaw = -self.pid_yaw(angles[0])

        self.body.set_motor_commands({
            "thrust": cmd_thrust,
            "roll": cmd_roll,
            "pitch": cmd_pitch,
            "yaw": cmd_yaw,
        })

    def step(self):
        self.body.step()


def run_flight(duration: float = 30.0, headless: bool = False):
    print("=" * 60)
    print(" Skydio X2 Autonomous Waypoint Flight Simulation")
    print("=" * 60)

    drone_sim = AutopilotDrone(target=np.array([0.0, 0.0, 1.0]))

    if headless:
        print(f"Running headless simulation for {duration:.1f} seconds...")
        start_time = time.time()
        step_count = 0
        while step_count * drone_sim.body.timestep < duration:
            sim_time = step_count * drone_sim.body.timestep
            if sim_time > 18.0:
                drone_sim.planner.update_target(np.array([-1.0, -1.0, 0.5]))
            elif sim_time > 10.0:
                drone_sim.planner.update_target(np.array([-1.0, 1.0, 2.0]))
            elif sim_time > 2.0:
                drone_sim.planner.update_target(np.array([1.0, 1.0, 1.0]))

            if step_count % 20 == 0:
                drone_sim.update_outer_control()
            drone_sim.update_inner_control()
            drone_sim.step()
            step_count += 1
        pos = drone_sim.body.read_sensors()["position"]
        print(f"Headless simulation complete. Final position: {pos}")
        return

    with mujoco.viewer.launch_passive(drone_sim.body.model, drone_sim.body.data) as viewer:
        start = time.time()
        step = 1

        while viewer.is_running() and (time.time() - start < duration):
            step_start = time.time()
            elapsed = time.time() - start

            if elapsed > 18.0:
                drone_sim.planner.update_target(np.array([-1.0, -1.0, 0.5]))
            elif elapsed > 10.0:
                drone_sim.planner.update_target(np.array([-1.0, 1.0, 2.0]))
            elif elapsed > 2.0:
                drone_sim.planner.update_target(np.array([1.0, 1.0, 1.0]))

            if step % 20 == 0:
                drone_sim.update_outer_control()
            drone_sim.update_inner_control()

            drone_sim.step()

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(drone_sim.body.data.time % 2)
            viewer.sync()

            step += 1
            dt_rem = drone_sim.body.timestep - (time.time() - step_start)
            if dt_rem > 0:
                time.sleep(dt_rem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skydio X2 PID Waypoint Flight")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Run without opening MuJoCo 3D viewer")
    args = parser.parse_args()
    run_flight(duration=args.duration, headless=args.headless)
