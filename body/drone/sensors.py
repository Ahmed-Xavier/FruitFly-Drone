"""
sensors.py
----------
Sensor abstraction for the MuJoCo Skydio X2 quadrotor.
Extracts IMU readings, rigid body poses, and telemetry directly from MjData.
"""

from typing import Dict, Any
import numpy as np
import mujoco


class DroneSensors:
    """
    Reads onboard sensors and rigid body telemetry from the Skydio X2 model.

    Supported sensor telemetry:
      - 'position': (3,) [x, y, z] in world frame
      - 'quaternion': (4,) [w, x, y, z] orientation of freejoint
      - 'velocity': (3,) [vx, vy, vz] linear velocity in world frame
      - 'angular_velocity': (3,) [wx, wy, wz] angular velocity in body frame
      - 'gyro': (3,) 3-axis gyroscope reading from IMU site
      - 'accelerometer': (3,) 3-axis accelerometer reading from IMU site
      - 'orientation': (4,) orientation quaternion from IMU site sensor
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Check sensor indices if defined in XML
        self._gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")
        self._acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")
        self._quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_quat")

    def read(self) -> Dict[str, np.ndarray]:
        """Read all current sensor signals and return them in a dictionary."""
        # Body position & orientation from generalized coordinates
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()

        # Velocities from generalized velocities
        vel = self.data.qvel[:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        # IMU hardware sensor outputs if available in MJModel
        if self._gyro_id != -1:
            gyro = self.data.sensor("body_gyro").data.copy()
        else:
            gyro = ang_vel.copy()

        if self._acc_id != -1:
            acc = self.data.sensor("body_linacc").data.copy()
        else:
            acc = np.zeros(3, dtype=np.float64)

        if self._quat_id != -1:
            imu_quat = self.data.sensor("body_quat").data.copy()
        else:
            imu_quat = quat.copy()

        return {
            "position": pos,
            "quaternion": quat,
            "velocity": vel,
            "angular_velocity": ang_vel,
            "gyro": gyro,
            "accelerometer": acc,
            "orientation": imu_quat,
        }

    def get_position(self) -> np.ndarray:
        """Returns [x, y, z] location in world frame."""
        return self.data.qpos[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Returns [vx, vy, vz] velocity vector."""
        return self.data.qvel[:3].copy()

    def get_orientation(self) -> np.ndarray:
        """Returns [w, x, y, z] quaternion."""
        return self.data.qpos[3:7].copy()
