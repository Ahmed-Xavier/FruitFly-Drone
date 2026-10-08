"""
target.py
---------
Configuration and kinematic management for the visual sugar cube target in MuJoCo.

Provides a clean abstraction for spawning, repositioning, and querying the
sugar cube target object in 3D world space.
"""

from typing import Tuple, Sequence, Optional
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class TargetConfig:
    """Configuration descriptor for the visual target object."""
    name: str = "sugar_cube"
    target_type: str = "sugar_cube"
    size: Tuple[float, float, float] = (0.05, 0.05, 0.05)  # half-lengths (5 cm cube)
    color_rgba: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # pure white
    initial_position: Tuple[float, float, float] = (2.5, 0.0, 1.0)  # world pos (x=2.5m ahead, z=1.0m altitude)


def get_target_position(model: mujoco.MjModel, data: mujoco.MjData, name: str = "sugar_cube") -> np.ndarray:
    """
    Get the 3D world position [x, y, z] of the named target body.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"Target body '{name}' not found in MuJoCo model.")
    return data.xpos[body_id].copy()


def set_target_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: Sequence[float],
    name: str = "sugar_cube",
) -> None:
    """
    Dynamically reposition the target body in 3D world space.
    If the body is mocap-enabled, updates data.mocap_pos.
    Otherwise updates data.body(name).xpos directly and steps forward kinematics.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id == -1:
        raise ValueError(f"Target body '{name}' not found in MuJoCo model.")

    mocap_id = model.body_mocapid[body_id]
    pos_arr = np.asarray(position, dtype=np.float64)

    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = pos_arr
    else:
        data.xpos[body_id] = pos_arr

    mujoco.mj_forward(model, data)


def get_relative_target_vector(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    drone_body_name: str = "x2",
    target_body_name: str = "sugar_cube",
) -> np.ndarray:
    """
    Returns the vector [dx, dy, dz] from the drone center to the target object in world frame.
    """
    drone_pos = get_target_position(model, data, name=drone_body_name)
    target_pos = get_target_position(model, data, name=target_body_name)
    return target_pos - drone_pos
