"""
manual_control.py
-----------------
Manual keyboard flight control + live drone-POV camera view for the Skydio X2.
AZERTY layout.

Controls  (focus the 'Drone POV' window):
  Z / S   - pitch forward / backward   (hold to keep applying)
  Q / D   - roll left / right
  A / E   - yaw CCW / CW
  Space   - thrust up
  X       - thrust down
  R       - reset to hover
  ESC     - quit
"""

import time
import sys
from pathlib import Path
import numpy as np
import cv2

import mujoco
import mujoco.viewer

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from body import Drone

# Initialize Drone through body package
drone = Drone()
m = drone.model
d = drone.data

# Control constants
HOVER_THRUST = 3.2495625
THRUST_DELTA = 0.3       # applied while Space/X is held
ROLL_VAL     = 0.3       # applied while Q/D is held
PITCH_VAL    = 0.3       # applied while Z/S is held
YAW_VAL      = 0.5       # applied while A/E is held
CTRL_RANGE   = (0.0, 13.0)


def compute_motors(thrust, roll, pitch, yaw):
    motors = np.array([
        thrust + roll + pitch - yaw,
        thrust - roll + pitch + yaw,
        thrust - roll - pitch - yaw,
        thrust + roll - pitch + yaw,
    ])
    return np.clip(motors, *CTRL_RANGE)


# Camera renderer (POV camera added to x2.xml as "drone_pov")
CAM_H, CAM_W = 360, 640
cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "drone_pov")
use_cam = cam_id != -1
renderer = mujoco.Renderer(m, height=CAM_H, width=CAM_W) if use_cam else None

# State
thrust_target = HOVER_THRUST
last_key_time = 0.0
KEY_TIMEOUT = 0.12


def render_pov(fps_val):
    if renderer is None:
        return
    renderer.update_scene(d, camera="drone_pov")
    bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    alt = d.qpos[2]
    spd = float(np.linalg.norm(d.qvel[:3]))
    cv2.putText(bgr, f"Alt: {alt:.2f} m", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(bgr, f"Spd: {spd:.2f} m/s", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(bgr, f"Thrust: {thrust_target:.2f} N", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(bgr, f"FPS: {fps_val:.0f}", (CAM_W - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(bgr, "Z/S: pitch  Q/D: roll  A/E: yaw  Space/X: thrust  R: reset",
                (15, CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.imshow("Skydio X2 - Drone POV", bgr)


def main():
    global thrust_target, last_key_time
    cv2.namedWindow("Skydio X2 - Drone POV", cv2.WINDOW_AUTOSIZE)

    active_roll  = 0.0
    active_pitch = 0.0
    active_yaw   = 0.0

    fps_count = 0
    fps_val = 0.0
    fps_timer = time.time()
    SIM_STEPS_PER_FRAME = 5

    print(__doc__)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            if key != 255:
                last_key_time = now
                if key == 27:
                    break
                elif key == ord('r'):
                    drone.reset()
                    thrust_target = HOVER_THRUST
                    active_roll = active_pitch = active_yaw = 0.0
                elif key == 32:
                    thrust_target = min(CTRL_RANGE[1], thrust_target + THRUST_DELTA)
                elif key == ord('x'):
                    thrust_target = max(CTRL_RANGE[0], thrust_target - THRUST_DELTA)
                elif key == ord('z'):
                    active_pitch = PITCH_VAL
                elif key == ord('s'):
                    active_pitch = -PITCH_VAL
                elif key == ord('q'):
                    active_roll = -ROLL_VAL
                elif key == ord('d'):
                    active_roll = ROLL_VAL
                elif key == ord('a'):
                    active_yaw = YAW_VAL
                elif key == ord('e'):
                    active_yaw = -YAW_VAL
            else:
                if now - last_key_time > KEY_TIMEOUT:
                    active_roll  = 0.0
                    active_pitch = 0.0
                    active_yaw   = 0.0

            d.ctrl[:4] = compute_motors(thrust_target, active_roll, active_pitch, active_yaw)

            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(m, d)

            viewer.sync()

            fps_count += 1
            if now - fps_timer >= 0.5:
                fps_val = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now

            render_pov(fps_val)

            elapsed = time.time() - step_start
            target_dt = m.opt.timestep * SIM_STEPS_PER_FRAME
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
