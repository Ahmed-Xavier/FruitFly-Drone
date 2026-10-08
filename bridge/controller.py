"""
controller.py
-------------
Flight stabilization controller and safety layer for the Skydio X2 quadrotor.

Takes high-level biological motor intent (from MotorDecoder) and translates it
into stabilized 4-rotor actuator commands, enforcing aerodynamic safety envelopes,
attitude stabilization, motor saturation limits, and numerical validity checks.

Architecture:
    Biological Motor Intent (thrust, roll, pitch, yaw)
            ↓
    Attitude & Rate Stabilization Controller (PID / PD scaffold)
            ↓
    Rate Limiting & Safety Envelope (NaN/Inf guard, tilt limit)
            ↓
    Aerodynamic Motor Mixer (X-configuration)
            ↓
    Clamped 4-Rotor Actuation ([0.0, 13.0] N per rotor)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class DroneFlightController:
    """
    Flight controller providing physical stabilization for biological neural intent.
    """

    # Physical drone parameters from body/drone/x2.xml and motors.py
    HOVER_THRUST = 3.2495625  # Nominal hover per motor (N)
    CTRL_MIN = 0.0            # Min rotor thrust (N)
    CTRL_MAX = 13.0           # Max rotor thrust (N)

    # Safety limits
    MAX_TILT_RAD = 0.5236     # Max allowed bank/pitch angle (30 degrees)
    MAX_YAW_RATE = 2.0        # Max yaw rate (rad/s)
    MAX_SLEW_RATE = 80.0      # Max motor thrust change rate (N/s)

    def __init__(
        self,
        kp_att: float = 4.0,
        kd_rate: float = 0.45,
        kp_alt: float = 2.5,
        kd_alt: float = 1.2,
        target_alt: float = 1.0,
    ):
        """
        Initialize PD gains for attitude and altitude stabilization.
        """
        self.kp_att = kp_att
        self.kd_rate = kd_rate
        self.kp_alt = kp_alt
        self.kd_alt = kd_alt
        self.target_alt = target_alt

        self.prev_motor_commands = np.array([self.HOVER_THRUST] * 4, dtype=np.float64)
        self.safety_triggered = False
        self.safety_reason = "OK"

    def reset(self, target_alt: float = 1.0):
        """Reset internal controller state."""
        self.target_alt = target_alt
        self.prev_motor_commands = np.array([self.HOVER_THRUST] * 4, dtype=np.float64)
        self.safety_triggered = False
        self.safety_reason = "OK"

    def compute(
        self,
        telemetry: Dict[str, np.ndarray],
        motor_intent: Dict[str, Any],
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Compute stabilized 4-rotor actuator commands from biological intent and telemetry.

        Parameters
        ----------
        telemetry : dict
            Sensors dictionary from Drone/Body containing 'position', 'velocity',
            'quaternion', and 'gyro'.
        motor_intent : dict
            Decoded intent from MotorDecoder containing 'thrust_delta',
            'pitch_intent', 'roll_intent', 'yaw_intent', 'escape_active'.
        dt : float
            Simulation timestep in seconds (default: 0.01 s = 10 ms).

        Returns
        -------
        dict with:
          - 'motor_commands': np.ndarray shape (4,) clamped to [0.0, 13.0]
          - 'flight_axes': dict with 'thrust', 'roll', 'pitch', 'yaw'
          - 'safety_triggered': bool
          - 'safety_reason': str
        """
        self.safety_triggered = False
        self.safety_reason = "OK"

        # ── 1. Numerical Validity & NaN/Inf Protection ────────────────────────
        pos = telemetry.get("position", np.array([0.0, 0.0, 0.0]))
        vel = telemetry.get("velocity", np.array([0.0, 0.0, 0.0]))
        quat = telemetry.get("quaternion", np.array([1.0, 0.0, 0.0, 0.0]))
        gyro = telemetry.get("gyro", np.array([0.0, 0.0, 0.0]))

        if any(not np.all(np.isfinite(x)) for x in (pos, vel, quat, gyro)):
            self.safety_triggered = True
            self.safety_reason = "NaN or Inf detected in sensor telemetry"
            fallback = np.array([self.HOVER_THRUST] * 4, dtype=np.float64)
            return {
                "motor_commands": fallback,
                "flight_axes": {"thrust": self.HOVER_THRUST, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "safety_triggered": True,
                "safety_reason": self.safety_reason,
            }

        # ── 2. Attitude Decomposition ─────────────────────────────────────────
        # MuJoCo quaternion: [qw, qx, qy, qz]
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        roll_angle = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch_angle = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

        # Check maximum tilt safety bound
        tilt_magnitude = np.sqrt(roll_angle * roll_angle + pitch_angle * pitch_angle)
        if tilt_magnitude > self.MAX_TILT_RAD * 1.5:
            self.safety_triggered = True
            self.safety_reason = f"Excessive tilt detected ({np.degrees(tilt_magnitude):.1f} deg)"

        # ── 3. Motor Intent Interpretation ────────────────────────────────────
        intent_thrust_delta = float(motor_intent.get("thrust_delta", 0.0))
        intent_roll = float(motor_intent.get("roll_intent", 0.0))
        intent_pitch = float(motor_intent.get("pitch_intent", 0.0))
        intent_yaw = float(motor_intent.get("yaw_intent", 0.0))
        escape_active = bool(motor_intent.get("escape_active", False))

        # Map normalized intents to physical setpoints
        target_roll = intent_roll * self.MAX_TILT_RAD
        # Positive pitch in NWU body is nose-up; forward drive corresponds to nose-down pitch:
        target_pitch = -intent_pitch * self.MAX_TILT_RAD
        target_yaw_rate = intent_yaw * self.MAX_YAW_RATE

        # ── 4. Attitude Stabilization Control Laws ────────────────────────────
        # Roll axis control: PD on angle and gyro roll rate (gyro[0])
        err_roll = target_roll - roll_angle
        cmd_roll = self.kp_att * err_roll - self.kd_rate * gyro[0]

        # Pitch axis control: PD on angle and gyro pitch rate (gyro[1])
        err_pitch = target_pitch - pitch_angle
        cmd_pitch = self.kp_att * err_pitch - self.kd_rate * gyro[1]

        # Yaw axis control: Rate feedback on gyro[2]
        err_yaw_rate = target_yaw_rate - gyro[2]
        cmd_yaw = 0.3 * err_yaw_rate

        # ── 5. Altitude & Thrust Modulation ───────────────────────────────────
        # Altitude feedback baseline + biological thrust delta
        current_alt = float(pos[2])
        alt_error = self.target_alt - current_alt
        alt_stabilization = self.kp_alt * alt_error - self.kd_alt * float(vel[2])

        if escape_active:
            # During emergency escape, override altitude hold with vertical climb surge
            total_thrust = self.HOVER_THRUST + (intent_thrust_delta / 4.0)
        else:
            # Normal flight: combine equilibrium hover + biological thrust delta + altitude hold
            total_thrust = self.HOVER_THRUST + (intent_thrust_delta / 4.0) + (alt_stabilization / 4.0)

        # ── 6. Aerodynamic Quadrotor Mixer (X-Configuration) ──────────────────
        # Actuator ordering matching body/drone/x2.xml and motors.py:
        #   thrust1: rear-right
        #   thrust2: rear-left
        #   thrust3: front-left
        #   thrust4: front-right
        raw_motors = np.array([
            total_thrust + cmd_roll + cmd_pitch - cmd_yaw,  # motor 0
            total_thrust - cmd_roll + cmd_pitch + cmd_yaw,  # motor 1
            total_thrust - cmd_roll - cmd_pitch - cmd_yaw,  # motor 2
            total_thrust + cmd_roll - cmd_pitch + cmd_yaw,  # motor 3
        ], dtype=np.float64)

        # ── 7. Slew Rate Limiting ─────────────────────────────────────────────
        max_delta = self.MAX_SLEW_RATE * dt
        clamped_motors = np.clip(
            raw_motors,
            self.prev_motor_commands - max_delta,
            self.prev_motor_commands + max_delta,
        )

        # ── 8. Absolute Actuator Saturation ───────────────────────────────────
        final_motors = np.clip(clamped_motors, self.CTRL_MIN, self.CTRL_MAX)
        self.prev_motor_commands = final_motors.copy()

        return {
            "motor_commands": final_motors,
            "flight_axes": {
                "thrust": total_thrust,
                "roll": float(cmd_roll),
                "pitch": float(cmd_pitch),
                "yaw": float(cmd_yaw),
            },
            "safety_triggered": self.safety_triggered,
            "safety_reason": self.safety_reason,
        }
