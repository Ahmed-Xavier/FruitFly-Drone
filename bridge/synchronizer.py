"""
synchronizer.py
---------------
Multiscale temporal synchronization coordinator interfacing the FlyWire v783
connectome (dt = 0.1 ms, 10 kHz) with MuJoCo quadrotor physics (dt = 10 ms, 100 Hz).

Manages the closed-loop execution cycle:
    1. Sample drone telemetry from Body
    2. Encode continuous sensory signals into neural firing rates (SensoryEncoder)
    3. Advance FlyWire brain simulation by N discrete neural steps
    4. Accumulate and smooth descending neuron (DN) rates over the window
    5. Decode biological motor intent into flight axes (MotorDecoder)
    6. Compute stabilized 4-rotor actuator commands (DroneFlightController)
    7. Step MuJoCo physical embodiment (Body)
    8. Record structured telemetry / logging frame
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import numpy as np

from body.body import Body
from brain.brain import FlyBrain, Brain
from bridge.sensory_encoder import SensoryEncoder
from bridge.motor_decoder import MotorDecoder
from bridge.controller import DroneFlightController


@dataclass
class BridgeTelemetry:
    """Structured telemetry snapshot capturing drone, brain, and bridge state."""
    time_sim_s: float
    time_brain_ms: float

    # Drone kinematics
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    gyro: np.ndarray
    accelerometer: np.ndarray
    motor_commands: np.ndarray

    # Brain state
    spikes_this_step: int
    dn_rates: Dict[str, float]
    population_rates: Dict[str, float]

    # Bridge state
    sensory_summary: Dict[str, Any]
    motor_intent: Dict[str, Any]
    flight_axes: Dict[str, float]
    safety_triggered: bool
    safety_reason: str


class BrainDroneSynchronizer:
    """
    Coordinates multiscale simulation stepping between Drosophila brain and MuJoCo drone.
    """

    def __init__(
        self,
        body: Optional[Body] = None,
        brain: Optional[FlyBrain] = None,
        neural_steps_per_body_step: int = 10,
        target_alt: float = 1.0,
        device: str = "cuda",
        plasticity: bool = False,
    ):
        """
        Parameters
        ----------
        body : Body, optional
            MuJoCo Skydio X2 Body instance. Created if None.
        brain : FlyBrain, optional
            FlyWire v783 connectome instance. Created if None.
        neural_steps_per_body_step : int
            Number of brain simulation steps (dt = 0.1 ms) per physical step (dt = 10 ms).
            e.g., N = 100 achieves 1:1 real-time temporal parity (10 ms neural per 10 ms physics).
            N = 10 achieves fast 10x accelerated simulation with 1.0 ms neural window.
        target_alt : float
            Target hover altitude in meters (default: 1.0 m).
        """
        self.body = body if body is not None else Body()
        self.brain = brain if brain is not None else Brain(device=device, plasticity=plasticity)

        self.neural_steps = max(1, neural_steps_per_body_step)
        self.dt_body = float(self.body.timestep)      # default 0.01 s = 10 ms
        self.dt_brain_ms = float(self.brain.timestep_ms) # default 0.1 ms

        # Bridge subsystems
        self.encoder = SensoryEncoder(use_full_populations=True)
        self.decoder = MotorDecoder()
        self.controller = DroneFlightController(target_alt=target_alt)

        self.step_count = 0
        self.total_sim_time = 0.0
        self.telemetry_history: List[BridgeTelemetry] = []

    def reset(self, keyframe_id: int = 0, target_alt: float = 1.0):
        """Reset drone, brain inputs, and controller state."""
        self.body.reset(keyframe_id=keyframe_id)
        self.brain.clear_input()
        self.controller.reset(target_alt=target_alt)
        self.step_count = 0
        self.total_sim_time = 0.0
        self.telemetry_history.clear()

    def step(
        self,
        looming_threat: Optional[Dict[str, float]] = None,
        external_force: Optional[np.ndarray] = None,
    ) -> BridgeTelemetry:
        """
        Execute one complete synchronized cycle.

        Parameters
        ----------
        looming_threat : dict, optional
            Simulated visual collision threat for LC4 visual pathway.
        external_force : np.ndarray, optional
            Optional force perturbation applied to drone body for disturbance testing.

        Returns
        -------
        BridgeTelemetry snapshot.
        """
        # ── 1. Read drone telemetry ───────────────────────────────────────────
        sensors = self.body.read_sensors()

        # ── 2. Encode continuous telemetry into biological firing rates ───────
        encoded = self.encoder.encode(sensors, looming_threat=looming_threat)
        rates_dict = encoded["rates"]
        sensory_summary = encoded["encoded_summary"]

        # ── 3. Step FlyWire brain N times with encoded input ──────────────────
        total_spikes = 0
        last_brain_out = None

        input_data = {"rates": rates_dict}

        for _ in range(self.neural_steps):
            brain_out = self.brain.step(input_data)
            total_spikes += int(brain_out["spikes"].sum())
            last_brain_out = brain_out

        # ── 4. Decode biological descending neuron activity into flight intent ─
        motor_intent = self.decoder.decode(last_brain_out)

        # ── 5. Compute stabilized 4-rotor actuator commands ───────────────────
        ctrl_out = self.controller.compute(
            telemetry=sensors,
            motor_intent=motor_intent,
            dt=self.dt_body,
        )
        motor_commands = ctrl_out["motor_commands"]

        # ── 6. Apply external perturbation if requested ───────────────────────
        if external_force is not None:
            # Apply force to main body in MuJoCo data
            self.body.data.xfrc_applied[1, :3] = external_force

        # ── 7. Advance MuJoCo physical embodiment ─────────────────────────────
        self.body.set_motor_commands(motor_commands)
        self.body.step()

        # Clear any applied transient perturbation
        if external_force is not None:
            self.body.data.xfrc_applied[1, :3] = 0.0

        self.step_count += 1
        self.total_sim_time += self.dt_body

        # ── 8. Assemble structured telemetry snapshot ─────────────────────────
        snapshot = BridgeTelemetry(
            time_sim_s=self.total_sim_time,
            time_brain_ms=last_brain_out["time_ms"],
            position=sensors["position"].copy(),
            velocity=sensors["velocity"].copy(),
            quaternion=sensors["quaternion"].copy(),
            gyro=sensors["gyro"].copy(),
            accelerometer=sensors["accelerometer"].copy(),
            motor_commands=motor_commands.copy(),
            spikes_this_step=total_spikes,
            dn_rates=dict(last_brain_out["dn_rates"]),
            population_rates=dict(last_brain_out["population_rates"]),
            sensory_summary=sensory_summary,
            motor_intent=motor_intent,
            flight_axes=ctrl_out["flight_axes"],
            safety_triggered=ctrl_out["safety_triggered"],
            safety_reason=ctrl_out["safety_reason"],
        )

        return snapshot
