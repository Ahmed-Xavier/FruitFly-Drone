"""
sensory_encoder.py
------------------
Sensory encoding layer mapping physical drone telemetry into biologically grounded
neural firing rates for the FlyWire v783 Drosophila connectome.

Supported biological sensory pathways:
1. Johnston's Organ (JO) mechanosensory pathway:
   - Encodes airspeed, aerodynamic drag, and angular velocity (differential airflow across antennae).
   - Bilateral lateralization: Left JO (JO-EV / JO-E) vs Right JO (JO-E).
2. Visual Looming Pathway (LC4):
   - Encodes looming optical expansion / collision threat (104 LC4 neurons).
   - Projects directly downstream to Giant Fiber (GF / DNp01) and escape circuits.
3. Optic Flow Tangential System (HS / VS):
   - Encodes wide-field angular rotations (Horizontal System for yaw, Vertical System for pitch/roll).

Scientific Classification:
- Biological Grounding: Selection of JO-E, LC4, and HS/VS neural populations from FlyWire connectome.
- Engineering Approximation: Linearized sigmoid transfer functions mapping SI kinematics
  (rad/s, m/s) to Poisson firing rates (Hz), clamped to physiological limits (0-350 Hz).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

from brain.connectome.loader import load_annotations
from brain.brain import STIMULI


class SensoryEncoder:
    """
    Translates physical drone sensor telemetry (IMU, velocities, looming threats)
    into FlyWire segment ID -> firing rate (Hz) mappings.
    """

    # Physiological limits for Drosophila sensory neurons
    MAX_FIRING_RATE_HZ = 350.0  # Max typical burst rate in fly mechanosensory / visual neurons
    BASELINE_RATE_HZ = 10.0     # Spontaneous baseline sensory firing rate

    def __init__(self, use_full_populations: bool = True):
        """
        Initialize sensory populations from FlyWire connectome annotations.

        Parameters
        ----------
        use_full_populations : bool
            If True, loads full wind_gravity and LC4 neuron sets from annotations table.
            If False, falls back to the pre-curated STIMULI subsets.
        """
        self.use_full_populations = use_full_populations

        # 1. Johnston's Organ (JO) Wind & Mechanosensory Populations
        self.jo_left_ids: List[int] = []
        self.jo_right_ids: List[int] = []

        # 2. LC4 Looming Visual Projection Neurons
        self.lc4_left_ids: List[int] = []
        self.lc4_right_ids: List[int] = []

        # 3. Horizontal & Vertical System Optic Flow Cells
        self.hs_left_ids: List[int] = []
        self.hs_right_ids: List[int] = []
        self.vs_left_ids: List[int] = []
        self.vs_right_ids: List[int] = []

        self._load_sensory_pathways()

    def _load_sensory_pathways(self):
        """Extract validated sensory neuron IDs from FlyWire annotations."""
        df = load_annotations()
        if df is not None and self.use_full_populations:
            # 1. Antennal Mechanosensory (Johnston's organ wind & gravity)
            wg = df[df["cell_sub_class"] == "wind_gravity"]
            self.jo_left_ids = wg[wg["side"] == "left"]["root_id"].astype(int).tolist()
            self.jo_right_ids = wg[wg["side"] == "right"]["root_id"].astype(int).tolist()

            # 2. LC4 Looming Neurons
            lc4 = df[df["cell_type"] == "LC4"]
            self.lc4_left_ids = lc4[lc4["side"] == "left"]["root_id"].astype(int).tolist()
            self.lc4_right_ids = lc4[lc4["side"] == "right"]["root_id"].astype(int).tolist()

            # 3. Optic Flow (Horizontal & Vertical System)
            hs = df[df["cell_type"].astype(str).str.startswith("HS")]
            self.hs_left_ids = hs[hs["side"] == "left"]["root_id"].astype(int).tolist()
            self.hs_right_ids = hs[hs["side"] == "right"]["root_id"].astype(int).tolist()

            vs = df[df["cell_type"].astype(str).str.startswith("VS")]
            self.vs_left_ids = vs[vs["side"] == "left"]["root_id"].astype(int).tolist()
            self.vs_right_ids = vs[vs["side"] == "right"]["root_id"].astype(int).tolist()
        else:
            # Fallback to curated STIMULI subsets in brain/brain.py
            if "jo" in STIMULI:
                self.jo_left_ids = list(STIMULI["jo"]["neurons"])
                self.jo_right_ids = list(STIMULI["jo"]["neurons"])
            if "lc4" in STIMULI:
                all_lc4 = list(STIMULI["lc4"]["neurons"])
                half = len(all_lc4) // 2
                self.lc4_left_ids = all_lc4[:half]
                self.lc4_right_ids = all_lc4[half:]

    def encode(
        self,
        telemetry: Dict[str, np.ndarray],
        looming_threat: Optional[Dict[str, float]] = None,
        stimulus_bias: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Encode physical drone state into biological input firing rates.

        Parameters
        ----------
        telemetry : dict
            Standard Drone sensor dictionary containing:
              - 'gyro': np.ndarray [gx, gy, gz] in rad/s (body frame: roll, pitch, yaw rates)
              - 'velocity': np.ndarray [vx, vy, vz] in m/s (world/body frame)
              - 'accelerometer': np.ndarray [ax, ay, az] in m/s^2 (specific force)
              - 'quaternion': np.ndarray [qw, qx, qy, qz]
        looming_threat : dict, optional
            Optional threat stimulus parameters:
              - 'threat_level': float in [0.0, 1.0]
              - 'side': str 'center', 'left', or 'right'

        Returns
        -------
        dict with:
          - 'rates': Dict[int, float] mapping FlyWire segment IDs to firing rate in Hz
          - 'encoded_summary': dict summarizing sensory channels for logging/telemetry
        """
        gyro = telemetry.get("gyro", np.zeros(3))
        vel = telemetry.get("velocity", np.zeros(3))
        acc = telemetry.get("accelerometer", np.zeros(3))

        # Kinematic metrics
        roll_rate = float(gyro[0])    # rad/s (+ = roll right)
        pitch_rate = float(gyro[1])   # rad/s (+ = pitch up)
        yaw_rate = float(gyro[2])     # rad/s (+ = CCW / turn left in NWU)

        airspeed_fwd = max(0.0, float(vel[0]))  # forward speed (m/s)
        speed_total = float(np.linalg.norm(vel))

        rates: Dict[int, float] = {}

        # ── 1. Johnston's Organ (JO) Aerodynamic & Mechanosensory Encoding ──────
        # BIOLOGICALLY GROUNDED: Bilateral differential airflow across left/right antennae.
        # Yaw rotation and lateral slip create asymmetric air velocity on antennae.
        # ENGINEERING APPROXIMATION: Linear sensitivity gain (30 Hz per rad/s yaw rate).
        symmetric_wind_rate = np.clip(
            self.BASELINE_RATE_HZ + 15.0 * speed_total,
            self.BASELINE_RATE_HZ,
            self.MAX_FIRING_RATE_HZ,
        )

        yaw_differential = 35.0 * yaw_rate     # +yaw (turn left) increases right antenna pressure
        roll_differential = 20.0 * roll_rate   # +roll (roll right) increases left antenna pressure

        # Stimulus bias (e.g. from keyboard navigation or high-level mission cues)
        steer_bias = float(stimulus_bias.get("steer", 0.0)) if stimulus_bias else 0.0
        fwd_bias = float(stimulus_bias.get("forward", 0.0)) if stimulus_bias else 0.0

        jo_left_rate = float(np.clip(
            symmetric_wind_rate - yaw_differential + roll_differential - steer_bias * 45.0,
            0.0,
            self.MAX_FIRING_RATE_HZ,
        ))
        jo_right_rate = float(np.clip(
            symmetric_wind_rate + yaw_differential - roll_differential + steer_bias * 45.0,
            0.0,
            self.MAX_FIRING_RATE_HZ,
        ))

        for nid in self.jo_left_ids:
            rates[nid] = jo_left_rate
        for nid in self.jo_right_ids:
            rates[nid] = jo_right_rate

        # Forward drive injection into P9 forward locomotion neurons if commanded
        if fwd_bias > 0.0:
            p9_rate = float(np.clip(fwd_bias * 120.0, 0.0, self.MAX_FIRING_RATE_HZ))
            p9_ids = STIMULI.get("p9", {}).get("neurons", [])
            for pid in p9_ids:
                rates[pid] = max(rates.get(pid, 0.0), p9_rate)


        # ── 2. Optic Flow (Horizontal & Vertical System Lobula Plate Cells) ────
        # BIOLOGICALLY GROUNDED: HS cells respond to horizontal visual yaw rotation;
        # VS cells respond to vertical visual pitch/roll rotation.
        hs_left_rate = float(np.clip(
            self.BASELINE_RATE_HZ + 25.0 * max(0.0, -yaw_rate),
            0.0,
            self.MAX_FIRING_RATE_HZ,
        ))
        hs_right_rate = float(np.clip(
            self.BASELINE_RATE_HZ + 25.0 * max(0.0, yaw_rate),
            0.0,
            self.MAX_FIRING_RATE_HZ,
        ))
        for nid in self.hs_left_ids:
            rates[nid] = hs_left_rate
        for nid in self.hs_right_ids:
            rates[nid] = hs_right_rate

        vs_rate = float(np.clip(
            self.BASELINE_RATE_HZ + 20.0 * abs(pitch_rate),
            0.0,
            self.MAX_FIRING_RATE_HZ,
        ))
        for nid in (self.vs_left_ids + self.vs_right_ids):
            rates[nid] = vs_rate

        # ── 3. Visual Looming Pathway (LC4 -> Giant Fiber Escape) ─────────────
        # BIOLOGICALLY GROUNDED: Impending collision triggers high-frequency burst in LC4 neurons.
        lc4_left_rate = 0.0
        lc4_right_rate = 0.0

        if looming_threat is not None:
            threat_level = np.clip(looming_threat.get("threat_level", 0.0), 0.0, 1.0)
            threat_side = looming_threat.get("side", "center")

            burst_rate = float(threat_level * 250.0)

            if threat_side in ("center", "both"):
                lc4_left_rate = burst_rate
                lc4_right_rate = burst_rate
            elif threat_side == "left":
                lc4_left_rate = burst_rate
                lc4_right_rate = burst_rate * 0.2
            elif threat_side == "right":
                lc4_left_rate = burst_rate * 0.2
                lc4_right_rate = burst_rate

            for nid in self.lc4_left_ids:
                rates[nid] = lc4_left_rate
            for nid in self.lc4_right_ids:
                rates[nid] = lc4_right_rate

        summary = {
            "jo_left_rate_hz": jo_left_rate,
            "jo_right_rate_hz": jo_right_rate,
            "hs_left_rate_hz": hs_left_rate,
            "hs_right_rate_hz": hs_right_rate,
            "vs_rate_hz": vs_rate,
            "lc4_left_rate_hz": lc4_left_rate,
            "lc4_right_rate_hz": lc4_right_rate,
            "total_sensory_neurons": len(rates),
        }

        return {
            "rates": rates,
            "encoded_summary": summary,
        }
