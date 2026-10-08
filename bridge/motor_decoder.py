"""
motor_decoder.py
----------------
Motor decoding layer translating biological Descending Neuron (DN) activity
into high-level flight motor intent for the quadrotor drone.

Biological Descending Neuron Groups utilized:
1. P9 / P9_oDN1 (Forward Locomotion / Propulsion):
   - Bilaterally active during forward walking and flight steering.
   - Decoded into forward pitch intent (nose-down forward velocity drive) and baseline thrust support.
2. DNa01 & DNa02 (Steering & Turning):
   - Ipsilateral turning drive. Left DN activation initiates leftward turns;
     right DN activation initiates rightward turns.
   - Bilateral differential (Right - Left) decodes into yaw torque and coordinated roll intent.
3. MDN (Moonwalker Descending Neurons):
   - Drives backward locomotion and deceleration braking.
   - Decoded into pitch-up braking / backward retreat intent.
4. Giant Fiber / DNp01 (Emergency Takeoff & Escape):
   - Large-diameter descending escape neuron activated downstream of looming visual pathways (LC4).
   - High-frequency burst decodes into emergency vertical thrust surge (takeoff reflex).

Scientific Classification:
- Biological Grounding: Directional assignments based on Ache et al. (2019), Rayshubskiy et al. (2020),
  Namiki et al. (2018), and von Reyn et al. (2014).
- Engineering Approximation: Normalization constants scaling spike rates (Hz) to bounded
  dimensionless motor intent [-1.0, +1.0] and thrust delta (N).
"""

from typing import Dict, Any, Optional
import numpy as np


class MotorDecoder:
    """
    Decodes biological Descending Neuron (DN) firing rates into high-level flight intent:
      - thrust_delta: vertical force offset (N)
      - pitch_intent: normalized [-1, 1] (+ = pitch forward, - = pitch backward)
      - roll_intent: normalized [-1, 1] (+ = roll right, - = roll left)
      - yaw_intent: normalized [-1, 1] (+ = turn right, - = turn left)
    """

    # Firing rate normalization constants (Hz)
    P9_RATE_NORM_HZ = 80.0       # Typical active walking/steering rate for P9
    DNA_RATE_NORM_HZ = 60.0      # Sustained steering rate for DNa01/DNa02
    MDN_RATE_NORM_HZ = 50.0      # Backward locomotion threshold
    GF_ESCAPE_THRESHOLD_HZ = 20.0 # Emergency escape trigger threshold for Giant Fiber

    def __init__(
        self,
        max_thrust_surge_n: float = 6.0,
        steer_gain: float = 1.0,
        pitch_gain: float = 1.0,
    ):
        self.max_thrust_surge_n = max_thrust_surge_n
        self.steer_gain = steer_gain
        self.pitch_gain = pitch_gain

    def decode(self, brain_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode brain simulation output into flight intent.

        Parameters
        ----------
        brain_output : dict
            Standard output dictionary from FlyBrain.step(), containing:
              - 'dn_rates': Dict[str, float] with smoothed firing rates per DN
              - 'population_rates': Dict[str, float] with group averages

        Returns
        -------
        dict with decoded intent:
          - 'thrust_delta': float (N)
          - 'pitch_intent': float in [-1.0, 1.0]
          - 'roll_intent': float in [-1.0, 1.0]
          - 'yaw_intent': float in [-1.0, 1.0]
          - 'escape_active': bool
          - 'dn_summary': dict of raw DN metrics for logging
        """
        dn_rates = brain_output.get("dn_rates", {})

        # Extract rates for specific functional descending neurons
        p9_l = dn_rates.get("P9_left", 0.0) + dn_rates.get("P9_oDN1_left", 0.0)
        p9_r = dn_rates.get("P9_right", 0.0) + dn_rates.get("P9_oDN1_right", 0.0)
        p9_mean = 0.5 * (p9_l + p9_r)

        dna_l = 0.5 * (dn_rates.get("DNa01_left", 0.0) + dn_rates.get("DNa02_left", 0.0))
        dna_r = 0.5 * (dn_rates.get("DNa01_right", 0.0) + dn_rates.get("DNa02_right", 0.0))

        mdn = 0.5 * (dn_rates.get("MDN_1", 0.0) + dn_rates.get("MDN_2", 0.0))

        gf_burst = max(
            dn_rates.get("GF_1", 0.0),
            dn_rates.get("GF_2", 0.0),
            dn_rates.get("DNp01_left", 0.0),
            dn_rates.get("DNp01_right", 0.0),
        )

        # ── 1. Steering intent from DNa bilateral differential ────────────────
        # BIOLOGICALLY GROUNDED: Left DNa drives left turn; Right DNa drives right turn.
        dna_diff = (dna_r - dna_l) / max(1.0, self.DNA_RATE_NORM_HZ)
        yaw_intent = float(np.clip(self.steer_gain * dna_diff, -1.0, 1.0))
        # Coordinated roll intent (bank into the turn)
        roll_intent = float(np.clip(0.6 * yaw_intent, -1.0, 1.0))

        # ── 2. Pitch / Forward-Backward intent from P9 vs MDN ──────────────────
        # BIOLOGICALLY GROUNDED: P9 drives forward walking; MDN drives backward walking.
        fwd_drive = p9_mean / max(1.0, self.P9_RATE_NORM_HZ)
        back_drive = mdn / max(1.0, self.MDN_RATE_NORM_HZ)
        net_pitch = (fwd_drive - back_drive) * self.pitch_gain
        pitch_intent = float(np.clip(net_pitch, -1.0, 1.0))

        # ── 3. Thrust intent and Giant Fiber Escape Surge ──────────────────────
        # BIOLOGICALLY GROUNDED: Giant Fiber firing produces massive wing depression / jump surge.
        escape_active = gf_burst >= self.GF_ESCAPE_THRESHOLD_HZ
        if escape_active:
            # Emergency upward surge proportional to GF activation
            escape_strength = min(1.0, gf_burst / 100.0)
            thrust_delta = float(self.max_thrust_surge_n * escape_strength)
        else:
            # Baseline thrust modulation correlated with forward drive
            thrust_delta = float(np.clip(1.5 * fwd_drive - 1.0 * back_drive, -2.0, 3.0))

        summary = {
            "p9_mean_hz": p9_mean,
            "dna_l_hz": dna_l,
            "dna_r_hz": dna_r,
            "mdn_hz": mdn,
            "gf_burst_hz": gf_burst,
            "escape_active": escape_active,
        }

        return {
            "thrust_delta": thrust_delta,
            "pitch_intent": pitch_intent,
            "roll_intent": roll_intent,
            "yaw_intent": yaw_intent,
            "escape_active": escape_active,
            "dn_summary": summary,
        }
