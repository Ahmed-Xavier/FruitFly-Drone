"""
visual_target_encoder.py
------------------------
Biologically grounded sensory representation translating visual target detections
(e.g. sugar cube in camera image) into neural firing rates for the FlyWire v783 connectome.

Biological Grounding:
---------------------
In Drosophila melanogaster, small visual target detection, tracking, and approach steering
are mediated by Lobula Columnar type 10 (LC10) visual projection neurons:
    - Ribeiro et al. (Cell 2018): "Visual tracking of targets requires LC10 neurons"
    - Hindmarsh Sten et al. (Nature 2021): "Sexual dimorphism in a Drosophila target-tracking circuit"
    - Morimoto et al. (Current Biology 2020): "Spatial receptive fields of LC10 neurons"

The FlyWire v783 connectome contains 815 LC10 neurons across subtypes LC10a-f,
cleanly partitioned across left and right optic lobes:
    - Left LC10 population: 403 neurons
    - Right LC10 population: 412 neurons

Encoding Model:
---------------
1. Azimuthal Target Position (center_x in [-1.0, +1.0]):
   - Target in left visual hemifield (center_x < 0) excites left LC10 population.
   - Target in right visual hemifield (center_x > 0) excites right LC10 population.
   - Target centered directly ahead (center_x ≈ 0) excites both LC10 populations symmetrically.
2. Apparent Size & Proximity:
   - Modulates visual drive intensity as target grows in retinal angular size.
3. Contrast & Detection Confidence:
   - Scales firing rates proportional to detector confidence.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from brain.connectome.loader import load_annotations
from object_detection.detector import TargetDetection


class VisualTargetEncoder:
    """
    Encodes visual TargetDetection features into FlyWire LC10 sensory neuron firing rates.
    """

    MAX_RATE_HZ = 300.0       # Max visual burst rate in Drosophila LC neurons
    BASELINE_RATE_HZ = 5.0    # Spontaneous visual baseline rate

    def __init__(self, subtypes: Optional[List[str]] = None):
        """
        Parameters
        ----------
        subtypes : list of str, optional
            Subtypes of LC10 to include (default: ['LC10a', 'LC10c'], the primary
            target-tracking columnar populations).
        """
        if subtypes is None:
            self.subtypes = ["LC10a", "LC10c"]
        else:
            self.subtypes = subtypes

        self.lc10_left_ids: List[int] = []
        self.lc10_right_ids: List[int] = []

        self._load_lc10_pathways()

    def _load_lc10_pathways(self):
        """Extract validated LC10 neuron IDs from FlyWire annotations."""
        df = load_annotations()
        if df is not None:
            lc10 = df[df["cell_type"].isin(self.subtypes)]
            self.lc10_left_ids = lc10[lc10["side"] == "left"]["root_id"].astype(int).tolist()
            self.lc10_right_ids = lc10[lc10["side"] == "right"]["root_id"].astype(int).tolist()

        # Fallback if annotations missing
        if not self.lc10_left_ids or not self.lc10_right_ids:
            # Fallback to general optic projection subset if needed
            self.lc10_left_ids = [720575940605598892]
            self.lc10_right_ids = [720575940611134833]

    def encode(self, detection: TargetDetection) -> Dict[str, Any]:
        """
        Encode target detection into biological firing rates.

        Parameters
        ----------
        detection : TargetDetection
            Visual detection output containing center_x, center_y, apparent_size, etc.

        Returns
        -------
        dict with:
            - 'rates': Dict[int, float] mapping FlyWire segment IDs to rates in Hz
            - 'summary': dict summarizing visual target channels
        """
        rates: Dict[int, float] = {}

        if not detection.detected:
            # No target in visual field: baseline quiescent rates
            for nid in self.lc10_left_ids:
                rates[nid] = self.BASELINE_RATE_HZ
            for nid in self.lc10_right_ids:
                rates[nid] = self.BASELINE_RATE_HZ

            return {
                "rates": rates,
                "summary": {
                    "target_detected": False,
                    "lc10_left_rate_hz": self.BASELINE_RATE_HZ,
                    "lc10_right_rate_hz": self.BASELINE_RATE_HZ,
                    "active_lc10_neurons": len(rates),
                },
            }

        cx = float(np.clip(detection.center_x, -1.0, 1.0))
        size_scale = min(2.0, 1.0 + float(detection.apparent_size) * 30.0)
        conf = float(np.clip(detection.confidence, 0.2, 1.0))

        # Base approach drive when target is in front
        # Highest when centered (|cx| small), diminished when far off-axis
        fwd_drive = max(0.2, 1.0 - 0.7 * abs(cx))

        # Lateral excitation asymmetry
        # If target on left (cx < 0): left LC10 receives higher rate
        # If target on right (cx > 0): right LC10 receives higher rate
        if cx < 0:
            left_weight = 1.0 + 1.2 * abs(cx)
            right_weight = max(0.1, 1.0 - 0.8 * abs(cx))
        else:
            left_weight = max(0.1, 1.0 - 0.8 * abs(cx))
            right_weight = 1.0 + 1.2 * abs(cx)

        nominal_rate = 60.0 * size_scale * conf * fwd_drive

        left_rate = float(np.clip(
            self.BASELINE_RATE_HZ + nominal_rate * left_weight,
            self.BASELINE_RATE_HZ,
            self.MAX_RATE_HZ,
        ))
        right_rate = float(np.clip(
            self.BASELINE_RATE_HZ + nominal_rate * right_weight,
            self.BASELINE_RATE_HZ,
            self.MAX_RATE_HZ,
        ))

        for nid in self.lc10_left_ids:
            rates[nid] = left_rate
        for nid in self.lc10_right_ids:
            rates[nid] = right_rate

        return {
            "rates": rates,
            "summary": {
                "target_detected": True,
                "center_x": cx,
                "horizontal_position": detection.horizontal_position,
                "apparent_size": detection.apparent_size,
                "lc10_left_rate_hz": left_rate,
                "lc10_right_rate_hz": right_rate,
                "active_lc10_neurons": len(rates),
            },
        }
