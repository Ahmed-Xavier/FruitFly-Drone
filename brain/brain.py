"""
brain.py
--------
Standalone fruit-fly brain simulation.

Wraps the FlyWire v783 LIF connectome model (138,639 neurons, 15M synapses)
with a clean step-by-step API and exposes all information needed for
inspection and future drone integration.

Public API:
    brain = Brain()   # or FlyBrain()
    output = brain.step(input_data)

input_data : dict with optional keys:
    'rates'   : {flyid: Hz, ...}   arbitrary neuron firing rates
    'stimulus': str                 named stimulus from STIMULI dict

output : dict with keys:
    'spikes'          : np.ndarray shape (num_neurons,)  - 0/1 per neuron
    'dn_spikes'       : dict {dn_name: 0|1}              - descending neurons
    'dn_rates'        : dict {dn_name: Hz}               - smoothed DN rates
    'population_rates': dict {group: mean_rate}          - DN group averages
    'time_ms'         : float                            - simulated time (ms)
"""

from typing import Dict, List, Optional, Union
from collections import deque
from pathlib import Path
import numpy as np
import torch

from .neurons import TorchModel
from .connectome import load_connectome
from .simulation import MODEL_PARAMS, DT, HEBB_BATCH, HEBB_ETA, HEBB_ALPHA

# ─────────────────────────────────────────────────────────────────────────────
# Descending Neuron (DN) definitions (FlyWire v783 segment IDs)
# ─────────────────────────────────────────────────────────────────────────────

DN_NEURONS = {
    # Forward walking (P9 / oDN1)
    "P9_left":        720575940627652358,
    "P9_right":       720575940635872101,
    "P9_oDN1_left":   720575940626730883,
    "P9_oDN1_right":  720575940620300308,
    # Sustained turning (DNa01)
    "DNa01_left":     720575940644438551,
    "DNa01_right":    720575940627787609,
    # Transient turning (DNa02)
    "DNa02_left":     720575940604737708,
    "DNa02_right":    720575940629327659,
    # Backward walking (MDN - Moonwalker Descending Neuron)
    "MDN_1":          720575940616026939,
    "MDN_2":          720575940631082808,
    # Escape / fast takeoff (Giant Fiber)
    "GF_1":           720575940626081498,
    "GF_2":           720575940628359487,
    # Antennal grooming (aDN1)
    "aDN1_left":      720575940614418659,
    "aDN1_right":     720575940623769165,
    # Proboscis extension / feeding motor neurons (MN9)
    "MN9_left":       720575940639908170,
    "MN9_right":      720575940625695026,
    # Wing coordination
    "DNp01_left":     720575940608681123,
    "DNp01_right":    720575940626388417,
}

DN_GROUPS = {
    "forward":  ["P9_left", "P9_right", "P9_oDN1_left", "P9_oDN1_right"],
    "turn_L":   ["DNa01_left", "DNa02_left"],
    "turn_R":   ["DNa01_right", "DNa02_right"],
    "backward": ["MDN_1", "MDN_2"],
    "escape":   ["GF_1", "GF_2"],
    "groom":    ["aDN1_left", "aDN1_right"],
    "feed":     ["MN9_left", "MN9_right"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Named stimuli (FlyWire segment IDs + baseline firing rate)
# ─────────────────────────────────────────────────────────────────────────────

STIMULI = {
    "sugar": {
        "neurons": [
            720575940624963786, 720575940630233916, 720575940637568838,
            720575940638202345, 720575940617000768, 720575940630797113,
            720575940632889389, 720575940621754367, 720575940621502051,
            720575940640649691, 720575940639332736, 720575940616885538,
            720575940639198653, 720575940639259967, 720575940617937543,
            720575940632425919, 720575940633143833, 720575940612670570,
            720575940628853239, 720575940629176663, 720575940611875570,
        ],
        "rate": 200.0,
        "description": "Sugar GRNs - 21 neurons at 200 Hz",
    },
    "p9": {
        "neurons": [720575940627652358, 720575940635872101],
        "rate": 100.0,
        "description": "P9 forward walking - 2 neurons at 100 Hz",
    },
    "lc4": {
        "neurons": [
            720575940605598892, 720575940611134833, 720575940612580977,
            720575940613256863, 720575940613260959, 720575940614914107,
            720575940615462587, 720575940617176321, 720575940617266722,
            720575940618807105, 720575940620795728, 720575940622108001,
            720575940624017251, 720575940625038090, 720575940625934973,
            720575940625991043, 720575940626605200, 720575940626626895,
            720575940628454522, 720575940628462340, 720575940630851036,
            720575940638496720, 720575940603637438, 720575940610522009,
            720575940612093351, 720575940612323025, 720575940612380723,
            720575940612498129, 720575940612518055, 720575940612968421,
            720575940613609484, 720575940613638041, 720575940614572742,
            720575940614582946, 720575940615053580, 720575940615127227,
            720575940615232217, 720575940615575007, 720575940616066705,
            720575940616713355, 720575940617026260, 720575940617348379,
            720575940618002644, 720575940618234704, 720575940618234715,
            720575940618266459, 720575940618267227, 720575940618275520,
            720575940618312606, 720575940618676440, 720575940618709158,
            720575940618723749, 720575940619397542, 720575940620314221,
            720575940620314612, 720575940620731380, 720575940620903551,
            720575940621145821, 720575940621522458, 720575940621753579,
            720575940622330582, 720575940622531767, 720575940622939836,
            720575940624111763, 720575940624790781, 720575940624856762,
            720575940625841351, 720575940625845447, 720575940625906702,
            720575940625932421, 720575940626553596, 720575940626916936,
            720575940627519107, 720575940628064260, 720575940628081541,
            720575940628419527, 720575940628518400, 720575940628599895,
            720575940628606713, 720575940628699560, 720575940628891863,
            720575940629753807, 720575940629964591, 720575940630154660,
            720575940630484495, 720575940630998339, 720575940631032657,
            720575940631338271, 720575940632475449, 720575940632715234,
            720575940632769180, 720575940633013355, 720575940633218863,
            720575940633580384, 720575940634517856, 720575940635835967,
            720575940636957006, 720575940638456227, 720575940639817947,
            720575940640612480, 720575940641213824, 720575940645821316,
            720575940649229433, 720575940652611745,
        ],
        "rate": 200.0,
        "description": "LC4 looming (visual threat) - 104 neurons at 200 Hz",
    },
    "jo": {
        "neurons": [
            720575940645106376, 720575940615272415, 720575940619869120,
            720575940620257345, 720575940620382889, 720575940630834683,
            720575940632449619, 720575940634020508, 720575940605530302,
            720575940607140035, 720575940608742409, 720575940615590843,
            720575940620410177, 720575940621870618, 720575940622344170,
            720575940623298559, 720575940626042149, 720575940627379333,
            720575940630080071, 720575940632128031, 720575940632307527,
            720575940634820703,
        ],
        "rate": 300.0,
        "description": "Johnston's organ touch - 22 neurons at 300 Hz",
    },
}


class FlyBrain:
    """
    Standalone fruit-fly brain simulation.

    Wraps the FlyWire v783 LIF connectome (138,639 neurons, 15M synapses)
    with a clean step-by-step interface for embodied-AI integration.
    """

    def __init__(
        self,
        device: str = "cuda",
        plasticity: bool = True,
        plastic_path: Optional[Union[str, Path]] = None,
        dn_window_ms: float = 50.0,
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.dt = DT          # ms
        self.time_ms = 0.0
        self._plasticity_enabled = plasticity

        # ── Load connectome ───────────────────────────────────────────────
        self.flyid2i, self.i2flyid, weights, self.num_neurons = load_connectome(
            device=self.device
        )

        # ── Build LIF model ───────────────────────────────────────────────
        self.model = TorchModel(
            batch=1, size=self.num_neurons,
            dt=self.dt, params=MODEL_PARAMS,
            weights=weights, device=self.device,
        )
        self.state = self.model.state_init()

        # ── Rate tensor ───────────────────────────────────────────────────
        self.rates = torch.zeros(1, self.num_neurons, device=self.device)

        # ── Map DN FlyWire IDs -> tensor indices ─────────────────────────
        self.dn_indices: Dict[str, int] = {}
        for name, flyid in DN_NEURONS.items():
            if flyid in self.flyid2i:
                self.dn_indices[name] = self.flyid2i[flyid]

        # ── Pre-map stimulus neuron IDs -> tensor indices ────────────────
        self.stim_indices: Dict[str, List[int]] = {}
        for sname, sinfo in STIMULI.items():
            self.stim_indices[sname] = [
                self.flyid2i[nid] for nid in sinfo["neurons"] if nid in self.flyid2i
            ]

        # ── DN sliding-window rate estimator ─────────────────────────────
        self._dn_window_steps = int(dn_window_ms / self.dt)
        self._dn_dt_s = self.dt / 1000.0
        self._dn_buffers: Dict[str, deque] = {
            name: deque(maxlen=self._dn_window_steps)
            for name in DN_NEURONS
        }
        self.dn_rates: Dict[str, float] = {name: 0.0 for name in DN_NEURONS}

        # ── Hebbian plasticity ────────────────────────────────────────────
        if plasticity:
            self._init_plasticity(plastic_path)

        print(f"[FlyBrain] {self.num_neurons:,} neurons | device={self.device} "
              f"| dt={self.dt} ms | plasticity={plasticity}")
        print(f"[FlyBrain] DN neurons mapped: {len(self.dn_indices)}/{len(DN_NEURONS)}")

    def _init_plasticity(self, plastic_path: Optional[Union[str, Path]] = None):
        w = self.model.weights
        self._row_ptr   = w.crow_indices()
        self._col_idx   = w.col_indices()
        self._syn_vals  = w.values()       # mutable in-place view
        self._sign_mask = torch.sign(self._syn_vals)
        self._abs_orig  = self._syn_vals.abs().clone()

        row_lengths  = self._row_ptr[1:] - self._row_ptr[:-1]
        self._post_idx = torch.repeat_interleave(
            torch.arange(self.num_neurons, device=self.device), row_lengths
        )
        self._spike_acc  = torch.zeros(self.num_neurons, device=self.device)
        self._hebb_count = 0

        max_mag = 3.0 * self._abs_orig
        self._clamp_min = torch.where(self._sign_mask < 0, -max_mag, torch.zeros_like(max_mag))
        self._clamp_max = torch.where(self._sign_mask > 0, max_mag, torch.zeros_like(max_mag))

        if plastic_path and Path(plastic_path).exists():
            saved = torch.load(plastic_path, map_location=self.device, weights_only=True)
            if saved.shape == self._syn_vals.shape:
                self._syn_vals.copy_(saved)
                self._sign_mask = torch.sign(self._syn_vals)
                print(f"[FlyBrain] Loaded plastic weights from {plastic_path}")

        print(f"[FlyBrain] Hebbian plasticity: {len(self._syn_vals):,} synapses")

    def _hebb_update(self):
        avg = self._spike_acc / HEBB_BATCH
        self._spike_acc.zero_()
        pre  = avg[self._col_idx]
        post = avg[self._post_idx]
        dW   = HEBB_ETA * pre * post * self._sign_mask - HEBB_ALPHA * self._syn_vals
        self._syn_vals.add_(dW)
        self._syn_vals.clamp_(min=self._clamp_min, max=self._clamp_max)

    def clear_input(self):
        """Zero all input firing rates."""
        self.rates.zero_()

    def set_stimulus(self, name: Optional[str]):
        """Set a named sensory stimulus. Pass None to clear all input."""
        self.rates.zero_()
        if name and name in STIMULI:
            idx = self.stim_indices.get(name, [])
            if idx:
                self.rates[0, idx] = STIMULI[name]["rate"]

    def set_input_rates(self, flyids: List[int], rates_hz: List[float]):
        """Inject arbitrary firing rates into specific neurons by FlyWire segment ID."""
        for fid, r in zip(flyids, rates_hz):
            if fid in self.flyid2i:
                self.rates[0, self.flyid2i[fid]] = r

    def set_input_by_index(self, indices: np.ndarray, rates_hz: np.ndarray):
        """Inject firing rates directly by tensor indices."""
        self.rates[0, indices] = torch.as_tensor(rates_hz, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def step(self, input_data: Optional[dict] = None) -> dict:
        """Advance the neural simulation by one timestep (dt = 0.1 ms)."""
        if input_data:
            if "stimulus" in input_data:
                self.set_stimulus(input_data["stimulus"])
            if "rates" in input_data:
                fids = list(input_data["rates"].keys())
                rhz  = list(input_data["rates"].values())
                self.set_input_rates(fids, rhz)

        # Physics step
        cond, dbuf, spk, v, ref = self.state
        self.state = self.model(self.rates, cond, dbuf, spk, v, ref)
        spikes_t = self.state[2]   # shape (1, num_neurons)

        # Plasticity
        if self._plasticity_enabled:
            self._spike_acc += spikes_t.squeeze(0)
            self._hebb_count += 1
            if self._hebb_count >= HEBB_BATCH:
                self._hebb_update()
                self._hebb_count = 0

        self.time_ms += self.dt
        spikes_np = spikes_t.squeeze(0).cpu().numpy()

        dn_spikes = {
            name: float(spikes_t[0, idx].item())
            for name, idx in self.dn_indices.items()
        }

        for name in self.dn_indices:
            self._dn_buffers[name].append(dn_spikes[name])
            buf = self._dn_buffers[name]
            n = len(buf)
            self.dn_rates[name] = sum(buf) / (n * self._dn_dt_s) if n > 0 else 0.0

        pop_rates = {}
        for group, members in DN_GROUPS.items():
            vals = [self.dn_rates[m] for m in members if m in self.dn_rates]
            pop_rates[group] = float(np.mean(vals)) if vals else 0.0

        return {
            "spikes":           spikes_np,
            "dn_spikes":        dn_spikes,
            "dn_rates":         dict(self.dn_rates),
            "population_rates": pop_rates,
            "time_ms":          self.time_ms,
        }

    @property
    def timestep_ms(self) -> float:
        return self.dt

    def get_neuron_ids(self) -> List[int]:
        return [self.i2flyid[i] for i in range(self.num_neurons)]

    def get_tensor_index(self, flyid: int) -> Optional[int]:
        return self.flyid2i.get(flyid)

    def get_dn_info(self) -> dict:
        return {
            name: {
                "flyid": DN_NEURONS[name],
                "tensor_idx": self.dn_indices.get(name),
                "rate_hz": self.dn_rates.get(name, 0.0),
            }
            for name in DN_NEURONS
        }

    def get_dn_groups(self) -> dict:
        return dict(DN_GROUPS)

    def get_stimulus_info(self) -> dict:
        return {
            name: {
                "description": info["description"],
                "n_neurons": len(self.stim_indices.get(name, [])),
                "rate_hz": info["rate"],
            }
            for name, info in STIMULI.items()
        }

    def save_plastic_weights(self, path: Path):
        if not self._plasticity_enabled:
            raise RuntimeError("Plasticity is disabled - nothing to save.")
        torch.save(self._syn_vals.detach().cpu(), path)
        print(f"[FlyBrain] Saved plastic weights -> {path}")


# Alias Brain to FlyBrain for the requested high-level interface
Brain = FlyBrain
