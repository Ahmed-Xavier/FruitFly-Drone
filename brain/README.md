# Fruit-fly Brain — Standalone Simulation

Minimal standalone simulation of the **FlyWire v783** *Drosophila melanogaster*
connectome: **138,639 LIF neurons · 15,091,983 directed weighted synapses**.

Stripped from [fly-brain-full](../fly-brain-full) — all fly-body, visualization,
and analysis code removed. Only the neural network remains.

---

## Project Structure

```
brain/
├── config/
│   ├── __init__.py
│   └── params.py              # Biological LIF & plasticity parameters
├── neurons/
│   └── __init__.py            # Neuron dynamics & spike generators
├── connections/
│   └── __init__.py            # Connectome loaders & synapse matrix builder
├── model/
│   ├── __init__.py
│   ├── neuron_model.py        # PyTorch LIF tensor implementations
│   ├── connectome_loader.py   # FlyWire parquet loader & pickle cache manager
│   └── brain.py               # FlyBrain simulation class & DN definitions
├── data/                      # FlyWire v783 dataset & tensor caches
├── run_brain.py               # Standalone runner with stimulus CLI
├── requirements.txt           # Minimal dependencies
└── README.md
```

## What was retained

| Component | File | Why |
|---|---|---|
| LIF neuron model | `neurons/`, `model/neuron_model.py` | Core neural dynamics |
| Connectome loader | `connections/`, `model/connectome_loader.py` | Loads FlyWire data & builds weights |
| FlyBrain API | `model/brain.py` | Clean step-by-step interface |
| Model parameters | `config/params.py` | LIF + plasticity constants |
| Connectome CSV | `data/2025_Completeness_783.csv` | Neuron table (138,639 rows) |
| Connectome parquet | `data/2025_Connectivity_783.parquet` | Synaptic edges (15M rows) |
| Annotations TSV | `data/flywire_annotations.tsv` | Neuron type labels |
| Hebbian plasticity | inside `brain.py` | Inherent tissue property |
| DN neuron IDs | inside `brain.py` | Future motor output interface |
| Stimulus definitions | inside `brain.py` | Known input neuron IDs |

## What was removed

| Removed | Reason |
|---|---|
| `fly_embodied.py` | Requires flygym / NeuroMechFly v2 body |
| `brain_body_bridge.py` | Body-specific DN→motor decoding |
| `two_flies.py`, `fly_alive.py`, `fly_walk.py`, `flight.py` | Body experiments |
| `visual_system.py`, `olfactory.py`, `gustatory.py`, `somatosensory.py`, `vocalization.py` | Fly-body sensor peripherals |
| `brain_monitor.py` | Real-time visualization (pygame) |
| `looming_arena.py`, `procedural_arena.py` | MuJoCo fly environment |
| `consciousness.py` | Analysis-only (not needed for sim) |
| `generate_paper*.py`, `analyze_*.py`, `compare_*.py` | Paper generation |
| `code/run_brian2_cuda.py`, `code/run_nestgpu.py` | Alternative backends |
| `code/benchmark.py`, `main.py` | Benchmark harness |
| `demo.mp4`, `demo_preview.gif` | Demo media |
| `paper_*.pdf`, `paper_figures/` | Paper assets |
| `consciousness_history/` | Experimental session data |
| `data/plastic_weights*.pt` | Trained weights (optional, can add back) |
| `data/eye_*.png` | Compound eye renders |
| `data/benchmark-results.csv` | Benchmark data |
| `docs/`, `scripts/setup_WSL_CUDA.sh` | Docs and setup scripts |

---

## Connectome

The FlyWire v783 connectome data lives in `data/` (symlinked or copied from
`fly-brain-full/data/`):

| File | Size | Contents |
|---|---|---|
| `2025_Completeness_783.csv` | 3.4 MB | 138,639 neurons with FlyWire segment IDs |
| `2025_Connectivity_783.parquet` | 100 MB | 15,091,983 directed weighted synaptic edges |
| `flywire_annotations.tsv` | 32 MB | Neuron type annotations (cell_type, super_class…) |

The **`Excitatory x Connectivity`** column in the parquet encodes sign:
- Positive values → excitatory (acetylcholine / glutamate)
- Negative values → inhibitory (GABA / glycine)

---

## Neural model

**Leaky Integrate-and-Fire (LIF)** with alpha-function synapses:

```
dv/dt = (v_rest - v + g) / tau_mem      membrane potential
dg/dt = -g / tau_syn                     synaptic conductance
```

| Parameter | Value | Source |
|---|---|---|
| Timestep | 0.1 ms (10 kHz) | |
| tau_mem | 20 ms | Kakaria & de Bivort 2017 |
| tau_syn | 5 ms | Jürgensen et al. |
| v_rest / v_reset | −52 mV | Kakaria & de Bivort 2017 |
| v_threshold | −45 mV | Kakaria & de Bivort 2017 |
| t_refrac | 2.2 ms | Lazar et al. |
| t_delay | 1.8 ms | Paul et al. 2015 |

**Hebbian plasticity** (runs on every synapse every 10 steps):
```
dW_ij = eta * (r_i * r_j) - alpha * W_ij
eta = 1e-4,  alpha = 1e-7
```

---

## Install

```bash
pip install torch numpy pandas pyarrow scipy
# or with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyarrow scipy
```

---

## Run

```bash
cd C:\My files\MuJoCo\brain

# Default: 200 steps, sugar GRN stimulus, GPU
python run_brain.py

# P9 forward-walking stimulus, 500 steps
python run_brain.py --stimulus p9 --steps 500

# LC4 looming (escape) stimulus
python run_brain.py --stimulus lc4 --steps 200

# No stimulus (spontaneous activity)
python run_brain.py --stimulus none --steps 100

# CPU only (no CUDA required, slower)
python run_brain.py --cpu --steps 50
```

---

## Python API

```python
from brain.model import FlyBrain

brain = FlyBrain(device='cuda')   # or 'cpu'

# Inspect
print(brain.num_neurons)          # 138639
print(brain.timestep_ms)          # 0.1
print(brain.get_dn_groups())      # forward / turn_L / turn_R / escape / …
print(brain.get_stimulus_info())  # available named stimuli

# Set a named stimulus
brain.set_stimulus('sugar')

# Inject arbitrary rates by FlyWire ID
brain.set_input_rates([720575940627652358], [150.0])

# Step
output = brain.step()
# or with inline input:
output = brain.step({'stimulus': 'lc4'})

# Read output
output['spikes']           # np.ndarray (138639,)  — binary
output['dn_spikes']        # {name: 0|1}           — descending neurons
output['dn_rates']         # {name: Hz}            — sliding-window rates
output['population_rates'] # {group: Hz}           — forward/escape/…
output['time_ms']          # float                 — simulated time
```

---

## Inspect neuron activity

```python
from brain.model import FlyBrain, DN_NEURONS, DN_GROUPS
import numpy as np

brain = FlyBrain(device='cpu')
brain.set_stimulus('p9')

spike_history = []
for _ in range(1000):
    out = brain.step()
    spike_history.append(out['spikes'])

arr = np.stack(spike_history)            # (1000, 138639)
active = np.where(arr.sum(0) > 0)[0]    # indices of neurons that fired
print(f"Active: {len(active)} neurons")

# Map back to FlyWire IDs
firing_ids = [brain.i2flyid[i] for i in active]
```

---

## Future interface: Drone → Brain → Drone

```
Drone sensors  (IMU, camera, sonar)
      ↓
sensory encoding  (to Hz firing rates)
      ↓  brain.set_input_rates(flyids, rates)
fruit-fly brain  (LIF connectome step)
      ↓  output['population_rates']['forward'], ['escape'], …
motor decoding  (DN rates → motor commands)
      ↓
drone controller
```

The `brain.step()` return dict already exposes all DN group rates needed
for this pipeline. No changes to the neural network are required.
