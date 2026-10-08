# Drosophila Brain <-> MuJoCo Skydio X2 Drone

An embodied-AI research platform investigating biological connectome control: a full **Drosophila melanogaster** (fruit-fly) brain model interfaced with a **MuJoCo Skydio X2** quadrotor drone.

![Drone Demo](assets/demo.gif)

---

## Architecture Overview

```text
project/
├── brain/
│   ├── connectome/         # FlyWire v783 dataset loader & sparse weight tensors
│   ├── neurons/            # Vectorized AlphaLIF, Poisson spike generators
│   ├── simulation/         # LIF numerical parameters & plasticity constants
│   ├── data/               # Connectome parquet/CSV and compiled tensor caches
│   ├── brain.py            # FlyBrain / Brain simulation class
│   └── run_brain.py        # Standalone connectome runner
│
├── body/
│   ├── drone/
│   │   ├── assets/         # 3D meshes and rotor textures
│   │   ├── scene.xml       # MuJoCo scene & lighting definition
│   │   ├── x2.xml          # Skydio X2 kinematics, sensors, and actuators
│   │   ├── sensors.py      # IMU, gyro, accelerometer, pose reader
│   │   ├── motors.py       # 4-motor quadrotor aerodynamic mixer & safety limits
│   │   └── drone.py        # Drone simulation encapsulation
│   └── body.py             # Top-level Body embodiment abstraction
│
├── bridge/
│   ├── README.md           # Specification of brain <-> body translation pipeline
│   └── __init__.py         # NOT IMPLEMENTED YET (boundary placeholder)
│
├── experiments/
│   ├── drone_pid_flight.py # Autonomous 3D waypoint mission (cascaded PID)
│   ├── manual_control.py   # Interactive keyboard flight with live camera HUD
│   ├── height_stabilization_test.py # Closed-loop altitude PD test
│   └── README.md           # Flight experiments guide
│
├── main.py                 # Top-level application entry point & integration loop
├── requirements.txt        # Python package dependencies
└── README.md
```

### Module Responsibilities

- **`brain/`** — **Biological/Neuromorphic Fly Brain:**
  Simulates the **FlyWire v783 connectome** (138,639 Leaky Integrate-and-Fire neurons, 15,091,983 directed synapses) with alpha-function conductances, axonal conduction delays ($t_{\text{delay}} = 1.8\text{ ms}$), and online Hebbian plasticity. Completely decoupled from drone mechanics.

- **`body/`** — **MuJoCo Skydio X2 Embodiment:**
  Owns the physics simulation, XML definitions, 3D assets, rigid-body state, IMU sensors (gyroscope, accelerometer, quaternion), and the 4-rotor actuator mixer. Completely decoupled from neural logic.

- **`bridge/`** — **Interface Between Brain and Body:**
  *NOT IMPLEMENTED YET.* Reserved for the scientific translation layer:
  $$\text{Sensors} \xrightarrow{\text{Encoding}} \text{Spikes} \xrightarrow{\text{Connectome}} \text{Neural Activity} \xrightarrow{\text{Decoding}} \text{Motor Commands}$$

- **`main.py`** — **Top-Level Application:**
  Coordinates component initialization and defines the architectural scaffold where the integration loop will live.

---

## Installation

```bash
pip install -r requirements.txt
```

*(For GPU acceleration with PyTorch CUDA, ensure CUDA-enabled PyTorch is installed).*

---

## Running the Project

### 1. Main Application & Verification
```bash
# Initialize and verify drone embodiment:
python main.py

# Initialize both Body and Brain (loads 138k connectome):
python main.py --with-brain

# Run with interactive 3D viewer:
python main.py --viewer
```

### 2. Drone Flight Scenarios
```bash
# Autonomous 3D waypoint navigation (PID controller):
python experiments/drone_pid_flight.py

# Manual keyboard flight + live POV camera (AZERTY: Z/S/Q/D/A/E/Space/X):
python experiments/manual_control.py

# Vertical altitude stabilization test:
python experiments/height_stabilization_test.py
```

### 3. Standalone Fruit-Fly Brain Connectome
```bash
# Visual looming threat (LC4 -> Giant Fiber escape circuit):
python brain/run_brain.py --stimulus lc4 --steps 300

# Gustatory sugar stimulus:
python brain/run_brain.py --stimulus sugar --steps 200

# Forward walking circuit stimulus (P9):
python brain/run_brain.py --stimulus p9 --steps 500
```