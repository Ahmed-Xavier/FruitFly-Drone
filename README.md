# Drosophila Brain <-> MuJoCo Skydio X2 Drone

An embodied-AI research platform investigating biological connectome control: a full **Drosophila melanogaster** (fruit-fly) brain model interfaced with a **MuJoCo Skydio X2** quadrotor drone.

---

## Architecture Overview

```text
FruitFly-Drone/
├── brain/                  # FlyWire v783 fruit-fly brain simulation
│   ├── connectome/         # FlyWire dataset loader & sparse weight tensors
│   ├── neurons/            # Vectorized AlphaLIF dynamics & spike generators
│   ├── simulation/         # LIF numerical constants & Hebbian parameters
│   ├── brain.py            # FlyBrain / Brain high-level simulation class
│   └── run_brain.py        # Standalone connectome runner
├── body/                   # MuJoCo Skydio X2 physical embodiment
│   ├── drone/              # Kinematics, actuators, sensors, and 3D assets
│   │   ├── assets/         # 3D meshes and rotor textures
│   │   ├── scene.xml       # MuJoCo scene & lighting definition
│   │   ├── x2.xml          # Skydio X2 kinematics, sensors, and actuators
│   │   ├── sensors.py      # IMU, gyro, accelerometer, pose reader
│   │   ├── motors.py       # 4-motor quadrotor aerodynamic mixer & limits
│   │   └── drone.py        # Drone simulation encapsulation
│   └── body.py             # Top-level Body embodiment abstraction
├── bridge/                 # Future brain-body interface (specification-only)
│   ├── README.md           # Interface boundary specifications & deferred mappings
│   └── __init__.py         # Package initialization
├── experiments/            # Standalone drone flight experiments
│   ├── drone_pid_flight.py # Autonomous 3D waypoint mission (cascaded PID)
│   ├── manual_control.py   # Interactive keyboard flight with live camera HUD
│   ├── height_stabilization_test.py # Closed-loop altitude PD test
│   └── README.md           # Flight experiments guide
├── main.py                 # Top-level entry point & architecture verification
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── README.md
```

### Module Responsibilities

- **`brain/`** — **FlyWire v783 Fruit-Fly Brain:**
  Simulates the **FlyWire v783 connectome** (138,639 Leaky Integrate-and-Fire neurons, 15,091,983 directed synapses) with alpha-function conductances, axonal conduction delays ($t_{\text{delay}} = 1.8\text{ ms}$), and online Hebbian plasticity. Completely decoupled from drone mechanics and independently runnable.

- **`body/`** — **MuJoCo Skydio X2:**
  Owns the physics simulation, XML definitions, 3D assets, rigid-body state, IMU sensors (gyroscope, accelerometer, quaternion), and the 4-rotor actuator mixer. Completely decoupled from neural logic and independently runnable.

- **`bridge/`** — **Future Brain-Body Interface:**
  Defines the architectural boundary between drone telemetry and neural circuits. Currently **specification-only** and intentionally **not implemented yet**.

- **`experiments/`** — **Standalone Experiments:**
  Flight scenarios, PID waypoint tracking, altitude stabilization tests, and interactive teleoperation tools for the Skydio X2 quadrotor.

- **`main.py`** — **Top-Level Entry Point:**
  Initializes and verifies the physical embodiment (`body/`) and optionally the connectome (`brain/`), demonstrating modular readiness without synthetic control bridges.

---

## Information Flow Pipeline

```text
Drone sensors
      ↓
Sensory encoding
      ↓
FlyWire brain
      ↓
Descending neurons
      ↓
Motor decoding
      ↓
Skydio X2 actuators
```

### Sensory-Driven Philosophy

The project does **NOT** give the fly high-level commands such as `"fly forward"` or `"turn left"`.

The intended architecture is strictly **sensory-driven**:
1. The drone provides physical sensory information (accelerations, angular velocities, optic flow / vision).
2. That information is encoded into neural inputs (synaptic currents, Poisson spike rates).
3. The biological network processes the activity through the whole-brain recurrent connectome.
4. Activity from descending neurons (DNs) is decoded into low-level flight control signals (motor thrusts / torques).

> **IMPORTANT:**
> The actual sensory and motor mappings are **not implemented yet**. They require detailed neurobiological literature grounding, response curve calibration, and empirical investigation, and will not be arbitrarily hardcoded.

---

## Installation

```bash
pip install -r requirements.txt
```

*(For GPU acceleration with PyTorch CUDA, ensure a CUDA-enabled PyTorch build matching your hardware is installed).*

---

## Running the Project

### 1. Main Entry Point
```bash
# Verify drone physics embodiment:
python main.py

# Verify both Body and Brain (loads 138k connectome):
python main.py --with-brain

# Run with interactive 3D viewer:
python main.py --viewer
```

### 2. Standalone Fruit-Fly Brain Connectome
```bash
# Sugar stimulus (gustatory receptor neurons):
python brain/run_brain.py --stimulus sugar --steps 200

# Visual looming threat (LC4 -> Giant Fiber escape circuit):
python brain/run_brain.py --stimulus lc4 --steps 200

# Forward walking circuit stimulus (P9):
python brain/run_brain.py --stimulus p9 --steps 200

# Spontaneous activity (no external stimulus):
python brain/run_brain.py --stimulus none --steps 100
```

### 3. Drone Flight Experiments
```bash
# Autonomous 3D waypoint navigation (cascaded PID controller):
python experiments/drone_pid_flight.py

# Manual keyboard flight + live POV camera:
python experiments/manual_control.py

# Vertical altitude stabilization test:
python experiments/height_stabilization_test.py
```