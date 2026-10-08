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
├── bridge/                 # Brain-body sensory-motor interface layer
│   ├── sensory_encoder.py  # IMU/velocity -> Johnston's Organ, LC4, HS/VS firing rates
│   ├── visual_target_encoder.py # Target detection -> LC10 visual projection neurons
│   ├── motor_decoder.py    # Descending neuron activity -> motor flight intent
│   ├── controller.py       # Safety envelopes, attitude PD stabilization, mixer
│   ├── synchronizer.py     # Multiscale timestep coordinator & telemetry logging
│   ├── __init__.py         # Package exports
│   └── README.md           # Interface boundary specifications & biological grounding
├── object_detection/       # Embodied vision capture and target localization
│   ├── camera_interface.py # MuJoCoCamera, PhysicalCamera (CSI/USB), MockCamera
│   ├── detector.py         # WhiteSugarDetector & TargetDetection dataclass
│   ├── target.py           # TargetConfig & MuJoCo target kinematics management
│   ├── __init__.py         # Package exports
│   └── README.md           # Vision pipeline documentation & hardware guide
├── experiments/            # Standalone and integrated flight experiments
│   ├── sugar_approach.py       # Controlled visual target approach battery (A-E)
│   ├── brain_drone_hover.py    # Closed-loop connectome hover experiment
│   ├── brain_drone_response.py # Controlled disturbance & recovery experiment
│   ├── brain_drone_escape.py   # Visual looming threat & escape reflex experiment
│   ├── drone_pid_flight.py     # Standalone autonomous 3D waypoint mission (PID)
│   ├── manual_control.py       # Standalone interactive keyboard flight
│   ├── height_stabilization_test.py # Standalone altitude PD test
│   └── README.md           # Flight experiments guide
├── tests/                  # Lightweight automated test suite (CPU-compatible)
│   ├── test_sugar_vision.py    # Vision pipeline, detection, LC10 encoding, regressions
│   └── __init__.py
├── main.py                 # Unified application launcher
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── README.md
```

---

## Information Flow Pipelines

### 1. Kinematic & Multimodal Sensor Pipeline
```text
                MUJOCO DRONE
                     |
              +------+------+
              |             |
             IMU          Looming Threat (LC4)
              |             |
              v             v
       SensoryEncoder (bridge/sensory_encoder.py)
              |
              v [Poisson Rates in Hz]
       FLYWIRE CONNECTOME (brain/brain.py)
              | [138,639 LIF Neurons · 15M Synapses]
              |
              v [Smoothed DN Firing Rates]
       MotorDecoder (bridge/motor_decoder.py)
              |
              v [Normalized Motor Intent: Thrust, Roll, Pitch, Yaw]
       DroneFlightController (bridge/controller.py)
              | [Attitude PD, Slew Rate Limiter, Safety Clamping]
              |
              v [Actuator Commands: [0.0, 13.0] N]
          MUJOCO 4-ROTOR ACTUATION
```

### 2. Embodied Visual Sugar Target Tracking Pipeline
```text
             Sugar Cube (Simulated / Physical)
                           ↓
                   Camera Interface (RGB)
                           ↓
             Sugar Detection & Image Localization
                           ↓
            Visual Target Features (center_x, size)
                           ↓
          Biological Sensory Pathway (LC10 Left/Right)
                           ↓
                 FlyWire v783 Connectome
                           ↓
         Descending Neurons (P9, DNa01, DNa02)
                           ↓
                    Motor Decoding
                           ↓
                 Flight Stabilization
                           ↓
                   Skydio X2 Motors
```

### Sensory-Driven Philosophy

The project does **NOT** give the fly high-level commands such as `"fly forward"` or `"turn left"`, nor does it hardcode `"sugar detected -> forward"`.

The architecture is strictly **sensory-driven**:
1. The camera provides raw images from which visual target features (azimuthal position, apparent size) are extracted.
2. Visual target features stimulate bilateral Lobula Columnar (`LC10`) visual projection neurons.
3. The biological network processes activity through the 138,639-neuron, 15-million synapse connectome.
4. Activity from descending steering (`DNa01`, `DNa02`) and propulsion (`P9`) neurons decodes into flight motor intent.

---

## Biological Grounding vs. Engineering Scaffolding

| Component | Biological Grounding | Engineering Scaffolding |
|---|---|---|
| **Visual Target Tracking** | 437 Lobula Columnar type 10 (LC10a/c) neurons (216 Left / 221 Right) mediating small-object tracking & orientation (*Ribeiro et al., 2018*) | Deterministic color thresholding and bounding box localization to compute visual coordinates |
| **Contact Gustation** | 21 Labellar receptor neurons (`LB3`) in Maxillary/Labial Nerve projecting to SEZ for feeding / proboscis extension | Preserved as separate `--stimulus sugar` pure connectome benchmark |
| **Mechanosensation** | 484 Johnston's Organ (JO) wind & gravity neurons (251 Left / 233 Right) | Linear transfer function mapping m/s and rad/s to Poisson rates |
| **Visual Looming** | 104 Lobula Columnar type 4 (LC4) neurons projecting to Giant Fiber | Optical expansion proxy driving LC4 firing |
| **Optic Flow** | Horizontal System (HS) and Vertical System (VS) lobula plate cells | Linear rate modulation based on gyro rates |
| **Motor Drive** | P9 (forward propulsion), DNa01/02 (steering), MDN (braking), GF (escape takeoff) | Normalized $[-1, 1]$ intent mapped to attitude PD setpoints |
| **Flight Control** | Brain governs motor intent and reactive steering/escape | PD controller & mixer provide aerodynamic stabilization |

> **Scientific Notice:**
> Direct camera-to-ommatidia pixel ray tracing is `[NOT BIOLOGICALLY VALIDATED]` and is deferred to future work. Visual targets stimulate validated `LC10` populations rather than arbitrary motor shortcuts.

---

## Installation

```bash
pip install -r requirements.txt
```

*(Ensure PyTorch with CUDA support is installed for real-time neural simulation).*

---

## Running the Project

### 1. Unified Launcher (`main.py`)
```bash
# Architecture and module verification:
python main.py

# Closed-loop brain-controlled drone simulation:
python main.py --integrated

# Embodied visual sugar-cube tracking:
python main.py --integrated --target sugar
python main.py --target sugar

# Standalone biological connectome simulation:
python main.py --brain --stimulus sugar

# Standalone drone physical embodiment:
python main.py --drone --steps 200

# Autonomous waypoint navigation demo:
python main.py --flight-demo
```

### 2. Controlled Visual Tracking Experiments (`experiments/sugar_approach.py`)
```bash
# Complete 5-condition comparative battery (A through E):
python experiments/sugar_approach.py --condition all

# Individual controlled conditions:
python experiments/sugar_approach.py --condition center   # Condition A: Centered ahead
python experiments/sugar_approach.py --condition left     # Condition B: Target to the left
python experiments/sugar_approach.py --condition right    # Condition C: Target to the right
python experiments/sugar_approach.py --condition none     # Condition D: Target absent
python experiments/sugar_approach.py --condition moving   # Condition E: Target moving laterally
```

### 3. Integrated Flight Experiments
```bash
# Closed-loop hover experiment:
python experiments/brain_drone_hover.py
python experiments/brain_drone_hover.py --viewer

# Disturbance recovery experiment:
python experiments/brain_drone_response.py --disturbance yaw
python experiments/brain_drone_response.py --disturbance roll

# Visual looming escape reflex experiment:
python experiments/brain_drone_escape.py --threat-side center
```

### 4. Running Tests
```bash
# Fast lightweight unit test suite (CPU-compatible, no GPU required):
python -m unittest tests/test_sugar_vision.py
```

### 3. Standalone Drone Experiments
```bash
# Autonomous 3D waypoint navigation (cascaded PID):
python experiments/drone_pid_flight.py

# Interactive keyboard flight with camera view:
python experiments/manual_control.py

# Altitude stabilization test:
python experiments/height_stabilization_test.py
```