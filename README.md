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
│   ├── motor_decoder.py    # Descending neuron activity -> motor flight intent
│   ├── controller.py       # Safety envelopes, attitude PD stabilization, mixer
│   ├── synchronizer.py     # Multiscale timestep coordinator & telemetry logging
│   ├── __init__.py         # Package exports
│   └── README.md           # Interface boundary specifications & biological grounding
├── experiments/            # Standalone and integrated flight experiments
│   ├── brain_drone_hover.py    # Closed-loop connectome hover experiment
│   ├── brain_drone_response.py # Controlled disturbance & recovery experiment
│   ├── brain_drone_escape.py   # Visual looming threat & escape reflex experiment
│   ├── drone_pid_flight.py     # Standalone autonomous 3D waypoint mission (PID)
│   ├── manual_control.py       # Standalone interactive keyboard flight
│   ├── height_stabilization_test.py # Standalone altitude PD test
│   └── README.md           # Flight experiments guide
├── main.py                 # Unified application launcher
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules
└── README.md
```

---

## Information Flow Pipeline

```text
                MUJOCO DRONE
                     |
              +------+------+
              |             |
             IMU          Looming / Camera
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

### Sensory-Driven Philosophy

The project does **NOT** give the fly high-level commands such as `"fly forward"` or `"turn left"`.

The architecture is strictly **sensory-driven**:
1. The drone provides physical sensory information (accelerations, angular velocities, optic flow / looming).
2. That information is encoded into neural inputs (synaptic currents, Poisson spike rates).
3. The biological network processes the activity through the whole-brain recurrent connectome.
4. Activity from descending neurons (DNs) is decoded into flight control signals (motor thrusts / torques).

---

## Biological Grounding vs. Engineering Scaffolding

| Component | Biological Grounding | Engineering Scaffolding |
|---|---|---|
| **Mechanosensation** | 484 Johnston's Organ (JO) wind & gravity neurons (251 Left / 233 Right) | Linear transfer function mapping m/s and rad/s to Poisson rates |
| **Visual Looming** | 104 Lobula Columnar type 4 (LC4) neurons projecting to Giant Fiber | Obstacle proximity proxy driving LC4 firing |
| **Optic Flow** | Horizontal System (HS) and Vertical System (VS) lobula plate cells | Linear rate modulation based on gyro rates |
| **Motor Drive** | P9 (forward propulsion), DNa01/02 (steering), MDN (braking), GF (escape takeoff) | Normalized $[-1, 1]$ intent mapped to attitude PD setpoints |
| **Flight Control** | Brain governs motor intent and reactive steering/escape | PD controller & mixer provide aerodynamic stabilization |

> **Scientific Notice:**
> Direct camera-to-ommatidia pixel ray tracing is `[NOT BIOLOGICALLY VALIDATED]` and is deferred to future work. Visual looming currently drives validated LC4 neural populations.

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

# Closed-loop brain-controlled drone with 3D viewer:
python main.py --integrated --viewer

# Standalone biological connectome simulation:
python main.py --brain --stimulus sugar

# Standalone drone physical embodiment:
python main.py --drone --steps 200

# Autonomous waypoint navigation demo:
python main.py --flight-demo
```

### 2. Integrated Experiments (`experiments/`)
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

### 3. Standalone Drone Experiments
```bash
# Autonomous 3D waypoint navigation (cascaded PID):
python experiments/drone_pid_flight.py

# Interactive keyboard flight with camera view:
python experiments/manual_control.py

# Altitude stabilization test:
python experiments/height_stabilization_test.py
```