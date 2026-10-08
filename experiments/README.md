# Experiments & Flight Scenarios

This directory contains standalone flight control scripts, interactive simulations, and testing harnesses for the MuJoCo Skydio X2 quadrotor.

---

## Available Scripts

### 1. Autonomous Waypoint Flight (`drone_pid_flight.py`)
Cascaded PID flight controller (outer horizontal velocity + inner attitude stabilization) tracking a 3D spatial waypoint trajectory.

```bash
# With 3D passive viewer:
python experiments/drone_pid_flight.py

# Headless verification mode (5 seconds):
python experiments/drone_pid_flight.py --duration 5.0 --headless
```

---

### 2. Manual Keyboard Flight (`manual_control.py`)
Interactive keyboard flight with real-time drone-POV camera feed (OpenCV HUD).

```bash
python experiments/manual_control.py
```
- **Controls (AZERTY):** `Z`/`S` pitch, `Q`/`D` roll, `A`/`E` yaw, `Space`/`X` thrust, `R` reset hover.

---

### 3. Height Stabilization PD Test (`height_stabilization_test.py`)
Closed-loop vertical velocity PD controller stabilizing altitude.

```bash
python experiments/height_stabilization_test.py
```
