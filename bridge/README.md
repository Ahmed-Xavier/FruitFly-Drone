# Bridge: Brain ↔ Body Interface Architecture

> **Status:** Architectural Boundary (Specification Only — Not Implemented Yet)

---

## Conceptual Pipeline

The `bridge/` layer will serve as the bidirectional translation interface between the biological/neuromorphic fruit-fly brain connectome (`brain/`) and the physical MuJoCo Skydio X2 quadrotor embodiment (`body/`).

```text
       MuJoCo Skydio X2 Drone
       [IMU, Gyro, Accel, Telemetry]
                   │
                   ▼
         ┌───────────────────┐
         │  Sensory Encoding │  (Maps physical continuous signals into
         │ (Body → Neurons)  │   spikes / Poisson rates in Hz)
         └───────────────────┘
                   │
                   ▼
       Drosophila LIF Connectome
       [138,639 Neurons · 15M Synapses]
                   │
                   ▼
         ┌───────────────────┐
         │  Motor Decoding   │  (Decodes descending neuron activity
         │ (Neurons → Motors)│   into rotor thrusts & flight torques)
         └───────────────────┘
                   │
                   ▼
       MuJoCo Skydio X2 Actuators
       [4 Rotors: Thrust, Roll, Pitch, Yaw]
```

---

## Explicit Design Boundaries

To preserve scientific rigor, the following decisions are deliberately deferred and **NOT implemented yet**:

1. **Sensory Channel Allocations:**
   - Which specific sensory neurons (e.g., Johnston's organ mechanosensory, visual lobula plate tangential cells, haltere-analogous gyroscopic inputs) receive drone accelerometer, angular velocity, and pose telemetry.
2. **Signal-to-Spike Encoding Functions:**
   - Mathematical transfer functions converting physical units (rad/s, m/s², m) into Poisson firing rates (Hz) or current injections.
3. **Descending Neuron (DN) Motor Decoding:**
   - Which specific descending populations (e.g., P9 forward walk, DNa01/DNa02 steering, MDN backward, Giant Fiber escape) drive quadrotor flight axes (collective thrust, roll torque, pitch torque, yaw torque).
4. **Timescale & Rate Normalization:**
   - Handling the rate mismatch between the brain's 10 kHz integration clock ($dt = 0.1\text{ ms}$) and the drone's 100 Hz physical control loop ($dt = 10.0\text{ ms}$).

These interfaces will be analyzed and designed systematically after profiling the biological response curves and physical flight envelope.
