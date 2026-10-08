# Bridge: Brain ↔ Body Interface Architecture

> **Status:** Specification Only — Architectural Boundary (Intentionally NOT Implemented Yet)

---

## Architectural Boundary Pipeline

The `bridge/` module defines the conceptual and architectural boundary between the physical drone embodiment (`body/`) and the biological neural network (`brain/`):

```text
DRONE SENSORS
      ↓
SENSORY ENCODING
      ↓
FLYWIRE BRAIN
      ↓
DESCENDING NEURON ACTIVITY
      ↓
MOTOR DECODING
      ↓
DRONE ACTUATORS
```

---

## Intentionally Deferred Interfaces

To maintain scientific integrity, the sensory and motor bridge mappings are **specification-only** and are **deliberately NOT implemented yet**:

1. **Biological sensory neuron / channel selection**: Identifying which specific biological sensory neurons and receptor types (e.g., Johnston's organ mechanoreceptors, lobula plate tangential cells, haltere analogs) should receive drone telemetry (IMU accelerations, angular rates, orientation).
2. **Sensor-to-neural-rate / spike encoding**: Designing physiologically plausible transfer functions that convert physical SI units into Poisson firing rates or current injection profiles.
3. **Descending-neuron-to-flight-command decoding**: Mapping activity across descending neurons (e.g., P9, DNa01, DNa02, MDN) into multi-rotor flight control commands (thrust, roll, pitch, yaw).
4. **Brain / body timestep synchronization**: Synchronizing the temporal disparity between the biological LIF simulation clock (10 kHz, $dt = 0.1\text{ ms}$) and the MuJoCo rigid-body flight physics simulation (100 Hz, $dt = 10.0\text{ ms}$).
5. **Camera / visual pathway integration**: Projecting drone camera frames or optic flow fields onto the compound eye retinotopic lattice and downstream optic lobe neuropils.

---

## Scientific Rigor Notice

> **IMPORTANT:**
> These neural and motor mappings require empirical scientific investigation, literature grounding, and systematic response-curve profiling. They should **not** be arbitrarily chosen or hardcoded as placeholder heuristics.
