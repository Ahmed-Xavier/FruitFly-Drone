# Bridge: Brain ↔ Body Sensory-Motor Integration

The `bridge/` package implements the bidirectional neuromorphic interface between the biological **FlyWire v783 Drosophila connectome** (`brain/`) and the **MuJoCo Skydio X2 quadrotor drone** (`body/`).

---

## 1. Information Flow Architecture

```text
                MUJOCO DRONE
                     │
              ┌──────┴──────┐
              │             │
             IMU          Looming / Camera
              │             │
              ▼             ▼
       SensoryEncoder (bridge/sensory_encoder.py)
              │
              ▼ [Poisson Rates in Hz]
       FLYWIRE CONNECTOME (brain/brain.py)
              │ [138,639 LIF Neurons · 15M Synapses]
              │
              ▼ [Smoothed DN Firing Rates]
       MotorDecoder (bridge/motor_decoder.py)
              │
              ▼ [Normalized Motor Intent: Thrust, Roll, Pitch, Yaw]
       DroneFlightController (bridge/controller.py)
              │ [Attitude PD, Slew Rate Limiter, Safety Clamping]
              │
              ▼ [Actuator Commands: [0.0, 13.0] N]
          MUJOCO 4-ROTOR ACTUATION
```

---

## 2. Biological Sensory Pathways Used

All sensory neurons used in the bridge are resolved directly from the **FlyWire v783 whole-brain connectome**:

### A. Antennal Mechanosensory (Johnston's Organ — JO)
* **Biological Structure:** Johnston's organ (JO) in the second antennal segment detects vibrations, sound, wind deflection, and gravitational forces.
* **Neurons Mapped:** **484 wind_gravity neurons** (subtypes `JO-C`, `JO-D`, `JO-E`, `JO-EV`):
  * **Left Antenna:** 251 neurons (FlyWire `wind_gravity`, `side="left"`).
  * **Right Antenna:** 233 neurons (FlyWire `wind_gravity`, `side="right"`).
* **Biological Grounding:** In flying *Drosophila*, forward airspeed produces symmetric antennal drag. Angular rotation ($\omega_z, \omega_x$) causes asymmetric aerodynamic airflow, deflecting the left and right aristae differentially (*Budick et al., 2007; Suver et al., 2012*).
* **Engineering Approximation:** Linearized transfer function mapping airspeed ($v$) and angular rates ($\omega$) to Poisson firing rates (10–350 Hz).

### B. Visual Looming Threat Pathway (LC4)
* **Biological Structure:** Lobula Columnar type 4 (LC4) visual projection neurons.
* **Neurons Mapped:** **104 LC4 neurons** (54 left, 50 right), identical to `STIMULI["lc4"]`.
* **Biological Grounding:** LC4 neurons specifically encode optical expansion rate and angular size $\theta \cdot \dot{\theta}$ of approaching objects, projecting monosynaptically to the Giant Fiber system and triggering emergency escape behaviors (*von Reyn et al., 2014, 2017; Ache et al., 2019*).

### C. Optic Flow Tangential System (HS / VS)
* **Biological Structure:** Lobula Plate Tangential Cells (LPTCs).
* **Neurons Mapped:** Horizontal System (`HS`) and Vertical System (`VS`) neurons.
* **Biological Grounding:** HS cells respond to wide-field horizontal yaw optic flow; VS cells respond to vertical pitch/roll optic flow (*Borst et al., 2010*).

### D. Visual Target Tracking Pathway (LC10) — Embodied Sugar Tracking
* **Biological Structure:** Lobula Columnar type 10 (LC10) visual projection neurons.
* **Neurons Mapped:** **437 LC10a/LC10c neurons** (216 left, 221 right) from a total of 815 LC10 neurons in the connectome.
* **Biological Grounding:** LC10 neurons specifically mediate visual tracking of, orientation toward, and approach to small salient visual objects in *Drosophila* (*Ribeiro et al., Cell 2018; Hindmarsh Sten et al., Nature 2021*). They project to the Anterior Optic Tubercle (AOTU) and down to descending steering neurons (`DNa01`, `DNa02`) and forward locomotion neurons (`P9`).
* **Visual Target Encoder:** [`bridge/visual_target_encoder.py`](visual_target_encoder.py) translates camera object detections (`center_x`, `apparent_size`, confidence) into differential bilateral LC10 activation.
* **Distinction from Contact Taste:** Strictly separated from `STIMULI["sugar"]` (21 labellar `LB3` gustatory receptor neurons that project to the SEZ to control proboscis extension and feeding).

---

## 3. Descending Neuron (DN) Motor Decoding

The decoder extracts smoothed firing rates from confirmed functional descending neurons in the connectome:

| Descending Neuron | Biological Function in *Drosophila* | Decoded Quadrotor Motor Intent | Status |
|---|---|---|---|
| **P9 / P9_oDN1** | Forward walking & forward propulsion drive (*Rayshubskiy et al., 2020*) | Pitch forward (nose-down) velocity drive + baseline thrust modulation | Biologically grounded |
| **DNa01 & DNa02** | Ipsilateral turning and steering (*Namiki et al., 2018; Rayshubskiy et al., 2020*) | Bilateral differential $(R - L)$ decodes to yaw rate & coordinated roll bank | Biologically grounded |
| **MDN (Moonwalker)** | Backward walking & retreat locomotion (*Bidaye et al., 2014*) | Pitch-up braking / deceleration retreat | Biologically grounded |
| **GF / DNp01 (Giant Fiber)** | Emergency collision takeoff & jump reflex (*von Reyn et al., 2014*) | Collective vertical takeoff surge ($\Delta T_{\text{escape}}$) | Biologically grounded |

---

## 4. Scientific Rigor & Validation Status

To ensure complete honesty, the following classification distinguishes validated science from engineering scaffolding:

### Validated Biological Features:
1. Whole-brain recurrent connectome topology (138,639 neurons, 15,091,983 directed synapses).
2. Identity and lateralization of Johnston's organ (251 L / 233 R) and LC4 (54 L / 50 R) neurons.
3. Functional roles of P9, DNa01, DNa02, MDN, and Giant Fiber (DNp01).
4. Differential bilateral sensory-motor coupling.

### Engineering Approximations:
1. Transfer function shapes converting physical SI kinematic units ($\text{rad/s}, \text{m/s}$) to firing rates ($\text{Hz}$).
2. Scaling constants mapping normalized motor intent $[-1.0, 1.0]$ to torque and thrust setpoints.
3. Quadrotor physical mass ($1.325\,\text{kg}$) differs by orders of magnitude from fruit-fly mass ($\approx 1\,\text{mg}$); the biological network governs high-level motor intent rather than raw micro-gram aerodynamics.

### [NOT BIOLOGICALLY VALIDATED]:
* Drone-to-fly visual camera pixel mapping: Current visual input uses LC4 looming rate injection rather than an end-to-end multi-ommatidial retinotopic lattice. Full photoreceptor-to-lamina camera ray-tracing is deferred to future visual sub-models.

---

## 5. Multiscale Timestep Synchronization

The two systems operate across different temporal scales:
* **FlyWire Biological Clock:** $\Delta t_{\text{brain}} = 0.1\,\text{ms}$ ($10\,\text{kHz}$).
* **MuJoCo Physical Clock:** $\Delta t_{\text{body}} = 10.0\,\text{ms}$ ($100\,\text{Hz}$).

`BrainDroneSynchronizer` coordinates the execution:
1. Sample drone telemetry ($100\,\text{Hz}$).
2. Encode sensory inputs to FlyWire firing rates.
3. Advance brain simulation by $N$ neural steps (default $N = 10$, configurable up to $N = 100$ for 1:1 real-time parity).
4. Smooth descending neuron firing rates across the sliding window.
5. Decode motor intent and compute stabilized rotor commands.
6. Step MuJoCo physics by $10\,\text{ms}$.

---

## 6. Safety Envelopes & Flight Controller

The biological connectome does not directly command motor wires. `DroneFlightController` provides:
* **Attitude PD Scaffold:** Stabilizes roll and pitch around biological target attitudes ($\pm 30^\circ$).
* **Slew Rate Limiting:** Enforces max thrust change ($\le 80\,\text{N/s}$) to prevent motor shock.
* **Actuator Saturation:** Clamps each rotor strictly to $[0.0, 13.0]\,\text{N}$.
* **NaN/Inf Guard:** Automatically falls back to equilibrium hover thrust ($3.25\,\text{N} \times 4$) if invalid telemetry is encountered.

---

## 7. Running Bridge Experiments

```bash
# 1. Closed-loop hover experiment:
python experiments/brain_drone_hover.py
python experiments/brain_drone_hover.py --viewer

# 2. Disturbance recovery experiment:
python experiments/brain_drone_response.py --disturbance yaw
python experiments/brain_drone_response.py --disturbance roll

# 3. Visual looming escape reflex experiment:
python experiments/brain_drone_escape.py --threat-side center
```
