# Visual Object Detection & Camera Interface (`object_detection/`)

Provides vision capture and visual target detection for the embodied **FruitFly-Drone** project, bridging physical/simulated targets with biological sensory pathways in the Drosophila FlyWire v783 connectome.

---

## 1. Information Flow Pipeline

```text
       Simulated / Physical Sugar Cube
                      ↓
           Camera Interface (RGB)
                      ↓
    Sugar Detection & Image Localization
                      ↓
  Biologically Grounded Sensory Representation
                      ↓
             FlyWire v783 Brain
                      ↓
              Descending Neurons
                      ↓
                Motor Decoding
                      ↓
           Flight Stabilization (PD)
                      ↓
             Skydio X2 Quadrotor
```

---

## 2. Architecture & Modules

| Module | Description |
| :--- | :--- |
| [`camera_interface.py`](camera_interface.py) | Unified abstraction supporting **`MuJoCoCamera`** (simulated `drone_pov`), **`PhysicalCamera`** (real USB/CSI hardware via OpenCV), and **`MockCamera`** (headless synthetic testing). |
| [`detector.py`](detector.py) | **`WhiteSugarDetector`** deterministic color/contour segmentation returning **`TargetDetection`** (normalized coordinates `[-1.0, 1.0]`, apparent size, confidence, and categorical descriptors). |
| [`target.py`](target.py) | Target configuration and kinematic management for MuJoCo mocap body `sugar_cube`. |

---

## 3. Scientific Grounding vs. Engineering Approximations

### [BIOLOGICALLY GROUNDED]
* **Target Detection Visual Projection Neurons (LC10):** In *Drosophila melanogaster*, visual tracking and orientation toward small visual objects are mediated by Lobula Columnar type 10 (`LC10`) projection neurons (*Ribeiro et al., Cell 2018*; *Hindmarsh Sten et al., Nature 2021*).
* **Connectome Preservation:** The full FlyWire v783 connectome contains 815 `LC10` neurons across left (403) and right (412) optic lobes. The model stimulates validated `LC10a` and `LC10c` subtypes (437 neurons: 216 left, 221 right).
* **Bilateral Asymmetry:** Off-axis target positions induce contralateral steering in *Drosophila* by selectively exciting the corresponding visual hemifield and driving descending steering neurons (`DNa01`, `DNa02`) and propulsion neurons (`P9`).
* **Distinction from Gustation:** Contact taste detection (`STIMULI["sugar"]`, 21 `LB3` gustatory receptor neurons in the SEZ) is strictly separated from distant visual tracking (`LC10` visual projection neurons in the lobula).

### [ENGINEERING APPROXIMATIONS]
* **Camera Model:** Pinhole perspective rendering (`drone_pov`, 320x240 RGB) approximating the fly compound eye's ~700 ommatidia without implementing non-uniform ommatidial facet optics.
* **Image Segmentation:** Deterministic HSV color-space thresholding and contour extraction (high brightness, low saturation) rather than biological photoreceptor (R1–R6 / R7–R8) and lamina/medulla motion-energy filtering.
* **Rate Encoding Transfer Function:** Linear proportional mapping between image-space coordinate `center_x` and population firing rates clamped to `[5.0, 300.0]` Hz.
* **Drone Flight Dynamics:** Quadrotor flight stabilization controller, rate limiters, and motor mixer.

---

## 4. Hardware Portability (Simulation to Real Drone)

The detector is strictly decoupled from MuJoCo physics and accepts arbitrary RGB image arrays:

```python
# Simulation
from object_detection import MuJoCoCamera, WhiteSugarDetector
camera = MuJoCoCamera(model, data, camera_name="drone_pov")
detector = WhiteSugarDetector()
detection = detector.detect(camera.get_frame())

# Real Physical Drone (USB or CSI Camera)
from object_detection import PhysicalCamera, WhiteSugarDetector
camera = PhysicalCamera(device_index=0, width=640, height=480)
detector = WhiteSugarDetector()
detection = detector.detect(camera.get_frame())
```

---

## 5. Running Experiments

```bash
# Run complete 5-condition comparative experiment battery:
python experiments/sugar_approach.py --condition all

# Run specific conditions:
python experiments/sugar_approach.py --condition center
python experiments/sugar_approach.py --condition left
python experiments/sugar_approach.py --condition right
python experiments/sugar_approach.py --condition none
python experiments/sugar_approach.py --condition moving

# Run integrated closed-loop drone flight with visual sugar tracking:
python main.py --target sugar
python main.py --headless --target sugar --duration 5.0
```
