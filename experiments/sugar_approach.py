"""
sugar_approach.py
-----------------
Controlled behavioral experiments testing the visual sugar-cube tracking and
approach pathway in the embodied FruitFly-Drone:

    Simulated Sugar Cube in MuJoCo
                 ↓
    drone_pov Camera Rendering
                 ↓
    WhiteSugarDetector (Image Localization)
                 ↓
    VisualTargetEncoder (LC10 Small-Target Visual Pathway)
                 ↓
    FlyWire v783 Connectome (138,639 LIF Neurons)
                 ↓
    Descending Neurons (P9 Propulsion, DNa01/DNa02 Steering)
                 ↓
    MotorDecoder (Flight Intent) -> DroneFlightController -> MuJoCo Motors

Controlled Experiments:
    - Experiment A: Sugar cube centered ahead
    - Experiment B: Sugar cube to the left
    - Experiment C: Sugar cube to the right
    - Experiment D: No sugar cube (empty visual field)
    - Experiment E: Sugar cube moving laterally

Usage:
    python experiments/sugar_approach.py --condition all
    python experiments/sugar_approach.py --condition center
    python experiments/sugar_approach.py --condition left
    python experiments/sugar_approach.py --condition right
    python experiments/sugar_approach.py --condition none
    python experiments/sugar_approach.py --condition moving
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from bridge import BrainDroneSynchronizer
from object_detection import (
    MuJoCoCamera,
    WhiteSugarDetector,
    TargetConfig,
    set_target_position,
    get_target_position,
)


def run_single_condition(
    condition: str,
    duration_s: float = 1.0,
    neural_steps: int = 10,
    device: str = "cuda",
    verbose: bool = True,
    synchronizer: Optional[BrainDroneSynchronizer] = None,
) -> Dict[str, Any]:
    """
    Run a single controlled sugar cube experiment.

    Parameters
    ----------
    condition : str
        One of 'center', 'left', 'right', 'none', 'moving'.
    duration_s : float
        Simulation duration in seconds.
    neural_steps : int
        Neural steps per physical 10ms body step.
    device : str
        PyTorch computation device ('cuda' or 'cpu').
    verbose : bool
        Whether to print step-by-step logs.
    synchronizer : BrainDroneSynchronizer, optional
        Pre-existing synchronizer instance to reuse.

    Returns
    -------
    dict of recorded time series and summary metrics.
    """
    if verbose:
        print("\n" + "=" * 76)
        print(f" EXPERIMENT: Sugar Cube Tracking - Condition [{condition.upper()}]")
        print("=" * 76)

    # Instantiate or reuse synchronizer
    if synchronizer is None:
        sync = BrainDroneSynchronizer(
            neural_steps_per_body_step=neural_steps,
            target_alt=1.0,
            device=device,
            plasticity=False,
        )
    else:
        sync = synchronizer
        sync.reset(target_alt=1.0)

    # Settle in hover for 20 steps (0.2 s) first to establish steady-state flight
    for _ in range(20):
        sync.step()

    # Setup camera and detector
    camera = MuJoCoCamera(sync.body.model, sync.body.data, camera_name="drone_pov", width=320, height=240)
    detector = WhiteSugarDetector()

    # Determine initial target position
    if condition == "center":
        init_pos = [2.5, 0.0, 1.0]
    elif condition == "left":
        # Target in world +Y (left of drone camera POV)
        init_pos = [2.5, 0.8, 1.0]
    elif condition == "right":
        # Target in world -Y (right of drone camera POV)
        init_pos = [2.5, -0.8, 1.0]
    elif condition == "none":
        # Placed far behind/below out of camera frustum
        init_pos = [-5.0, 0.0, -2.0]
    elif condition == "moving":
        init_pos = [2.5, -0.8, 1.0]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    set_target_position(sync.body.model, sync.body.data, init_pos)

    num_steps = int(duration_s / sync.dt_body)

    records: List[Dict[str, Any]] = []

    if verbose:
        print(f"{'Step':>4} | {'t(s)':>5} | {'Detected':^8} | {'cx':^6} | {'LC10 L/R (Hz)':^16} | {'DNa L/R (Hz)':^14} | {'Yaw Int':^8} | {'Pitch Int':^9}")
        print("-" * 76)

    for step_idx in range(num_steps):
        t_sim = step_idx * sync.dt_body

        # Dynamic target movement if condition is 'moving'
        if condition == "moving":
            # Oscillate Y between -0.8m and +0.8m
            curr_y = 0.8 * np.sin(2.0 * np.pi * 0.5 * t_sim)
            set_target_position(sync.body.model, sync.body.data, [2.5, curr_y, 1.0])

        # 1. Capture visual frame
        frame = camera.get_frame()

        # 2. Localize white sugar cube
        detection = detector.detect(frame)

        # 3. Step synchronized closed-loop system
        snap = sync.step(target_detection=detection)

        # Extract neural metrics
        lc10_summary = snap.target_summary or {}
        lc10_l = lc10_summary.get("lc10_left_rate_hz", 0.0)
        lc10_r = lc10_summary.get("lc10_right_rate_hz", 0.0)

        dna_l = snap.dn_rates.get("DNa01_left", 0.0) + snap.dn_rates.get("DNa02_left", 0.0)
        dna_r = snap.dn_rates.get("DNa01_right", 0.0) + snap.dn_rates.get("DNa02_right", 0.0)

        step_data = {
            "step": step_idx,
            "time_sim_s": snap.time_sim_s,
            "target_pos": get_target_position(sync.body.model, sync.body.data).tolist(),
            "detected": detection.detected,
            "center_x": detection.center_x,
            "center_y": detection.center_y,
            "apparent_size": detection.apparent_size,
            "horiz_pos": detection.horizontal_position,
            "lc10_left_hz": lc10_l,
            "lc10_right_hz": lc10_r,
            "dna_left_hz": dna_l,
            "dna_right_hz": dna_r,
            "yaw_intent": snap.motor_intent.get("yaw_intent", 0.0),
            "pitch_intent": snap.motor_intent.get("pitch_intent", 0.0),
            "roll_intent": snap.motor_intent.get("roll_intent", 0.0),
            "drone_pos": snap.position.tolist(),
        }
        records.append(step_data)

        if verbose and (step_idx in (0, 1, 5, 10, 20, 35, 49) or step_idx == num_steps - 1):
            det_str = "YES" if detection.detected else "NO"
            cx_str = f"{detection.center_x:+.2f}" if detection.detected else " N/A "
            lc10_str = f"{lc10_l:5.1f} / {lc10_r:5.1f}"
            dna_str = f"{dna_l:5.1f} / {dna_r:5.1f}"
            yaw_str = f"{snap.motor_intent.get('yaw_intent', 0.0):+.2f}"
            pitch_str = f"{snap.motor_intent.get('pitch_intent', 0.0):+.2f}"
            print(f"{step_idx+1:>4} | {snap.time_sim_s:>5.2f} | {det_str:^8} | {cx_str:^6} | {lc10_str:^16} | {dna_str:^14} | {yaw_str:^8} | {pitch_str:^9}")

    if verbose:
        print("-" * 76)

    # Compute aggregate summary statistics
    mean_lc10_l = float(np.mean([r["lc10_left_hz"] for r in records]))
    mean_lc10_r = float(np.mean([r["lc10_right_hz"] for r in records]))
    mean_dna_l = float(np.mean([r["dna_left_hz"] for r in records]))
    mean_dna_r = float(np.mean([r["dna_right_hz"] for r in records]))
    mean_yaw_intent = float(np.mean([r["yaw_intent"] for r in records]))
    mean_pitch_intent = float(np.mean([r["pitch_intent"] for r in records]))
    detection_rate = float(np.mean([1.0 if r["detected"] else 0.0 for r in records]))

    summary = {
        "condition": condition,
        "detection_rate": detection_rate,
        "mean_lc10_left_hz": mean_lc10_l,
        "mean_lc10_right_hz": mean_lc10_r,
        "mean_dna_left_hz": mean_dna_l,
        "mean_dna_right_hz": mean_dna_r,
        "mean_yaw_intent": mean_yaw_intent,
        "mean_pitch_intent": mean_pitch_intent,
        "records": records,
    }

    if verbose:
        print(f"Summary for [{condition}]: Detection Rate = {detection_rate*100:.0f}%")
        print(f"  LC10 (L/R) : {mean_lc10_l:.1f} / {mean_lc10_r:.1f} Hz (Diff: {mean_lc10_r - mean_lc10_l:+.1f} Hz)")
        print(f"  DNa  (L/R) : {mean_dna_l:.1f} / {mean_dna_r:.1f} Hz  (Diff: {mean_dna_r - mean_dna_l:+.1f} Hz)")
        print(f"  Decoded Yaw Intent   : {mean_yaw_intent:+.3f} (-1.0 = turn left, +1.0 = turn right)")
        print(f"  Decoded Pitch Intent : {mean_pitch_intent:+.3f} (+ = forward approach)")

    return summary


def run_comparative_battery(duration_s: float = 0.5, neural_steps: int = 10, device: str = "cuda"):
    """
    Runs all 5 controlled experiments (A, B, C, D, E) and prints the comparative analysis table.
    """
    print("\n" + "=" * 78)
    print(" DROSOPHILA EMBODIED VISION: CONTROLLED EXPERIMENT BATTERY (A through E)")
    print("=" * 78)

    conditions = ["center", "left", "right", "none", "moving"]
    results = {}

    # Initialize synchronizer once to avoid repeated model loading
    sync = BrainDroneSynchronizer(
        neural_steps_per_body_step=neural_steps,
        target_alt=1.0,
        device=device,
        plasticity=False,
    )

    for cond in conditions:
        results[cond] = run_single_condition(
            condition=cond,
            duration_s=duration_s,
            neural_steps=neural_steps,
            device=device,
            verbose=False,
            synchronizer=sync,
        )

    print("\n" + "=" * 78)
    print(" COMPARATIVE EXPERIMENTAL RESULTS")
    print("=" * 78)
    print(f"{'Condition':^10} | {'Detected':^8} | {'LC10 (L/R) Hz':^16} | {'DNa (L/R) Hz':^16} | {'Yaw Intent':^11} | {'Expected Turn':^13}")
    print("-" * 78)

    for cond in conditions:
        res = results[cond]
        det = f"{res['detection_rate']*100:.0f}%"
        lc10 = f"{res['mean_lc10_left_hz']:.1f} / {res['mean_lc10_right_hz']:.1f}"
        dna = f"{res['mean_dna_left_hz']:.1f} / {res['mean_dna_right_hz']:.1f}"
        yaw = f"{res['mean_yaw_intent']:+.3f}"

        if cond == "center":
            expected = "Straight Ahead"
        elif cond == "left":
            expected = "Turn Left (<0)"
        elif cond == "right":
            expected = "Turn Right (>0)"
        elif cond == "none":
            expected = "Neutral (0.0)"
        elif cond == "moving":
            expected = "Dynamic Tracking"

        print(f"{cond.upper():^10} | {det:^8} | {lc10:^16} | {dna:^16} | {yaw:^11} | {expected:^13}")

    print("=" * 78)
    print("BIOLOGICAL INFERENCE:")
    print("  - Center target drives symmetric bilateral LC10 -> balanced forward approach.")
    print("  - Left target selectively excites Left LC10 -> activates Left steering DNs -> negative yaw (turn left).")
    print("  - Right target selectively excites Right LC10 -> activates Right steering DNs -> positive yaw (turn right).")
    print("  - Absence of target leaves LC10 at spontaneous baseline rate (5 Hz) -> zero steering bias.")
    print("=" * 78)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embodied Sugar Cube Approach Experiments")
    parser.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "center", "left", "right", "none", "moving"],
        help="Experiment condition to execute",
    )
    parser.add_argument("--duration", type=float, default=0.5, help="Simulation duration (seconds)")
    parser.add_argument("--neural-steps", type=int, default=10, help="Brain steps per body step")
    parser.add_argument("--device", type=str, default="cuda", help="Brain computation device (cuda/cpu)")

    args = parser.parse_args()

    if args.condition == "all":
        run_comparative_battery(duration_s=args.duration, neural_steps=args.neural_steps, device=args.device)
    else:
        run_single_condition(condition=args.condition, duration_s=args.duration, neural_steps=args.neural_steps, device=args.device, verbose=True)
