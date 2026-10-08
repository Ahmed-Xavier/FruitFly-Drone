"""
brain_drone_hover.py
--------------------
Integrated experiment testing closed-loop hover maintenance with the FlyWire connectome.

Signal Path:
    Body sensors -> SensoryEncoder (JO wind/gravity, HS/VS) -> FlyBrain LIF ->
    DN activity (P9, DNa01, MDN) -> MotorDecoder -> DroneFlightController ->
    MuJoCo 4-rotor actuation.

Usage:
    python experiments/brain_drone_hover.py
    python experiments/brain_drone_hover.py --viewer
    python experiments/brain_drone_hover.py --duration 5.0 --neural-steps 10
"""

import time
import argparse
import numpy as np
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from bridge import BrainDroneSynchronizer


def run_hover_experiment(
    duration: float = 3.0,
    neural_steps: int = 10,
    viewer: bool = False,
    device: str = "cuda",
):
    print("=" * 68)
    print(" Brain-Controlled Drone: Closed-Loop Hover Experiment")
    print("=" * 68)
    print(f"Target Duration: {duration:.1f} s | Neural steps per body step: {neural_steps}")

    synchronizer = BrainDroneSynchronizer(
        neural_steps_per_body_step=neural_steps,
        target_alt=1.0,
        device=device,
        plasticity=False,
    )

    steps = int(duration / synchronizer.dt_body)
    print(f"Executing {steps} synchronized physics steps ({steps * synchronizer.dt_body:.1f} s)...")

    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(synchronizer.body.model, synchronizer.body.data) as v:
            for s in range(steps):
                t0 = time.time()
                snap = synchronizer.step()
                v.sync()

                if s % max(1, steps // 5) == 0 or s == steps - 1:
                    print(f"  [{snap.time_sim_s:5.2f}s] Alt: {snap.position[2]:.3f} m | "
                          f"Spikes: {snap.spikes_this_step:4d} | "
                          f"P9 rate: {snap.dn_rates.get('P9_left', 0.0):.1f} Hz | "
                          f"Rotor 0: {snap.motor_commands[0]:.2f} N")

                elapsed = time.time() - t0
                rem = synchronizer.dt_body - elapsed
                if rem > 0:
                    time.sleep(rem)
    else:
        for s in range(steps):
            snap = synchronizer.step()
            if s % max(1, steps // 5) == 0 or s == steps - 1:
                print(f"  [{snap.time_sim_s:5.2f}s] Alt: {snap.position[2]:.3f} m | "
                      f"Spikes: {snap.spikes_this_step:4d} | "
                      f"P9 rate: {snap.dn_rates.get('P9_left', 0.0):.1f} Hz | "
                      f"Rotor 0: {snap.motor_commands[0]:.2f} N")

    print("\n" + "=" * 68)
    print(" HOVER EXPERIMENT RESULTS")
    print("=" * 68)
    final_snap = synchronizer.step()
    print(f"  Final Position   : {final_snap.position}")
    print(f"  Final Velocity   : {final_snap.velocity}")
    print(f"  Final Rotor Thrusts: {np.round(final_snap.motor_commands, 3)} N")
    print(f"  Total Simulated Physics Time: {final_snap.time_sim_s:.2f} s")
    print(f"  Total Neural Time           : {final_snap.time_brain_ms:.2f} ms")
    print("  [OK] Complete brain-body loop executed successfully.")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain-Drone Closed-Loop Hover Experiment")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds (default: 3.0)")
    parser.add_argument("--neural-steps", type=int, default=10, help="Brain steps per body step (default: 10)")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D viewer")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    dev = "cpu" if args.cpu else "cuda"
    run_hover_experiment(
        duration=args.duration,
        neural_steps=args.neural_steps,
        viewer=args.viewer,
        device=dev,
    )
