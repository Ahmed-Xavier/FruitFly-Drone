"""
main.py
-------
Top-level application entry point for the Drosophila Brain <-> Skydio X2 Drone project.

Supported execution modes:
    1. Default / Architecture check:
       python main.py
    2. Standalone Drone physical embodiment:
       python main.py --drone
       python main.py --drone --viewer
    3. Standalone FlyWire biological connectome:
       python main.py --brain
       python main.py --brain --stimulus sugar
    4. Closed-loop Integrated Brain-Drone Simulation:
       python main.py --integrated
       python main.py --integrated --viewer
    5. Autonomous waypoint flight mission:
       python main.py --flight-demo
"""

import time
import argparse
import numpy as np

from body import Body
from brain import Brain, FlyBrain, STIMULI
from bridge import BrainDroneSynchronizer


def run_drone_mode(steps: int = 200, viewer: bool = False):
    """Run standalone drone physics simulation without neural connectome."""
    print("=" * 68)
    print(" Standalone MuJoCo Skydio X2 Drone Embodiment")
    print("=" * 68)
    body = Body()
    initial_sensors = body.read_sensors()
    print(f"  Model loaded: {body.model.ngeom} geoms, {body.model.nbody} bodies")
    print(f"  Initial position: {initial_sensors['position']}")
    print(f"  Physics timestep: {body.timestep * 1000.0:.1f} ms ({1.0 / body.timestep:.0f} Hz)")
    print(f"  Stepping drone physics ({steps} steps with hover equilibrium thrust)...")

    hover_thrust = [3.2495625] * 4

    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(body.model, body.data) as v:
            for _ in range(steps):
                t0 = time.time()
                body.set_motor_commands(hover_thrust)
                body.step()
                v.sync()
                dt_rem = body.timestep - (time.time() - t0)
                if dt_rem > 0:
                    time.sleep(dt_rem)
    else:
        for _ in range(steps):
            body.set_motor_commands(hover_thrust)
            body.step()

    final_sensors = body.read_sensors()
    print(f"  Final altitude: {final_sensors['position'][2]:.3f} m")
    print("  [OK] Drone physics stepped successfully.")
    print("=" * 68)


def run_brain_mode(stimulus: str = "sugar", steps: int = 100, cpu: bool = False):
    """Run standalone FlyWire connectome simulation without drone physics."""
    print("=" * 68)
    print(" Standalone Drosophila melanogaster Connectome (FlyWire v783)")
    print("=" * 68)
    device = "cpu" if cpu else "cuda"
    brain = FlyBrain(device=device, plasticity=False)
    print(f"  Neurons: {brain.num_neurons:,} | Synapses: 15,091,983 | Device: {brain.device.upper()}")
    print(f"  Stimulus: '{stimulus}' | Steps: {steps} ({steps * brain.timestep_ms:.1f} ms neural time)")

    input_data = {"stimulus": stimulus} if stimulus != "none" else {}
    total_spikes = 0

    for s in range(steps):
        out = brain.step(input_data)
        total_spikes += int(out["spikes"].sum())
        if (s + 1) % max(1, steps // 5) == 0:
            print(f"    Step {s+1:>4}/{steps} | t={out['time_ms']:.1f} ms | Spikes: {int(out['spikes'].sum()):>5}")

    print(f"  Total spikes produced: {total_spikes:,}")
    print("  Descending neuron activity:")
    for group, rate in out["population_rates"].items():
        print(f"    {group:<10} rate: {rate:6.1f} Hz")
    print("  [OK] Biological connectome simulation complete.")
    print("=" * 68)


def run_integrated_mode(
    duration: float = 3.0,
    neural_steps: int = 10,
    viewer: bool = False,
    cpu: bool = False,
):
    """Run closed-loop integrated brain-drone simulation through the bridge."""
    print("=" * 68)
    print(" Closed-Loop Integrated Simulation: FlyWire Brain <-> Skydio X2 Drone")
    print("=" * 68)
    print(" Information Flow Pipeline:")
    print("   Drone sensors (IMU, Velocity, Looming)")
    print("         |")
    print("         v")
    print("   Sensory encoder (JO wind/gravity, LC4 looming, HS/VS)")
    print("         |")
    print("         v")
    print("   FlyWire brain (138,639 LIF neurons)")
    print("         |")
    print("         v")
    print("   Descending neurons (P9, DNa01, MDN, Giant Fiber)")
    print("         |")
    print("         v")
    print("   Motor decoder (Thrust, Roll, Pitch, Yaw intent)")
    print("         |")
    print("         v")
    print("   Flight controller (Safety limits, Attitude PD)")
    print("         |")
    print("         v")
    print("   MuJoCo actuators (4 rotors)")
    print("=" * 68)


    dev = "cpu" if cpu else "cuda"
    synchronizer = BrainDroneSynchronizer(
        neural_steps_per_body_step=neural_steps,
        target_alt=1.0,
        device=dev,
        plasticity=False,
    )

    steps = int(duration / synchronizer.dt_body)
    print(f"\nRunning {steps} synchronized cycles ({duration:.1f} s physics, {neural_steps} brain steps/cycle)...")

    if viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(synchronizer.body.model, synchronizer.body.data) as v:
            for s in range(steps):
                t0 = time.time()
                snap = synchronizer.step()
                v.sync()

                if (s + 1) % max(1, steps // 5) == 0:
                    print(f"  [{snap.time_sim_s:5.2f}s] Alt: {snap.position[2]:.3f} m | "
                          f"Spikes: {snap.spikes_this_step:4d} | "
                          f"JO Left/Right: {snap.sensory_summary['jo_left_rate_hz']:.1f}/{snap.sensory_summary['jo_right_rate_hz']:.1f} Hz | "
                          f"Rotor 0: {snap.motor_commands[0]:.2f} N")

                elapsed = time.time() - t0
                rem = synchronizer.dt_body - elapsed
                if rem > 0:
                    time.sleep(rem)
    else:
        for s in range(steps):
            snap = synchronizer.step()
            if (s + 1) % max(1, steps // 5) == 0:
                print(f"  [{snap.time_sim_s:5.2f}s] Alt: {snap.position[2]:.3f} m | "
                      f"Spikes: {snap.spikes_this_step:4d} | "
                      f"JO Left/Right: {snap.sensory_summary['jo_left_rate_hz']:.1f}/{snap.sensory_summary['jo_right_rate_hz']:.1f} Hz | "
                      f"Rotor 0: {snap.motor_commands[0]:.2f} N")

    final_snap = synchronizer.step()
    print("\n" + "=" * 68)
    print(" INTEGRATED SIMULATION RESULTS")
    print("=" * 68)
    print(f"  Final Position      : {np.round(final_snap.position, 3)}")
    print(f"  Final Velocity      : {np.round(final_snap.velocity, 3)}")
    print(f"  Final Motor Commands: {np.round(final_snap.motor_commands, 3)} N")
    print(f"  Safety Triggered    : {final_snap.safety_triggered} ({final_snap.safety_reason})")
    print("  [OK] Closed-loop brain-to-drone pipeline successfully executed.")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Drosophila Brain <-> Skydio X2 Drone Simulation Launcher"
    )
    # Mode selectors
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Run closed-loop integrated brain-drone simulation through the bridge",
    )
    parser.add_argument(
        "--brain",
        "--with-brain",
        action="store_true",
        dest="brain",
        help="Run standalone FlyWire biological connectome simulation",
    )
    parser.add_argument(
        "--drone",
        action="store_true",
        help="Run standalone MuJoCo drone embodiment simulation",
    )
    parser.add_argument(
        "--flight-demo",
        action="store_true",
        help="Run autonomous waypoint flight mission in 3D viewer",
    )

    # Simulation options
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch interactive MuJoCo 3D passive viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of simulation steps to execute (default: 200)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration in seconds for integrated mode (default: 3.0)",
    )
    parser.add_argument(
        "--neural-steps",
        type=int,
        default=10,
        help="Number of brain steps per body step in integrated mode (default: 10)",
    )
    parser.add_argument(
        "--stimulus",
        type=str,
        default="sugar",
        choices=list(STIMULI.keys()) + ["none"],
        help="Sensory stimulus for brain-only mode (default: sugar)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (slow for connectome)",
    )
    args = parser.parse_args()

    # 1. Autonomous flight mission demo
    if args.flight_demo:
        from experiments.drone_pid_flight import run_flight
        run_flight(duration=30.0, headless=not args.viewer)
        return

    # 2. Integrated closed-loop simulation
    if args.integrated:
        run_integrated_mode(
            duration=args.duration,
            neural_steps=args.neural_steps,
            viewer=args.viewer,
            cpu=args.cpu,
        )
        return

    # 3. Standalone brain mode
    if args.brain:
        run_brain_mode(
            stimulus=args.stimulus,
            steps=args.steps,
            cpu=args.cpu,
        )
        return

    # 4. Standalone drone mode
    if args.drone:
        run_drone_mode(
            steps=args.steps,
            viewer=args.viewer,
        )
        return

    # 5. Default: Clean Architecture Check
    print("=" * 68)
    print(" Drosophila Brain <-> MuJoCo Skydio X2 Quadrotor Project")
    print("=" * 68)
    print("\n[1/3] Verifying Physical Embodiment (body/)...")
    body = Body()
    initial_sensors = body.read_sensors()
    print(f"  MuJoCo Model: {body.model.ngeom} geoms, {body.model.nbody} bodies loaded.")
    print(f"  Initial Position: {initial_sensors['position']} m | Timestep: {body.timestep*1000:.1f} ms")

    print("\n[2/3] Verifying Brain-Body Bridge Architecture (bridge/)...")
    from bridge import SensoryEncoder, MotorDecoder, DroneFlightController
    enc = SensoryEncoder()
    dec = MotorDecoder()
    ctrl = DroneFlightController()
    print(f"  Sensory Encoder : {len(enc.jo_left_ids)+len(enc.jo_right_ids)} JO wind neurons, {len(enc.lc4_left_ids)+len(enc.lc4_right_ids)} LC4 visual neurons active.")
    print("  Motor Decoder   : P9 forward, DNa01/02 steering, MDN braking, GF escape mapped.")
    print("  Flight Controller: Safety envelope, rate limiter, and X-configuration mixer ready.")

    print("\n[3/3] Testing Open-Loop Physics Baseline (50 steps)...")
    body.set_motor_commands([3.2495625] * 4)
    for _ in range(50):
        body.step()
    final_sensors = body.read_sensors()
    print(f"  Baseline physics stepped successfully. Altitude: {final_sensors['position'][2]:.3f} m")

    print("\n" + "=" * 68)
    print(" SYSTEM MODULARITY VERIFIED")
    print("=" * 68)
    print("Available execution modes:")
    print("  python main.py --integrated          # Closed-loop brain-controlled drone")
    print("  python main.py --integrated --viewer # Live 3D interactive viewer with brain control")
    print("  python main.py --drone               # Standalone drone physics simulation")
    print("  python main.py --brain               # Standalone biological connectome simulation")
    print("  python experiments/brain_drone_hover.py    # Closed-loop hover experiment")
    print("  python experiments/brain_drone_response.py # Disturbance recovery experiment")
    print("  python experiments/brain_drone_escape.py   # Visual looming escape reflex")
    print("=" * 68)


if __name__ == "__main__":
    main()
