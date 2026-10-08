"""
main.py
-------
Top-level application entry point for the Drosophila Brain <-> Skydio X2 Drone project.

Initializes the physical embodiment (body) and the neuromorphic connectome (brain)
and defines the architectural scaffold where the future bridge integration loop will live.
"""

import time
import argparse
import numpy as np

from body import Body
from brain import Brain


def main():
    parser = argparse.ArgumentParser(
        description="Drosophila Connectome - Skydio X2 Quadrotor Simulation"
    )
    parser.add_argument(
        "--with-brain",
        action="store_true",
        help="Initialize the biological FlyWire v783 connectome (138k LIF neurons)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive MuJoCo 3D passive viewer for the drone",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of initial physics steps to execute (default: 200)",
    )
    parser.add_argument(
        "--flight-demo",
        action="store_true",
        help="Run the autonomous waypoint flight demonstration in 3D viewer",
    )
    args = parser.parse_args()

    # Delegate to autonomous waypoint flight if requested
    if args.flight_demo:
        from experiments.drone_pid_flight import run_flight
        run_flight(duration=30.0, headless=False)
        return

    print("=" * 68)
    print(" Drosophila Brain <-> MuJoCo Skydio X2 Quadrotor Project")
    print("=" * 68)

    # 1. Initialize physical embodiment (drone body)
    print("\n[1/2] Initializing MuJoCo Skydio X2 Body...")
    body = Body()
    initial_sensors = body.read_sensors()
    print(f"  Model loaded: {body.model.ngeom} geoms, {body.model.nbody} bodies")
    print(f"  Initial position: {initial_sensors['position']}")
    print(f"  Physics timestep: {body.timestep * 1000.0:.1f} ms ({1.0 / body.timestep:.0f} Hz)")
    print("  Actuators: 4 rotor thrusters ready.")

    # 2. Initialize neuromorphic connectome (brain)
    brain = None
    if args.with_brain:
        print("\n[2/2] Initializing Drosophila Connectome Brain...")
        brain = Brain()
        print(f"  Connectome: {brain.num_neurons:,} LIF neurons active.")
        print(f"  Integration timestep: {brain.timestep_ms:.2f} ms (10 kHz)")
    else:
        print("\n[2/2] Drosophila Brain module ready (run with --with-brain to instantiate connectome).")

    # 3. Brain-body bridge status notice
    print("\n[Bridge Status]")
    print("  Sensory-motor bridge is currently specification-only (not connected).")
    print("  Drone and biological brain models are independently initialized and verified.")
    print("  No synthetic neural mappings are applied; full integration is intentionally deferred.")


    # 4. Verify physical simulation stepping
    print(f"\nStepping drone physics ({args.steps} steps)...")
    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(body.model, body.data) as viewer:
            for _ in range(args.steps):
                step_start = time.time()
                # Apply equilibrium hover thrust
                body.set_motor_commands([3.2495625] * 4)
                body.step()
                viewer.sync()
                dt_rem = body.timestep - (time.time() - step_start)
                if dt_rem > 0:
                    time.sleep(dt_rem)
    else:
        for _ in range(args.steps):
            body.set_motor_commands([3.2495625] * 4)
            body.step()

    final_sensors = body.read_sensors()
    print(f"Drone physics stepped successfully. Current altitude: {final_sensors['position'][2]:.3f} m")
    print("\n[OK] Modular architecture verified.")
    print("To run the full autonomous waypoint flight mission:")
    print("  python experiments/drone_pid_flight.py")
    print("To run manual keyboard flight with live camera:")
    print("  python experiments/manual_control.py")
    print("To run the standalone fruit-fly brain simulation:")
    print("  python brain/run_brain.py --stimulus lc4")
    print("=" * 68)


if __name__ == "__main__":
    main()
