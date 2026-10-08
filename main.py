"""
main.py
-------
Top-level application entry point for the Drosophila Brain <-> Skydio X2 Drone project.

By default, running `python main.py` launches the interactive 3D simulation with the
biological connectome controlling the drone in real time.

Supported execution modes:
    1. Default / Interactive Launch:
       python main.py                        # Launches 3D viewer with brain-controlled drone
    2. Headless closed-loop simulation:
       python main.py --headless --duration 5.0
    3. Standalone Drone physical embodiment:
       python main.py --drone
       python main.py --drone --headless
    4. Standalone FlyWire biological connectome:
       python main.py --brain
       python main.py --brain --stimulus sugar
    5. Autonomous waypoint flight mission demo:
       python main.py --flight-demo
    6. System modularity & architecture check:
       python main.py --check
"""

import time
import argparse
from typing import Optional
import numpy as np

from body import Body
from brain import Brain, FlyBrain, STIMULI
from bridge import BrainDroneSynchronizer


def run_drone_mode(steps: Optional[int] = None, viewer: bool = True):
    """Run standalone drone physics simulation without neural connectome."""
    print("=" * 68)
    print(" Standalone MuJoCo Skydio X2 Drone Embodiment")
    print("=" * 68)
    body = Body()
    initial_sensors = body.read_sensors()
    print(f"  Model loaded: {body.model.ngeom} geoms, {body.model.nbody} bodies")
    print(f"  Initial position: {initial_sensors['position']}")
    print(f"  Physics timestep: {body.timestep * 1000.0:.1f} ms ({1.0 / body.timestep:.0f} Hz)")

    hover_thrust = [3.2495625] * 4

    if viewer:
        print("  Launching interactive 3D viewer (close window to exit)...")
        import mujoco.viewer
        with mujoco.viewer.launch_passive(body.model, body.data) as v:
            step_count = 0
            while v.is_running():
                t0 = time.time()
                body.set_motor_commands(hover_thrust)
                body.step()
                v.sync()
                step_count += 1
                if steps is not None and step_count >= steps:
                    break
                dt_rem = body.timestep - (time.time() - t0)
                if dt_rem > 0:
                    time.sleep(dt_rem)
    else:
        n_steps = steps if steps is not None else 200
        print(f"  Stepping drone physics ({n_steps} steps headless)...")
        for _ in range(n_steps):
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
    duration: Optional[float] = None,
    neural_steps: int = 10,
    viewer: bool = True,
    cpu: bool = False,
    target: Optional[str] = None,
):
    """Run closed-loop integrated brain-drone simulation through the bridge."""
    print("=" * 68)
    print(" Closed-Loop Integrated Simulation: FlyWire Brain <-> Skydio X2 Drone")
    print("=" * 68)
    print(" Information Flow Pipeline:")
    if target == "sugar":
        print("   Visual Target (White Sugar Cube in MuJoCo scene)")
        print("         |")
        print("         v")
        print("   drone_pov camera (MuJoCoCamera 320x240 RGB)")
        print("         |")
        print("         v")
        print("   Sugar detector (WhiteSugarDetector image localization)")
        print("         |")
        print("         v")
        print("   Target encoder (LC10 small-target tracking pathway)")
        print("         |")
        print("         v")
    else:
        print("   Drone sensors (IMU, Velocity, Looming)")
        print("         |")
        print("         v")
        print("   Sensory encoder (JO wind/gravity, LC4 looming, HS/VS)")
        print("         |")
        print("         v")
    print("   FlyWire brain (138,639 LIF neurons)")
    print("         |")
    print("         v")
    print("   Descending neurons (P9, DNa01/02, MDN, Giant Fiber)")
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

    camera = None
    detector = None
    if target == "sugar":
        from object_detection import MuJoCoCamera, WhiteSugarDetector
        camera = MuJoCoCamera(synchronizer.body.model, synchronizer.body.data, "drone_pov")
        detector = WhiteSugarDetector()
        print("  [Vision] Embodied Target Tracking Active: 'sugar' -> drone_pov -> LC10")

    if viewer:
        print("\n  Interactive Controls (focus the 3D Viewer window):")
        print("    W / Up    : Forward propulsion drive (stimulates P9 neurons)")
        print("    S / Down  : Backward braking drive (stimulates MDN neurons)")
        print("    A / Left  : Turn left (applies antennal wind differential)")
        print("    D / Right : Turn right (applies antennal wind differential)")
        print("    Space     : Visual looming hazard (stimulates LC4 -> Giant Fiber escape surge)")
        print("    R         : Reset to initial hover")
        print("    Close window to exit\n")

        fwd_bias = 0.0
        steer_bias = 0.0
        active_looming = None
        last_key_time = 0.0

        def key_callback(keycode: int):
            nonlocal fwd_bias, steer_bias, active_looming, last_key_time
            last_key_time = time.time()
            if keycode in (87, ord('w'), 265):
                fwd_bias = min(1.0, fwd_bias + 0.4)
                print("  [Stimulus] >> Forward Drive (P9 activated)")
            elif keycode in (83, ord('s'), 264):
                fwd_bias = max(-1.0, fwd_bias - 0.4)
                print("  [Stimulus] >> Backward Braking (MDN activated)")
            elif keycode in (65, ord('a'), 263):
                steer_bias = max(-1.0, steer_bias - 0.4)
                print("  [Stimulus] >> Turn Left (Antennal differential)")
            elif keycode in (68, ord('d'), 262):
                steer_bias = min(1.0, steer_bias + 0.4)
                print("  [Stimulus] >> Turn Right (Antennal differential)")
            elif keycode == 32:
                active_looming = {"threat_level": 1.0, "side": "center"}
                print("  [Stimulus] >> Visual Looming Hazard (LC4 -> Giant Fiber ESCAPE surge!)")
            elif keycode in (82, ord('r')):
                fwd_bias = 0.0
                steer_bias = 0.0
                active_looming = None
                synchronizer.reset()
                print("  [Reset] >> Hover state reset")

        import mujoco.viewer
        with mujoco.viewer.launch_passive(
            synchronizer.body.model,
            synchronizer.body.data,
            key_callback=key_callback,
        ) as v:
            step_idx = 0
            while v.is_running():
                t0 = time.time()

                # Smoothly decay steering and forward impulses after release
                if time.time() - last_key_time > 0.35:
                    fwd_bias *= 0.88
                    steer_bias *= 0.88
                    active_looming = None

                target_det = None
                if camera is not None and detector is not None:
                    target_det = detector.detect(camera.get_frame())

                stim_bias = {"forward": fwd_bias, "steer": steer_bias}
                snap = synchronizer.step(
                    looming_threat=active_looming,
                    stimulus_bias=stim_bias,
                    target_detection=target_det,
                )
                v.sync()
                step_idx += 1

                if step_idx % 25 == 0:
                    status_line = (
                        f"  [{snap.time_sim_s:5.2f}s] Alt: {snap.position[2]:.3f} m | "
                        f"Spikes: {snap.spikes_this_step:4d} | "
                        f"JO L/R: {snap.sensory_summary['jo_left_rate_hz']:.1f}/{snap.sensory_summary['jo_right_rate_hz']:.1f} Hz"
                    )
                    if snap.target_summary is not None:
                        status_line += (
                            f" | LC10 L/R: {snap.target_summary['lc10_left_rate_hz']:.1f}/"
                            f"{snap.target_summary['lc10_right_rate_hz']:.1f} Hz"
                        )
                    status_line += f" | Rotor 0: {snap.motor_commands[0]:.2f} N"
                    print(status_line)

                if duration is not None and snap.time_sim_s >= duration:
                    break

                elapsed = time.time() - t0
                rem = synchronizer.dt_body - elapsed
                if rem > 0:
                    time.sleep(rem)

    else:
        run_dur = duration if duration is not None else 3.0
        steps = int(run_dur / synchronizer.dt_body)
        print(f"\n  Running {steps} synchronized cycles ({run_dur:.1f} s physics headless)...")
        for s in range(steps):
            target_det = None
            if camera is not None and detector is not None:
                target_det = detector.detect(camera.get_frame())

            snap = synchronizer.step(target_detection=target_det)
            if (s + 1) % max(1, steps // 5) == 0:
                status_line = (
                    f"  [{snap.time_sim_s:5.2f}s] Alt: {snap.position[2]:.3f} m | "
                    f"Spikes: {snap.spikes_this_step:4d} | "
                    f"JO L/R: {snap.sensory_summary['jo_left_rate_hz']:.1f}/{snap.sensory_summary['jo_right_rate_hz']:.1f} Hz"
                )
                if snap.target_summary is not None:
                    status_line += (
                        f" | LC10 L/R: {snap.target_summary['lc10_left_rate_hz']:.1f}/"
                        f"{snap.target_summary['lc10_right_rate_hz']:.1f} Hz"
                    )
                status_line += f" | Rotor 0: {snap.motor_commands[0]:.2f} N"
                print(status_line)

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


def run_check_mode():
    """Run fast modularity and architecture verification."""
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
    print("To launch the live interactive 3D simulation with brain control:")
    print("  python main.py")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Drosophila Brain <-> Skydio X2 Drone Simulation Launcher"
    )
    # Mode selectors
    parser.add_argument(
        "--integrated",
        action="store_true",
        help="Run closed-loop integrated brain-drone simulation through the bridge (default)",
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run quick system modularity and architecture verification without launching viewer",
    )

    # Simulation options
    parser.add_argument(
        "--viewer",
        action="store_true",
        default=None,
        help="Force launch of interactive MuJoCo 3D passive viewer",
    )
    parser.add_argument(
        "--headless",
        "--no-viewer",
        action="store_true",
        dest="headless",
        help="Run headless without opening a 3D viewer window",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps to execute",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds for simulation",
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
        "--target",
        type=str,
        default=None,
        choices=["sugar", "none"],
        help="Visual target object for embodied tracking experiment (e.g. sugar)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (slow for connectome)",
    )
    args = parser.parse_args()

    # Determine viewer flag: Default is True (interactive) unless --headless is passed
    use_viewer = not args.headless
    if args.viewer is True:
        use_viewer = True

    # 1. Modularity check mode
    if args.check:
        run_check_mode()
        return

    # 2. Autonomous flight mission demo
    if args.flight_demo:
        from experiments.drone_pid_flight import run_flight
        run_flight(duration=args.duration or 30.0, headless=not use_viewer)
        return

    # 3. Standalone brain mode
    if args.brain:
        run_brain_mode(
            stimulus=args.stimulus,
            steps=args.steps or 100,
            cpu=args.cpu,
        )
        return

    # 4. Standalone drone mode
    if args.drone:
        run_drone_mode(
            steps=args.steps,
            viewer=use_viewer,
        )
        return

    # 5. Default / Integrated mode: Launch the live brain-controlled drone simulation!
    run_integrated_mode(
        duration=args.duration,
        neural_steps=args.neural_steps,
        viewer=use_viewer,
        cpu=args.cpu,
        target=args.target,
    )


if __name__ == "__main__":
    main()
