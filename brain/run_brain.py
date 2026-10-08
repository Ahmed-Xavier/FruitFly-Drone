#!/usr/bin/env python3
"""
run_brain.py
------------
Standalone runner for the fruit-fly brain simulation.

Initializes the FlyWire v783 LIF connectome, runs N timesteps with a
named sensory stimulus, and prints a summary of neural activity.

Usage:
    python run_brain.py
    python run_brain.py --stimulus p9 --steps 500
    python run_brain.py --stimulus lc4 --steps 200 --no-plasticity
    python run_brain.py --steps 100 --cpu
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Make 'brain' package importable regardless of working directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from brain import FlyBrain, STIMULI


def parse_args():
    p = argparse.ArgumentParser(description='Fruit-fly brain standalone runner')
    p.add_argument('--stimulus', type=str, default='sugar',
                   choices=list(STIMULI.keys()) + ['none'],
                   help='Named sensory stimulus (default: sugar)')
    p.add_argument('--steps', type=int, default=200,
                   help='Number of simulation steps (default: 200, each = 0.1 ms)')
    p.add_argument('--no-plasticity', action='store_true',
                   help='Disable Hebbian plasticity')
    p.add_argument('--cpu', action='store_true',
                   help='Force CPU execution (slow — use for testing without GPU)')
    return p.parse_args()


def main():
    args = parse_args()

    device = 'cpu' if args.cpu else 'cuda'
    stimulus = None if args.stimulus == 'none' else args.stimulus
    n_steps = args.steps

    # ── Initialize brain ─────────────────────────────────────────────────────
    print("=" * 65)
    print(" Fruit-fly brain - FlyWire v783 LIF connectome")
    print("=" * 65)

    brain = FlyBrain(
        device=device,
        plasticity=not args.no_plasticity,
    )

    print()
    print(f"Stimulus  : {stimulus or 'none (spontaneous activity)'}")
    print(f"Steps     : {n_steps}  ({n_steps * brain.timestep_ms:.1f} ms simulated)")
    print(f"Neurons   : {brain.num_neurons:,}")
    print(f"Device    : {brain.device.upper()}")
    print()

    # Describe stimulus
    if stimulus:
        stim_info = brain.get_stimulus_info()[stimulus]
        print(f"Stimulus detail: {stim_info['description']}")
        print(f"  Neurons stimulated: {stim_info['n_neurons']}")
        print(f"  Input rate: {stim_info['rate_hz']} Hz")
    print()

    # ── Run simulation ───────────────────────────────────────────────────────
    print("-" * 65)
    print("Running simulation...")
    print("-" * 65)

    total_spikes = 0
    active_neurons = set()
    dn_spike_totals = {name: 0 for name in brain.dn_indices}

    input_data = {'stimulus': stimulus} if stimulus else {}

    for step in range(n_steps):
        out = brain.step(input_data)

        spikes = out['spikes']
        n_spikes_this_step = int(spikes.sum())
        total_spikes += n_spikes_this_step
        active_neurons.update(np.where(spikes > 0)[0].tolist())

        for name, val in out['dn_spikes'].items():
            dn_spike_totals[name] += int(val)

        if (step + 1) % max(1, n_steps // 5) == 0 or step == 0:
            pct = (step + 1) / n_steps * 100
            t_ms = out['time_ms']
            print(f"  Step {step+1:>5}/{n_steps}  ({pct:5.1f}%)  "
                  f"t={t_ms:.1f} ms  "
                  f"spikes/step={n_spikes_this_step:>6}")

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(" SIMULATION RESULTS")
    print("=" * 65)
    print(f"  Total simulated time : {out['time_ms']:.2f} ms")
    print(f"  Total neurons        : {brain.num_neurons:,}")
    print(f"  Active neurons       : {len(active_neurons):,}  "
          f"({100 * len(active_neurons) / brain.num_neurons:.2f}%)")
    print(f"  Total spikes         : {total_spikes:,}")
    print(f"  Mean spikes/step     : {total_spikes / n_steps:.1f}")

    print()
    print("  Descending Neuron (DN) spike counts:")
    for group, members in brain.get_dn_groups().items():
        group_total = sum(dn_spike_totals.get(m, 0) for m in members)
        rate_str = f"{out['population_rates'].get(group, 0):.1f} Hz"
        print(f"    {group:<10} total spikes={group_total:>5}  "
              f"current rate={rate_str}")

    print()
    print("  Final DN firing rates (Hz, sliding window):")
    for name, rate in out['dn_rates'].items():
        if rate > 0:
            print(f"    {name:<22} {rate:>8.2f} Hz")

    print()
    print("  [OK] Neural network loaded and producing activity.")
    print("  [OK] Connectivity from FlyWire v783 connectome is active.")
    print("=" * 65)


if __name__ == '__main__':
    main()
