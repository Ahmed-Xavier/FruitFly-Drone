"""
params.py
---------
Biological LIF neuron parameters and synaptic plasticity constants for
the Drosophila melanogaster connectome simulation.

Sources: Kakaria & de Bivort 2017, Juergensen et al., Lazar et al., Paul et al.
"""

MODEL_PARAMS = {
    "tauSyn": 5.0,        # ms - synaptic time constant (Juergensen et al.)
    "tDelay": 1.8,        # ms - axonal conduction delay (Paul et al. 2015)
    "v0": -52.0,          # mV - initial membrane potential
    "vReset": -52.0,      # mV - reset potential
    "vRest": -52.0,       # mV - resting potential
    "vThreshold": -45.0,  # mV - spike threshold (Kakaria & de Bivort 2017)
    "tauMem": 20.0,       # ms - membrane time constant
    "tRefrac": 2.2,       # ms - absolute refractory period (Lazar et al.)
    "scalePoisson": 250,  # Poisson spike scaling factor
    "wScale": 0.275,      # mV - per-synapse weight multiplier
}

DT = 0.1  # ms - simulation timestep (10 kHz clock)

# Hebbian plasticity parameters (Rojas Aliaga 2026)
HEBB_BATCH = 10    # Apply weight update every N brain steps
HEBB_ETA = 1e-4    # Learning rate (correlated firing potentiates)
HEBB_ALPHA = 1e-7  # Homeostatic weight decay
