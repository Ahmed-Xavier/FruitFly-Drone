# Fruit-fly brain – LIF model parameters
# Source: Kakaria & de Bivort 2017, Jürgensen et al., Lazar et al., Paul et al.

MODEL_PARAMS = {
    'tauSyn':      5.0,    # ms  — synaptic time constant (Jürgensen et al.)
    'tDelay':      1.8,    # ms  — axonal conduction delay (Paul et al. 2015)
    'v0':        -52.0,    # mV  — resting / reset potential (Kakaria & de Bivort 2017)
    'vReset':    -52.0,    # mV
    'vRest':     -52.0,    # mV
    'vThreshold':-45.0,    # mV  — spike threshold (Kakaria & de Bivort 2017)
    'tauMem':     20.0,    # ms  — membrane time constant
    'tRefrac':     2.2,    # ms  — absolute refractory period (Lazar et al.)
    'scalePoisson': 250,   # scaling factor for Poisson spike injection
    'wScale':      0.275,  # mV  — per-synapse weight multiplier
}

DT = 0.1  # ms — simulation timestep  (10 kHz → matches Brian2 defaultclock)

# Hebbian plasticity constants (Rojas Aliaga 2026)
HEBB_BATCH = 10    # accumulate spikes for N brain steps before applying update
HEBB_ETA   = 1e-4  # learning rate
HEBB_ALPHA = 1e-7  # weight decay (homeostatic)
