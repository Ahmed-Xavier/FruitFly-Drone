"""
brain.simulation - Simulation parameters, numerical integration constants, and runners.
"""

from .params import (
    MODEL_PARAMS,
    DT,
    HEBB_BATCH,
    HEBB_ETA,
    HEBB_ALPHA,
)

__all__ = [
    "MODEL_PARAMS",
    "DT",
    "HEBB_BATCH",
    "HEBB_ETA",
    "HEBB_ALPHA",
]
