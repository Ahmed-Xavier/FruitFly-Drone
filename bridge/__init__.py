"""
bridge - Bidirectional Sensory-Motor Interface between Drosophila Brain and MuJoCo Drone.

Architecture:
    Drone Telemetry (IMU, Velocity, Looming)
            ↓
    SensoryEncoder (Biological rate injection into JO, LC4, HS/VS)
            ↓
    FlyWire v783 Connectome (138k LIF neurons)
            ↓
    MotorDecoder (Descending neuron decoding into flight intent)
            ↓
    DroneFlightController (Safety envelopes, attitude PD stabilization, mixer)
            ↓
    MuJoCo Skydio X2 Actuation (4 rotors)
"""

from .sensory_encoder import SensoryEncoder
from .motor_decoder import MotorDecoder
from .controller import DroneFlightController
from .synchronizer import BrainDroneSynchronizer, BridgeTelemetry

__all__ = [
    "SensoryEncoder",
    "MotorDecoder",
    "DroneFlightController",
    "BrainDroneSynchronizer",
    "BridgeTelemetry",
]
