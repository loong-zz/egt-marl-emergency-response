"""
DisasterSim-2026: High-fidelity disaster simulation environment for medical resource allocation.
"""

from .disaster_sim import DisasterSim
from .disaster_scenarios import (
    EarthquakeScenario,
    FloodScenario,
    HurricaneScenario,
)

# Delay import of DisasterVisualizer to speed up initial import
# Import it only when needed using: from environments import DisasterVisualizer

__all__ = [
    "DisasterSim",
    "EarthquakeScenario",
    "FloodScenario",
    "HurricaneScenario",
    "DisasterVisualizer",
]

__version__ = "1.0.0"

def __getattr__(name):
    if name == "DisasterVisualizer":
        from .visualization import DisasterVisualizer
        return DisasterVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
