"""
EGT-MARL Algorithm Package
==========================

This package contains the core algorithms for the EGT-MARL framework:
- EGTMARL: Main two-layer algorithm
- MARLLayer: Multi-agent reinforcement learning layer
- EGTLayer: Evolutionary game theory layer
- AntiSpoofing: Anti-spoofing mechanism
- ImprovedQMIX: Improved QMIX implementation
- DynamicFrontier: Dynamic Pareto frontier optimization
"""

from .egt_marl import EGTMARL
from .marl_layer import MARLLayer
from .egt_layer import EGTLayer
from .anti_spoofing import AntiSpoofing
from .qmix_improved import ImprovedQMIX
from .dynamic_frontier import DynamicFrontier

__all__ = [
    'EGTMARL',
    'MARLLayer',
    'EGTLayer',
    'AntiSpoofing',
    'ImprovedQMIX',
    'DynamicFrontier',
]