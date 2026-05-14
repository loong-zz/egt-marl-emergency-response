"""
Entities module for disaster simulation.

This module contains all entity classes that represent objects in the simulation:
- Casualty: Injured victims
- RescueAgent: Rescue personnel/vehicles/drones
- ResourceDepot: Resource storage locations
- AffectedArea: Areas affected by the disaster
"""

from .casualty import Casualty
from .agent import RescueAgent
from .depot import ResourceDepot
from .area import AffectedArea

__all__ = ['Casualty', 'RescueAgent', 'ResourceDepot', 'AffectedArea']
