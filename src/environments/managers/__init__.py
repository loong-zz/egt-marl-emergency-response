"""
Managers module for disaster simulation.

This module contains manager classes that handle specific aspects of the simulation:
- ResourceManager: Resource allocation and management
- TreatmentManager: Casualty treatment logic
- DroneManager: Drone behavior coordination
"""

from .resource_manager import ResourceManager
from .treatment_manager import TreatmentManager
from .drone_manager import DroneManager

__all__ = ['ResourceManager', 'TreatmentManager', 'DroneManager']
