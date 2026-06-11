"""
Managers module for disaster simulation.

This module contains manager classes that handle specific aspects of the simulation:
- ResourceManager: Resource allocation and management
- TreatmentManager: Casualty treatment logic
- DroneManager: Drone behavior coordination
- EGTManager: Evolutionary Game Theory meta-controller for fairness-efficiency trade-off
- ReputationManager: Incentive-compatible reputation system with Bayesian verification
- ParetoFrontierManager: Dynamic Pareto frontier for efficiency-fairness trade-off
- CommunicationManager: Agent-to-agent communication for casualty information sharing
- CommunicationInterference: Communication interference model with delay and packet loss
"""

from .resource_manager import ResourceManager
from .treatment_manager import TreatmentManager
from .drone_manager import DroneManager
from .egt_manager import EGTManager
from .reputation_manager import ReputationManager
from .pareto_manager import ParetoFrontierManager
from .communication_manager import CommunicationManager
from .communication_interference import CommunicationInterference

__all__ = ['ResourceManager', 'TreatmentManager', 'DroneManager', 'EGTManager', 'ReputationManager', 'ParetoFrontierManager', 'CommunicationManager', 'CommunicationInterference']
