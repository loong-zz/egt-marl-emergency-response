"""
Agents module for EGT-MARL disaster resource allocation system.

This module contains implementations of various agents used in the disaster simulation:
- BaseAgent: Abstract base class for all agents
- RescueAgent: Rescue agents (drones, ambulances, mobile hospitals)
- MaliciousAgent: Malicious agents for robustness testing
"""

from .base_agent import BaseAgent
from .rescue_agent import RescueAgent
from .malicious_agent import MaliciousAgent

__all__ = [
    "BaseAgent",
    "RescueAgent", 
    "MaliciousAgent",
]

__version__ = "1.0.0"