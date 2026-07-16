"""entity dataclasses for the redesigned DisasterSim."""

from .agent_v2 import Agent
from .casualty_v2 import Casualty, Severity
from .depot_v2 import Depot
from .area_v2 import Area

__all__ = ["Agent", "Casualty", "Severity", "Depot", "Area"]