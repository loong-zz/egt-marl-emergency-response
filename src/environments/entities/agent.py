"""
RescueAgent entity class for disaster simulation.

Represents a rescue agent (personnel, vehicle, or drone) in the disaster scenario.
"""

import numpy as np
from typing import Dict, List, Optional
from ..config.constants import AgentType, ResourceType, AGENT_SPEEDS, AGENT_CAPACITY, RESOURCE_ABBR, AGENT_DETECTION_RANGE, COMMUNICATION_RANGE, MODERATE_RESOURCES_NEEDED


class RescueAgent:
    """Base class for rescue agents."""

    def __init__(
        self,
        agent_id: int,
        position: np.ndarray,
        map_size: Optional[float] = None,
        agent_type: AgentType = AgentType.PERSONNEL
    ):
        self.id = agent_id
        self.position = position
        self.agent_type = agent_type
        self.velocity = np.zeros(2)

        # Set resource capacity based on agent type
        # VEHICLE > PERSONNEL > DRONE
        resource_capacity = AGENT_CAPACITY.get(agent_type, 8.0)

        self.capacity = {rt: resource_capacity for rt in ResourceType}
        self.max_capacity = {rt: resource_capacity for rt in ResourceType}
        self.endurance = 100.0
        self.max_endurance = 100.0
        self.current_mission = None
        self.connected_agents: List[int] = []
        self.map_size = map_size
        self.route: List[np.ndarray] = []
        self.known_casualties: Dict[int, Dict] = {}
        self.rescued_count = 0  # Number of casualties rescued by this agent
        self._has_refilled = False  # Flag indicating recent depot refill

        # Set behavior strategy based on agent type
        self._set_behavior_strategy()

    def _set_behavior_strategy(self) -> None:
        """Set behavior strategy based on agent type.

        Uses lazy import to avoid circular dependencies.
        """
        from ..behaviors import PersonnelBehavior, DroneBehavior, VehicleBehavior

        if self.agent_type == AgentType.DRONE:
            self.behavior = DroneBehavior()
        elif self.agent_type == AgentType.VEHICLE:
            self.behavior = VehicleBehavior()
        else:
            self.behavior = PersonnelBehavior()

    def process(self, environment) -> None:
        """
        Process the agent's behavior for one time step.

        Delegates to the behavior strategy object.

        Args:
            environment: The simulation environment
        """
        self.behavior.process(self, environment)

    def get_max_speed(self) -> float:
        """Get maximum speed of the agent in m/s based on type."""
        return AGENT_SPEEDS.get(self.agent_type, 5.0)

    def get_detection_range(self) -> float:
        """Get detection range in meters based on type."""
        return AGENT_DETECTION_RANGE.get(self.agent_type, 10.0)

    def move(self, time_step: float) -> None:
        """Move the agent based on velocity."""
        self.position += self.velocity * time_step

    def can_communicate(self, other_position: np.ndarray) -> bool:
        """Check if agent can communicate with another position."""
        distance = np.linalg.norm(self.position - other_position)
        return distance < COMMUNICATION_RANGE

    def get_total_resources(self) -> float:
        """Calculate total resources carried by the agent."""
        return sum(self.capacity.values())

    def is_resource_low(self, threshold: float = 0.2) -> bool:
        """Check if total resources are below a threshold percentage."""
        total = self.get_total_resources()
        max_total = sum(self.max_capacity.values())
        return total < max_total * threshold

    def has_full_resources(self) -> bool:
        """Check if all resources are at maximum capacity."""
        return all(self.capacity[rt] >= self.max_capacity[rt] * 0.95 for rt in ResourceType)

    def has_enough_resources_for_moderate(self) -> bool:
        """Check if agent has enough resources to treat a MODERATE casualty."""
        for rt, needed in MODERATE_RESOURCES_NEEDED.items():
            if self.capacity.get(rt, 0.0) < needed:
                return False
        return True

    def needs_resources(self) -> bool:
        """Check if agent needs resources (doesn't have enough for a MODERATE casualty)."""
        return not self.has_enough_resources_for_moderate()

    def distance_to(self, target_position: np.ndarray) -> float:
        """Calculate distance to a target position."""
        return np.linalg.norm(self.position - target_position)

    def add_known_casualty(self, casualty_id: int, severity: str, distance: float) -> None:
        """Add a casualty to the agent's known list."""
        self.known_casualties[casualty_id] = {
            'severity': severity,
            'distance': distance,
            'discovered_at': None,
            'treated': False
        }

    def remove_known_casualty(self, casualty_id: int) -> None:
        """Remove a casualty from the agent's known list."""
        if casualty_id in self.known_casualties:
            del self.known_casualties[casualty_id]

    def update_known_casualty_distance(self, casualty_id: int, distance: float) -> None:
        """Update the distance to a known casualty."""
        if casualty_id in self.known_casualties:
            self.known_casualties[casualty_id]['distance'] = distance

    def get_nearest_known_casualty(self) -> Optional[int]:
        """Get the ID of the nearest known casualty."""
        if not self.known_casualties:
            return None

        nearest = min(
            self.known_casualties.items(),
            key=lambda x: x[1]['distance']
        )
        return nearest[0]

    def format_resource_log(self) -> str:
        """Format resource information for logging."""
        return ", ".join(
            f"{RESOURCE_ABBR.get(rt.name, rt.name[:4])}:{self.capacity[rt]:.2f}"
            for rt in ResourceType
        )

    def format_position_log(self) -> str:
        """Format position information for logging."""
        return f"[{self.position[0]:.1f},{self.position[1]:.1f}]"

    def format_log_line(self) -> str:
        """Format agent state as a single log line."""
        mission = getattr(self, 'current_mission', 'None')
        resources = self.format_resource_log()
        known_casualties = list(self.known_casualties.keys())
        rescued_count = getattr(self, 'rescued_count', 0)

        known_list = ",".join(str(cid) for cid in sorted(known_casualties)) if known_casualties else "[]"

        return (
            f"AGENT {self.id}/{self.agent_type.name} | "
            f"Status={mission} | "
            f"Pos={self.format_position_log()} | "
            f"Rescued={rescued_count} | "
            f"Known=[{known_list}] | "
            f"Resources={resources}"
        )
