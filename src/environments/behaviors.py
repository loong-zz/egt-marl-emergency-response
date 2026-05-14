"""
Agent Behavior Strategies - Strategy Pattern Implementation

This module contains behavior strategies for different agent types,
following the Strategy design pattern.

Each behavior class encapsulates the complete behavior logic for a specific agent type.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np
import logging

from .config.constants import TREATMENT_RANGE, MODERATE_RESOURCES_NEEDED, ARRIVAL_RANGE, MIN_MOVE_DISTANCE, POSITION_CHANGE_THRESHOLD, ResourceType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .disaster_sim import DisasterSim
    from .entities.agent import RescueAgent


class AgentBehavior(ABC):
    """
    Abstract base class for agent behavior strategies.

    Defines the interface for all agent behaviors.
    """

    @abstractmethod
    def process(self, agent: 'RescueAgent', environment: 'DisasterSim') -> None:
        """
        Process the agent's behavior for one time step.

        Args:
            agent: The agent to process
            environment: The simulation environment
        """
        pass

    def _detect_casualties_on_move(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Detect casualties within detection range while moving."""
        detection_range = agent.get_detection_range()

        for casualty in env.casualties.values():
            if casualty.id in agent.known_casualties:
                continue
            dist = np.linalg.norm(agent.position - casualty.position)
            if dist > detection_range:
                continue
            is_first_discoverer = casualty.discovered_by is None
            if is_first_discoverer:
                casualty.discovered_by = agent.id
                casualty.discovered_at = env.current_time
            agent.known_casualties[casualty.id] = {
                'position': casualty.position.copy(),
                'severity': casualty.severity,
                'survival_probability': casualty.survival_probability,
                'discovered_at': env.current_time if is_first_discoverer else casualty.discovered_at
            }


class PersonnelBehavior(AgentBehavior):
    """
    Behavior strategy for personnel rescue agents.

    Handles:
    - Treatment of casualties
    - Movement to targets
    - Resource management
    - Depot resupply
    - Casualty discovery
    """

    def process(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Process personnel agent behavior - main entry point."""
        self._detect_casualties_on_move(agent, env)

        if self._handle_existing_mission(agent, env):
            return

        if self._needs_refill(agent):
            if self._assign_depot_mission(agent, env):
                return

        if self._assign_treatment_mission(agent, env):
            return

        self._search_for_casualties(agent, env)

    def _handle_existing_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Handle existing mission. Returns True if mission was handled."""
        mission = getattr(agent, 'current_mission', None)
        if not mission:
            return False

        if mission.startswith("go_to_depot_"):
            return self._handle_depot_mission(agent, env, mission)

        if mission.startswith("treat_casualty_"):
            casualty_id = int(mission.replace("treat_casualty_", ""))
            return self._handle_treat_mission(agent, env, casualty_id)

        if mission.startswith("searching_casualty_"):
            casualty_id = int(mission.replace("searching_casualty_", ""))
            return self._handle_search_mission(agent, env, casualty_id)

        if mission.startswith("go_to_casualty_"):
            casualty_id = int(mission.replace("go_to_casualty_", ""))
            return self._handle_go_to_casualty_mission(agent, env, casualty_id)

        return False

    def _handle_depot_mission(self, agent: 'RescueAgent', env: 'DisasterSim', mission: str) -> bool:
        """Handle go_to_depot mission."""
        depot_id = int(mission.replace("go_to_depot_", ""))
        if depot_id not in env.resource_depots:
            agent.current_mission = None
            return False

        depot = env.resource_depots[depot_id]
        distance = np.linalg.norm(agent.position - depot.position)
        if distance <= ARRIVAL_RANGE:
            env.resource_manager.refill_from_depot(agent, depot)
            agent.current_mission = None
            agent._has_refilled = True
            return True
        else:
            self._move_towards(agent, depot.position, env)
            return True

    def _handle_treat_mission(self, agent: 'RescueAgent', env: 'DisasterSim', casualty_id: int) -> bool:
        """Handle treat_casualty mission."""
        if casualty_id not in env.casualties:
            agent.current_mission = None
            return False

        casualty = env.casualties[casualty_id]

        if casualty.treated or not casualty.is_alive(env.current_time):
            agent.current_mission = None
            return False

        if casualty.treating_agent_id is not None and casualty.treating_agent_id != agent.id:
            agent.current_mission = None
            return False

        distance = np.linalg.norm(agent.position - casualty.position)
        if distance > TREATMENT_RANGE:
            self._move_towards(agent, casualty.position, env)
            return True

        if env.treatment_manager.can_treat_casualty(agent, casualty):
            env.treatment_manager.process_treatment_step(agent, casualty, env.current_time)
            return True
        else:
            agent.current_mission = None
            return False

    def _handle_search_mission(self, agent: 'RescueAgent', env: 'DisasterSim', casualty_id: int) -> bool:
        """Handle searching_casualty mission."""
        if casualty_id not in env.casualties:
            agent.current_mission = None
            return False

        casualty = env.casualties[casualty_id]

        if casualty.treated or not casualty.is_alive(env.current_time):
            agent.current_mission = None
            return False

        distance = np.linalg.norm(agent.position - casualty.position)

        if casualty.discovered_by is None:
            if distance <= agent.get_detection_range():
                casualty.discovered_by = agent.id
                casualty.discovered_at = env.current_time
                agent.known_casualties[casualty.id] = {
                    'position': casualty.position.copy(),
                    'severity': casualty.severity,
                    'survival_probability': casualty.survival_probability,
                    'discovered_at': env.current_time
                }

        if distance > TREATMENT_RANGE:
            self._move_towards(agent, casualty.position, env)
        elif casualty.treating_agent_id is None or casualty.treating_agent_id == agent.id:
            if env.treatment_manager.can_treat_casualty(agent, casualty):
                env.treatment_manager.process_treatment_step(agent, casualty, env.current_time)
                agent.current_mission = f"treat_casualty_{casualty.id}"
        return True

    def _handle_go_to_casualty_mission(self, agent: 'RescueAgent', env: 'DisasterSim', casualty_id: int) -> bool:
        """Handle go_to_casualty mission."""
        if casualty_id not in env.casualties:
            agent.current_mission = None
            return False

        casualty = env.casualties[casualty_id]

        if casualty.treated or not casualty.is_alive(env.current_time):
            agent.current_mission = None
            return False

        distance = np.linalg.norm(agent.position - casualty.position)
        if distance > TREATMENT_RANGE:
            self._move_towards(agent, casualty.position, env)
            return True

        if env.treatment_manager.can_treat_casualty(agent, casualty):
            env.treatment_manager.process_treatment_step(agent, casualty, env.current_time)
            agent.current_mission = f"treat_casualty_{casualty.id}"
        else:
            agent.current_mission = None
        return True

    def _needs_refill(self, agent: 'RescueAgent') -> bool:
        """Check if agent needs to refill resources. Returns True if any resource is insufficient for MODERATE casualty."""
        if getattr(agent, '_has_refilled', False):
            return False

        for rt, needed in MODERATE_RESOURCES_NEEDED.items():
            if agent.capacity.get(rt, 0.0) < needed:
                return True
        return False

    def _assign_depot_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Assign depot refill mission. Returns True if mission was assigned."""
        nearest_depot = self._find_nearest_depot(agent, env)
        if nearest_depot:
            agent.current_mission = f"go_to_depot_{nearest_depot.id}"
            return True
        return False

    def _assign_treatment_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Find highest priority known casualty and assign treatment mission. Returns True if mission was assigned."""
        priority_casualty = self._find_highest_priority_casualty(agent, env)
        if priority_casualty:
            distance = np.linalg.norm(agent.position - priority_casualty.position)
            if distance <= TREATMENT_RANGE:
                if env.treatment_manager.can_treat_casualty(agent, priority_casualty):
                    env.treatment_manager.process_treatment_step(agent, priority_casualty, env.current_time)
                    agent.current_mission = f"treat_casualty_{priority_casualty.id}"
                    return True
            self._move_towards(agent, priority_casualty.position, env)
            agent.current_mission = f"treat_casualty_{priority_casualty.id}"
            return True

        if agent.known_casualties:
            nearest = self._find_nearest_known_casualty(agent, env)
            if nearest:
                self._move_towards(agent, nearest.position, env)
                agent.current_mission = f"go_to_casualty_{nearest.id}"
                return True

        return False

    def _find_nearest_depot(self, agent: 'RescueAgent', env: 'DisasterSim') -> Optional['ResourceDepot']:
        """Find the nearest resource depot."""
        nearest = None
        min_dist = float('inf')
        for depot in env.resource_depots.values():
            dist = np.linalg.norm(agent.position - depot.position)
            if dist < min_dist:
                min_dist = dist
                nearest = depot
        return nearest

    def _find_highest_priority_casualty(self, agent: 'RescueAgent', env: 'DisasterSim') -> Optional['Casualty']:
        """Find the highest priority known casualty that can be treated."""
        best = None
        best_priority = -float('inf')

        severity_weight = {
            'CRITICAL': 1000,
            'SEVERE': 100,
            'MODERATE': 10,
            'MILD': 1
        }

        for casualty_id in agent.known_casualties:
            if casualty_id not in env.casualties:
                continue
            casualty = env.casualties[casualty_id]
            if casualty.treated or not casualty.is_alive(env.current_time):
                continue
            if casualty.treating_agent_id is not None and casualty.treating_agent_id != agent.id:
                continue

            dist = np.linalg.norm(agent.position - casualty.position)
            priority = severity_weight.get(casualty.severity.name, 0) - dist / 10.0

            if priority > best_priority:
                best_priority = priority
                best = casualty

        return best

    def _find_nearest_known_casualty(self, agent: 'RescueAgent', env: 'DisasterSim') -> Optional['Casualty']:
        """Find the nearest known untreated casualty."""
        nearest = None
        min_dist = float('inf')

        for casualty_id in agent.known_casualties:
            if casualty_id not in env.casualties:
                continue
            casualty = env.casualties[casualty_id]
            if casualty.treated or not casualty.is_alive(env.current_time):
                continue
            dist = np.linalg.norm(agent.position - casualty.position)
            if dist < min_dist:
                min_dist = dist
                nearest = casualty

        return nearest

    def _search_for_casualties(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Search for undiscovered casualties within detection range."""
        detection_range = agent.get_detection_range()

        for casualty in env.casualties.values():
            if casualty.discovered_by is not None:
                continue
            dist = np.linalg.norm(agent.position - casualty.position)
            if dist <= detection_range:
                casualty.discovered_by = agent.id
                casualty.discovered_at = env.current_time
                agent.known_casualties[casualty.id] = {
                    'position': casualty.position.copy(),
                    'severity': casualty.severity,
                    'survival_probability': casualty.survival_probability,
                    'discovered_at': env.current_time
                }

        if not agent.known_casualties:
            self._random_explore(agent, env)

    def _move_towards(self, agent: 'RescueAgent', target_position: np.ndarray, env: 'DisasterSim') -> None:
        """Move agent towards target position."""
        direction = target_position - agent.position
        distance = np.linalg.norm(direction)
        if distance > MIN_MOVE_DISTANCE:
            direction = direction / distance
            max_speed = agent.get_max_speed()
            old_position = agent.position.copy()
            agent.position += direction * max_speed * env.config.time_step
            map_size = env.map_size[0] if isinstance(env.map_size, (tuple, list, np.ndarray)) else env.map_size
            agent.position = np.clip(agent.position, 0, map_size)

            position_changed = np.linalg.norm(agent.position - old_position) > POSITION_CHANGE_THRESHOLD
            if position_changed:
                self._detect_casualties_on_move(agent, env)

    def _random_explore(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Move agent using grid-based exploration strategy."""
        map_size = env.map_size[0] if isinstance(env.map_size, (tuple, list, np.ndarray)) else env.map_size

        grid_size = 100.0
        num_grids = int(map_size / grid_size)

        if not hasattr(agent, '_explored_grids'):
            agent._explored_grids = set()
        if not hasattr(agent, '_current_grid_target') or agent._current_grid_target is None:
            agent._current_grid_target = self._select_unexplored_grid(agent, num_grids, map_size)

        self._move_towards(agent, agent._current_grid_target, env)

        distance = np.linalg.norm(agent.position - agent._current_grid_target)
        if distance < TREATMENT_RANGE:
            current_grid = self._get_grid_coords(agent.position, grid_size)
            agent._explored_grids.add(current_grid)
            agent._current_grid_target = self._select_unexplored_grid(agent, num_grids, map_size)

    def _get_grid_coords(self, position: np.ndarray, grid_size: float) -> tuple:
        """Get grid coordinates for a given position."""
        return (int(position[0] / grid_size), int(position[1] / grid_size))

    def _select_unexplored_grid(self, agent: 'RescueAgent', num_grids: int, map_size: float) -> np.ndarray:
        """Select an unexplored grid cell as the next exploration target."""
        grid_size = map_size / num_grids

        unexplored = []
        for i in range(num_grids):
            for j in range(num_grids):
                if (i, j) not in agent._explored_grids:
                    unexplored.append((i, j))

        if unexplored:
            grid_i, grid_j = unexplored[np.random.randint(len(unexplored))]
            return np.array([grid_i * grid_size + grid_size / 2,
                            grid_j * grid_size + grid_size / 2])
        else:
            return np.random.rand(2) * map_size


class VehicleBehavior(PersonnelBehavior):
    """
    Behavior strategy for vehicle rescue agents.

    Vehicles have higher capacity and speed than personnel.
    Handles the same behaviors but with vehicle-specific characteristics.
    """

    def process(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Process vehicle agent behavior."""
        super().process(agent, env)


class DroneBehavior(AgentBehavior):
    """
    Behavior strategy for drone agents.

    Handles:
    - Resource delivery to agents
    - Casualty search
    - Depot resupply
    - Patrol behavior
    """

    def process(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Process drone agent behavior."""
        self._detect_casualties_on_move(agent, env)

        total_resources = sum(agent.capacity.values())
        total_max = sum(agent.max_capacity.values())

        if total_resources < total_max * env.config.drone_resource_threshold:
            env.drone_manager.return_to_depot(agent, env.resource_depots)
            return

        mission = getattr(agent, 'current_mission', None)
        if mission and mission.startswith("go_to_agent_"):
            target_agent_id = int(mission.replace("go_to_agent_", ""))
            if target_agent_id in env.rescue_agents:
                target_agent = env.rescue_agents[target_agent_id]
                if env.drone_manager.deliver_resources(agent, target_agent):
                    agent.current_mission = None
            return

        needy_agent = env.drone_manager.find_needy_agent(agent, env.rescue_agents)
        if needy_agent:
            distance = np.linalg.norm(agent.position - needy_agent.position)
            if distance > 10.0:
                env.drone_manager.move_to_target(agent, needy_agent.position)
                agent.current_mission = f"go_to_agent_{needy_agent.id}"
            else:
                env.drone_manager.deliver_resources(agent, needy_agent)
            return

        if not getattr(agent, 'current_mission', None):
            env.drone_manager.search_casualties(agent, env.casualties)
