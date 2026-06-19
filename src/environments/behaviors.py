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

from .config.constants import TREATMENT_RANGE, ARRIVAL_RANGE, MIN_MOVE_DISTANCE, POSITION_CHANGE_THRESHOLD, ResourceType, CasualtySeverity

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


class PersonnelBehavior(AgentBehavior):
    """
    Behavior strategy for personnel rescue agents.

    Handles:
    - Casualty detection and tracking
    - Treatment of casualties
    - Resource management
    - Depot resupply
    - Exploration

    Process flow:
        1. Detect casualties in range (unified detection)
        2. Handle existing mission (move/treat/supply)
        3. Assign new mission (prioritize treatment > supply > explore)
    """

    def process(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Process personnel agent behavior - main entry point."""
        self._detect_casualties(agent, env)

        if self._handle_existing_mission(agent, env):
            return

        if self._needs_refill(agent):
            if self._assign_depot_mission(agent, env):
                return

        if self._assign_go_to_casualty_mission(agent, env):
            return

        self._explore(agent, env)

    def _detect_casualties(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """
        Detect casualties within detection range.
        Unified detection for both stationary and moving agents.
        """
        detection_range = agent.get_detection_range()
        
        # Get set of casualties this agent has marked as untreatable
        untreatable_casualties = getattr(agent, '_untreatable_casualties', set())

        for casualty in env.casualties.values():
            if casualty.id in agent.known_casualties:
                continue
            # Skip casualties marked as untreatable (resources insufficient even when full)
            if casualty.id in untreatable_casualties:
                continue
            # Skip already treated casualties
            if casualty.treated:
                continue
            # Skip dead casualties
            if not casualty.is_alive(env.current_time):
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

    def _handle_existing_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Handle existing mission. Returns True if mission was handled."""
        mission = getattr(agent, 'current_mission', None)
        if not mission:
            return False

        if mission.startswith("go_to_depot_"):
            return self._handle_depot_mission(agent, env, mission)

        if mission.startswith("treat_casualty_"):
            casualty_id = int(mission.replace("treat_casualty_", ""))
            return self._handle_treatment_mission(agent, env, casualty_id)

        if mission.startswith("go_to_casualty_"):
            casualty_id = int(mission.replace("go_to_casualty_", ""))
            return self._handle_go_to_casualty(agent, env, casualty_id)

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
            self._navigate_to(agent, depot.position, env)
            return True

    def _handle_treatment_mission(self, agent: 'RescueAgent', env: 'DisasterSim', casualty_id: int) -> bool:
        """Handle treat_casualty mission - agent has arrived and should treat."""
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

        completed = env.treatment_manager.process_treatment_step(agent, casualty, env.current_time)
        if completed:
            agent.current_mission = None
        return True

    def _handle_go_to_casualty(self, agent: 'RescueAgent', env: 'DisasterSim', casualty_id: int) -> bool:
        """Handle go_to_casualty mission - navigate to casualty location."""
        if casualty_id not in env.casualties:
            agent.current_mission = None
            return False

        casualty = env.casualties[casualty_id]

        if casualty.treated or not casualty.is_alive(env.current_time):
            agent.current_mission = None
            return False

        distance = np.linalg.norm(agent.position - casualty.position)
        if distance > ARRIVAL_RANGE:
            self._navigate_to(agent, casualty.position, env)
            return True

        if env.treatment_manager.can_treat_casualty(agent, casualty):
            # Switch to treatment mission; actual treatment starts next step
            agent.current_mission = f"treat_casualty_{casualty_id}"
            return True
        else:
            lower_casualty = self._find_lower_resource_casualty(agent, env, casualty.severity)
            if lower_casualty:
                agent.current_mission = f"go_to_casualty_{lower_casualty.id}"
                logger.debug(
                    f"[GO TO CASUALTY] Agent{agent.id} cannot treat Casualty{casualty_id} "
                    f"(Severity={casualty.severity.name}) - Lower severity casualty {lower_casualty.id}"
                )
                return True
            
            # No lower priority casualty available
            # Check if resources are already full (cannot benefit from depot)
            if agent.has_full_resources():
                # Record this casualty as untreatable (resource capacity insufficient)
                if not hasattr(agent, '_untreatable_casualties'):
                    agent._untreatable_casualties = set()
                agent._untreatable_casualties.add(casualty_id)
                
                logger.debug(
                    f"[GO TO CASUALTY] Agent{agent.id} cannot treat Casualty{casualty_id} "
                    f"(Severity={casualty.severity.name}) - Resources full but insufficient, "
                    f"marking as untreatable and exploring"
                )
                self._explore(agent, env)
                return True
            
            # Resources not full - go to depot to refill
            return self._assign_depot_mission(agent, env)

    def _handle_exploration(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Handle exploring mission - grid-based exploration."""
        map_size = env.map_size[0] if isinstance(env.map_size, (tuple, list, np.ndarray)) else env.map_size
        grid_size = 100.0
        num_grids = int(map_size / grid_size)

        if not hasattr(agent, '_explored_grids'):
            agent._explored_grids = set()
        if not hasattr(agent, '_exploration_target') or agent._exploration_target is None:
            agent._exploration_target = self._select_unexplored_grid(agent, num_grids, map_size)

        distance = np.linalg.norm(agent.position - agent._exploration_target)
        if distance <= ARRIVAL_RANGE:
            current_grid = self._get_grid_coords(agent.position, grid_size)
            agent._explored_grids.add(current_grid)
            agent._exploration_target = self._select_unexplored_grid(agent, num_grids, map_size)
        else:
            self._navigate_to(agent, agent._exploration_target, env)

        return True

    def _needs_refill(self, agent: 'RescueAgent') -> bool:
        """Check if agent needs to refill resources."""
        if getattr(agent, '_has_refilled', False):
            return False

        return agent.needs_resources()

    def _assign_depot_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Assign go_to_depot mission."""
        nearest_depot = self._find_nearest_depot(agent, env)
        if nearest_depot:
            agent.current_mission = f"go_to_depot_{nearest_depot.id}"
            logger.debug(
                f"[GO TO DEPOT] Agent{agent.id} Go to depot {nearest_depot.id}"
            )
            return True
        return False

    def _assign_go_to_casualty_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Assign go_to_casualty mission."""
        if not agent.known_casualties:
            return False
        
        logger.debug(f"[ASSIGN CASUALTY] Agent{agent.id} known_casualties count: {len(agent.known_casualties)}")
        
        priority_casualty = self._find_highest_priority_casualty(agent, env)
        if priority_casualty:
            agent.current_mission = f"go_to_casualty_{priority_casualty.id}"
            logger.debug(
                f"[ASSIGN CASUALTY] Agent{agent.id} assigned priority casualty {priority_casualty.id} "
                f"(Severity={priority_casualty.severity.name})"
            )
            return True
        
        # logger.debug(f"[ASSIGN CASUALTY] Agent{agent.id} no priority casualty found")

        nearest = self._find_nearest_known_casualty(agent, env)
        if nearest:
            agent.current_mission = f"go_to_casualty_{nearest.id}"
            logger.debug(
                f"[ASSIGN CASUALTY] Agent{agent.id} assigned nearest casualty {nearest.id} "
                f"(Severity={nearest.severity.name})"
            )
            return True
        
        logger.debug(f"[ASSIGN CASUALTY] Agent{agent.id} known_casualties not assignable (all filtered out)")
        return False

    def _explore(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Start or continue exploration when no other missions."""
        agent.current_mission = "exploring"
        if not hasattr(agent, '_exploration_target'):
            agent._exploration_target = None
        self._handle_exploration(agent, env)

    def _navigate_to(self, agent: 'RescueAgent', target_position: np.ndarray, env: 'DisasterSim') -> None:
        """Navigate towards target position with detection during movement."""
        direction = target_position - agent.position
        distance = np.linalg.norm(direction)
        
        if distance <= ARRIVAL_RANGE:
            agent.position = target_position.copy()
            return

        direction = direction / distance
        max_speed = agent.get_max_speed()
        old_position = agent.position.copy()
        
        step_distance = max_speed * env.config.time_step
        if step_distance > distance:
            agent.position = target_position.copy()
        else:
            agent.position += direction * step_distance

        map_size = env.map_size[0] if isinstance(env.map_size, (tuple, list, np.ndarray)) else env.map_size
        agent.position = np.clip(agent.position, 0, map_size)

        if np.linalg.norm(agent.position - old_position) > POSITION_CHANGE_THRESHOLD:
            self._detect_casualties(agent, env)

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

        # Get set of casualties this agent has marked as untreatable
        untreatable_casualties = getattr(agent, '_untreatable_casualties', set())

        # Track filtered casualties for debugging and cleanup
        filtered_reasons = []
        to_remove_from_known = []

        for casualty_id in agent.known_casualties:
            if casualty_id not in env.casualties:
                filtered_reasons.append(f"{casualty_id}:not_in_env")
                to_remove_from_known.append(casualty_id)
                continue
            # Skip casualties marked as untreatable (resources insufficient even when full)
            if casualty_id in untreatable_casualties:
                filtered_reasons.append(f"{casualty_id}:untreatable")
                to_remove_from_known.append(casualty_id)
                continue
            casualty = env.casualties[casualty_id]
            if casualty.treated:
                filtered_reasons.append(f"{casualty_id}:treated")
                to_remove_from_known.append(casualty_id)
                continue
            if not casualty.is_alive(env.current_time):
                filtered_reasons.append(f"{casualty_id}:dead")
                to_remove_from_known.append(casualty_id)
                continue
            if casualty.treating_agent_id is not None and casualty.treating_agent_id != agent.id:
                filtered_reasons.append(f"{casualty_id}:being_treated_by_{casualty.treating_agent_id}")
                continue

            dist = np.linalg.norm(agent.position - casualty.position)
            priority = severity_weight.get(casualty.severity.name, 0) - dist / 10.0

            if priority > best_priority:
                best_priority = priority
                best = casualty

        # Remove unassignable casualties from known list
        for casualty_id in to_remove_from_known:
            if casualty_id in agent.known_casualties:
                del agent.known_casualties[casualty_id]

        if to_remove_from_known:
            logger.debug(f"[FIND PRIORITY] Agent{agent.id} removed from known: {', '.join(map(str, to_remove_from_known))}")

        if agent.known_casualties and not best and filtered_reasons:
            logger.debug(f"[FIND PRIORITY] Agent{agent.id} filtered: {', '.join(filtered_reasons)}")

        return best

    def _find_lower_resource_casualty(
        self,
        agent: 'RescueAgent',
        env: 'DisasterSim',
        current_severity: 'CasualtySeverity'
    ) -> Optional['Casualty']:
        """Find a casualty with lower resource requirements that can be treated."""
        severity_order = [
            CasualtySeverity.MILD,
            CasualtySeverity.MODERATE,
            CasualtySeverity.SEVERE,
            CasualtySeverity.CRITICAL
        ]

        try:
            current_idx = severity_order.index(current_severity)
        except ValueError:
            return None

        for casualty_id in agent.known_casualties:
            if casualty_id not in env.casualties:
                continue
            casualty = env.casualties[casualty_id]
            
            if severity_order.index(casualty.severity) >= current_idx:
                continue
            
            if casualty.treated or not casualty.is_alive(env.current_time):
                continue
            if casualty.treating_agent_id is not None and casualty.treating_agent_id != agent.id:
                continue
            if env.treatment_manager.can_treat_casualty(agent, casualty):
                return casualty

        return None

    def _find_nearest_known_casualty(self, agent: 'RescueAgent', env: 'DisasterSim') -> Optional['Casualty']:
        """Find the nearest known untreated casualty."""
        nearest = None
        min_dist = float('inf')

        # Get set of casualties this agent has marked as untreatable
        untreatable_casualties = getattr(agent, '_untreatable_casualties', set())

        # Track filtered casualties for debugging and cleanup
        filtered_reasons = []
        to_remove_from_known = []

        for casualty_id in agent.known_casualties:
            if casualty_id not in env.casualties:
                filtered_reasons.append(f"{casualty_id}:not_in_env")
                to_remove_from_known.append(casualty_id)
                continue
            # Skip casualties marked as untreatable (resources insufficient even when full)
            if casualty_id in untreatable_casualties:
                filtered_reasons.append(f"{casualty_id}:untreatable")
                to_remove_from_known.append(casualty_id)
                continue
            casualty = env.casualties[casualty_id]
            if casualty.treated:
                filtered_reasons.append(f"{casualty_id}:treated")
                to_remove_from_known.append(casualty_id)
                continue
            if not casualty.is_alive(env.current_time):
                filtered_reasons.append(f"{casualty_id}:dead")
                to_remove_from_known.append(casualty_id)
                continue
            if casualty.treating_agent_id is not None and casualty.treating_agent_id != agent.id:
                filtered_reasons.append(f"{casualty_id}:being_treated_by_{casualty.treating_agent_id}")
                continue

            dist = np.linalg.norm(agent.position - casualty.position)
            if dist < min_dist:
                min_dist = dist
                nearest = casualty

        # Remove unassignable casualties from known list
        for casualty_id in to_remove_from_known:
            if casualty_id in agent.known_casualties:
                del agent.known_casualties[casualty_id]

        if to_remove_from_known:
            logger.debug(f"[FIND NEAREST] Agent{agent.id} removed from known: {', '.join(map(str, to_remove_from_known))}")

        if agent.known_casualties and not nearest and filtered_reasons:
            logger.debug(f"[FIND NEAREST] Agent{agent.id} filtered: {', '.join(filtered_reasons)}")

        return nearest

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


class DroneBehavior(PersonnelBehavior):
    """
    Behavior strategy for drone agents.

    Handles:
    - Resource delivery to agents
    - Casualty search
    - Depot resupply
    - Exploration

    Process flow:
        1. Detect casualties in range (inherited from parent)
        2. Handle existing mission (go_to_depot / go_to_agent / exploring)
        3. Assign new mission (prioritize delivery > supply > explore)

    Drone-specific missions:
        - go_to_depot_{id}: Return to depot for refill
        - go_to_agent_{id}: Deliver resources to needy agent
        - exploring: Search for casualties
    """

    def process(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Process drone agent behavior."""
        self._detect_casualties(agent, env)

        if self._handle_existing_mission(agent, env):
            return

        if self._needs_refill(agent, env):
            if self._assign_depot_mission(agent, env):
                return

        if self._assign_delivery_mission(agent, env):
            return

        self._explore(agent, env)

    def _handle_existing_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Handle existing mission. Returns True if mission was handled."""
        mission = getattr(agent, 'current_mission', None)
        if not mission:
            return False

        if mission.startswith("go_to_depot_"):
            return self._handle_depot_mission(agent, env, mission)

        if mission.startswith("go_to_agent_"):
            return self._handle_go_to_agent_mission(agent, env, mission)

        if mission == "exploring":
            return self._handle_exploration(agent, env)

        return False

    def _handle_go_to_agent_mission(self, agent: 'RescueAgent', env: 'DisasterSim', mission: str) -> bool:
        """Handle go_to_agent mission - navigate to needy agent location."""
        target_agent_id = int(mission.replace("go_to_agent_", ""))
        if target_agent_id not in env.rescue_agents:
            agent.current_mission = None
            return False

        target_agent = env.rescue_agents[target_agent_id]
        distance = np.linalg.norm(agent.position - target_agent.position)

        if distance > ARRIVAL_RANGE:
            self._navigate_to(agent, target_agent.position, env)
            return True

        if env.drone_manager.deliver_resources(agent, target_agent):
            agent.current_mission = None
            return True

        agent.current_mission = None
        return False

    def _needs_refill(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Check if drone needs to return to depot for refill."""
        total_resources = sum(agent.capacity.values())
        total_max = sum(agent.max_capacity.values())
        return total_resources < total_max * env.config.drone_resource_threshold

    def _assign_delivery_mission(self, agent: 'RescueAgent', env: 'DisasterSim') -> bool:
        """Assign go_to_agent mission for resource delivery."""
        needy_agent = env.drone_manager.find_needy_agent(agent, env.rescue_agents)
        if needy_agent:
            agent.current_mission = f"go_to_agent_{needy_agent.id}"
            return True
        return False

    def _explore(self, agent: 'RescueAgent', env: 'DisasterSim') -> None:
        """Start or continue exploration when no other missions."""
        agent.current_mission = "exploring"
        
        map_size = env.map_size[0] if isinstance(env.map_size, (tuple, list, np.ndarray)) else env.map_size
        grid_size = 100.0
        num_grids = int(map_size / grid_size)

        if not hasattr(agent, '_explored_grids'):
            agent._explored_grids = set()
        if not hasattr(agent, '_exploration_target') or agent._exploration_target is None:
            agent._exploration_target = self._select_unexplored_grid(agent, num_grids, map_size)

        distance = np.linalg.norm(agent.position - agent._exploration_target)
        if distance <= ARRIVAL_RANGE:
            current_grid = self._get_grid_coords(agent.position, grid_size)
            agent._explored_grids.add(current_grid)
            agent._exploration_target = self._select_unexplored_grid(agent, num_grids, map_size)
        else:
            self._navigate_to(agent, agent._exploration_target, env)

    def _navigate_to(self, agent: 'RescueAgent', target_position: np.ndarray, env: 'DisasterSim') -> None:
        """Navigate towards target position."""
        direction = target_position - agent.position
        distance = np.linalg.norm(direction)
        if distance <= ARRIVAL_RANGE:
            return

        direction = direction / distance
        max_speed = agent.get_max_speed()
        old_position = agent.position.copy()
        agent.position += direction * max_speed * env.config.time_step

        map_size = env.map_size[0] if isinstance(env.map_size, (tuple, list, np.ndarray)) else env.map_size
        agent.position = np.clip(agent.position, 0, map_size)

        if np.linalg.norm(agent.position - old_position) > POSITION_CHANGE_THRESHOLD:
            self._detect_casualties(agent, env)
