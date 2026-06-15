"""
Main disaster simulation environment.

This module contains the core DisasterSim class that orchestrates the simulation,
using managers and entities from the refactored modules.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Any, Optional, List
from .config.constants import (
    CasualtySeverity, ResourceType, AgentType, SimulationConfig, DEFAULT_CONFIG,
    TREATMENT_DURATION, RESOURCES_NEEDED
)
from .entities.casualty import Casualty
from .entities.agent import RescueAgent
from .entities.depot import ResourceDepot
from .entities.area import AffectedArea
from .managers.resource_manager import ResourceManager
from .managers.treatment_manager import TreatmentManager
from .managers.drone_manager import DroneManager


class DisasterSim:
    """Main disaster simulation environment."""
    
    def __init__(
        self, 
        scenario: str = "earthquake_standard", 
        map_size: Tuple[float, float] = (10000.0, 10000.0), 
        num_agents: int = 5, 
        num_victims: int = 100, 
        num_resources: int = 4, 
        num_hospitals: int = 2, 
        disaster_type: str = 'earthquake', 
        severity: str = 'medium',
        config: Optional[SimulationConfig] = None
    ):
        """Initialize the disaster simulation environment."""
        # Use provided config or default
        self.config = config if config is not None else DEFAULT_CONFIG
        
        # Override config with constructor parameters
        self.scenario = scenario
        self.map_size = map_size
        self.config.num_agents = num_agents
        self.config.num_victims = num_victims
        self.config.num_resources = num_resources
        self.config.num_hospitals = num_hospitals
        self.config.disaster_type = disaster_type
        self.config.severity = severity
        
        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        
        # Track previous statistics for incremental reward calculation
        self.prev_total_rescued = 0
        self.prev_total_deaths = 0
        
        # Track individual agent rescue contributions for reward attribution
        self.prev_agent_rescued: Dict[int, int] = {}
        
        # Managers
        self.resource_manager = ResourceManager(self.config)
        self.treatment_manager = TreatmentManager(self.config)
        self.drone_manager = DroneManager(self.config, self)
        
        # Components
        self.affected_areas: Dict[int, AffectedArea] = {}
        self.resource_depots: Dict[int, ResourceDepot] = {}
        self.rescue_agents: Dict[int, RescueAgent] = {}
        self.casualties: Dict[int, Casualty] = {}
        self.road_network = nx.Graph()
        self.statistics = self._initialize_statistics()

        # Initialize environment
        self._initialize_affected_areas()
        self._initialize_resource_depots()
        self._initialize_rescue_agents()
        self._initialize_casualties()
        self._initialize_road_network()

        # Track initial resources for metrics calculation
        self._track_initial_resources()
    
    def _initialize_statistics(self) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            'total_rescued': 0,
            'total_deaths': 0,
            'total_resources_replenished': 0.0,
            'response_times': [],
            'resource_utilization': {},
            'fairness_metrics': {
                'gini': []
            },
            'resources_used': 0.0
        }
    
    def _initialize_affected_areas(self) -> None:
        """Initialize affected areas based on scenario."""
        num_areas = max(3, self.config.num_hospitals)
        map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
        
        for i in range(num_areas):
            angle = 2 * np.pi * i / num_areas
            radius = map_size * 0.15
            position = np.array([
                map_size / 2 + radius * np.cos(angle),
                map_size / 2 + radius * np.sin(angle)
            ])
            
            area = AffectedArea(
                id=i,
                position=position,
                size=map_size * 0.35,
                population=1000 + i * 200,
                building_damage=0.3 + i * 0.1,
                road_accessibility=0.8 - i * 0.1
            )
            self.affected_areas[i] = area
    
    def _initialize_resource_depots(self) -> None:
        """Initialize resource depots with dynamic resource calculation based on num_victims."""
        from environments.config.constants import (
            RESOURCE_SUPPLY_RATIO, EXPECTED_SEVERITY_DISTRIBUTION,
            RESOURCES_NEEDED, ResourceType, CasualtySeverity
        )
        
        map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
        depot_positions = [
            np.array([map_size * 0.2, map_size * 0.2]),
            np.array([map_size * 0.8, map_size * 0.8])
        ]
        
        # Calculate total expected resource demand based on num_victims
        num_victims = getattr(self.config, 'num_victims', 100)
        total_demand = {rt: 0.0 for rt in ResourceType}
        
        for severity, ratio in EXPECTED_SEVERITY_DISTRIBUTION.items():
            count = int(num_victims * ratio)
            for rt, amount in RESOURCES_NEEDED[severity].items():
                total_demand[rt] += amount * count
        
        # Apply supply ratio to create scarcity (zero-sum environment)
        total_supply = {rt: demand * RESOURCE_SUPPLY_RATIO for rt, demand in total_demand.items()}
        
        # Divide between depots (each depot gets half)
        num_depots = len(depot_positions)
        per_depot = {rt: supply / num_depots for rt, supply in total_supply.items()}
        
        for i, position in enumerate(depot_positions):
            resources = per_depot.copy()
            
            depot = ResourceDepot(
                id=i,
                position=position,
                resources=resources
            )
            self.resource_depots[i] = depot

    def _track_initial_resources(self) -> None:
        """Track initial resources for metrics calculation."""
        self.initial_resources = {
            depot_id: depot.resources.copy()
            for depot_id, depot in self.resource_depots.items()
        }
        # 计算agent初始实际资源总量（capacity初始值为满载资源量）
        self.initial_agent_resources = sum(
            sum(agent.capacity.values())
            for agent in self.rescue_agents.values()
        )
    
    def _initialize_rescue_agents(self) -> None:
        """Initialize rescue agents."""
        map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
        
        # Create agents with different types
        num_drones = max(1, self.config.num_agents // 10)
        num_vehicles = max(1, self.config.num_agents // 5)
        num_personnel = self.config.num_agents - num_drones - num_vehicles
        
        agent_id = 0
        
        # Create personnel agents
        for i in range(num_personnel):
            position = np.array([
                map_size * (0.3 + i * 0.1),
                map_size * 0.5
            ])
            agent = RescueAgent(
                agent_id=agent_id,
                position=position,
                map_size=map_size,
                agent_type=AgentType.PERSONNEL
            )
            self.rescue_agents[agent_id] = agent
            agent_id += 1
        
        # Create vehicle agents
        for i in range(num_vehicles):
            position = np.array([
                map_size * (0.6 + i * 0.15),
                map_size * 0.5
            ])
            agent = RescueAgent(
                agent_id=agent_id,
                position=position,
                map_size=map_size,
                agent_type=AgentType.VEHICLE
            )
            self.rescue_agents[agent_id] = agent
            agent_id += 1
        
        # Create drone agents
        for i in range(num_drones):
            position = np.array([
                map_size * 0.5,
                map_size * (0.3 + i * 0.2)
            ])
            agent = RescueAgent(
                agent_id=agent_id,
                position=position,
                map_size=map_size,
                agent_type=AgentType.DRONE
            )
            self.rescue_agents[agent_id] = agent
            agent_id += 1
    
    def _initialize_casualties(self) -> None:
        """Initialize casualties in affected areas."""
        casualty_id = 0
        
        # Use configured num_victims if available, otherwise calculate based on affected areas
        if hasattr(self.config, 'num_victims') and self.config.num_victims > 0:
            total_casualties = self.config.num_victims
        else:
            total_casualties = max(10, len(self.affected_areas) * 20)
        
        num_casualties_per_area = max(1, total_casualties // len(self.affected_areas))
        remaining_casualties = total_casualties % len(self.affected_areas)
        
        for idx, area in enumerate(self.affected_areas.values()):
            # Distribute casualties across severity levels
            num_casualties = num_casualties_per_area
            if idx < remaining_casualties:
                num_casualties += 1
            
            for _ in range(num_casualties):
                # Random position within affected area
                offset = (np.random.rand(2) - 0.5) * area.size * 0.5
                position = area.position + offset
                
                # Random severity
                severity = np.random.choice([
                    CasualtySeverity.CRITICAL,
                    CasualtySeverity.SEVERE,
                    CasualtySeverity.MODERATE,
                    CasualtySeverity.MILD
                ], p=[0.15, 0.25, 0.35, 0.25])
                
                # Calculate resources needed based on severity
                resources_needed = self._calculate_resources_needed(severity)
                
                casualty = Casualty(
                    id=casualty_id,
                    position=position,
                    severity=severity,
                    injury_time=0.0,
                    resources_needed=resources_needed
                )
                
                self.casualties[casualty_id] = casualty
                area.add_casualty(casualty)
                casualty_id += 1
    
    def _calculate_resources_needed(self, severity: CasualtySeverity) -> Dict[ResourceType, float]:
        """Calculate resource requirements for a casualty based on severity."""
        return RESOURCES_NEEDED[severity].copy()

    def _initialize_road_network(self) -> None:
        """Initialize road network graph."""
        self.road_network = nx.Graph()
        
        # Add depot nodes
        for depot in self.resource_depots.values():
            self.road_network.add_node(f"depot_{depot.id}", position=depot.position)
        
        # Add area nodes
        for area in self.affected_areas.values():
            self.road_network.add_node(f"area_{area.id}", position=area.position)
        
        # Add connections
        nodes = list(self.road_network.nodes())
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:
                    pos1 = self.road_network.nodes[node1]['position']
                    pos2 = self.road_network.nodes[node2]['position']
                    distance = np.linalg.norm(pos1 - pos2)
                    self.road_network.add_edge(node1, node2, weight=distance)
    
    def step(self, actions: Optional[Dict[int, Dict[str, Any]]] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment.

        Args:
            actions: Dictionary mapping agent IDs to actions

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Increment time first - this ensures treatment timing is correct
        self.current_time += self.config.time_step
        self.step_count += 1

        # Update casualty survival probabilities
        self._update_casualties()

        # Apply algorithm actions to agents
        if actions is not None:
            self._apply_actions(actions)

        # Process agent behaviors using strategy pattern
        for agent in self.rescue_agents.values():
            agent.process(self)

        # Update statistics
        self._update_statistics()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.config.max_steps

        # Generate observation
        observation = self._generate_observation()

        # Prepare info
        info = {
            'step_count': self.step_count,
            'current_time': self.current_time,
            'statistics': self.statistics,
            'num_casualties': len(self.casualties),
            'num_rescue_agents': len(self.rescue_agents),
            'num_affected_areas': len(self.affected_areas),
            'num_resource_depots': len(self.resource_depots)
        }

        return observation, reward, terminated, truncated, info

    def _apply_actions(self, actions: Dict[int, Dict[str, Any]]) -> None:
        """Apply algorithm actions to agents.

        Args:
            actions: Dictionary mapping agent IDs to action dictionaries
        """
        for agent_id, action in actions.items():
            if agent_id not in self.rescue_agents:
                continue

            agent = self.rescue_agents[agent_id]

            # Allow algorithm actions to override existing missions, especially exploration
            # This ensures the MARL algorithm can take control of agent behavior
            current_mission = getattr(agent, 'current_mission', None)
            
            # Only allow override for exploring mission or no mission
            # Other missions like treating or going to depot should not be interrupted
            if current_mission is not None and not current_mission == "exploring":
                continue

            tactical = action.get('tactical', 0)

            if tactical == 0:
                # Stay/idle action - clear mission to let agent continue normal behavior
                agent.current_mission = None
            elif tactical == 1:
                nearest = self._find_nearest_untreated_casualty(agent)
                if nearest:
                    agent.current_mission = f"go_to_casualty_{nearest.id}"
            elif tactical == 2:
                nearest_depot = self._find_nearest_depot(agent)
                if nearest_depot:
                    agent.current_mission = f"go_to_depot_{nearest_depot.id}"
            elif tactical == 3:
                nearest_unknown = self._find_nearest_unknown_casualty(agent)
                if nearest_unknown:
                    agent.current_mission = f"go_to_casualty_{nearest_unknown.id}"
            elif tactical == 4:
                # Go to nearest known casualty
                if agent.known_casualties:
                    nearest_known = self._find_nearest_known_casualty(agent)
                    if nearest_known:
                        agent.current_mission = f"go_to_casualty_{nearest_known.id}"
            elif tactical == 5:
                # Return to base (first depot)
                if self.resource_depots:
                    first_depot = next(iter(self.resource_depots.values()))
                    agent.current_mission = f"go_to_depot_{first_depot.id}"

    def _find_nearest_untreated_casualty(self, agent: 'RescueAgent') -> Optional['Casualty']:
        """Find nearest untreated casualty."""
        min_dist = float('inf')
        nearest = None
        for casualty in self.casualties.values():
            if casualty.treated or not casualty.is_alive(self.current_time):
                continue
            dist = np.linalg.norm(agent.position - casualty.position)
            if dist < min_dist:
                min_dist = dist
                nearest = casualty
        return nearest

    def _find_nearest_depot(self, agent: 'RescueAgent') -> Optional['ResourceDepot']:
        """Find nearest resource depot."""
        min_dist = float('inf')
        nearest = None
        for depot in self.resource_depots.values():
            dist = np.linalg.norm(agent.position - depot.position)
            if dist < min_dist:
                min_dist = dist
                nearest = depot
        return nearest

    def _find_nearest_unknown_casualty(self, agent: 'RescueAgent') -> Optional['Casualty']:
        """Find nearest casualty not yet discovered by this agent."""
        min_dist = float('inf')
        nearest = None
        for casualty in self.casualties.values():
            if casualty.discovered_by == agent.id or not casualty.is_alive(self.current_time):
                continue
            dist = np.linalg.norm(agent.position - casualty.position)
            if dist < min_dist:
                min_dist = dist
                nearest = casualty
        return nearest

    def _find_nearest_known_casualty(self, agent: 'RescueAgent') -> Optional['Casualty']:
        """Find nearest casualty known to this agent."""
        min_dist = float('inf')
        nearest = None
        for casualty_id in agent.known_casualties:
            if casualty_id not in self.casualties:
                continue
            casualty = self.casualties[casualty_id]
            if casualty.treated or not casualty.is_alive(self.current_time):
                continue
            dist = np.linalg.norm(agent.position - casualty.position)
            if dist < min_dist:
                min_dist = dist
                nearest = casualty
        return nearest
    
    def _update_casualties(self) -> None:
        """Update survival probabilities for all casualties."""
        for casualty in self.casualties.values():
            casualty.update_survival_probability(self.current_time)
    
    def _update_statistics(self) -> None:
        """Update simulation statistics."""
        rescued = sum(1 for c in self.casualties.values() if c.treated)
        dead = sum(1 for c in self.casualties.values() if not c.is_alive(self.current_time))

        self.statistics['total_rescued'] = rescued
        self.statistics['total_deaths'] = dead
        
        # Update resources used from treatment manager
        self.statistics['resources_used'] = self.treatment_manager.total_resources_used
        
        # Update response times for newly treated casualties
        for casualty in self.casualties.values():
            if casualty.treated and casualty.treatment_start is not None and casualty.injury_time is not None:
                response_time = casualty.treatment_start - casualty.injury_time
                if response_time > 0 and response_time not in self.statistics['response_times']:
                    self.statistics['response_times'].append(response_time)
    
    def _calculate_reward(self) -> Dict[int, float]:
        """Calculate individual rewards for each agent.
        
        Returns rewards based on newly rescued casualties and deaths in this step,
        with individual attribution to reward the agents who actually performed rescues.
        
        This solves the "free-rider" problem where inactive agents would get the same
        reward as active rescuers.
        
        Returns:
            Dictionary mapping agent_id to individual reward
        """
        # Calculate incremental changes
        new_rescued = self.statistics['total_rescued'] - self.prev_total_rescued
        new_deaths = self.statistics['total_deaths'] - self.prev_total_deaths
        
        # Update previous values for next step
        self.prev_total_rescued = self.statistics['total_rescued']
        self.prev_total_deaths = self.statistics['total_deaths']
        
        # Global base reward components
        global_rescue_reward = new_rescued  # +1 for each rescue
        global_death_penalty = new_deaths * 10  # -10 for each death
        time_penalty = 0.01  # Small penalty per step to encourage efficiency
        
        # Calculate individual rewards
        individual_rewards: Dict[int, float] = {}
        
        # Initialize prev_agent_rescued if not done yet
        if not self.prev_agent_rescued:
            for agent_id, agent in self.rescue_agents.items():
                self.prev_agent_rescued[agent_id] = 0
        
        # Calculate each agent's contribution
        for agent_id, agent in self.rescue_agents.items():
            # Get agent's rescue count change
            current_rescued = getattr(agent, 'rescued_count', 0)
            prev_rescued = self.prev_agent_rescued.get(agent_id, 0)
            agent_new_rescued = current_rescued - prev_rescued
            
            # Update previous rescued count for next step
            self.prev_agent_rescued[agent_id] = current_rescued
            
            # Individual reward components:
            # 1. Share of global reward (distributed equally)
            num_agents = len(self.rescue_agents)
            global_share = (global_rescue_reward - global_death_penalty) / num_agents
            
            # 2. Individual rescue bonus (agent who rescued gets extra reward)
            rescue_bonus = agent_new_rescued * 0.5  # Extra reward for each rescue
            
            # 3. Time penalty (applied to all agents)
            individual_rewards[agent_id] = global_share + rescue_bonus - time_penalty
        
        return individual_rewards
    
    def _check_termination(self) -> bool:
        """Check if simulation should terminate."""
        all_processed = all(
            not c.is_alive(self.current_time) or c.treated
            for c in self.casualties.values()
        )
        return all_processed
    
    def get_state_dimension(self) -> int:
        """Get the dimension of the state vector."""
        num_agents = len(self.rescue_agents)
        num_casualties = len(self.casualties)
        return (num_agents + num_casualties) * 3

    def get_num_agents(self) -> int:
        """Get the number of agents in the simulation."""
        return len(self.rescue_agents)

    def _generate_observation(self) -> np.ndarray:
        """Generate observation vector for the environment.
        
        Returns:
            2D array of shape (num_agents, state_dim) for MARL compatibility.
        """
        num_agents = len(self.rescue_agents)
        num_casualties = len(self.casualties)
        state_dim = (num_agents + num_casualties) * 3

        observations = np.zeros((num_agents, state_dim), dtype=np.float32)

        agent_list = list(self.rescue_agents.values())
        for i, agent in enumerate(agent_list):
            observations[i, 0:2] = agent.position
            observations[i, 2] = sum(agent.capacity.values())

        for j, casualty in enumerate(self.casualties.values()):
            base_idx = num_agents * 3 + j * 3
            if base_idx + 2 < state_dim:
                observations[:, base_idx] = casualty.position[0]
                observations[:, base_idx + 1] = casualty.position[1]
                observations[:, base_idx + 2] = casualty.survival_probability

        return observations
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Reinitialize all components
        self.current_time = 0.0
        self.step_count = 0
        self.statistics = self._initialize_statistics()
        
        # Reset previous statistics for incremental reward calculation
        self.prev_total_rescued = 0
        self.prev_total_deaths = 0
        
        # Reset individual agent rescue tracking
        self.prev_agent_rescued = {}
        
        # Reset treatment manager statistics
        self.treatment_manager.total_resources_used = 0.0
        
        self.affected_areas.clear()
        self.resource_depots.clear()
        self.rescue_agents.clear()
        self.casualties.clear()
        self.road_network = nx.Graph()
        
        self._initialize_affected_areas()
        self._initialize_resource_depots()
        self._initialize_rescue_agents()
        self._initialize_casualties()
        self._initialize_road_network()
        self._track_initial_resources()

        observation = self._generate_observation()
        info = {
            'num_casualties': len(self.casualties),
            'num_rescue_agents': len(self.rescue_agents),
            'num_resource_depots': len(self.resource_depots),
            'statistics': self.statistics.copy()
        }
        return observation, info
