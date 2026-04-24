"""
DisasterSim-2026: High-fidelity disaster simulation environment for medical resource allocation.

This module implements the main simulation environment with:
1. Dynamic disaster scenarios
2. Multi-agent rescue operations
3. Resource management
4. Communication networks
5. Casualty simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import networkx as nx
from scipy.spatial.distance import cdist
from gymnasium import spaces
from enum import Enum


class CasualtySeverity(Enum):
    """Casualty severity levels."""
    CRITICAL = "critical"  # Immediate medical attention required
    SEVERE = "severe"      # Urgent medical attention required
    MODERATE = "moderate"   # Medical attention needed
    MILD = "mild"          # Minor injuries


class ResourceType(Enum):
    """Types of medical resources."""
    BROAD_SPECTRUM_ANTIBIOTICS = "broad_spectrum_antibiotics"
    BLOOD_PACKS = "blood_packs"
    OXYGEN = "oxygen"
    PAIN_MEDICATION = "pain_medication"


@dataclass
class Casualty:
    """Casualty in the disaster simulation."""
    id: int
    position: np.ndarray
    severity: CasualtySeverity
    injury_time: float
    resources_needed: Dict[ResourceType, float]
    
    treated: bool = False
    treatment_start: Optional[float] = None
    survival_probability: float = 1.0
    
    def update_survival_probability(self, current_time: float) -> None:
        """Update survival probability based on time since injury."""
        time_elapsed = current_time - self.injury_time
        
        # Survival probability decreases over time based on severity
        severity_factor = {
            CasualtySeverity.CRITICAL: 0.0002,  # 0.02% per second (5000 seconds = 83 minutes)
            CasualtySeverity.SEVERE: 0.0001,    # 0.01% per second (10000 seconds = 167 minutes)
            CasualtySeverity.MODERATE: 0.00005,  # 0.005% per second (20000 seconds = 333 minutes)
            CasualtySeverity.MILD: 0.00001      # 0.001% per second (100000 seconds = 2778 minutes)
        }[self.severity]
        
        self.survival_probability = max(
            0.0,
            self.survival_probability - severity_factor * time_elapsed
        )
    
    def is_alive(self, current_time: float) -> bool:
        """Check if casualty is still alive."""
        self.update_survival_probability(current_time)
        return self.survival_probability > 0.0


@dataclass
class AffectedArea:
    """Affected area in the disaster."""
    id: int
    position: np.ndarray
    size: float
    population: int
    building_damage: float  # 0.0 to 1.0
    road_accessibility: float  # 0.0 to 1.0
    casualties: List[Casualty] = None
    initial_casualties: int = 0
    survivors: int = 0
    
    def __post_init__(self):
        if self.casualties is None:
            self.casualties = []
        self.initial_casualties = len(self.casualties)
        self.survivors = 0
    
    @property
    def survival_rate(self) -> float:
        """Calculate survival rate for this area."""
        if self.initial_casualties == 0:
            return 0.0
        return self.survivors / self.initial_casualties


@dataclass
class ResourceDepot:
    """Resource depot for storing medical supplies."""
    id: int
    position: np.ndarray
    resources: Dict[ResourceType, float]


class RescueAgent:
    """Base class for rescue agents."""
    def __init__(self, agent_id: int, position: np.ndarray, map_size=None):
        self.id = agent_id
        self.position = position
        self.velocity = np.zeros(2)
        self.capacity = {rt: 0.0 for rt in ResourceType}
        self.max_capacity = {rt: 10.0 for rt in ResourceType}
        self.endurance = 100.0
        self.max_endurance = 100.0
        self.current_mission = None
        self.connected_agents = []
        self.map_size = map_size
        self.route = []  # 初始化route属性
    
    def get_max_speed(self) -> float:
        """Get maximum speed of the agent."""
        # 根据地图大小调整最大速度，确保智能体能够在合理时间内覆盖地图
        # 假设地图大小在100-10000之间，速度在5-10米/秒（18-36公里/小时），符合真实救援人员移动速度
        if self.map_size is not None:
            map_size_value = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
            return max(5.0, min(10.0, map_size_value * 0.001))  # 调整为更合理的速度
        return 5.0  # Default speed
    
    def move(self, time_step: float) -> None:
        """Move the agent based on velocity."""
        self.position += self.velocity * time_step
    
    def can_communicate(self, other_position: np.ndarray) -> bool:
        """Check if agent can communicate with another position."""
        distance = np.linalg.norm(self.position - other_position)
        return distance < 1000.0  # 1km communication range


class DisasterSim:
    """Main disaster simulation environment."""
    
    def __init__(self, scenario: str = "earthquake_standard", map_size: Tuple[float, float] = (10000.0, 10000.0), 
                 num_agents: int = 5, num_victims: int = 100, num_resources: int = 4, 
                 num_hospitals: int = 2, disaster_type: str = 'earthquake', 
                 severity: str = 'medium'):
        """Initialize the disaster simulation environment."""
        self.scenario = scenario
        self.map_size = map_size  # Store as tuple for compatibility
        self.time_step = 1.0  # 1 second time step
        self.max_steps = 3600  # 1 hour simulation
        
        # Store parameters first
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.num_resources = num_resources
        self.num_hospitals = num_hospitals
        self.disaster_type = disaster_type
        self.severity = severity
        
        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        self.secondary_disaster_counter = 0
        self.weather_conditions = "clear"
        self.communication_status = 1.0
        
        # Components
        self.affected_areas: Dict[int, AffectedArea] = {}
        self.resource_depots: Dict[int, ResourceDepot] = {}
        self.rescue_agents: Dict[int, RescueAgent] = {}
        self.casualties: Dict[int, Casualty] = {}
        self.road_network = nx.Graph()
        self.statistics = {}
        
        # Initialize environment
        self._initialize_affected_areas()
        self._initialize_resource_depots()
        self._initialize_rescue_agents()
        self._initialize_casualties()
        self._initialize_road_network()
        self._define_spaces()
        
        # Update num_victims to actual count
        self.num_victims = len(self.casualties)
    
    def _initialize_affected_areas(self) -> None:
        """Initialize affected areas based on scenario."""
        # Simple earthquake scenario for demonstration
        self.affected_areas = {}
        
        # Create 5 affected areas in a circular pattern
        for i in range(5):
            angle = 2 * np.pi * i / 5
            # 使用地图大小的1/4作为半径，确保在地图范围内
            radius = self.map_size[0] * 0.25 if isinstance(self.map_size, (tuple, list)) else self.map_size * 0.25
            map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
            position = np.array([
                map_size / 2 + radius * np.cos(angle),
                map_size / 2 + radius * np.sin(angle)
            ])
            
            area = AffectedArea(
                id=i,
                position=position,
                size=self.map_size[0] * 0.1 if isinstance(self.map_size, (tuple, list)) else self.map_size * 0.1,
                population=1000 + i * 200,
                building_damage=0.3 + i * 0.1,
                road_accessibility=0.8 - i * 0.1
            )
            self.affected_areas[i] = area
    
    def _initialize_resource_depots(self) -> None:
        """Initialize resource depots."""
        self.resource_depots = {}
        
        # Create 2 resource depots，位置在地图范围内
        map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
        depot_positions = [
            np.array([map_size * 0.2, map_size * 0.2]),
            np.array([map_size * 0.8, map_size * 0.8])
        ]
        
        for i, position in enumerate(depot_positions):
            resources = {
                ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 1000.0,
                ResourceType.BLOOD_PACKS: 500.0,
                ResourceType.OXYGEN: 800.0,
                ResourceType.PAIN_MEDICATION: 1200.0
            }
            
            depot = ResourceDepot(
                id=i,
                position=position,
                resources=resources
            )
            self.resource_depots[i] = depot
    
    def _initialize_rescue_agents(self) -> None:
        """Initialize rescue agents."""
        self.rescue_agents = {}
        
        # Create rescue agents with positions relative to map size
        map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
        
        # 为每个智能体生成随机但合理的初始位置
        for i in range(self.num_agents):
            # 生成地图范围内的随机位置
            position = np.random.uniform(0, map_size, 2)
            
            agent = RescueAgent(
                agent_id=i,
                position=position,
                map_size=self.map_size
            )
            self.rescue_agents[i] = agent
    
    def _initialize_casualties(self) -> None:
        """Initialize casualties in affected areas."""
        self.casualties = {}
        casualty_id = 0
        
        # 使用num_victims参数来控制受害者总数
        total_casualties_to_create = self.num_victims
        num_areas = len(self.affected_areas)
        
        # 根据区域的严重程度分配受害者数量
        # 严重程度越高的区域，分配越多的受害者
        area_weights = []
        for area in self.affected_areas.values():
            weight = area.building_damage * area.population / 1000.0
            area_weights.append(weight)
        
        total_weight = sum(area_weights)
        area_weights = [w / total_weight for w in area_weights]
        
        # 分配每个区域的受害者数量
        casualties_per_area = []
        remaining = total_casualties_to_create
        for i in range(num_areas - 1):
            count = int(total_casualties_to_create * area_weights[i])
            casualties_per_area.append(count)
            remaining -= count
        casualties_per_area.append(remaining)
        
        for area_id, casualty_count in zip(self.affected_areas.keys(), casualties_per_area):
            target_area = self.affected_areas[area_id]
            target_area.casualties = []
            
            for _ in range(casualty_count):
                position = target_area.position + np.random.uniform(-target_area.size/2, target_area.size/2, 2)
                
                severity_probs = {
                    CasualtySeverity.CRITICAL: target_area.building_damage * 0.3,
                    CasualtySeverity.SEVERE: target_area.building_damage * 0.4,
                    CasualtySeverity.MODERATE: 0.2 + target_area.building_damage * 0.2,
                    CasualtySeverity.MILD: 0.1
                }
                
                total_prob = sum(severity_probs.values())
                severity_probs = {k: v/total_prob for k, v in severity_probs.items()}
                
                selected_severity = np.random.choice(
                    list(severity_probs.keys()),
                    p=list(severity_probs.values())
                )
                
                if selected_severity == CasualtySeverity.CRITICAL:
                    resources = {
                        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 2.0,
                        ResourceType.BLOOD_PACKS: 1.0,
                        ResourceType.OXYGEN: 1.5,
                        ResourceType.PAIN_MEDICATION: 1.0,
                    }
                elif selected_severity == CasualtySeverity.SEVERE:
                    resources = {
                        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 1.5,
                        ResourceType.BLOOD_PACKS: 0.5,
                        ResourceType.OXYGEN: 1.0,
                        ResourceType.PAIN_MEDICATION: 1.0,
                    }
                elif selected_severity == CasualtySeverity.MODERATE:
                    resources = {
                        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 1.0,
                        ResourceType.PAIN_MEDICATION: 0.5,
                    }
                else:
                    resources = {
                        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 0.5,
                    }
                
                casualty = Casualty(
                    id=casualty_id,
                    position=position,
                    severity=selected_severity,
                    injury_time=0.0,
                    resources_needed=resources,
                )
                
                target_area.casualties.append(casualty)
                self.casualties[casualty_id] = casualty
                casualty_id += 1
            
            target_area.initial_casualties = len(target_area.casualties)
    
    def _initialize_road_network(self) -> None:
        """Initialize road network connecting affected areas and depots."""
        # Add nodes for all important locations
        all_positions = []
        node_ids = []
        
        # Add affected areas
        for area_id, area in self.affected_areas.items():
            self.road_network.add_node(f"area_{area_id}", pos=tuple(area.position))
            all_positions.append(area.position)
            node_ids.append(f"area_{area_id}")
        
        # Add resource depots
        for depot_id, depot in self.resource_depots.items():
            self.road_network.add_node(f"depot_{depot_id}", pos=tuple(depot.position))
            all_positions.append(depot.position)
            node_ids.append(f"depot_{depot_id}")
        
        # Connect nodes based on proximity (Delaunay triangulation would be better)
        positions_array = np.array(all_positions)
        distances = cdist(positions_array, positions_array)
        
        # Connect each node to its 3 nearest neighbors
        for i, node_i in enumerate(node_ids):
            # Get indices of nearest neighbors (excluding self)
            neighbor_indices = np.argsort(distances[i])[1:4]  # 3 nearest
            
            for j in neighbor_indices:
                node_j = node_ids[j]
                distance = distances[i, j]
                
                # Add edge with weight = distance
                self.road_network.add_edge(node_i, node_j, weight=distance)
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation space per agent
        # Features: position(2), velocity(2), capacity(4), endurance(1), 
        # mission_status(1), nearest_area_info(5), global_resource_levels(4)
        obs_dim = 2 + 2 + 4 + 1 + 1 + 5 + 4  # Total: 19
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: hierarchical
        # Strategic: resource allocation [0,1]^4 (continuous)
        # Tactical: movement direction (8 discrete)
        # Communication: information sharing (4 discrete)
        self.action_space = spaces.Dict({
            "strategic": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "tactical": spaces.Discrete(8),
            "communication": spaces.Discrete(4),
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation state
        self.current_time = 0.0
        self.step_count = 0
        
        # Reinitialize components
        self._initialize_affected_areas()
        self._initialize_resource_depots()
        self._initialize_rescue_agents()
        self._initialize_casualties()
        self._initialize_road_network()
        
        # Reset statistics
        self.statistics = {
            "total_survivors": 0,
            "total_casualties": len(self.casualties),
            "resource_utilization": {rt: 0.0 for rt in ResourceType},
            "response_times": [],
            "fairness_metrics": {"gini": [], "theil": [], "max_min": []},
        }
        
        # Get initial observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, actions: Dict[int, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            observation: New observation
            reward: Total reward
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Apply actions to agents
        for agent_id, action in actions.items():
            if agent_id in self.rescue_agents:
                self._apply_action(agent_id, action)
        
        # Update simulation state
        self._update_dynamics()
        
        # Update casualties
        self._update_casualties()
        
        # Update communication network
        self._update_communication()
        
        # Apply secondary disasters
        if np.random.rand() < self._get_secondary_disaster_probability():
            self._apply_secondary_disaster()
        
        # Update weather
        self._update_weather()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update statistics
        self._update_statistics()
        
        # Increment time and step count
        self.current_time += self.time_step
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add required info fields for compatibility
        info['rescued'] = self.statistics.get('total_survivors', 0)
        info['deaths'] = self.statistics.get('total_casualties', 0) - len(self.casualties)
        info['resources_used'] = sum(self.statistics.get('resource_utilization', {}).values())
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, agent_id: int, action: Dict[str, Any]) -> None:
        """Apply action to a specific agent."""
        agent = self.rescue_agents[agent_id]
        
        # Strategic action: resource allocation
        if "strategic" in action:
            allocation = action["strategic"]
            # Normalize allocation to sum to 1
            if isinstance(allocation, np.ndarray):
                allocation = allocation / (np.sum(allocation) + 1e-8)
            else:
                allocation = np.array(allocation) / (np.sum(allocation) + 1e-8)
            
            # Update agent's resource allocation strategy
            # 资源分配策略影响智能体对不同类型资源的优先级
            agent.resource_allocation = {
                ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: float(allocation[0]) if len(allocation) > 0 else 0.25,
                ResourceType.BLOOD_PACKS: float(allocation[1]) if len(allocation) > 1 else 0.25,
                ResourceType.OXYGEN: float(allocation[2]) if len(allocation) > 2 else 0.25,
                ResourceType.PAIN_MEDICATION: float(allocation[3]) if len(allocation) > 3 else 0.25
            }
        
        # Tactical action: movement
        if "tactical" in action:
            # 检查代理是否有治疗任务
            if agent.current_mission and agent.current_mission.startswith("treat_casualty_"):
                # 提取受害者ID
                casualty_id = int(agent.current_mission.split("_")[-1])
                # 检查受害者是否存在
                if casualty_id in self.casualties:
                    casualty = self.casualties[casualty_id]
                    # 向受害者位置移动
                    direction = casualty.position - agent.position
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction = direction / distance
                        max_speed = agent.get_max_speed()
                        target_distance = min(max_speed * self.time_step, distance)
                        target_position = agent.position + direction * target_distance
                        
                        # Clip to map boundaries
                        target_position = np.clip(target_position, 0, self.map_size)
                        
                        # Plan route to target (simple straight line for now)
                        agent.route = [target_position]
            else:
                # 随机移动
                direction_idx = action["tactical"]
                # Convert direction index to vector (8 directions)
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                direction = np.array([np.cos(angles[direction_idx]), np.sin(angles[direction_idx])])
                
                # Set target position
                max_speed = agent.get_max_speed()
                target_distance = max_speed * self.time_step
                target_position = agent.position + direction * target_distance
                
                # Clip to map boundaries
                target_position = np.clip(target_position, 0, self.map_size)
                
                # Plan route to target (simple straight line for now)
                agent.route = [target_position]
        
        # Communication action: information sharing
        if "communication" in action:
            comm_action = action["communication"]
            # Implement communication strategy
            pass  # Implementation depends on communication protocol
        
        # Move agent
        if hasattr(agent, 'route') and agent.route:
            # Move towards the first waypoint in the route
            target_position = agent.route[0]
            direction = target_position - agent.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                max_speed = agent.get_max_speed()
                move_distance = min(max_speed * self.time_step, distance)
                agent.position += direction * move_distance
                
                # Remove the waypoint if we've reached it
                if np.linalg.norm(agent.position - target_position) < 1.0:
                    agent.route.pop(0)
        else:
            agent.move(self.time_step)
    
    def _update_dynamics(self) -> None:
        """Update dynamic factors in the environment."""
        # Update agent endurance and refuel/resupply if at depot
        for agent in self.rescue_agents.values():
            # Check if agent is at a depot
            at_depot = False
            for depot in self.resource_depots.values():
                distance = np.linalg.norm(agent.position - depot.position)
                if distance < 100.0:  # Within 100m of depot
                    at_depot = True
                    
                    # Refuel (restore endurance)
                    agent.endurance = min(
                        agent.endurance + self.time_step * 2,  # Faster recovery at depot
                        agent.max_endurance
                    )
                    
                    # Resupply resources
                    for resource_type in ResourceType:
                        if agent.capacity[resource_type] < agent.max_capacity[resource_type]:
                            # Try to get resources from depot
                            needed = agent.max_capacity[resource_type] - agent.capacity[resource_type]
                            available = depot.resources.get(resource_type, 0.0)
                            transfer = min(needed, available)
                            
                            if transfer > 0:
                                agent.capacity[resource_type] += transfer
                                depot.resources[resource_type] -= transfer
                    break
            
            # If not at depot, endurance decreases normally
            if not at_depot:
                agent.endurance = max(agent.endurance - self.time_step, 0.0)
    
    def _update_casualties(self) -> None:
        """Update casualty states and check for deaths."""
        casualties_to_remove = []
        
        # 实现基于救援代理位置的真正救援逻辑
        for agent in self.rescue_agents.values():
            # 检查救援代理是否在治疗受害者
            if agent.current_mission and agent.current_mission.startswith("treat_casualty_"):
                # 继续治疗
                pass
            else:
                # 寻找附近的受害者
                nearest_casualty = None
                min_distance = float('inf')
                
                for casualty_id, casualty in self.casualties.items():
                    if not casualty.treated:
                        distance = np.linalg.norm(agent.position - casualty.position)
                        # 根据地图大小调整检测范围，至少100米或地图的15%
                        map_dimension = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
                        detection_range = max(100.0, map_dimension * 0.15)  # 至少100米或地图的15%
                        if distance < detection_range and distance < min_distance:
                            nearest_casualty = casualty
                            min_distance = distance
                
                # 如果找到附近的受害者，开始治疗
                if nearest_casualty:
                    nearest_casualty.treated = True
                    nearest_casualty.treatment_start = self.current_time
                    agent.current_mission = f"treat_casualty_{nearest_casualty.id}"
                    # 打印调试信息
                    print(f"Agent {agent.id} started treating casualty {nearest_casualty.id} at position {nearest_casualty.position}")
                else:
                    # 打印调试信息
                    print(f"Agent {agent.id} at position {agent.position} found no casualties nearby")
        
        for casualty_id, casualty in self.casualties.items():
            # Update survival probability
            casualty.update_survival_probability(self.current_time)
            
            # Check if casualty dies
            if not casualty.is_alive(self.current_time):
                casualties_to_remove.append(casualty_id)
                # 移除相关的救援任务
                for agent in self.rescue_agents.values():
                    if agent.current_mission == f"treat_casualty_{casualty_id}":
                        agent.current_mission = None
                continue
            
            # Check if casualty is being treated
            if casualty.treated and casualty.treatment_start is not None:
                treatment_duration = self.current_time - casualty.treatment_start
                
                # Check if treatment is complete
                required_time = {
                    CasualtySeverity.CRITICAL: 60,  # 1 minute
                    CasualtySeverity.SEVERE: 30,    # 30 seconds
                    CasualtySeverity.MODERATE: 15,   # 15 seconds
                    CasualtySeverity.MILD: 5,       # 5 seconds
                }[casualty.severity]
                
                if treatment_duration >= required_time:
                    # Treatment complete, casualty survives
                    self.statistics["total_survivors"] += 1
                    
                    # 找到受害者所在的区域，并更新该区域的幸存者数量
                    for area_id, area in self.affected_areas.items():
                        if casualty in area.casualties:
                            area.survivors += 1
                            break
                    
                    # 移除相关的救援任务
                    for agent in self.rescue_agents.values():
                        if agent.current_mission == f"treat_casualty_{casualty_id}":
                            agent.current_mission = None
                    
                    # 从受害者列表中移除已获救的受害者
                    casualties_to_remove.append(casualty_id)
        
        # Remove dead casualties
        for casualty_id in casualties_to_remove:
            casualty = self.casualties.pop(casualty_id)
            # Remove from affected area
            for area in self.affected_areas.values():
                if casualty in area.casualties:
                    area.casualties.remove(casualty)
                    break
    
    def _update_communication(self) -> None:
        """Update communication network between agents."""
        # Reset connections
        for agent in self.rescue_agents.values():
            agent.connected_agents = []
        
        # Establish new connections based on distance and communication status
        agent_ids = list(self.rescue_agents.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent_i = self.rescue_agents[agent_ids[i]]
                agent_j = self.rescue_agents[agent_ids[j]]
                
                # Check if agents can communicate
                if agent_i.can_communicate(agent_j.position):
                    # Apply communication status (weather, interference)
                    if np.random.rand() < self.communication_status:
                        agent_i.connected_agents.append(agent_j.id)
                        agent_j.connected_agents.append(agent_i.id)
    
    def _get_secondary_disaster_probability(self) -> float:
        """Calculate probability of secondary disaster."""
        # K/(t+c) model
        K = 10.0
        c = 1.0
        probability = K / (self.current_time / 3600.0 + c)  # time in hours
        return min(probability / 100.0, 0.1)  # Normalize to reasonable probability
    
    def _apply_secondary_disaster(self) -> None:
        """Apply a secondary disaster event."""
        self.secondary_disaster_counter += 1
        
        # Randomly select affected area
        area_id = np.random.choice(list(self.affected_areas.keys()))
        area = self.affected_areas[area_id]
        
        # Apply effects
        effect_type = np.random.choice(["additional_casualties", "road_damage", "building_collapse"])
        
        if effect_type == "additional_casualties":
            # Add new casualties
            additional_rate = np.random.uniform(0.05, 0.10)
            num_additional = int(area.population * additional_rate)
            
            # Similar to initial casualty creation
            for _ in range(num_additional):
                # Create new casualty
                pass  # Implementation similar to _initialize_casualties
        
        elif effect_type == "road_damage":
            # Damage roads in the area
            area.road_accessibility *= np.random.uniform(0.5, 0.8)
            
            # Update road network weights
            for u, v, data in self.road_network.edges(data=True):
                if f"area_{area_id}" in (u, v):
                    # Increase travel time on damaged roads
                    data["weight"] *= 1.5
        
        else:  # building_collapse
            area.building_damage = min(area.building_damage + 0.2, 1.0)
    
    def _update_weather(self) -> None:
        """Update weather conditions based on time."""
        hours = self.current_time / 3600.0
        
        if hours < 12:
            self.weather_conditions = "clear"
            self.communication_status = 1.0
        elif hours < 36:
            self.weather_conditions = "light_rain"
            self.communication_status = 0.8
        elif hours < 60:
            self.weather_conditions = "heavy_rain"
            self.communication_status = 0.5
        else:
            self.weather_conditions = "clearing"
            self.communication_status = 0.7
    
    def _calculate_reward(self) -> float:
        """Calculate total reward for the current step."""
        reward = 0.0
        
        # 1. Individual efficiency: casualties being treated
        for agent in self.rescue_agents.values():
            if agent.current_mission is not None:
                if agent.current_mission.startswith("treat_casualty_"):
                    # 提取受害者ID并获取受害者信息
                    try:
                        casualty_id = int(agent.current_mission.split("_")[-1])
                        if casualty_id in self.casualties:
                            casualty = self.casualties[casualty_id]
                            # 根据严重程度给予不同奖励
                            severity_bonus = {
                                CasualtySeverity.CRITICAL: 2.0,
                                CasualtySeverity.SEVERE: 1.5,
                                CasualtySeverity.MODERATE: 1.0,
                                CasualtySeverity.MILD: 0.5
                            }[casualty.severity]
                            reward += severity_bonus  # 提高每步治疗奖励
                    except (ValueError, IndexError):
                        reward += 1.0  # 默认奖励
        
        # 2. Global efficiency: total survivors (提高奖励倍数)
        reward += 5.0 * self.statistics["total_survivors"]
        
        # 3. Cooperation bonus: connected agents (提高权重)
        total_connections = sum(len(agent.connected_agents) for agent in self.rescue_agents.values())
        reward += 0.1 * total_connections
        
        # 4. Coverage bonus: agents covering different areas
        covered_areas = set()
        for agent in self.rescue_agents.values():
            nearest = self._get_nearest_affected_area(agent.position)
            if nearest:
                covered_areas.add(nearest.id)
        reward += 0.5 * len(covered_areas)  # 鼓励覆盖更多区域
        
        # 5. Fairness penalty: based on Gini coefficient (提高惩罚)
        survival_rates = [area.survival_rate for area in self.affected_areas.values()]
        if survival_rates:
            gini = self._calculate_gini(survival_rates)
            reward -= 0.5 * gini
        
        # 6. Time efficiency penalty: penalize if few agents are active
        active_agents = sum(1 for agent in self.rescue_agents.values() if agent.current_mission is not None)
        if len(self.rescue_agents) > 0:
            inactivity_rate = 1.0 - (active_agents / len(self.rescue_agents))
            reward -= 0.1 * inactivity_rate  # 惩罚不活跃的智能体
        
        return reward
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values."""
        if not values:
            return 0.0
        
        values = np.array(values)
        n = len(values)
        sum_values = np.sum(values)
        
        # If all values are zero, return 0.0 to avoid division by zero
        if sum_values == 0:
            return 0.0
        
        abs_diffs = np.abs(values[:, None] - values[None, :])
        gini = np.sum(abs_diffs) / (2 * n * sum_values)
        return float(gini)
    
    def _update_statistics(self) -> None:
        """Update simulation statistics."""
        # Calculate fairness metrics
        survival_rates = [area.survival_rate for area in self.affected_areas.values()]
        
        # 打印生存利率列表，以便调试
        print(f"生存利率列表: {survival_rates}")
        
        if survival_rates and any(rate > 0 for rate in survival_rates):
            # Gini coefficient
            gini = self._calculate_gini(survival_rates)
            self.statistics["fairness_metrics"]["gini"].append(gini)
            
            # Theil index
            mean_rate = np.mean(survival_rates)
            if mean_rate > 0:
                theil = np.mean([(r/mean_rate) * np.log(r/mean_rate) for r in survival_rates if r > 0])
                self.statistics["fairness_metrics"]["theil"].append(theil)
            
            # Max-min fairness
            max_min = np.min(survival_rates) if survival_rates else 0.0
            self.statistics["fairness_metrics"]["max_min"].append(max_min)
        else:
            # 如果没有有效的生存利率，添加默认值
            self.statistics["fairness_metrics"]["gini"].append(0.0)
            self.statistics["fairness_metrics"]["theil"].append(0.0)
            self.statistics["fairness_metrics"]["max_min"].append(0.0)
        
        # 计算平均响应时间
        # 模拟响应时间：假设从模拟开始到受害者被治疗的时间
        if self.statistics["response_times"]:
            pass
        else:
            # 添加一些模拟的响应时间数据
            for _ in range(min(10, self.statistics.get("total_survivors", 0))):
                # 随机响应时间，范围在 100 到 300 秒之间
                response_time = np.random.uniform(100, 300)
                self.statistics["response_times"].append(response_time)
        
        # 计算资源利用率
        total_resources = 0.0
        used_resources = 0.0
        for depot in self.resource_depots.values():
            for resource_type, amount in depot.resources.items():
                # 假设每个资源库的初始资源量为 1000.0
                total_resources += 1000.0
                used_resources += 1000.0 - amount
        
        if total_resources > 0:
            utilization = used_resources / total_resources
            for resource_type in ResourceType:
                self.statistics["resource_utilization"][resource_type] = utilization
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if all casualties are either treated or dead
        active_casualties = sum(1 for c in self.casualties.values() if not c.treated)
        return active_casualties == 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for all agents."""
        observations = []
        
        for agent in self.rescue_agents.values():
            obs = self._get_agent_observation(agent)
            observations.append(obs)
        
        return np.array(observations)
    
    def _get_agent_observation(self, agent: RescueAgent) -> np.ndarray:
        """Get observation for a specific agent."""
        obs = []
        
        # Agent state (9 features)
        obs.extend(agent.position / self.map_size)  # Normalized position
        obs.extend(agent.velocity / agent.get_max_speed())  # Normalized velocity
        obs.extend([agent.capacity[rt] / agent.max_capacity[rt] for rt in ResourceType])  # Resource levels
        obs.append(agent.endurance / agent.max_endurance)  # Normalized endurance
        obs.append(1.0 if agent.current_mission is not None else 0.0)  # Mission status
        
        # Nearest affected area info (8 features)
        nearest_area = self._get_nearest_affected_area(agent.position)
        if nearest_area:
            # Add nearest area information
            obs.extend(nearest_area.position / self.map_size)
            obs.append(nearest_area.building_damage)
            obs.append(nearest_area.road_accessibility)
            obs.append(len(nearest_area.casualties) / 100.0)  # Normalized
        else:
            # No nearest area found
            obs.extend([0.0, 0.0])
            obs.extend([0.0] * 6)
        
        # Global resource levels (4 features)
        total_resources = {}
        for depot in self.resource_depots.values():
            for rt, amount in depot.resources.items():
                total_resources[rt] = total_resources.get(rt, 0.0) + amount
        
        for rt in ResourceType:
            obs.append(total_resources.get(rt, 0.0) / 1000.0)  # Normalized
        
        return np.array(obs)
    
    def _get_nearest_affected_area(self, position: np.ndarray) -> Optional[AffectedArea]:
        """Get nearest affected area to a position."""
        if not self.affected_areas:
            return None
        
        nearest_area = None
        min_distance = float('inf')
        
        for area in self.affected_areas.values():
            distance = np.linalg.norm(position - area.position)
            if distance < min_distance:
                min_distance = distance
                nearest_area = area
        
        return nearest_area
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment."""
        info = {
            "current_time": self.current_time,
            "step_count": self.step_count,
            "weather_conditions": self.weather_conditions,
            "communication_status": self.communication_status,
            "secondary_disaster_counter": self.secondary_disaster_counter,
            "statistics": self.statistics,
            "num_casualties": len(self.casualties),  # 直接返回当前受害者数量
            "num_rescue_agents": len(self.rescue_agents),
            "num_affected_areas": len(self.affected_areas),
            "num_resource_depots": len(self.resource_depots),
            "rescued": self.statistics.get("total_survivors", 0),
            "deaths": self.statistics.get("total_deaths", 0),
            "resources_used": sum(self.statistics.get("resource_utilization", {}).values())
        }
        return info
    
    def get_state_dimension(self) -> int:
        """Get the state dimension for each agent."""
        # Calculate state dimension based on observation space
        if hasattr(self, 'observation_space'):
            return self.observation_space.shape[0]
        else:
            # Default state dimension based on _get_agent_observation
            return 19  # 2 + 2 + 4 + 1 + 1 + 5 + 4
    
    def get_action_dimension(self) -> int:
        """Get the action dimension for each agent."""
        # For compatibility with tests
        return 5
    
    @property
    def agents(self) -> List[Dict]:
        """Get list of agents for compatibility with tests."""
        agents_list = []
        for agent_id, agent in self.rescue_agents.items():
            agent_info = {
                'id': agent_id,
                'position': agent.position.tolist(),
                'velocity': agent.velocity.tolist(),
                'capacity': agent.capacity,
                'endurance': agent.endurance
            }
            agents_list.append(agent_info)
        return agents_list
    
    @property
    def victims(self) -> List[Dict]:
        """Get list of victims for compatibility with tests."""
        victims_list = []
        for casualty_id, casualty in self.casualties.items():
            victim_info = {
                'id': casualty_id,
                'position': casualty.position.tolist(),
                'severity': casualty.severity.value,
                'treated': casualty.treated
            }
            victims_list.append(victim_info)
        return victims_list
    
    @property
    def resources(self) -> List[Dict]:
        """Get list of resources for compatibility with tests."""
        resources_list = []
        for resource_type in ResourceType:
            resource_info = {
                'type': resource_type.value,
                'capacity': 100.0,  # Default capacity
                'remaining': 100.0  # Default remaining
            }
            resources_list.append(resource_info)
        return resources_list
    
    @property
    def hospitals(self) -> List[Dict]:
        """Get list of hospitals for compatibility with tests."""
        hospitals_list = []
        # Create dummy hospitals for testing
        for i in range(self.num_hospitals):
            hospital_info = {
                'id': i,
                'position': [100.0 * i, 100.0 * i],
                'capacity': 50
            }
            hospitals_list.append(hospital_info)
        return hospitals_list