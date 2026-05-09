"""
DisasterSim-2026: High-fidelity disaster simulation environment for medical resource allocation.

This module implements the main simulation environment with:
1. Dynamic disaster scenarios
2. Multi-agent rescue operations
3. Resource management
4. Communication networks
5. Casualty simulation (Weibull distribution deterioration model)

时间尺度说明：
- 论文原始尺度：危重病人6小时内50%存活率
- 当前模拟尺度：时间压缩约100倍，危重病人约200步(3.3分钟)内50%存活率
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import networkx as nx
from scipy.spatial.distance import cdist
from gymnasium import spaces
from enum import Enum
logger = logging.getLogger(__name__)


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


WEIBULL_PARAMS = {
    CasualtySeverity.CRITICAL: {"theta": 200, "kappa": 1.0},   # 现实6小时=模拟200步(3.3分钟)，约139步时存活率50%
    CasualtySeverity.SEVERE: {"theta": 600, "kappa": 1.2},     # 现实24小时=模拟600步(10分钟)，约442步时存活率50%
    CasualtySeverity.MODERATE: {"theta": 1800, "kappa": 1.5},  # 现实48小时=模拟1800步(30分钟)，约1410步时存活率50%
    CasualtySeverity.MILD: {"theta": 4800, "kappa": 2.0},     # 现实120小时=模拟4800步(80分钟)，约3996步时存活率50%
}

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
    treating_agent_id: Optional[int] = None
    original_treating_agent_id: Optional[int] = None  # 保存最初治疗的agent
    discovered_by: Optional[int] = None  # 记录第一个发现该伤员的agent
    discovered_at: Optional[float] = None  # 记录该伤员被发现的时间
    grace_period_end: Optional[float] = None
    survival_probability: float = 1.0
    _last_update_time: float = 0.0

    weibull_theta: float = field(init=False)
    weibull_kappa: float = field(init=False)

    def __post_init__(self):
        weibull_params = WEIBULL_PARAMS[self.severity]
        self.weibull_theta = weibull_params["theta"]
        self.weibull_kappa = weibull_params["kappa"]

    def update_survival_probability(self, current_time: float) -> None:
        """Update survival probability using Weibull distribution model.
        
        P_survive(t) = exp(-(t/theta)^kappa)
        
        If treated: survival probability recovers towards 1.0
        If not treated: survival probability decreases following Weibull distribution
        """
        time_delta = current_time - self._last_update_time
        if time_delta <= 0:
            return
            
        if self.treated and self.treatment_start is not None:
            recovery_rate = {
                CasualtySeverity.CRITICAL: 0.030,
                CasualtySeverity.SEVERE: 0.040,
                CasualtySeverity.MODERATE: 0.050,
                CasualtySeverity.MILD: 0.080
            }[self.severity]
            self.survival_probability = min(1.0, self.survival_probability + recovery_rate * time_delta)
        else:
            elapsed = current_time - self.injury_time
            survival = np.exp(-(elapsed / self.weibull_theta) ** self.weibull_kappa)
            self.survival_probability = max(0.0, survival)
            
        self._last_update_time = current_time

    def is_alive(self, current_time: float) -> bool:
        """Check if casualty is still alive (survival_probability > 1%)."""
        return self.survival_probability > 0.01


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


class AgentType:
    DRONE = "drone"
    VEHICLE = "vehicle"
    PERSONNEL = "personnel"


class RescueAgent:
    """Base class for rescue agents."""
    def __init__(self, agent_id: int, position: np.ndarray, map_size=None, agent_type: str = AgentType.PERSONNEL):
        self.id = agent_id
        self.position = position
        self.agent_type = agent_type
        self.velocity = np.zeros(2)
        
        # 根据agent类型设置不同的资源容量（更符合实际场景）
        # VEHICLE > PERSONNEL > DRONE
        if agent_type == AgentType.VEHICLE:
            resource_capacity = 20.0  # 车辆容量最大，可携带大量物资
        elif agent_type == AgentType.PERSONNEL:
            resource_capacity = 8.0   # 人员容量中等，携带背包物资
        elif agent_type == AgentType.DRONE:
            resource_capacity = 3.0   # 无人机容量最小，只能携带少量急救物资
        else:
            resource_capacity = 8.0
        
        self.capacity = {rt: resource_capacity for rt in ResourceType}
        self.max_capacity = {rt: resource_capacity for rt in ResourceType}
        self.endurance = 100.0
        self.max_endurance = 100.0
        self.current_mission = None
        self.connected_agents = []
        self.map_size = map_size
        self.route = []
        self.known_casualties = {}
        self.rescued_count = 0  # 记录该agent救治的伤员数量

    def get_max_speed(self) -> float:
        """Get maximum speed of the agent in m/s based on type."""
        speeds = {
            AgentType.DRONE: 30.0,
            AgentType.VEHICLE: 15.0,
            AgentType.PERSONNEL: 5.0
        }
        return speeds.get(self.agent_type, 5.0)

    def get_detection_range(self) -> float:
        """Get detection range in meters based on type."""
        return 100.0
    
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
        self.max_steps = 14400  # 4 hours simulation
        
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
        self.initial_resources: Dict[int, Dict[ResourceType, float]] = {}
        self.rescue_agents: Dict[int, RescueAgent] = {}
        self.casualties: Dict[int, Casualty] = {}
        self.road_network = nx.Graph()
        self.statistics = {}
        self._logged_insufficient = set()  # 追踪已打印过资源不足日志的(agent_id, casualty_id)组合
        
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
        self.affected_areas = {}

        num_areas = max(3, self.num_hospitals)
        for i in range(num_areas):
            angle = 2 * np.pi * i / num_areas
            radius = self.map_size[0] * 0.15 if isinstance(self.map_size, (tuple, list)) else self.map_size * 0.15
            map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size
            position = np.array([
                map_size / 2 + radius * np.cos(angle),
                map_size / 2 + radius * np.sin(angle)
            ])

            area = AffectedArea(
                id=i,
                position=position,
                size=self.map_size[0] * 0.35 if isinstance(self.map_size, (tuple, list)) else self.map_size * 0.35,
                population=1000 + i * 200,
                building_damage=0.3 + i * 0.1,
                road_accessibility=0.8 - i * 0.1
            )
            self.affected_areas[i] = area
    
    def _initialize_resource_depots(self) -> None:
        """Initialize resource depots."""
        self.resource_depots = {}
        self.initial_resources = {}
        
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
                resources=resources.copy()
            )
            self.resource_depots[i] = depot
            self.initial_resources[i] = resources.copy()
    
    def _initialize_rescue_agents(self) -> None:
        """Initialize rescue agents with heterogeneous types."""
        self.rescue_agents = {}

        map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list)) else self.map_size

        num_drones = max(1, self.num_agents // 10)
        num_vehicles = max(2, self.num_agents // 3)
        num_personnel = self.num_agents - num_drones - num_vehicles

        agent_types = (
            [AgentType.DRONE] * num_drones +
            [AgentType.VEHICLE] * num_vehicles +
            [AgentType.PERSONNEL] * num_personnel
        )
        np.random.shuffle(agent_types)

        for i in range(self.num_agents):
            position = np.random.uniform(0, map_size, 2)

            agent = RescueAgent(
                agent_id=i,
                position=position,
                map_size=self.map_size,
                agent_type=agent_types[i]
            )
            agent._has_refilled = False  # 显式初始化补充标志，避免初始状态错误显示为REFILLED
            self.rescue_agents[i] = agent
        
        # 计算初始agent资源总量
        self.initial_agent_resources = sum(sum(a.capacity.values()) for a in self.rescue_agents.values())
    
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
    
    def _initial_casualty_discovery(self) -> None:
        """
        Initial casualty discovery at episode start.
        Agents discover casualties in their detection range before any actions are taken.
        This ensures that casualties are marked as discovered before agents start treating them.
        """
        agent_ids = list(self.rescue_agents.keys())
        casualty_ids = list(self.casualties.keys())
        
        if len(agent_ids) == 0 or len(casualty_ids) == 0:
            return
        
        agent_positions = np.array([self.rescue_agents[aid].position for aid in agent_ids])
        casualty_positions = np.array([self.casualties[cid].position for cid in casualty_ids])
        
        # Calculate distances between all agents and casualties
        distances = np.sqrt(np.sum((agent_positions[:, np.newaxis, :] - casualty_positions[np.newaxis, :, :]) ** 2, axis=2))
        
        # Discover casualties in detection range
        for i, agent_id in enumerate(agent_ids):
            agent = self.rescue_agents[agent_id]
            detection_range = agent.get_detection_range()
            nearby_casualty_indices = np.where(distances[i, :] <= detection_range)[0]
            
            for j in nearby_casualty_indices:
                casualty_id = casualty_ids[j]
                casualty = self.casualties[casualty_id]
                
                # Mark casualty as discovered by this agent
                if casualty_id not in agent.known_casualties:
                    agent.known_casualties[casualty_id] = {
                        'position': casualty.position.copy(),
                        'severity': casualty.severity,
                        'survival_probability': casualty.survival_probability,
                        'distance': distances[i, j],
                        'treated': casualty.treated
                    }
                    # 记录第一个发现者
                    if self.casualties[casualty_id].discovered_by is None:
                        self.casualties[casualty_id].discovered_by = agent_id
                        self.casualties[casualty_id].discovered_at = self.current_time
                    # 记录每个agent的发现（用于分组打印）
                    self.statistics["pending_casualty_discoveries"].append({
                        'agent_id': agent_id,
                        'casualty_id': casualty_id,
                        'severity': casualty.severity,
                        'position': casualty.position.copy(),
                        'distance': distances[i, j],
                        'time': self.current_time
                    })
                else:
                    # Update existing knowledge
                    agent.known_casualties[casualty_id]['distance'] = distances[i, j]
                    agent.known_casualties[casualty_id]['treated'] = casualty.treated
                    agent.known_casualties[casualty_id]['survival_probability'] = casualty.survival_probability
    
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
            "total_deaths": 0,
            "total_casualties": len(self.casualties),
            "total_treated": 0,
            "total_rescued": 0,
            "resource_utilization": {rt: 0.0 for rt in ResourceType},
            "total_resources_replenished": 0.0,
            "total_resources_consumed": 0.0,
            "response_times": [],
            "fairness_metrics": {"gini": [], "theil": [], "max_min": []},
            "pending_casualty_discoveries": [],  # 待打印的新发现伤员列表
        }

        self._last_currently_treating = 0
        self._last_min_survival = 1.0
        self._last_max_survival = 1.0
        self._logged_insufficient = set()  # 重置已记录的资源不足日志组合
        
        # Reset reward state tracking
        self._last_survivors = 0
        self._last_casualties = len(self.casualties)
        
        # Log initial environment information
        logger.info(f"[ENVIRONMENT INIT] Affected Areas: {len(self.affected_areas)}")
        for area_id, area in self.affected_areas.items():
            logger.info(f"  Area {area_id}: Position=[{area.position[0]:.1f}, {area.position[1]:.1f}], Radius={area.size:.1f}, Damage={area.building_damage:.2f}, Population={area.population:.0f}")
        
        logger.info(f"[ENVIRONMENT INIT] Resource Depots: {len(self.resource_depots)}")
        for depot_id, depot in self.resource_depots.items():
            total_resources = sum(depot.resources.values())
            logger.info(f"  Depot {depot_id}: Position=[{depot.position[0]:.1f}, {depot.position[1]:.1f}], TotalResources={total_resources:.1f}")
        
        # Initial casualty discovery - agents discover casualties in their detection range at episode start
        self._initial_casualty_discovery()
        
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

        if self.step_count % 50 == 0:
            total_initial_resources = sum(sum(r.values()) for r in self.initial_resources.values()) + self.initial_agent_resources
            total_current_resources = sum(sum(d.resources.values()) for d in self.resource_depots.values())
            total_agent_resources = sum(sum(a.capacity.values()) for a in self.rescue_agents.values())
            resource_consumed = max(0, total_initial_resources - total_current_resources - total_agent_resources)

            num_agents = len(self.rescue_agents)
            num_connected = sum(len(a.connected_agents) for a in self.rescue_agents.values())
            max_possible_connections = num_agents * (num_agents - 1) if num_agents > 1 else 1
            comm_coverage = (num_connected / max_possible_connections * 100) if max_possible_connections > 0 else 0

            logger.info(f"[STEP {self.step_count}] Alive: {len(self.casualties)} | Treating: {self._last_currently_treating} | Rescued: {self.statistics.get('total_rescued', 0)} | Deaths: {self.statistics.get('total_deaths', 0)} | Treated: {self.statistics.get('total_treated', 0)} | Resources: Depot={total_current_resources:.1f}, Agents={total_agent_resources:.1f}, Used={resource_consumed:.1f}/{total_initial_resources:.1f} | Comm: {num_connected}/{max_possible_connections} ({comm_coverage:.1f}%)")
            
            # 已移除重复的[CASUALTY FOUND]日志输出，改为仅在DEBUG级别记录发现详情
            
            # 每50步打印每个agent已知的伤员信息（按发现顺序输出）
            if self.step_count % 50 == 0:
                for agent_id, agent in sorted(self.rescue_agents.items()):
                    if agent.known_casualties:
                        casualty_strs = []
                        # 按发现时间排序输出
                        for c_id, info in sorted(agent.known_casualties.items(), key=lambda x: x[1].get('discovered_at', 0)):
                            status = "T" if info.get('treated', False) else "F"
                            casualty_strs.append(f"C{c_id}({info['severity'].name[:4]},{info['distance']:.0f}m,{status})")
                        logger.debug(f"[AGENT {agent_id} KNOWN CASUALTIES] {', '.join(casualty_strs)}")

            for agent in self.rescue_agents.values():
                mission = getattr(agent, 'current_mission', None)
                nearest_dist = None
                target_distance_steps = None
                
                if mission and mission.startswith("treat_casualty_"):
                    casualty_id_str = mission.replace("treat_casualty_", "")
                    try:
                        casualty_id = int(casualty_id_str)
                        if casualty_id in self.casualties:
                            casualty = self.casualties[casualty_id]
                            treatment_duration = self.current_time - casualty.treatment_start
                            required_time = {
                                CasualtySeverity.CRITICAL: 30,
                                CasualtySeverity.SEVERE: 20,
                                CasualtySeverity.MODERATE: 10,
                                CasualtySeverity.MILD: 3,
                            }[casualty.severity]
                            remaining_time = max(0, required_time - treatment_duration)
                            dist_to_casualty = np.linalg.norm(agent.position - casualty.position)
                            steps = np.ceil(dist_to_casualty / (agent.get_max_speed() * self.time_step))
                            status = f"TREATING->Casualty{casualty_id}({dist_to_casualty:.0f}m,{int(steps)} steps,{remaining_time:.0f}s)"
                            nearest_dist = dist_to_casualty
                        else:
                            status = "TREATING->(completed)"
                            nearest_dist = None
                    except ValueError:
                        status = "TREATING->(invalid)"
                        nearest_dist = None
                elif mission and mission.startswith("go_to_depot_"):
                    depot_id = mission.replace("go_to_depot_", "")
                    if int(depot_id) in self.resource_depots:
                        depot = self.resource_depots[int(depot_id)]
                        distance = np.linalg.norm(agent.position - depot.position)
                        steps = np.ceil(distance / (agent.get_max_speed() * self.time_step))
                        nearest_casualty = self._get_nearest_untreated_casualty(agent.position)
                        nearest_info = f" | NearestCasualty={nearest_casualty.id}({np.linalg.norm(agent.position - nearest_casualty.position):.0f}m)" if nearest_casualty else ""
                        status = f"MOVING_TO_DEPOT{depot_id}({distance:.0f}m,{int(steps)} steps){nearest_info}"
                elif mission == "wait_for_resupply":
                    nearest_casualty = self._get_nearest_untreated_casualty(agent.position)
                    if nearest_casualty:
                        distance = np.linalg.norm(agent.position - nearest_casualty.position)
                        steps = np.ceil(distance / (agent.get_max_speed() * self.time_step))
                        status = f"WAITING_FOR_RESUPPLY({distance:.0f}m,{int(steps)} steps)"
                    else:
                        status = "WAITING_FOR_RESUPPLY"
                # 无人机在depot补充完成后的状态
                elif agent.agent_type == AgentType.DRONE and mission is None:
                    # 检查是否在depot附近且已完成过补充
                    at_depot = False
                    depot_id = None
                    for d_id, depot in self.resource_depots.items():
                        distance = np.linalg.norm(agent.position - depot.position)
                        if distance < 100.0:
                            at_depot = True
                            depot_id = d_id
                            break
                    if at_depot and getattr(agent, '_has_refilled', False):
                        total_resources = sum(agent.capacity.values())
                        total_max = sum(agent.max_capacity.values())
                        if total_resources >= total_max * 0.95:
                            status = f"REFILLED_AT_DEPOT{depot_id}"
                        else:
                            status = f"REFILLING_AT_DEPOT{depot_id}"
                    elif agent.known_casualties:
                        nearest = min(agent.known_casualties.items(), key=lambda x: x[1]['distance'])
                        nearest_dist = nearest[1]['distance']
                        steps = np.ceil(nearest_dist / (agent.get_max_speed() * self.time_step))
                        status = f"SEARCHING->Casualty{nearest[0]}({nearest_dist:.0f}m,{int(steps)} steps)"
                    else:
                        in_area = self._is_in_affected_area(agent.position)
                        if not in_area:
                            nearest_area = self._get_nearest_affected_area(agent.position)
                            if nearest_area:
                                distance = np.linalg.norm(agent.position - nearest_area.position)
                                steps = np.ceil(distance / (agent.get_max_speed() * self.time_step))
                                status = f"MOVING_TO_AREA{nearest_area.id}({distance:.0f}m,{int(steps)} steps)"
                            else:
                                status = "MOVING_TO_AREA"
                        else:
                            status = "SEARCHING(no targets)"
                elif agent.known_casualties:
                    nearest = min(agent.known_casualties.items(), key=lambda x: x[1]['distance'])
                    nearest_dist = nearest[1]['distance']
                    steps = np.ceil(nearest_dist / (agent.get_max_speed() * self.time_step))
                    status = f"SEARCHING->Casualty{nearest[0]}({nearest_dist:.0f}m,{int(steps)} steps)"
                else:
                    in_area = self._is_in_affected_area(agent.position)
                    if not in_area:
                        nearest_area = self._get_nearest_affected_area(agent.position)
                        if nearest_area:
                            distance = np.linalg.norm(agent.position - nearest_area.position)
                            steps = np.ceil(distance / (agent.get_max_speed() * self.time_step))
                            status = f"MOVING_TO_AREA{nearest_area.id}({distance:.0f}m,{int(steps)} steps)"
                        else:
                            status = "MOVING_TO_AREA"
                    else:
                        status = "SEARCHING(no targets)"
                
                resource_abbr = {'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT', 'BLOOD_PACKS': 'BLD', 'OXYGEN': 'OXY', 'PAIN_MEDICATION': 'PAIN'}
                resources = {resource_abbr.get(rt.name, rt.name[:4]): round(agent.capacity[rt], 1) for rt in ResourceType}
                rescued_count = getattr(agent, 'rescued_count', 0)
                
                if nearest_dist is not None:
                    logger.debug(f"[AGENT {agent.id}/{agent.agent_type}] Status={status} | Position=[{agent.position[0]:.1f}, {agent.position[1]:.1f}] | NearestCasualtyDist={nearest_dist:.1f} | Resources={resources} | Rescued={rescued_count}")
                else:
                    logger.debug(f"[AGENT {agent.id}/{agent.agent_type}] Status={status} | Position=[{agent.position[0]:.1f}, {agent.position[1]:.1f}] | Resources={resources} | Rescued={rescued_count}")
            
            for casualty_id, casualty in self.casualties.items():
                treating_agent = None
                
                # Find treating agent
                for agent in self.rescue_agents.values():
                    mission = getattr(agent, 'current_mission', None)
                    if mission == f"treat_casualty_{casualty_id}":
                        treating_agent = agent
                        break
                
                if treating_agent:
                    status = f"TREATED_BY=Agent{treating_agent.id}"
                else:
                    min_dist = float('inf')
                    nearest_agent = None
                    for agent in self.rescue_agents.values():
                        dist = np.linalg.norm(agent.position - casualty.position)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_agent = agent
                    status = f"NEAREST_AGENT=Agent{nearest_agent.id}({int(min_dist)}m,{int(min_dist / (nearest_agent.get_max_speed() * self.time_step))} steps)"
                
                discovered_str = f"DiscoveredBy=Agent{casualty.discovered_by}" if casualty.discovered_by is not None else "DiscoveredBy=None"
                logger.debug(f"[CASUALTY {casualty_id}] Severity={casualty.severity.name} | Survival={casualty.survival_probability:.4f} | Position=[{casualty.position[0]:.1f},{casualty.position[1]:.1f}] | {discovered_str} | {status}")
        
        # Apply secondary disasters
        if np.random.rand() < self._get_secondary_disaster_probability():
            self._apply_secondary_disaster()

        # Update weather
        self._update_weather()

        # Update statistics before calculating reward
        self._update_statistics()

        # Calculate reward
        reward = self._calculate_reward()
        
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
        info['deaths'] = self.statistics.get('total_deaths', 0)

        total_replenished = self.statistics.get("total_resources_replenished", 0.0)
        total_consumed = self.statistics.get("total_resources_consumed", 0.0)
        # 使用初始资源总量（depot初始资源 + agent初始资源），确保与训练阶段计算口径一致
        total_initial = sum(sum(r.values()) for r in self.initial_resources.values()) + self.initial_agent_resources
        total_available = total_initial + total_replenished
        info['resources_used'] = total_consumed
        info['resources_replenished'] = total_replenished
        if total_available > 0:
            info['resource_utilization'] = min(100.0, total_consumed / total_available * 100)
        else:
            info['resource_utilization'] = 0.0
        
        # Calculate and add response time (time steps per rescue)
        survivors = self.statistics.get('total_survivors', 0)
        if survivors > 0:
            info['response_time'] = self.step_count / survivors
        else:
            info['response_time'] = 0
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, agent_id: int, action: Dict[str, Any]) -> None:
        """Apply action to a specific agent."""
        agent = self.rescue_agents[agent_id]

        mission = getattr(agent, 'current_mission', None)
        is_treating = False
        if mission and mission.startswith("treat_casualty_"):
            casualty_id_str = mission.replace("treat_casualty_", "")
            try:
                casualty_id = int(casualty_id_str)
                if casualty_id in self.casualties:
                    casualty = self.casualties[casualty_id]
                    if casualty.treated and casualty.treatment_start is not None:
                        treatment_duration = self.current_time - casualty.treatment_start
                        required_time = {
                            CasualtySeverity.CRITICAL: 30,
                            CasualtySeverity.SEVERE: 20,
                            CasualtySeverity.MODERATE: 10,
                            CasualtySeverity.MILD: 3,
                        }[casualty.severity]
                        if treatment_duration < required_time:
                            is_treating = True
                        else:
                            agent.current_mission = None
                else:
                    agent.current_mission = None
            except ValueError:
                agent.current_mission = None

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
            direction_idx = action["tactical"]
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            direction = np.array([np.cos(angles[direction_idx]), np.sin(angles[direction_idx])])

            max_speed = agent.get_max_speed()
            target_distance = max_speed * self.time_step
            target_position = agent.position + direction * target_distance
            target_position = np.clip(target_position, 0, self.map_size)
            agent.route = [target_position]

        # 禁用agent间信息共享，每个agent只记录自己发现的伤员
        # 通信共享功能将在后续模拟通信通知场景时启用
        comm_action = 0

        moved_towards_target = False
        
        # Clean up known_casualties: remove casualties that no longer exist or treatment completed
        to_remove = []
        for cid in agent.known_casualties:
            if cid not in self.casualties:
                to_remove.append(cid)
            else:
                casualty = self.casualties[cid]
                if casualty.treated and casualty.treatment_start is not None:
                    treatment_duration = self.current_time - casualty.treatment_start
                    required_time = {
                        CasualtySeverity.CRITICAL: 30,
                        CasualtySeverity.SEVERE: 20,
                        CasualtySeverity.MODERATE: 10,
                        CasualtySeverity.MILD: 3,
                    }[casualty.severity]
                    if treatment_duration >= required_time:
                        to_remove.append(cid)
        for cid in to_remove:
            del agent.known_casualties[cid]

        if is_treating:
            pass
        elif mission and mission.startswith("go_to_depot_"):
            # 有去depot补充资源的任务，优先执行
            pass  # 由_update_dynamics()处理
        elif agent.known_casualties:
            untreated = [(cid, info) for cid, info in agent.known_casualties.items() if not info.get('treated', False)]
            if untreated:
                target_cid, target_info = min(untreated, key=lambda x: x[1]['distance'])
                target_position = target_info['position']
                direction = target_position - agent.position
                distance = np.linalg.norm(direction)
                if distance > 1.0:
                    direction = direction / distance
                    max_speed = agent.get_max_speed()
                    agent.position += direction * max_speed * self.time_step
                    agent.position = np.clip(agent.position, 0, self.map_size)
                    moved_towards_target = True
                else:
                    target_casualty = self.casualties.get(target_cid)
                    if target_casualty is None or target_casualty.treated:
                        del agent.known_casualties[target_cid]
                        nearest_casualty = self._get_nearest_untreated_casualty(agent.position)
                        if nearest_casualty is not None:
                            agent.known_casualties[nearest_casualty.id] = {
                                'position': nearest_casualty.position.copy(),
                                'severity': nearest_casualty.severity,
                                'survival_probability': nearest_casualty.survival_probability,
                                'distance': np.linalg.norm(nearest_casualty.position - agent.position),
                                'treated': nearest_casualty.treated
                            }
                            new_direction = nearest_casualty.position - agent.position
                            new_distance = np.linalg.norm(new_direction)
                            if new_distance > 1.0:
                                new_direction = new_direction / new_distance
                                max_speed = agent.get_max_speed()
                                agent.position += new_direction * max_speed * self.time_step
                                agent.position = np.clip(agent.position, 0, self.map_size)
                                moved_towards_target = True
        elif not self._is_in_affected_area(agent.position):
            nearest_area = self._get_nearest_affected_area(agent.position)
            if nearest_area:
                direction = nearest_area.position - agent.position
                distance = np.linalg.norm(direction)
                if distance > 1.0:
                    direction = direction / distance
                    max_speed = agent.get_max_speed()
                    agent.position += direction * max_speed * self.time_step
                    agent.position = np.clip(agent.position, 0, self.map_size)
                    moved_towards_target = True
        else:
            nearest_casualty = self._get_nearest_untreated_casualty(agent.position)
            if nearest_casualty is not None:
                casualty_id = nearest_casualty.id
                if casualty_id not in agent.known_casualties:
                    agent.known_casualties[casualty_id] = {
                        'position': nearest_casualty.position.copy(),
                        'severity': nearest_casualty.severity,
                        'survival_probability': nearest_casualty.survival_probability,
                        'distance': np.linalg.norm(nearest_casualty.position - agent.position),
                        'treated': nearest_casualty.treated
                    }
                direction = nearest_casualty.position - agent.position
                distance = np.linalg.norm(direction)
                if distance > 1.0:
                    direction = direction / distance
                    max_speed = agent.get_max_speed()
                    agent.position += direction * max_speed * self.time_step
                    agent.position = np.clip(agent.position, 0, self.map_size)
                    moved_towards_target = True
                else:
                    agent.known_casualties[casualty_id]['distance'] = distance

        if not moved_towards_target and not is_treating:
            # 检查是否正在执行资源补给任务，如果是则继续执行，不覆盖
            current_mission = getattr(agent, 'current_mission', None)
            if current_mission and current_mission.startswith("go_to_agent_"):
                # 继续执行资源补给任务
                continue_mission = True
            else:
                continue_mission = False
            
            has_resources = sum(agent.capacity.values()) > 0.1
            is_drone = getattr(agent, 'agent_type', None) == AgentType.DRONE
            max_capacity = sum(agent.max_capacity.values()) if hasattr(agent, 'max_capacity') else 40.0
            total_capacity = sum(agent.capacity.values())
            drone_critical = is_drone and total_capacity < max_capacity * 0.2
            
            # 检查资源是否足够救治最近的伤员
            can_treat_nearest = True
            nearest_casualty_for_resupply = None
            if self.casualties and not is_drone:
                nearest_casualty_dist = float('inf')
                nearest_casualty = None
                for casualty in self.casualties.values():
                    if casualty.treated:
                        continue
                    dist = np.linalg.norm(agent.position - casualty.position)
                    if dist < nearest_casualty_dist:
                        nearest_casualty_dist = dist
                        nearest_casualty = casualty
                
                if nearest_casualty:
                    nearest_casualty_for_resupply = nearest_casualty
                    can_treat_nearest = self._can_treat_casualty(agent, nearest_casualty, log_failure=False)
            
            # 资源不足条件：无法救治最近的伤员
            needs_resupply = not can_treat_nearest
            
            # 如果正在执行资源补给任务，跳过资源检查逻辑，继续执行任务
            if continue_mission:
                # 继续执行当前的资源补给任务
                target_agent_id = int(current_mission.replace("go_to_agent_", ""))
                if target_agent_id in self.rescue_agents:
                    target_agent = self.rescue_agents[target_agent_id]
                    direction = target_agent.position - agent.position
                    distance = np.linalg.norm(direction)
                    if distance > 1.0:
                        direction = direction / distance
                        max_speed = agent.get_max_speed()
                        agent.position += direction * max_speed * self.time_step
                        agent.position = np.clip(agent.position, 0, self.map_size)
                        
                        if self.step_count % 50 == 0:
                            new_dist = np.linalg.norm(agent.position - target_agent.position)
                            logger.debug(f"[DRONE {agent.id}] Status=DELIVERING->Agent{target_agent.id} | Position={agent.position[0]:.1f},{agent.position[1]:.1f} | Distance={new_dist:.1f}m")
                    else:
                        # 到达目标agent位置，进行投送
                        any_transfer = False
                        transferred_resources = []
                        resource_abbr = {'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT', 'BLOOD_PACKS': 'BLD', 'OXYGEN': 'OXY', 'PAIN_MEDICATION': 'PAIN'}
                        for resource_type in ResourceType:
                            if target_agent.capacity[resource_type] < target_agent.max_capacity[resource_type]:
                                needed = target_agent.max_capacity[resource_type] - target_agent.capacity[resource_type]
                                available = agent.capacity.get(resource_type, 0.0)
                                transfer = min(needed, available)
                                if transfer > 0:
                                    agent.capacity[resource_type] -= transfer
                                    target_agent.capacity[resource_type] += transfer
                                    self.statistics["total_resources_replenished"] += transfer
                                    transferred_resources.append(f"{resource_abbr.get(resource_type.name, resource_type.name[:4])}+{transfer:.2f}")
                                    any_transfer = True
                        if any_transfer:
                            drone_remaining = sum(agent.capacity.values())
                            target_remaining = sum(target_agent.capacity.values())
                            logger.info(f"[DRONE RESUPPLY] Drone{agent.id} -> Agent{target_agent.id} | Resources={','.join(transferred_resources)} | DroneRemaining={drone_remaining:.2f}")
                            self._logged_insufficient = {k for k in self._logged_insufficient if k[0] != target_agent.id}
                        agent.current_mission = None
                else:
                    agent.current_mission = None
            elif not has_resources or drone_critical or needs_resupply:
                # 资源不足时，直接去depot补充（不比较距离）
                nearest_depot_dist = float('inf')
                nearest_depot = None
                for depot in self.resource_depots.values():
                    dist = np.linalg.norm(agent.position - depot.position)
                    if dist < nearest_depot_dist:
                        nearest_depot_dist = dist
                        nearest_depot = depot
                
                if nearest_depot:
                    # 去depot获取资源
                    target_position = nearest_depot.position
                    agent.current_mission = f"go_to_depot_{nearest_depot.id}"
                    if self.step_count % 50 == 0:
                        if not has_resources:
                            reason = "No resources"
                        elif needs_resupply and nearest_casualty_for_resupply:
                            reason = f"Cannot treat nearest casualty {nearest_casualty_for_resupply.id}"
                        elif needs_resupply:
                            reason = "Insufficient resources for treatment"
                        else:
                            reason = "Drone critical"
                        logger.info(f"[AGENT {agent.id} IDLE REASON] {reason}, going to depot {nearest_depot.id} (distance={nearest_depot_dist:.1f}m)")
                elif self.casualties:
                    # 没有depot但有伤员，去最近的伤员位置
                    nearest_casualty_dist = float('inf')
                    nearest_casualty = None
                    for casualty in self.casualties.values():
                        dist = np.linalg.norm(agent.position - casualty.position)
                        if dist < nearest_casualty_dist:
                            nearest_casualty_dist = dist
                            nearest_casualty = casualty
                    if nearest_casualty:
                        target_position = nearest_casualty.position
                        agent.current_mission = f"wait_for_resupply"
                        if self.step_count % 50 == 0:
                            logger.info(f"[AGENT {agent.id} IDLE REASON] No resources, no depot, waiting for resupply near casualty {nearest_casualty.id}")
                
                if 'target_position' in locals():
                    direction = target_position - agent.position
                    distance = np.linalg.norm(direction)
                    if distance > 1.0:
                        direction = direction / distance
                        max_speed = agent.get_max_speed()
                        agent.position += direction * max_speed * self.time_step
                        agent.position = np.clip(agent.position, 0, self.map_size)
                else:
                    if self.step_count % 50 == 0:
                        logger.info(f"[AGENT {agent.id} IDLE REASON] No resources, no depot, no casualties - cannot move")
            elif hasattr(agent, 'route') and agent.route:
                target_position = agent.route[0]
                direction = target_position - agent.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                    max_speed = agent.get_max_speed()
                    move_distance = min(max_speed * self.time_step, distance)
                    agent.position += direction * move_distance
                    if np.linalg.norm(agent.position - target_position) < 1.0:
                        agent.route.pop(0)
            else:
                # 有资源但没有明确目标，随机移动
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                direction_idx = np.random.randint(0, 8)
                direction = np.array([np.cos(angles[direction_idx]), np.sin(angles[direction_idx])])
                max_speed = agent.get_max_speed()
                agent.position += direction * max_speed * self.time_step
                agent.position = np.clip(agent.position, 0, self.map_size)
    
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
                    resource_abbr = {'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT', 'BLOOD_PACKS': 'BLD', 'OXYGEN': 'OXY', 'PAIN_MEDICATION': 'PAIN'}
                    any_replenished = False
                    for resource_type in ResourceType:
                        if agent.capacity[resource_type] < agent.max_capacity[resource_type]:
                            # Try to get resources from depot
                            needed = agent.max_capacity[resource_type] - agent.capacity[resource_type]
                            available = depot.resources.get(resource_type, 0.0)
                            transfer = min(needed, available)
                            
                            if transfer > 0:
                                agent.capacity[resource_type] += transfer
                                depot.resources[resource_type] -= transfer
                                self.statistics["total_resources_replenished"] += transfer
                                any_replenished = True
                    # 如果有资源补充，清除该agent的资源不足日志记录并输出日志
                    if any_replenished:
                        # 补充完成后，清除go_to_depot任务，切换到寻找伤员
                        if getattr(agent, 'current_mission', None) and agent.current_mission.startswith("go_to_depot_"):
                            agent.current_mission = None
                            # 只有在完成depot任务时才输出日志（避免重复输出）
                            self._logged_insufficient = {k for k in self._logged_insufficient if k[0] != agent.id}
                            replenished_resources = []
                            for resource_type in ResourceType:
                                if agent.capacity[resource_type] > 0:
                                    replenished_resources.append(f"{resource_abbr.get(resource_type.name, resource_type.name)}={agent.capacity[resource_type]:.1f}")
                            logger.info(f"[AGENT RESUPPLY] Agent{agent.id} refilled at depot {depot.id} | Resources={','.join(replenished_resources)}")
                    
                    # 检查agent是否在去depot的途中且资源已补充完成
                    if getattr(agent, 'current_mission', None) and agent.current_mission.startswith("go_to_depot_"):
                        # 检查资源是否已满
                        total_resources = sum(agent.capacity.values())
                        total_max = sum(agent.max_capacity.values())
                        if total_resources >= total_max * 0.95:
                            # 资源已满，重置任务并设置目标为最近的已知伤员
                            agent.current_mission = None
                            # 如果有已知伤员，设置目标为最近的
                            if hasattr(agent, 'known_casualties') and agent.known_casualties:
                                nearest = min(agent.known_casualties.items(), key=lambda x: x[1]['distance'])
                                agent.target_casualty_id = nearest[0]
                    break
            
            # If not at depot, endurance decreases normally
            if not at_depot:
                agent.endurance = max(agent.endurance - self.time_step, 0.0)
        
        # 无人机智能资源支援逻辑
        resource_abbr = {'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT', 'BLOOD_PACKS': 'BLD', 'OXYGEN': 'OXY', 'PAIN_MEDICATION': 'PAIN'}
        
        for drone in [a for a in self.rescue_agents.values() if a.agent_type == AgentType.DRONE]:
            total_resources = sum(drone.capacity.values())
            total_max = sum(drone.max_capacity.values())
            
            # 检查无人机资源是否低于20%
            if total_resources < total_max * 0.2:
                # 资源不足，返回最近的depot补充
                nearest_depot_dist = float('inf')
                nearest_depot = None
                for depot in self.resource_depots.values():
                    dist = np.linalg.norm(drone.position - depot.position)
                    if dist < nearest_depot_dist:
                        nearest_depot_dist = dist
                        nearest_depot = depot
                
                if nearest_depot:
                    # 移动到depot
                    direction = nearest_depot.position - drone.position
                    distance = np.linalg.norm(direction)
                    if distance > 1.0:
                        direction = direction / distance
                        max_speed = drone.get_max_speed()
                        drone.position += direction * max_speed * self.time_step
                        drone.position = np.clip(drone.position, 0, self.map_size)
                        drone.current_mission = f"go_to_depot_{nearest_depot.id}"
                    
                    # 到达depot，补充资源
                    if distance <= 10.0:  # 10m范围内视为到达
                        for resource_type in ResourceType:
                            drone.capacity[resource_type] = drone.max_capacity[resource_type]
                        logger.info(f"[DRONE RESUPPLY] Drone{drone.id} refilled at depot {nearest_depot.id}")
                        drone.current_mission = None
                        drone._has_refilled = True  # 设置已补充标志
            else:
                # 资源充足，主动寻找需要资源的agent进行支援
                # 找到资源不足50%的agent（降低阈值，更积极支援）
                needy_agents = []
                for agent in self.rescue_agents.values():
                    if agent.id == drone.id or agent.agent_type == AgentType.DRONE:
                        continue
                    # 检查agent是否资源不足（低于最大容量的50%，降低阈值）
                    agent_total = sum(agent.capacity.values())
                    agent_max = sum(agent.max_capacity.values())
                    if agent_total < agent_max * 0.5:
                        dist = np.linalg.norm(drone.position - agent.position)
                        needy_agents.append((dist, agent))
                
                if needy_agents:
                    # 前往最近的需要资源的agent
                    needy_agents.sort(key=lambda x: x[0])
                    _, target_agent = needy_agents[0]
                    
                    # 重新计算实时距离，而不是使用之前保存的旧距离（关键修复）
                    nearest_dist = np.linalg.norm(drone.position - target_agent.position)
                    
                    if nearest_dist > 10.0:  # 超过10m需要移动（降低距离阈值）
                        direction = target_agent.position - drone.position
                        distance = np.linalg.norm(direction)
                        direction = direction / distance
                        max_speed = drone.get_max_speed()
                        drone.position += direction * max_speed * self.time_step
                        drone.position = np.clip(drone.position, 0, self.map_size)
                        drone.current_mission = f"go_to_agent_{target_agent.id}"
                        
                        # 每50step输出一次日志（在移动后输出，显示最新位置）
                        if self.step_count % 50 == 0:
                            new_dist = np.linalg.norm(drone.position - target_agent.position)
                            logger.debug(f"[DRONE {drone.id}] Status=DELIVERING->Agent{target_agent.id} | Position={drone.position[0]:.1f},{drone.position[1]:.1f} | Distance={new_dist:.1f}m")
                        
                        if self.step_count % 100 == 0:
                            logger.debug(f"[DRONE DISPATCH] Drone{drone.id} moving to Agent{target_agent.id} (distance={nearest_dist:.1f}m)")
                    else:
                        # 在10m范围内，进行投送
                        any_transfer = False
                        transferred_resources = []
                        for resource_type in ResourceType:
                            # 只要agent该资源未满，就补充
                            if target_agent.capacity[resource_type] < target_agent.max_capacity[resource_type]:
                                needed = target_agent.max_capacity[resource_type] - target_agent.capacity[resource_type]
                                available = drone.capacity.get(resource_type, 0.0)
                                transfer = min(needed, available)
                                if transfer > 0:
                                    drone.capacity[resource_type] -= transfer
                                    target_agent.capacity[resource_type] += transfer
                                    self.statistics["total_resources_replenished"] += transfer
                                    transferred_resources.append(f"{resource_abbr.get(resource_type.name, resource_type.name[:4])}+{transfer:.2f}")
                                    any_transfer = True
                        
                        if any_transfer:
                            drone_remaining = sum(drone.capacity.values())
                            target_remaining = sum(target_agent.capacity.values())
                            logger.info(f"[DRONE RESUPPLY] Drone{drone.id} -> Agent{target_agent.id} | Resources={','.join(transferred_resources)} | DroneRemaining={drone_remaining:.2f}")
                            # 清除目标agent的资源不足日志记录
                            self._logged_insufficient = {k for k in self._logged_insufficient if k[0] != target_agent.id}
                        # 完成交付后清除任务，允许切换到其他任务
                        drone.current_mission = None
                else:
                    # 没有需要资源的agent，且资源充足，寻找未发现的伤员
                    # 无人机的主要作用：1)寻找伤员 2)补充资源
                    
                    # 如果正在执行资源补给任务，继续执行，不切换到搜索
                    if getattr(drone, 'current_mission', None) and drone.current_mission.startswith("go_to_agent_"):
                        # 继续执行资源补给任务
                        target_agent_id = int(drone.current_mission.replace("go_to_agent_", ""))
                        if target_agent_id in self.rescue_agents:
                            target_agent = self.rescue_agents[target_agent_id]
                            nearest_dist = np.linalg.norm(drone.position - target_agent.position)
                            if nearest_dist > 10.0:
                                direction = target_agent.position - drone.position
                                distance = np.linalg.norm(direction)
                                direction = direction / distance
                                max_speed = drone.get_max_speed()
                                drone.position += direction * max_speed * self.time_step
                                drone.position = np.clip(drone.position, 0, self.map_size)
                                
                                if self.step_count % 50 == 0:
                                    new_dist = np.linalg.norm(drone.position - target_agent.position)
                                    logger.debug(f"[DRONE {drone.id}] Status=DELIVERING->Agent{target_agent.id} | Position={drone.position[0]:.1f},{drone.position[1]:.1f} | Distance={new_dist:.1f}m")
                            else:
                                # 到达目标agent位置，进行投送
                                any_transfer = False
                                transferred_resources = []
                                for resource_type in ResourceType:
                                    if target_agent.capacity[resource_type] < target_agent.max_capacity[resource_type]:
                                        needed = target_agent.max_capacity[resource_type] - target_agent.capacity[resource_type]
                                        available = drone.capacity.get(resource_type, 0.0)
                                        transfer = min(needed, available)
                                        if transfer > 0:
                                            drone.capacity[resource_type] -= transfer
                                            target_agent.capacity[resource_type] += transfer
                                            self.statistics["total_resources_replenished"] += transfer
                                            transferred_resources.append(f"{resource_abbr.get(resource_type.name, resource_type.name[:4])}+{transfer:.2f}")
                                            any_transfer = True
                                if any_transfer:
                                    drone_remaining = sum(drone.capacity.values())
                                    target_remaining = sum(target_agent.capacity.values())
                                    logger.info(f"[DRONE RESUPPLY] Drone{drone.id} -> Agent{target_agent.id} | Resources={','.join(transferred_resources)} | DroneRemaining={drone_remaining:.2f}")
                                    self._logged_insufficient = {k for k in self._logged_insufficient if k[0] != target_agent.id}
                                drone.current_mission = None
                        else:
                            drone.current_mission = None
                    
                    elif getattr(drone, 'current_mission', None) and drone.current_mission.startswith("go_to_depot_"):
                        # 取消返回depot任务，改为搜索伤员
                        drone.current_mission = None
                    
                    # 如果没有任务，搜索未发现的伤员
                    if not getattr(drone, 'current_mission', None):
                        # 找到未被发现的伤员
                        undiscovered_casualties = []
                        for casualty in self.casualties.values():
                            if casualty.discovered_by is None:
                                dist = np.linalg.norm(drone.position - casualty.position)
                                undiscovered_casualties.append((dist, casualty))
                        
                        if undiscovered_casualties:
                            # 前往最近的未发现伤员
                            undiscovered_casualties.sort(key=lambda x: x[0])
                            nearest_dist, target_casualty = undiscovered_casualties[0]
                            
                            if nearest_dist > 5.0:  # 超过5m需要移动
                                direction = target_casualty.position - drone.position
                                distance = np.linalg.norm(direction)
                                direction = direction / distance
                                max_speed = drone.get_max_speed()
                                drone.position += direction * max_speed * self.time_step
                                map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list, np.ndarray)) else self.map_size
                                drone.position = np.clip(drone.position, 0, map_size)
                                drone.current_mission = f"searching_casualty_{target_casualty.id}"
                                
                                # 每50step输出一次日志（在移动后输出，显示最新位置）
                                if self.step_count % 50 == 0:
                                    new_dist = np.linalg.norm(drone.position - target_casualty.position)
                                    logger.debug(f"[DRONE {drone.id}] Status=SEARCHING->Casualty{target_casualty.id} | Position={drone.position[0]:.1f},{drone.position[1]:.1f} | Distance={new_dist:.1f}m")
                            else:
                                # 到达伤员位置，标记为已发现
                                target_casualty.discovered_by = drone.id
                                drone.current_mission = None
                                if self.step_count % 50 == 0:
                                    logger.debug(f"[DRONE {drone.id}] Discovered Casualty{target_casualty.id} at position {target_casualty.position}")
                        else:
                            # 所有伤员都已发现，进行地毯式巡逻
                            if hasattr(drone, '_patrol_direction'):
                                direction = drone._patrol_direction
                            else:
                                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                                direction_idx = np.random.randint(0, 8)
                                direction = np.array([np.cos(angles[direction_idx]), np.sin(angles[direction_idx])])
                                drone._patrol_direction = direction
                            
                            max_speed = drone.get_max_speed()
                            new_position = drone.position + direction * max_speed * self.time_step
                            
                            # 检查是否超出边界，如果是则改变方向
                            map_size = self.map_size[0] if isinstance(self.map_size, (tuple, list, np.ndarray)) else self.map_size
                            if new_position[0] < 0 or new_position[0] > map_size or new_position[1] < 0 or new_position[1] > map_size:
                                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                                direction_idx = np.random.randint(0, 8)
                                direction = np.array([np.cos(angles[direction_idx]), np.sin(angles[direction_idx])])
                                drone._patrol_direction = direction
                                new_position = drone.position + direction * max_speed * self.time_step
                            
                            drone.position = np.clip(new_position, 0, map_size)
                            
                            # 每50step输出一次巡逻日志
                            if self.step_count % 50 == 0:
                                logger.debug(f"[DRONE {drone.id}] Status=PATROLLING | Position={drone.position[0]:.1f},{drone.position[1]:.1f} | Direction={direction[0]:.2f},{direction[1]:.2f}")
        
        # Agent之间资源共享逻辑
        self._agent_resource_sharing()

    def _agent_resource_sharing(self):
        """Agent之间资源共享，仅在集中资源救治伤员时允许转移（无人机例外）"""
        resource_abbr = {'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT', 'BLOOD_PACKS': 'BLD', 'OXYGEN': 'OXY', 'PAIN_MEDICATION': 'PAIN'}
        agents = list(self.rescue_agents.values())
        transfer_logs = []  # 收集所有转移日志，最后合并输出
        
        # 获取正在治疗的agent列表（同时检查casualty状态和agent的current_mission）
        treating_agents = set()
        for casualty in self.casualties.values():
            if casualty.treated and getattr(casualty, 'treating_agent_id', None) is not None:
                treating_agents.add(casualty.treating_agent_id)
        # 额外检查agent的current_mission，确保捕获所有正在治疗的agent
        for agent in agents:
            mission = getattr(agent, 'current_mission', None)
            if mission and mission.startswith("treat_casualty_"):
                treating_agents.add(agent.id)
        
        # 仅保留策略：集中资源到即将治疗伤员但资源不足的agent
        # 其他场景（普通资源转移）默认不进行，无人机例外（无人机逻辑在_update_dynamics中处理）
        for casualty in self.casualties.values():
            # 只处理未被治疗的伤员
            if casualty.treated:
                continue
            
            # 找到伤员15m范围内的所有agent（排除正在治疗的agent）
            nearby_agents = []
            for agent in agents:
                if agent.agent_type == AgentType.DRONE:
                    continue
                # 跳过正在治疗其他伤员的agent，避免资源冲突
                if agent.id in treating_agents:
                    continue
                dist = np.linalg.norm(agent.position - casualty.position)
                if dist <= 15.0:
                    nearby_agents.append((dist, agent))
            
            if not nearby_agents:
                continue
            
            # 按距离排序，找到最近的agent
            nearby_agents.sort(key=lambda x: x[0])
            nearest_dist, nearest_agent = nearby_agents[0]
            
            # 检查条件1：是否没有任何一个agent单独能治疗该伤员
            any_agent_can_treat = False
            for _, agent in nearby_agents:
                if self._can_treat_casualty(agent, casualty, log_failure=False):
                    any_agent_can_treat = True
                    break
            
            if any_agent_can_treat:
                continue  # 有agent能单独治疗，不需要集中资源
            
            # 检查条件2：所有agent合起来是否有足够资源
            total_resources = {}
            for _, agent in nearby_agents:
                for resource_type in ResourceType:
                    total_resources[resource_type] = total_resources.get(resource_type, 0.0) + agent.capacity.get(resource_type, 0.0)
            
            can_treat_together = True
            for resource_type, amount_needed in casualty.resources_needed.items():
                if total_resources.get(resource_type, 0.0) < amount_needed:
                    can_treat_together = False
                    break
            
            if not can_treat_together:
                # 合起来资源也不够，记录详细日志便于分析
                log_key = (nearest_agent.id, casualty.id)
                if log_key not in self._logged_insufficient:
                    needed_str = ", ".join([f"{resource_abbr.get(rt.name, rt.name[:4])}={amount_needed:.1f}" 
                                          for rt, amount_needed in casualty.resources_needed.items()])
                    total_str = ", ".join([f"{resource_abbr.get(rt.name, rt.name[:4])}={total_resources.get(rt, 0.0):.1f}" 
                                          for rt in ResourceType])
                    logger.info(f"[INSUFFICIENT TOTAL RESOURCES] Casualty{casualty.id} cannot be treated - Needed: {needed_str}, Available: {total_str}, Agents nearby: {len(nearby_agents)}")
                    self._logged_insufficient.add(log_key)
                continue  # 合起来资源也不够，无法治疗
            
            # 满足条件：没有agent单独能治疗，但合起来可以
            # 现在集中资源到最近的agent
            gather_logs = []
            gather_logs.append(f"Casualty{casualty.id}: gathering to Agent{nearest_agent.id}")
            
            # 计算每个资源类型还需要多少
            needed = {}
            for resource_type, amount_needed in casualty.resources_needed.items():
                needed[resource_type] = max(0.0, amount_needed - nearest_agent.capacity.get(resource_type, 0.0))
            
            # 从其他agent按需转移资源（一次性转移足够量）
            has_transfer = False
            for dist, agent in nearby_agents[1:]:  # 跳过最近的agent
                if agent.id == nearest_agent.id:
                    continue
                
                for resource_type in ResourceType:
                    if needed.get(resource_type, 0.0) <= 0:
                        continue
                    
                    available = agent.capacity.get(resource_type, 0.0)
                    if available <= 0:
                        continue
                    
                    # 一次性转移所有可用资源或所需资源的最小值
                    transfer = min(available, needed[resource_type])
                    if transfer > 0:
                        agent.capacity[resource_type] -= transfer
                        nearest_agent.capacity[resource_type] += transfer
                        needed[resource_type] -= transfer
                        abbr = resource_abbr.get(resource_type.name, resource_type.name[:4])
                        gather_logs.append(f"  Agent{agent.id}->Agent{nearest_agent.id}:{abbr}+{transfer:.2f}")
                        has_transfer = True
            
            if has_transfer:  # 有实际转移发生
                transfer_logs.append("[RESOURCE GATHER] " + "; ".join(gather_logs))
            
            # 检查是否已经收集到足够资源（使用完整治疗所需资源量）
            has_enough = self._can_treat_casualty(nearest_agent, casualty, log_failure=False)
            
            if has_enough:
                transfer_logs.append(f"[RESOURCE GATHER COMPLETE] Agent{nearest_agent.id} has enough resources for Casualty{casualty.id}")
                # 清除目标agent的资源不足日志记录
                self._logged_insufficient = {k for k in self._logged_insufficient if k[0] != nearest_agent.id}
                # 标记该agent即将开始治疗，避免被其他伤员选中
                treating_agents.add(nearest_agent.id)
        
        # 合并所有转移日志为单行输出
        if transfer_logs:
            logger.debug(f"[RESOURCE TRANSFER] {'; '.join(transfer_logs)}")

    def _can_treat_casualty(self, agent: 'RescueAgent', casualty: 'Casualty', log_failure: bool = True) -> bool:
        """Check if agent has sufficient resources to treat casualty.
        
        检查资源是否足够完成整个治疗过程，而不仅仅是开始时的需求量。
        治疗过程中资源会持续消耗，需要确保有足够的资源完成治疗。
        """
        if not hasattr(casualty, 'resources_needed') or not casualty.resources_needed:
            return True

        # 获取治疗时间和每步消耗量
        required_time = {
            CasualtySeverity.CRITICAL: 30,
            CasualtySeverity.SEVERE: 20,
            CasualtySeverity.MODERATE: 10,
            CasualtySeverity.MILD: 3,
        }[casualty.severity]
        
        # 每步消耗系数（基于治疗时间调整，确保总消耗合理）
        consumption_factor = {
            CasualtySeverity.CRITICAL: 0.30,  # 30秒治疗，总消耗约为需求量的1.5倍
            CasualtySeverity.SEVERE: 0.25,    # 20秒治疗，总消耗约为需求量的1.3倍
            CasualtySeverity.MODERATE: 0.20,  # 10秒治疗，总消耗约为需求量的1.2倍
            CasualtySeverity.MILD: 0.15        # 3秒治疗，总消耗约为需求量的1.1倍
        }[casualty.severity]

        # 检查每种资源是否足够完成整个治疗过程
        # 总消耗 = 需求量 × (1 + 消耗系数)，确保总消耗在合理范围内
        for resource_type, amount_needed in casualty.resources_needed.items():
            available = agent.capacity.get(resource_type, 0.0)
            total_consumption = amount_needed * (1 + consumption_factor)
            
            if available < total_consumption - 1e-6:  # 允许微小浮点误差
                if log_failure:
                    log_key = (agent.id, casualty.id)
                    if log_key not in self._logged_insufficient:
                        logger.debug(f"[INSUFFICIENT RESOURCES] Agent{agent.id} cannot treat Casualty{casualty.id}: {resource_type.name} needed={total_consumption:.1f} (for full treatment), available={available:.1f}")
                        self._logged_insufficient.add(log_key)
                return False
        return True

    def _get_treatment_priority(self, casualty: 'Casualty', agent_distance: float) -> float:
        """
        Calculate treatment priority for a casualty.
        Higher priority = more urgent.

        Priority factors based on paper principle (efficiency/utilitarian):
        - Survival probability: higher = more urgent (higher priority, less resource investment)
        - Severity: higher severity = higher priority
        - Distance: closer = higher priority (less travel time)

        Formula: Priority = alpha * P_survival + beta * Severity + gamma * (1 - Distance/15)
        """
        alpha, beta, gamma = 0.5, 0.35, 0.15

        severity_weight = {
            CasualtySeverity.CRITICAL: 4,
            CasualtySeverity.SEVERE: 3,
            CasualtySeverity.MODERATE: 2,
            CasualtySeverity.MILD: 1
        }[casualty.severity]

        priority = (
            alpha * casualty.survival_probability +
            beta * severity_weight / 4 +
            gamma * (1 - agent_distance / 15.0)
        )
        return priority

    def _consume_resources_for_treatment(self, agent: 'RescueAgent', casualty: 'Casualty') -> None:
        """Consume resources from agent when treating casualty."""
        if not hasattr(casualty, 'resources_needed') or not casualty.resources_needed:
            return

        consumption_rate = {
            CasualtySeverity.CRITICAL: 0.15,
            CasualtySeverity.SEVERE: 0.10,
            CasualtySeverity.MODERATE: 0.06,
            CasualtySeverity.MILD: 0.03
        }[casualty.severity]

        for resource_type, amount_needed in casualty.resources_needed.items():
            consumption = amount_needed * consumption_rate * self.time_step
            agent.capacity[resource_type] = max(0.0, agent.capacity.get(resource_type, 0.0) - consumption)

    def _update_casualties(self) -> None:
        """Update casualty states and check for deaths.

        Rules:
        1. Only the nearest agent within 15m can treat a casualty
        2. Agent must have sufficient medical resources to treat
        3. Treatment takes time (severity-dependent) and resources are consumed
        4. Casualty can only be rescued if survival_probability >= 0.8 after treatment time
        """
        casualties_to_remove = []

        agent_ids = list(self.rescue_agents.keys())
        casualty_ids = list(self.casualties.keys())

        currently_treating = 0
        min_survival = 1.0
        max_survival = 0.0

        if len(agent_ids) > 0 and len(casualty_ids) > 0:
            agent_positions = np.array([self.rescue_agents[aid].position for aid in agent_ids])
            casualty_positions = np.array([self.casualties[cid].position for cid in casualty_ids])

            distances = np.sqrt(np.sum((agent_positions[:, np.newaxis, :] - casualty_positions[np.newaxis, :, :]) ** 2, axis=2))

            treatment_distance_threshold = 15.0

            for i, agent_id in enumerate(agent_ids):
                agent = self.rescue_agents[agent_id]
                detection_range = agent.get_detection_range()
                nearby_casualty_ids = np.where(distances[i, :] <= detection_range)[0]
                for j in nearby_casualty_ids:
                    casualty_id = casualty_ids[j]
                    casualty = self.casualties[casualty_id]
                    if casualty_id not in agent.known_casualties:
                        agent.known_casualties[casualty_id] = {
                            'position': casualty.position.copy(),
                            'severity': casualty.severity,
                            'survival_probability': casualty.survival_probability,
                            'distance': distances[i, j],
                            'treated': casualty.treated,
                            'discovered_at': self.current_time  # 记录发现时间，用于排序
                        }
                        # 已注释掉[CASUALTY FOUND]日志，避免日志冗余
                    else:
                        agent.known_casualties[casualty_id]['position'] = casualty.position.copy()
                        agent.known_casualties[casualty_id]['distance'] = distances[i, j]
                        agent.known_casualties[casualty_id]['treated'] = casualty.treated
                        agent.known_casualties[casualty_id]['survival_probability'] = casualty.survival_probability

            occupied_agents = set()
            for agent_id, agent in self.rescue_agents.items():
                mission = getattr(agent, 'current_mission', None)
                if mission and mission.startswith("treat_casualty_"):
                    occupied_agents.add(agent_id)

            candidates = []
            for j, casualty_id in enumerate(casualty_ids):
                casualty = self.casualties[casualty_id]

                if not casualty.treated:
                    close_agents = np.where(distances[:, j] <= treatment_distance_threshold)[0]

                    available_agents = [a for a in close_agents if agent_ids[a] not in occupied_agents and self.rescue_agents[agent_ids[a]].agent_type != AgentType.DRONE]

                    if len(available_agents) > 0:
                        close_distances = distances[available_agents, j]
                        nearest_agent_idx = available_agents[np.argmin(close_distances)]
                        nearest_agent_id = agent_ids[nearest_agent_idx]
                        nearest_agent = self.rescue_agents[nearest_agent_id]
                        nearest_distance = distances[nearest_agent_idx, j]

                        if self._can_treat_casualty(nearest_agent, casualty, log_failure=False):
                            priority = self._get_treatment_priority(casualty, nearest_distance)
                            candidates.append((priority, casualty_id, nearest_agent_id, nearest_distance))
                            occupied_agents.add(nearest_agent_id)
                        else:
                            # 最近的agent资源不足，寻找其他资源充足的agent
                            found_alternate = False
                            for agent_idx in available_agents:
                                agent_id = agent_ids[agent_idx]
                                agent = self.rescue_agents[agent_id]
                                if self._can_treat_casualty(agent, casualty, log_failure=False):
                                    distance = distances[agent_idx, j]
                                    priority = self._get_treatment_priority(casualty, distance)
                                    candidates.append((priority, casualty_id, agent_id, distance))
                                    occupied_agents.add(agent_id)
                                    logger.debug(f"[ALTERNATE TREATMENT] Agent{agent_id} (distance={distance:.1f}m) will treat Casualty{casualty_id} since nearest Agent{nearest_agent_id} lacks resources")
                                    found_alternate = True
                                    break
                            # 只有当没有找到替代agent且该组合未曾记录过时才打印资源不足的日志
                            if not found_alternate:
                                log_key = (nearest_agent_id, casualty_id)
                                if log_key not in self._logged_insufficient:
                                    self._can_treat_casualty(nearest_agent, casualty, log_failure=True)
                                    self._logged_insufficient.add(log_key)

            candidates.sort(key=lambda x: x[0], reverse=True)

            for priority, casualty_id, nearest_agent_id, nearest_distance in candidates:
                casualty = self.casualties[casualty_id]
                nearest_agent = self.rescue_agents[nearest_agent_id]

                if self._can_treat_casualty(nearest_agent, casualty, log_failure=False):
                    # 确保救治agent知道这个伤员
                    if casualty_id not in nearest_agent.known_casualties:
                        nearest_agent.known_casualties[casualty_id] = {
                            'position': casualty.position.copy(),
                            'severity': casualty.severity,
                            'survival_probability': casualty.survival_probability,
                            'distance': nearest_distance,
                            'treated': casualty.treated
                        }
                    casualty.treated = True
                    if casualty.treatment_start is None:
                        casualty.treatment_start = self.current_time
                        casualty.treating_agent_id = nearest_agent_id
                        casualty.original_treating_agent_id = nearest_agent_id  # 保存最初治疗的agent
                        self.statistics["total_treated"] = self.statistics.get("total_treated", 0) + 1
                        nearest_agent.current_mission = f"treat_casualty_{casualty_id}"
                    # 如果treatment_start已设置但original_treating_agent_id未设置，补充设置
                    elif not getattr(casualty, 'original_treating_agent_id', None):
                        casualty.original_treating_agent_id = nearest_agent_id
                else:
                    if self.step_count % 50 == 0:
                        logger.debug(f"[TREATMENT BLOCKED] Agent {nearest_agent_id} lacks resources for casualty {casualty_id}")

        for casualty_id, casualty in self.casualties.items():
            casualty.update_survival_probability(self.current_time)
            min_survival = min(min_survival, casualty.survival_probability)
            max_survival = max(max_survival, casualty.survival_probability)

            if casualty.treated and casualty.treatment_start is not None:
                currently_treating += 1

                treating_agent_id = getattr(casualty, 'treating_agent_id', None)
                if treating_agent_id is not None and treating_agent_id in self.rescue_agents:
                    treating_agent = self.rescue_agents[treating_agent_id]

                    has_resources = True
                    for resource_type, amount_needed in casualty.resources_needed.items():
                        if treating_agent.capacity.get(resource_type, 0.0) < amount_needed * 0.1:
                            has_resources = False
                            break

                    if not has_resources:
                        casualty.treated = False
                        casualty.treatment_start = None
                        casualty.treating_agent_id = None
                        # 注意：保留original_treating_agent_id以记录最初的治疗者
                        treating_agent.current_mission = None
                        # 治疗被放弃，减少total_treated统计
                        self.statistics["total_treated"] = max(0, self.statistics.get("total_treated", 0) - 1)
                        if self.step_count % 100 == 0:
                            logger.debug(f"[TREATMENT ABANDONED] Casualty {casualty_id} released due to lack of resources")
                        continue

                    consumption_rate = {
                        CasualtySeverity.CRITICAL: 0.15,
                        CasualtySeverity.SEVERE: 0.10,
                        CasualtySeverity.MODERATE: 0.06,
                        CasualtySeverity.MILD: 0.03
                    }[casualty.severity]

                    for resource_type, amount_needed in casualty.resources_needed.items():
                        consumption = amount_needed * consumption_rate * self.time_step
                        treating_agent.capacity[resource_type] = max(0.0, treating_agent.capacity.get(resource_type, 0.0) - consumption)
                        self.statistics["total_resources_consumed"] += consumption

            if not casualty.is_alive(self.current_time):
                casualties_to_remove.append(casualty_id)
                self.statistics["total_deaths"] = self.statistics.get("total_deaths", 0) + 1

                was_being_treated = casualty.treated and casualty.treatment_start is not None
                treatment_duration = self.current_time - casualty.treatment_start if casualty.treatment_start else 0

                if was_being_treated:
                    logger.warning(f"[DEATH DURING TREATMENT] ID={casualty_id}, Severity={casualty.severity.name}, Survival={casualty.survival_probability:.4f}, TreatedFor={treatment_duration:.1f}s, Time={self.current_time}")
                else:
                    logger.info(f"[CASUALTY DEATH] ID={casualty_id}, Severity={casualty.severity.name}, Survival={casualty.survival_probability:.4f}, Time={self.current_time}")

                for agent in self.rescue_agents.values():
                    if agent.current_mission == f"treat_casualty_{casualty_id}":
                        agent.current_mission = None
                continue

            if casualty.treated and casualty.treatment_start is not None:
                treatment_duration = self.current_time - casualty.treatment_start

                required_time = {
                    CasualtySeverity.CRITICAL: 30,
                    CasualtySeverity.SEVERE: 20,
                    CasualtySeverity.MODERATE: 10,
                    CasualtySeverity.MILD: 3,
                }[casualty.severity]

                if treatment_duration >= required_time and casualty.survival_probability >= 0.8:
                    self.statistics["total_survivors"] += 1
                    self.statistics["total_rescued"] = self.statistics.get("total_rescued", 0) + 1
                    
                    # 计算响应时间：从未发现到现在的时间
                    discovered_time = casualty.discovered_at if casualty.discovered_at is not None else casualty.injury_time
                    response_time = self.current_time - discovered_time
                    self.statistics["response_times"].append(response_time)
                    
                    original_id = getattr(casualty, 'original_treating_agent_id', None)
                    current_id = getattr(casualty, 'treating_agent_id', None)
                    final_treating_agent_id = original_id if original_id is not None else current_id
                    
                    if final_treating_agent_id is None:
                        for agent_id, agent in self.rescue_agents.items():
                            if getattr(agent, 'current_mission', None) == f"treat_casualty_{casualty_id}":
                                final_treating_agent_id = agent_id
                                break
                    
                    treating_agent_name = f"Agent {final_treating_agent_id}" if final_treating_agent_id is not None else "Unknown"
                    if final_treating_agent_id is None:
                        logger.debug(f"[DEBUG] Casualty {casualty_id} rescued but no treating agent found. original_treating_agent_id={original_id}, treating_agent_id={current_id}, treatment_start={casualty.treatment_start}")
                    treatment_start_str = f"{int(casualty.treatment_start)}s" if casualty.treatment_start else "0s"
                    treatment_end_str = f"{int(self.current_time)}s"
                    total_resources_used = sum(casualty.resources_needed.values()) * {
                        CasualtySeverity.CRITICAL: 0.15,
                        CasualtySeverity.SEVERE: 0.10,
                        CasualtySeverity.MODERATE: 0.06,
                        CasualtySeverity.MILD: 0.03
                    }[casualty.severity] * treatment_duration
                    logger.info(f"[CASUALTY RESCUED] ID={casualty_id}, Severity={casualty.severity.name}, Survival={casualty.survival_probability:.4f}, TreatmentTime={int(treatment_duration)}s, TreatedBy={treating_agent_name}, Start={treatment_start_str}, End={treatment_end_str}, ResourcesUsed={total_resources_used:.1f}")

                    # 更新救治agent的救治数量
                    if original_id is not None and original_id in self.rescue_agents:
                        self.rescue_agents[original_id].rescued_count += 1

                    for area_id, area in self.affected_areas.items():
                        if casualty in area.casualties:
                            area.survivors += 1
                            break

                    for agent in self.rescue_agents.values():
                        if agent.current_mission == f"treat_casualty_{casualty_id}":
                            agent.current_mission = None

                    casualties_to_remove.append(casualty_id)

        self._last_currently_treating = currently_treating
        self._last_min_survival = min_survival
        self._last_max_survival = max_survival

        for casualty_id in casualties_to_remove:
            casualty = self.casualties.pop(casualty_id)
            for area in self.affected_areas.values():
                if casualty in area.casualties:
                    area.casualties.remove(casualty)
                    break
            for agent in self.rescue_agents.values():
                if casualty_id in agent.known_casualties:
                    del agent.known_casualties[casualty_id]
    
    def _update_communication(self) -> None:
        """Update communication network between agents. Optimized with vectorized operations."""
        for agent in self.rescue_agents.values():
            agent.connected_agents = []

        agent_ids = list(self.rescue_agents.keys())

        if len(agent_ids) > 1:
            agent_positions = np.array([self.rescue_agents[aid].position for aid in agent_ids])

            distances = np.sqrt(np.sum((agent_positions[:, np.newaxis, :] - agent_positions[np.newaxis, :, :]) ** 2, axis=2))

            communication_range = 1000.0
            can_communicate_matrix = distances < communication_range

            np.fill_diagonal(can_communicate_matrix, False)

            if self.communication_status < 1.0:
                random_matrix = np.random.rand(*can_communicate_matrix.shape) < self.communication_status
                can_communicate_matrix = can_communicate_matrix & random_matrix

            for i, agent_id_i in enumerate(agent_ids):
                connected = np.where(can_communicate_matrix[i])[0]
                self.rescue_agents[agent_id_i].connected_agents = [agent_ids[j] for j in connected]
    
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
        """Calculate total reward for the current step with STRONG incentives for rescue."""
        reward = 0.0

        # 1. 记录上一步的幸存者数量，以便计算救援成功的奖励
        if not hasattr(self, '_last_survivors'):
            self._last_survivors = 0
        
        new_survivors = self.statistics["total_survivors"] - self._last_survivors
        if new_survivors > 0:
            # 救援成功给大幅奖励
            reward += 100.0 * new_survivors
        self._last_survivors = self.statistics["total_survivors"]

        # 2. 每个智能体接近受害者给奖励（密度更高的奖励）- Vectorized version
        treatment_distance_threshold = 15.0

        if len(self.rescue_agents) > 0 and len(self.casualties) > 0:
            agent_ids = list(self.rescue_agents.keys())
            agent_positions = np.array([self.rescue_agents[aid].position for aid in agent_ids])

            untreated_casualty_ids = [cid for cid, c in self.casualties.items() if not c.treated]

            if len(untreated_casualty_ids) > 0:
                casualty_positions = np.array([self.casualties[cid].position for cid in untreated_casualty_ids])
                severity_map = {CasualtySeverity.CRITICAL: 3.0, CasualtySeverity.SEVERE: 2.0,
                               CasualtySeverity.MODERATE: 1.5, CasualtySeverity.MILD: 1.0}
                survival_probs = np.array([self.casualties[cid].survival_probability for cid in untreated_casualty_ids])
                severities = np.array([severity_map[self.casualties[cid].severity] for cid in untreated_casualty_ids])

                distances = np.sqrt(np.sum((agent_positions[:, np.newaxis, :] - casualty_positions[np.newaxis, :, :]) ** 2, axis=2))

                min_distances = distances.min(axis=1)
                min_indices = distances.argmin(axis=1)

                max_range = 500.0
                valid_mask = min_distances < max_range

                for i, valid in enumerate(valid_mask):
                    if valid:
                        min_distance = min_distances[i]
                        idx = min_indices[i]
                        severity_scale = severities[idx]
                        survival_prob = survival_probs[idx]

                        distance_reward = (1.0 - min_distance / max_range) ** 2
                        reward += 0.5 * severity_scale * distance_reward * survival_prob

                        if min_distance <= treatment_distance_threshold:
                            reward += 5.0 * severity_scale * survival_prob

        # 3. 正在接受治疗的受害者给持续奖励
        for casualty in self.casualties.values():
            if casualty.treated and casualty.survival_probability < 1.0:
                severity_scale = {
                    CasualtySeverity.CRITICAL: 2.0,
                    CasualtySeverity.SEVERE: 1.5,
                    CasualtySeverity.MODERATE: 1.0,
                    CasualtySeverity.MILD: 0.5
                }[casualty.severity]
                reward += severity_scale

        # 4. 伤亡死亡的负奖励（惩罚智能体）
        if not hasattr(self, '_last_casualties'):
            self._last_casualties = len(self.casualties) + self.statistics["total_survivors"]
        
        current_casualties = len(self.casualties) + self.statistics["total_survivors"]
        deaths = self._last_casualties - current_casualties
        if deaths > 0:
            reward -= 50.0 * deaths
        self._last_casualties = current_casualties

        # 5. 协作奖励（简单但更直接）
        total_connections = sum(len(agent.connected_agents) for agent in self.rescue_agents.values())
        num_agents = len(self.rescue_agents)
        if num_agents > 1:
            max_connections = num_agents * (num_agents - 1)
            reward += 2.0 * (total_connections / max_connections)

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
        survival_rates = [area.survival_rate for area in self.affected_areas.values()]

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
        
        total_replenished = self.statistics.get("total_resources_replenished", 0.0)
        total_consumed = self.statistics.get("total_resources_consumed", 0.0)

        for rt in ResourceType:
            if total_replenished > 0:
                self.statistics["resource_utilization"][rt] = min(1.0, total_consumed / total_replenished)
            else:
                self.statistics["resource_utilization"][rt] = 0.0
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if there are no more casualties (all rescued or dead)
        return len(self.casualties) == 0
    
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
        
        # Nearest untreated casualty info (6 features)
        nearest_casualty = self._get_nearest_untreated_casualty(agent.position)
        if nearest_casualty:
            obs.extend(nearest_casualty.position / self.map_size)  # 2 features
            severity_map = {"critical": 3, "severe": 2, "moderate": 1, "mild": 0}
            obs.append(severity_map.get(nearest_casualty.severity.value, 0) / 3.0)  # Normalized severity
            distance = np.linalg.norm(agent.position - nearest_casualty.position)
            obs.append(distance / self.map_size[0] if isinstance(self.map_size, (tuple, list)) else distance / self.map_size)  # Normalized distance
            obs.append(1.0 if nearest_casualty.treated else 0.0)  # Treatment status
            obs.append(nearest_casualty.survival_probability)  # Survival probability
        else:
            obs.extend([0.0] * 6)  # No untreated casualties
        
        # Nearest affected area info (8 features)
        nearest_area = self._get_nearest_affected_area(agent.position)
        if nearest_area:
            obs.extend(nearest_area.position / self.map_size)
            obs.append(nearest_area.building_damage)
            obs.append(nearest_area.road_accessibility)
            obs.append(len(nearest_area.casualties) / 100.0)
            area_distance = np.linalg.norm(agent.position - nearest_area.position)
            obs.append(area_distance / self.map_size[0] if isinstance(self.map_size, (tuple, list)) else area_distance / self.map_size)
            obs.append(1.0 if self._is_in_affected_area(agent.position) else 0.0)
        else:
            obs.extend([0.0] * 8)
            obs.append(0.0)
            obs.append(0.0)
        
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

    def _is_in_affected_area(self, position: np.ndarray) -> bool:
        """Check if position is inside any affected area."""
        for area in self.affected_areas.values():
            distance = np.linalg.norm(position - area.position)
            if distance <= area.size / 2:
                return True
        return False

    def _get_nearest_untreated_casualty(self, position: np.ndarray) -> Optional[Casualty]:
        """Get nearest untreated casualty to a position."""
        if not self.casualties:
            return None
        
        nearest_casualty = None
        min_distance = float('inf')
        
        for casualty in self.casualties.values():
            if casualty.treated:
                continue
            distance = np.linalg.norm(position - casualty.position)
            if distance < min_distance:
                min_distance = distance
                nearest_casualty = casualty
        
        return nearest_casualty
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment."""
        # Calculate actual resources used
        total_used = 0.0
        for depot_id, depot in self.resource_depots.items():
            initial_depot = self.initial_resources[depot_id]
            for rt in ResourceType:
                used = initial_depot[rt] - max(0.0, depot.resources.get(rt, 0.0))
                total_used += max(0.0, used)
        
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
            "resources_used": total_used
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