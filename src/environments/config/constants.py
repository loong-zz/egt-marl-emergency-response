"""
Constants and configuration for the disaster simulation environment.

This module contains all centralized constants, enums, and configuration defaults.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# ==================== Enums ====================

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


class AgentType(Enum):
    """Types of rescue agents."""
    DRONE = "drone"
    VEHICLE = "vehicle"
    PERSONNEL = "personnel"


# ==================== Simulation Constants ====================

# Weibull distribution parameters for survival probability
# 修复3+：压缩生存时间；MILD 也设上限确保 episode 可自然终止
WEIBULL_PARAMS = {
    CasualtySeverity.CRITICAL: {"theta": 200, "kappa": 1.0},   # max≈140步
    CasualtySeverity.SEVERE: {"theta": 240, "kappa": 1.2},     # max≈168步
    CasualtySeverity.MODERATE: {"theta": 420, "kappa": 1.5},   # max≈294步
    CasualtySeverity.MILD: {"theta": 800, "kappa": 2.0},       # max≈560步 (episode ≤800 步内可自然终止)
}

# Treatment duration in seconds
TREATMENT_DURATION = {
    CasualtySeverity.CRITICAL: 30,
    CasualtySeverity.SEVERE: 20,
    CasualtySeverity.MODERATE: 10,
    CasualtySeverity.MILD: 3,
}

# Resource consumption rate per second (proportion of total needed)
CONSUMPTION_RATE = {
    CasualtySeverity.CRITICAL: 0.0433,   # (1 + 0.30) / 30 ≈ 0.0433
    CasualtySeverity.SEVERE: 0.0625,     # (1 + 0.25) / 20 = 0.0625
    CasualtySeverity.MODERATE: 0.12,     # (1 + 0.20) / 10 = 0.12
    CasualtySeverity.MILD: 0.3833,       # (1 + 0.15) / 3 ≈ 0.3833
}

# Resource consumption factor (total consumption = demand * (1 + factor))
CONSUMPTION_FACTOR = {
    CasualtySeverity.CRITICAL: 0.30,  # Total consumption ≈ 1.5x demand
    CasualtySeverity.SEVERE: 0.25,    # Total consumption ≈ 1.3x demand
    CasualtySeverity.MODERATE: 0.20,  # Total consumption ≈ 1.2x demand
    CasualtySeverity.MILD: 0.15,      # Total consumption ≈ 1.1x demand
}

# Resource type abbreviations for logging
RESOURCE_ABBR = {
    'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT',
    'BLOOD_PACKS': 'BLD',
    'OXYGEN': 'OXY',
    'PAIN_MEDICATION': 'PN'
}

# Arrival range (in meters) - agent considered arrived at target when within this distance
ARRIVAL_RANGE = 1.0

# Treatment range (agent must be within this distance to treat casualty)
TREATMENT_RANGE = ARRIVAL_RANGE

# Resources needed for treating casualties by severity level
RESOURCES_NEEDED = {
    CasualtySeverity.CRITICAL: {
        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 6.0,
        ResourceType.BLOOD_PACKS: 4.0,
        ResourceType.OXYGEN: 8.0,
        ResourceType.PAIN_MEDICATION: 3.0
    },
    CasualtySeverity.SEVERE: {
        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 4.0,
        ResourceType.BLOOD_PACKS: 2.0,
        ResourceType.OXYGEN: 5.0,
        ResourceType.PAIN_MEDICATION: 2.0
    },
    CasualtySeverity.MODERATE: {
        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 2.0,
        ResourceType.BLOOD_PACKS: 1.0,
        ResourceType.OXYGEN: 3.0,
        ResourceType.PAIN_MEDICATION: 1.5
    },
    CasualtySeverity.MILD: {
        ResourceType.BROAD_SPECTRUM_ANTIBIOTICS: 0.5,
        ResourceType.BLOOD_PACKS: 0.0,
        ResourceType.OXYGEN: 1.0,
        ResourceType.PAIN_MEDICATION: 0.5
    }
}

# Convenience reference for moderate casualty resources (used for refill threshold check)
MODERATE_RESOURCES_NEEDED = RESOURCES_NEEDED[CasualtySeverity.MODERATE]

# Resource supply ratio for creating scarcity (0.7-0.8 is recommended for balanced challenge)
# This creates a "zero-sum" environment where resources are insufficient but achievable through optimization
RESOURCE_SUPPLY_RATIO = 0.75

# Expected severity distribution (used for initial resource calculation)
# Can be overridden by config if casualty generation uses different distribution
EXPECTED_SEVERITY_DISTRIBUTION = {
    CasualtySeverity.CRITICAL: 0.25,
    CasualtySeverity.SEVERE: 0.25,
    CasualtySeverity.MODERATE: 0.30,
    CasualtySeverity.MILD: 0.20
}

# Agent speed in m/s
AGENT_SPEEDS = {
    AgentType.DRONE: 30.0,
    AgentType.VEHICLE: 15.0,
    AgentType.PERSONNEL: 5.0
}

# Agent resource capacity
AGENT_CAPACITY = {
    AgentType.DRONE: 3.0,    # Smallest capacity
    AgentType.PERSONNEL: 8.0,  # Medium capacity
    AgentType.VEHICLE: 20.0   # Largest capacity
}

# Agent detection range by type (in meters)
AGENT_DETECTION_RANGE = {
    AgentType.DRONE: 30.0,    # Drones have larger detection range
    AgentType.PERSONNEL: 10.0,  # Personnel have standard detection range
    AgentType.VEHICLE: 15.0   # Vehicles have slightly larger range
}

# Communication range (in meters)
COMMUNICATION_RANGE = 1000.0

# Minimum movement distance threshold (in meters) - avoid division by zero and jitter
MIN_MOVE_DISTANCE = 1.0

# Position change threshold for casualty detection (in meters)
POSITION_CHANGE_THRESHOLD = 0.1

# ==================== Manager Configuration Constants ====================

# EGT Manager 参数（演化博弈论元控制器）
EGT_CONFIG = {
    'kappa': 0.01,           # 调整率
    'tau_0': 0.3,            # 初始阈值
    'nu': 0.001,             # 衰减率
    'lambda_min': 0.0,       # 公平权重最小值
    'lambda_max': 1.0,       # 公平权重最大值
    'delta': 0.02,           # 滞后阈值
    'initial_lambda': 0.5    # 初始公平-效率权重
}

# Reputation Manager 参数（激励相容机制）
REPUTATION_CONFIG = {
    'initial_reputation': 0.5,      # 初始信誉值
    'honesty_bonus': 0.05,          # 诚实奖励
    'dishonesty_penalty': 0.1,      # 不诚实惩罚
    'decay_rate': 0.01,             # 衰减率
    'forgetting_factor': 0.95,      # 遗忘因子
    'penalty_factor': 0.7,          # 惩罚因子
    'anomaly_threshold': 2.0        # 异常检测阈值
}

# Pareto Manager 参数（动态帕累托前沿）
PARETO_CONFIG = {
    'initial_efficiency': 0.5,           # 初始效率权重
    'initial_fairness': 0.5,             # 初始公平权重
    'efficiency_weight_phase1': 0.9,     # 灾情初期效率权重
    'efficiency_weight_phase2': 0.6,     # 灾情中期效率权重
    'efficiency_weight_phase3': 0.3      # 灾情恢复期效率权重
}

# Communication Manager 参数（信息共享机制）
COMMUNICATION_CONFIG = {
    'communication_range': 50.0,      # 通信范围（米）
    'broadcast_frequency': 5,         # 广播频率（每N步）
    'max_broadcast_size': 10          # 最大广播伤员数量
}

# Interference Manager 参数（通信干扰模型）
INTERFERENCE_CONFIG = {
    'min_delay_mean': 0.5,           # 最小通信延迟均值（秒）
    'max_delay_mean': 2.0,           # 最大通信延迟均值（秒）
    'min_packet_loss': 0.05,         # 最小丢包率
    'max_packet_loss': 0.20,         # 最大丢包率
    'interruption_probability': 0.10, # 余震中断概率
    'improvement_rate': 0.001        # 通信质量改善率
}

# 区域配置（用于多区域适应度计算）
NUM_REGIONS = 4  # 灾难场景划分为4个区域


# ==================== EGT-MARL Algorithm Constants ====================
# 这些是算法侧的"硬常量",与仿真侧常量(SimulationConfig)平行。
# 在代码里改这些值会改变算法行为,所以不要随意调。
#
# 配置优先级(高→低):
#   1. YAML 配置文件 (src/configs/egt_marl.yaml, src/experiments/configs/*.yaml)
#   2. 命令行参数 (--num_episodes 等)
#   3. 本文件常量 (作为 yaml 缺失字段时的兜底默认值)
# 即:yaml 显式给出值时优先用 yaml,没给才回退到本常量。

# EGT 演化博弈的策略数量。修改此值需同时更新:
#   - src/configs/egt_marl.yaml 的 egt.num_strategies
#   - 策略名顺序(见 STRATEGY_NAMES,需保持长度 == NUM_STRATEGIES)
# 当前固定 3(Fairness / Efficiency / Balanced),这是 EGTLayer 实际
# 跑通后被广泛使用的值。
NUM_STRATEGIES = 3

# 策略名映射,长度必须 == NUM_STRATEGIES
# 顺序与 egt_layer.get_fairness_efficiency_weights 的索引对应:
#   0 = Fairness, 1 = Efficiency, 2 = Balanced
STRATEGY_NAMES = ['Fairness', 'Efficiency', 'Balanced']

# ==================== Configuration Dataclass ====================

@dataclass
class SimulationConfig:
    """
    Centralized configuration for the disaster simulation.

    All configurable parameters should be defined here.
    """

    # Map configuration
    map_size: Tuple[float, float] = (10000.0, 10000.0)
    time_step: float = 1.0  # Seconds per simulation step
    max_steps: int = 14400  # 4 hours simulation

    # Agent configuration
    num_agents: int = 10
    agent_types: Tuple[AgentType, ...] = (AgentType.PERSONNEL, AgentType.VEHICLE, AgentType.DRONE)

    # Casualty configuration
    num_victims: int = 100

    # Resource depot configuration
    num_resources: int = 4
    # num_areas: count of affected disaster areas.
    # Used to bound the number of affected areas (``max(3, num_areas)`` in DisasterSim).
    num_areas: int = 3

    # Disaster configuration
    disaster_type: str = 'earthquake'
    severity: str = 'medium'

    # Treatment configuration
    treatment_duration: Dict[CasualtySeverity, int] = field(
        default_factory=lambda: TREATMENT_DURATION.copy()
    )
    consumption_rate: Dict[CasualtySeverity, float] = field(
        default_factory=lambda: CONSUMPTION_RATE.copy()
    )
    consumption_factor: Dict[CasualtySeverity, float] = field(
        default_factory=lambda: CONSUMPTION_FACTOR.copy()
    )

    # Drone configuration
    drone_resource_threshold: float = 0.2  # Return to depot when below this percentage
    drone_delivery_range: float = 10.0    # Delivery distance threshold

    # Resource configuration
    initial_resource_level: float = 10.0

    # Depot configuration
    depot_refill_threshold: float = 0.95  # Consider refilled when at this percentage

    # Secondary disaster configuration
    secondary_disaster_probability: float = 0.001  # Probability per step

    # Manager configurations
    egt_config: Dict = field(default_factory=lambda: EGT_CONFIG.copy())
    reputation_config: Dict = field(default_factory=lambda: REPUTATION_CONFIG.copy())
    pareto_config: Dict = field(default_factory=lambda: PARETO_CONFIG.copy())
    communication_config: Dict = field(default_factory=lambda: COMMUNICATION_CONFIG.copy())
    interference_config: Dict = field(default_factory=lambda: INTERFERENCE_CONFIG.copy())

    # Region configuration
    num_regions: int = NUM_REGIONS


# ==================== Default Config Instance ====================

DEFAULT_CONFIG = SimulationConfig()
