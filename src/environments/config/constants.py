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
WEIBULL_PARAMS = {
    CasualtySeverity.CRITICAL: {"theta": 200, "kappa": 1.0},   # ~6 hours (200 steps), 50% survival at ~139 steps
    CasualtySeverity.SEVERE: {"theta": 600, "kappa": 1.2},     # ~24 hours (600 steps), 50% survival at ~442 steps
    CasualtySeverity.MODERATE: {"theta": 1800, "kappa": 1.5},  # ~48 hours (1800 steps), 50% survival at ~1410 steps
    CasualtySeverity.MILD: {"theta": 4800, "kappa": 2.0},      # ~120 hours (4800 steps), 50% survival at ~3996 steps
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
    num_hospitals: int = 2

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


# ==================== Default Config Instance ====================

DEFAULT_CONFIG = SimulationConfig()
