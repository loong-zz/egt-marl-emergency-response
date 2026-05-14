"""
Casualty entity class for disaster simulation.

Represents an injured victim in the disaster scenario.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from ..config.constants import (
    CasualtySeverity, ResourceType, WEIBULL_PARAMS
)


@dataclass
class Casualty:
    """Casualty in the disaster simulation."""
    
    id: int
    position: np.ndarray
    severity: CasualtySeverity
    injury_time: float
    resources_needed: Dict[ResourceType, float]
    
    # State variables
    treated: bool = False
    treatment_start: Optional[float] = None
    treating_agent_id: Optional[int] = None
    original_treating_agent_id: Optional[int] = None  # Original treating agent
    discovered_by: Optional[int] = None  # First agent to discover this casualty
    discovered_at: Optional[float] = None  # Time when discovered
    grace_period_end: Optional[float] = None
    survival_probability: float = 1.0
    _last_update_time: float = 0.0
    
    # Weibull distribution parameters (initialized post-hoc)
    weibull_theta: float = field(init=False)
    weibull_kappa: float = field(init=False)
    
    def __post_init__(self):
        """Initialize Weibull parameters after dataclass creation."""
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
            # Recovery phase - survival probability increases
            recovery_rate = {
                CasualtySeverity.CRITICAL: 0.030,
                CasualtySeverity.SEVERE: 0.040,
                CasualtySeverity.MODERATE: 0.050,
                CasualtySeverity.MILD: 0.080
            }[self.severity]
            self.survival_probability = min(1.0, self.survival_probability + recovery_rate * time_delta)
        else:
            # Survival phase - probability decreases based on Weibull distribution
            elapsed = current_time - self.injury_time
            survival = np.exp(-(elapsed / self.weibull_theta) ** self.weibull_kappa)
            self.survival_probability = max(0.0, survival)
        
        self._last_update_time = current_time
    
    def is_alive(self, current_time: float) -> bool:
        """Check if casualty is still alive (survival_probability > 1%)."""
        return self.survival_probability > 0.01
    
    def is_discovered(self) -> bool:
        """Check if casualty has been discovered by any agent."""
        return self.discovered_by is not None
    
    def start_treatment(self, agent_id: int, start_time: float) -> None:
        """Start treatment by an agent."""
        self.treatment_start = start_time
        self.treating_agent_id = agent_id
        if self.original_treating_agent_id is None:
            self.original_treating_agent_id = agent_id

    def stop_treatment(self) -> None:
        """Stop current treatment."""
        self.treating_agent_id = None
    
    def discover(self, agent_id: int, discovery_time: float) -> None:
        """Mark casualty as discovered by an agent."""
        if self.discovered_by is None:
            self.discovered_by = agent_id
            self.discovered_at = discovery_time
    
    def format_position_log(self) -> str:
        """Format position information for logging."""
        return f"[{self.position[0]:.1f},{self.position[1]:.1f}]"
    
    def format_log_line(self, nearest_agent_info: Optional[Dict[str, Any]] = None) -> str:
        """Format casualty state as a single log line."""
        status = 'TREATED' if self.treated else 'DISCOVERED' if self.discovered_by else 'UNKNOWN'
        
        nearest_info = ""
        if nearest_agent_info:
            nearest_info = f"| Nearest={nearest_agent_info.get('agent_id', 'None')}({nearest_agent_info.get('distance', 0.0):.1f}m)"
        
        return (
            f"CASUALTY {self.id} | "
            f"Pos={self.format_position_log()} | "
            f"Sev={self.severity.name} | "
            f"Status={status} | "
            f"Survival={self.survival_probability:.2f} | "
            f"DiscoveredBy={self.discovered_by or 'None'} | "
            f"Treating={self.treating_agent_id or 'None'}"
            f"{nearest_info}"
        )
