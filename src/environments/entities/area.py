"""
AffectedArea entity class for disaster simulation.

Represents an area affected by the disaster.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .casualty import Casualty


@dataclass
class AffectedArea:
    """Affected area in the disaster."""
    
    id: int
    position: np.ndarray
    size: float
    population: int
    building_damage: float  # 0.0 to 1.0
    road_accessibility: float  # 0.0 to 1.0
    casualties: List[Casualty] = field(default_factory=list)
    initial_casualties: int = 0
    survivors: int = 0
    
    def __post_init__(self):
        """Initialize derived properties."""
        self.initial_casualties = len(self.casualties)
    
    @property
    def survival_rate(self) -> float:
        """Calculate survival rate for this area."""
        if self.initial_casualties == 0:
            return 0.0
        return self.survivors / self.initial_casualties
    
    @property
    def remaining_casualties(self) -> int:
        """Count remaining untreated casualties."""
        return sum(1 for c in self.casualties if not c.treated or not c.is_alive(self.casualties[0]._last_update_time))
    
    def add_casualty(self, casualty: Casualty) -> None:
        """Add a casualty to this area."""
        self.casualties.append(casualty)
        self.initial_casualties = len(self.casualties)
    
    def remove_casualty(self, casualty_id: int) -> Optional[Casualty]:
        """Remove a casualty from this area by ID."""
        for i, casualty in enumerate(self.casualties):
            if casualty.id == casualty_id:
                return self.casualties.pop(i)
        return None
    
    def update_casualties(self, current_time: float) -> None:
        """Update survival probabilities for all casualties in this area."""
        for casualty in self.casualties:
            casualty.update_survival_probability(current_time)
    
    def is_accessible(self) -> bool:
        """Check if the area is accessible via roads."""
        return self.road_accessibility > 0.1
    
    def get_total_resources_needed(self) -> float:
        """Calculate total resources needed for all casualties."""
        total = 0.0
        for casualty in self.casualties:
            total += sum(casualty.resources_needed.values())
        return total
