"""
ResourceDepot entity class for disaster simulation.

Represents a storage location for medical supplies.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
from ..config.constants import ResourceType


@dataclass
class ResourceDepot:
    """Resource depot for storing medical supplies."""
    
    id: int
    position: np.ndarray
    resources: Dict[ResourceType, float]
    
    @property
    def total_resources(self) -> float:
        """Calculate total resources in the depot."""
        return sum(self.resources.values())
    
    def replenish(self, resource_type: ResourceType, amount: float) -> None:
        """Add resources to the depot."""
        if resource_type in self.resources:
            self.resources[resource_type] += amount
    
    def consume(self, resource_type: ResourceType, amount: float) -> float:
        """
        Consume resources from the depot.
        
        Returns:
            The actual amount consumed (may be less than requested if insufficient)
        """
        if resource_type not in self.resources:
            return 0.0
        
        available = self.resources[resource_type]
        consumed = min(amount, available)
        self.resources[resource_type] -= consumed
        return consumed
    
    def is_empty(self) -> bool:
        """Check if the depot has any resources."""
        return self.total_resources <= 0.0
    
    def distance_to(self, position: np.ndarray) -> float:
        """Calculate distance to a given position."""
        return np.linalg.norm(self.position - position)
