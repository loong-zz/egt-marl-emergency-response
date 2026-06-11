"""
Pareto Frontier Manager for dynamic fairness-efficiency trade-off.

This module implements the dynamic Pareto frontier that:
1. Generates Pareto-optimal solutions for efficiency vs fairness
2. Selects operating point based on disaster phase
3. Supports context-aware trade-off adjustment
"""

import numpy as np
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


class ParetoFrontierManager:
    """
    Pareto Frontier Manager for dynamic fairness-efficiency trade-off.
    
    This class implements:
    1. Multi-objective optimization for efficiency and fairness
    2. Phase-based operating point selection
    3. Context-aware trade-off adjustment based on resource scarcity and social unrest
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Pareto frontier manager.
        
        Args:
            config: Configuration dictionary with Pareto parameters
        """
        # Phase time thresholds (in hours, converted to steps)
        self.phase1_threshold = config.get('phase1_threshold', 24)    # 0-24h: early phase
        self.phase2_threshold = config.get('phase2_threshold', 72)    # 24-72h: mid phase
        # After 72h: recovery phase
        
        # Weights for each phase (efficiency_weight, fairness_weight)
        self.phase_weights = {
            'early': config.get('early_weights', (0.9, 0.1)),
            'mid': config.get('mid_weights', (0.6, 0.4)),
            'recovery': config.get('recovery_weights', (0.3, 0.7))
        }
        
        # Context sensitivity parameters
        self.resource_scarcity_sensitivity = config.get('resource_scarcity_sensitivity', 0.5)
        self.social_unrest_sensitivity = config.get('social_unrest_sensitivity', 0.3)
        
        # Current weights
        self.current_efficiency_weight = 0.5
        self.current_fairness_weight = 0.5
        
        # History tracking
        self.phase_history = []
        self.weight_history = []
        
    def determine_phase(self, hours_elapsed: float) -> str:
        """
        Determine the current disaster phase based on time elapsed.
        
        Args:
            hours_elapsed: Number of hours since disaster started
            
        Returns:
            Phase name: 'early', 'mid', or 'recovery'
        """
        if hours_elapsed < self.phase1_threshold:
            return 'early'
        elif hours_elapsed < self.phase2_threshold:
            return 'mid'
        else:
            return 'recovery'
    
    def get_base_weights(self, phase: str) -> Tuple[float, float]:
        """
        Get the base weights for a given phase.
        
        Args:
            phase: Disaster phase ('early', 'mid', 'recovery')
            
        Returns:
            (efficiency_weight, fairness_weight)
        """
        return self.phase_weights.get(phase, (0.5, 0.5))
    
    def adjust_weights_for_context(self, efficiency_weight: float, fairness_weight: float,
                                   resource_scarcity: float, social_unrest: float) -> Tuple[float, float]:
        """
        Adjust weights based on contextual factors.
        
        Args:
            efficiency_weight: Base efficiency weight
            fairness_weight: Base fairness weight
            resource_scarcity: Resource scarcity level [0, 1] (1 = very scarce)
            social_unrest: Social unrest level [0, 1] (1 = high unrest)
            
        Returns:
            Adjusted (efficiency_weight, fairness_weight)
        """
        # Higher resource scarcity -> more focus on efficiency
        scarcity_adjustment = self.resource_scarcity_sensitivity * resource_scarcity
        
        # Higher social unrest -> more focus on fairness
        unrest_adjustment = self.social_unrest_sensitivity * social_unrest
        
        # Adjust weights
        new_efficiency = efficiency_weight + scarcity_adjustment - unrest_adjustment
        new_fairness = fairness_weight - scarcity_adjustment + unrest_adjustment
        
        # Normalize to ensure weights sum to 1
        total = new_efficiency + new_fairness
        if total > 0:
            new_efficiency /= total
            new_fairness /= total
        
        # Clamp to [0, 1]
        new_efficiency = max(0.0, min(1.0, new_efficiency))
        new_fairness = max(0.0, min(1.0, new_fairness))
        
        return (new_efficiency, new_fairness)
    
    def update_weights(self, hours_elapsed: float, resource_scarcity: float = 0.0, 
                       social_unrest: float = 0.0):
        """
        Update the current weights based on disaster phase and context.
        
        Args:
            hours_elapsed: Number of hours since disaster started
            resource_scarcity: Resource scarcity level [0, 1]
            social_unrest: Social unrest level [0, 1]
        """
        # Determine current phase
        phase = self.determine_phase(hours_elapsed)
        
        # Get base weights for this phase
        base_efficiency, base_fairness = self.get_base_weights(phase)
        
        # Adjust for context
        adjusted_efficiency, adjusted_fairness = self.adjust_weights_for_context(
            base_efficiency, base_fairness, resource_scarcity, social_unrest
        )
        
        # Update current weights
        self.current_efficiency_weight = adjusted_efficiency
        self.current_fairness_weight = adjusted_fairness
        
        # Record history
        self.phase_history.append(phase)
        self.weight_history.append((adjusted_efficiency, adjusted_fairness))
        
        logger.debug(f"[PARETO] Phase={phase}, Hours={hours_elapsed:.1f}, "
                     f"Efficiency={adjusted_efficiency:.4f}, Fairness={adjusted_fairness:.4f}")
    
    def get_current_weights(self) -> Tuple[float, float]:
        """Get the current efficiency and fairness weights."""
        return (self.current_efficiency_weight, self.current_fairness_weight)
    
    def calculate_priority_score(self, efficiency_score: float, fairness_score: float) -> float:
        """
        Calculate a weighted priority score combining efficiency and fairness.
        
        Args:
            efficiency_score: Efficiency component [0, 1]
            fairness_score: Fairness component [0, 1]
            
        Returns:
            Combined priority score [0, 1]
        """
        return (self.current_efficiency_weight * efficiency_score +
                self.current_fairness_weight * fairness_score)
    
    def get_pareto_frontier(self, num_points: int = 10) -> List[Tuple[float, float]]:
        """
        Generate a discrete approximation of the Pareto frontier.
        
        The frontier represents optimal trade-offs between efficiency and fairness.
        
        Args:
            num_points: Number of points to generate
            
        Returns:
            List of (efficiency, fairness) tuples representing the frontier
        """
        frontier = []
        
        for i in range(num_points + 1):
            efficiency = i / num_points
            # Fairness is derived from efficiency in a convex trade-off
            # This is a simplified frontier; real implementation would use MOO
            fairness = 1.0 - (1.0 - efficiency) ** 2
            frontier.append((efficiency, fairness))
        
        return frontier
    
    def select_operating_point(self, frontier: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Select the current operating point from the Pareto frontier.
        
        Args:
            frontier: List of (efficiency, fairness) tuples
            
        Returns:
            The selected operating point closest to current weights
        """
        target_efficiency, target_fairness = self.get_current_weights()
        
        # Find the point closest to the target
        best_point = None
        best_distance = float('inf')
        
        for point in frontier:
            distance = ((point[0] - target_efficiency) ** 2 + 
                       (point[1] - target_fairness) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_point = point
        
        return best_point
    
    def get_pareto_metrics(self) -> Dict:
        """Get current Pareto metrics for logging/monitoring."""
        return {
            'current_efficiency_weight': self.current_efficiency_weight,
            'current_fairness_weight': self.current_fairness_weight,
            'phase_weights': self.phase_weights,
            'phase1_threshold': self.phase1_threshold,
            'phase2_threshold': self.phase2_threshold,
            'history_length': len(self.weight_history),
            'sensitivity_parameters': {
                'resource_scarcity': self.resource_scarcity_sensitivity,
                'social_unrest': self.social_unrest_sensitivity
            }
        }
    
    def reset(self):
        """Reset the Pareto frontier manager to initial state."""
        self.current_efficiency_weight = 0.5
        self.current_fairness_weight = 0.5
        self.phase_history = []
        self.weight_history = []
        logger.debug("[PARETO] Manager reset")