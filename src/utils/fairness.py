"""
Fairness measurement and analysis utilities for EGT-MARL disaster resource allocation system.

This module provides functions for calculating various fairness metrics:
- Gini coefficient
- Theil index
- Max-min fairness
- Fairness-efficiency tradeoff analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def gini_coefficient(x: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    gini = 0.5 * rmad
    return float(gini)


def theil_index(x: np.ndarray) -> float:
    """Calculate Theil index (generalized entropy index with α=1)."""
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0.0
    
    # Avoid division by zero and log(0)
    x_normalized = x / mean_x
    x_normalized = np.where(x_normalized > 0, x_normalized, 1e-10)
    
    theil = np.mean(x_normalized * np.log(x_normalized))
    return float(theil)


def max_min_fairness(x: np.ndarray) -> float:
    """Calculate max-min fairness ratio."""
    if len(x) == 0:
        return 1.0
    
    min_val = np.min(x)
    max_val = np.max(x)
    
    if max_val > 0:
        return min_val / max_val
    else:
        return 1.0


def calculate_fairness_efficiency_tradeoff(
    fairness_scores: List[float],
    efficiency_scores: List[float]
) -> Dict[str, Any]:
    """
    Calculate fairness-efficiency tradeoff metrics.
    
    Args:
        fairness_scores: List of fairness scores
        efficiency_scores: List of efficiency scores
        
    Returns:
        Dictionary containing tradeoff metrics
    """
    if not fairness_scores or not efficiency_scores:
        return {
            'pareto_frontier': [],
            'tradeoff_ratio': 1.0,
            'correlation': 0.0
        }
    
    # Calculate Pareto frontier
    pareto_frontier = _calculate_pareto_frontier(
        fairness_scores, efficiency_scores
    )
    
    # Calculate tradeoff ratio
    tradeoff_ratio = _calculate_tradeoff_ratio(
        fairness_scores, efficiency_scores
    )
    
    # Calculate correlation between fairness and efficiency
    correlation = np.corrcoef(fairness_scores, efficiency_scores)[0, 1]
    
    return {
        'pareto_frontier': pareto_frontier,
        'tradeoff_ratio': tradeoff_ratio,
        'correlation': float(correlation)
    }


def _calculate_pareto_frontier(
    fairness_scores: List[float],
    efficiency_scores: List[float]
) -> List[Tuple[float, float]]:
    """Calculate Pareto frontier points."""
    # Combine scores into list of tuples
    points = list(zip(fairness_scores, efficiency_scores))
    
    # Sort by efficiency, then by fairness
    points.sort(key=lambda x: (-x[1], -x[0]))
    
    # Find Pareto optimal points
    pareto = []
    max_fairness = -float('inf')
    
    for fairness, efficiency in points:
        if fairness > max_fairness:
            pareto.append((fairness, efficiency))
            max_fairness = fairness
    
    return pareto


def _calculate_tradeoff_ratio(
    fairness_scores: List[float],
    efficiency_scores: List[float]
) -> float:
    """Calculate fairness-efficiency tradeoff ratio."""
    if len(fairness_scores) < 2:
        return 1.0
    
    # Calculate normalized scores
    norm_fairness = np.array(fairness_scores) / max(max(fairness_scores), 1e-10)
    norm_efficiency = np.array(efficiency_scores) / max(max(efficiency_scores), 1e-10)
    
    # Calculate tradeoff ratio as the ratio of mean fairness to mean efficiency
    mean_fairness = np.mean(norm_fairness)
    mean_efficiency = np.mean(norm_efficiency)
    
    if mean_efficiency > 0:
        return float(mean_fairness / mean_efficiency)
    else:
        return 1.0


class FairnessMetrics:
    """Fairness metrics calculator."""
    
    def gini_coefficient(self, x: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        return gini_coefficient(x)
    
    def max_min_fairness(self, x: np.ndarray) -> float:
        """Calculate max-min fairness ratio."""
        return max_min_fairness(x)
    
    def theil_index(self, x: np.ndarray) -> float:
        """Calculate Theil index."""
        return theil_index(x)
    
    def jain_fairness_index(self, x: np.ndarray) -> float:
        """Calculate Jain fairness index."""
        if len(x) == 0:
            return 1.0
        
        sum_x = np.sum(x)
        if sum_x == 0:
            return 1.0
        
        sum_squared = np.sum(x ** 2)
        jain = (sum_x ** 2) / (len(x) * sum_squared)
        return float(jain)
    
    def atkinson_index(self, x: np.ndarray, epsilon: float = 1.0) -> float:
        """Calculate Atkinson index."""
        if len(x) == 0:
            return 0.0
        
        mean_x = np.mean(x)
        if mean_x == 0:
            return 0.0
        
        if epsilon == 0:
            return 0.0
        elif epsilon == 1:
            # Use log form for epsilon=1
            log_mean = np.log(mean_x)
            mean_log = np.mean(np.log(x))
            return float(1 - np.exp(mean_log - log_mean))
        else:
            # General case
            x_normalized = x / mean_x
            term = np.mean(x_normalized ** (1 - epsilon))
            atkinson = 1 - term ** (1 / (1 - epsilon))
            return float(atkinson)
    
    def compute_all(self, x: np.ndarray) -> Dict[str, float]:
        """Compute all fairness metrics."""
        metrics = {
            'gini': self.gini_coefficient(x),
            'maxmin': self.max_min_fairness(x),
            'theil': self.theil_index(x),
            'jain': self.jain_fairness_index(x),
            'atkinson_0.5': self.atkinson_index(x, epsilon=0.5),
            'atkinson_1.0': self.atkinson_index(x, epsilon=1.0)
        }
        return metrics
    
    def fairness_efficiency_tradeoff(self, efficiency_scores: np.ndarray, fairness_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate fairness-efficiency tradeoff."""
        result = calculate_fairness_efficiency_tradeoff(
            fairness_scores.tolist(),
            efficiency_scores.tolist()
        )
        
        # Add additional tradeoff metrics
        correlation = np.corrcoef(efficiency_scores, fairness_scores)[0, 1]
        
        return {
            'correlation': float(correlation),
            'tradeoff_index': result['tradeoff_ratio'],
            'pareto_frontier_size': len(result['pareto_frontier'])
        }
