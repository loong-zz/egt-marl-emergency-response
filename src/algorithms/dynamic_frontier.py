"""
Dynamic Pareto Frontier for EGT-MARL disaster resource allocation.

This module implements a dynamic Pareto frontier that adaptively balances:
1. Efficiency (total survivors, response time)
2. Fairness (equitable resource distribution)
3. Robustness (system stability under stress)

Key features:
- Adaptive weight adjustment based on performance feedback
- Evolutionary algorithm for frontier optimization
- Real-time frontier update during training
- Multi-objective optimization with constraints
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import copy
from scipy.spatial.distance import cdist


@dataclass
class ParetoPoint:
    """Data class representing a point on the Pareto frontier."""
    efficiency: float  # Normalized efficiency score (0-1)
    fairness: float    # Normalized fairness score (0-1)
    robustness: float  # Normalized robustness score (0-1)
    weights: np.ndarray  # Weight vector for objectives
    solution: Optional[Any] = None  # Associated solution/parameters
    dominance_count: int = 0  # Number of points this dominates
    dominated_by: int = 0     # Number of points that dominate this
    rank: int = 0             # Non-dominated sorting rank


@dataclass
class FrontierMetrics:
    """Metrics for evaluating Pareto frontier quality."""
    hypervolume: float  # Hypervolume indicator
    spread: float       # Spread/diversity of solutions
    convergence: float  # Convergence to true Pareto front
    uniformity: float   # Uniformity of distribution
    cardinality: int    # Number of non-dominated solutions


class DynamicParetoFrontier:
    """
    Dynamic Pareto frontier with adaptive weight adjustment.
    
    The frontier evolves during training to find optimal trade-offs between:
    1. Efficiency: Maximize survivors, minimize response time
    2. Fairness: Minimize inequality in resource distribution
    3. Robustness: Maximize system stability and fault tolerance
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        
        if config is None:
            config = {}
        self.config = config
        
        # Frontier parameters
        self.frontier_size = config.get('frontier_size', 50)
        self.num_objectives = config.get('num_objectives', 3)
        
        # Adaptive weight parameters
        self.weight_adaptation_rate = config.get('weight_adaptation_rate', 0.05)
        self.min_weight = config.get('min_weight', 0.1)
        self.max_weight = config.get('max_weight', 0.8)
        
        # Evolutionary algorithm parameters
        self.mutation_strength = config.get('mutation_strength', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.elitism_rate = config.get('elitism_rate', 0.1)
        self.population_size = config.get('population_size', 100)
        
        # Reference point for hypervolume calculation
        self.reference_point = np.array([0.0, 0.0, 0.0])
        
        # Initialize frontier
        self.frontier: List[ParetoPoint] = []
        self.archive: List[ParetoPoint] = []  # Archive of all evaluated points
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.weight_history: List[np.ndarray] = []
        
        # Initialize with random weights
        self._initialize_frontier()
    
    def _initialize_frontier(self):
        """Initialize frontier with random weight vectors."""
        for _ in range(self.frontier_size):
            # Generate random weights that sum to 1
            weights = np.random.dirichlet(np.ones(self.num_objectives))
            
            # Ensure weights are within bounds
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / weights.sum()
            
            point = ParetoPoint(
                efficiency=0.0,
                fairness=0.0,
                robustness=0.0,
                weights=weights
            )
            self.frontier.append(point)
    
    def update_frontier(self,
                       new_solutions: List[Dict[str, Any]],
                       performance_metrics: Dict[str, float]):
        """
        Update Pareto frontier with new solutions.
        
        Args:
            new_solutions: List of new solution evaluations
            performance_metrics: Current performance metrics
        """
        # Evaluate new solutions
        new_points = []
        for solution in new_solutions:
            point = self._evaluate_solution(solution, performance_metrics)
            new_points.append(point)
        
        # Add to archive
        self.archive.extend(new_points)
        
        # Combine with current frontier
        all_points = self.frontier + new_points
        
        # Perform non-dominated sorting
        ranked_points = self._non_dominated_sorting(all_points)
        
        # Select new frontier
        self.frontier = self._select_new_frontier(ranked_points)
        
        # Update weights based on performance
        self._adapt_weights(performance_metrics)
        
        # Track performance
        self.performance_history.append(performance_metrics.copy())
        if self.frontier:
            self.weight_history.append(self.frontier[0].weights.copy())
    
    def _evaluate_solution(self,
                          solution: Dict[str, Any],
                          metrics: Dict[str, float]) -> ParetoPoint:
        """Evaluate a solution and create Pareto point."""
        # Extract objective values from solution
        efficiency = self._calculate_efficiency_score(solution, metrics)
        fairness = self._calculate_fairness_score(solution, metrics)
        robustness = self._calculate_robustness_score(solution, metrics)
        
        # Get weights from solution or generate new ones
        if 'weights' in solution:
            weights = np.array(solution['weights'])
        else:
            # Generate weights based on performance
            weights = self._generate_weights_from_performance(metrics)
        
        # Ensure weights are valid
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()
        
        return ParetoPoint(
            efficiency=efficiency,
            fairness=fairness,
            robustness=robustness,
            weights=weights,
            solution=solution
        )
    
    def _calculate_efficiency_score(self,
                                  solution: Dict[str, Any],
                                  metrics: Dict[str, float]) -> float:
        """Calculate efficiency score (0-1)."""
        # Survivor efficiency
        survivors = metrics.get('total_survivors', 0)
        survivor_score = min(1.0, survivors / 100.0)  # Normalize to 100 survivors
        
        # Response time efficiency
        response_time = metrics.get('mean_response_time', 60.0)
        response_score = max(0.0, 1.0 - response_time / 120.0)  # Normalize to 120 minutes
        
        # Resource utilization efficiency
        utilization = metrics.get('overall_resource_utilization', 0.5)
        # Target utilization around 70%
        utilization_score = 1.0 - abs(utilization - 0.7)
        
        # Combined efficiency score
        efficiency_score = (
            0.5 * survivor_score +
            0.3 * response_score +
            0.2 * utilization_score
        )
        
        return float(np.clip(efficiency_score, 0.0, 1.0))
    
    def _calculate_fairness_score(self,
                                solution: Dict[str, Any],
                                metrics: Dict[str, float]) -> float:
        """Calculate fairness score (0-1)."""
        # Gini coefficient (0 is perfectly equal)
        gini = metrics.get('gini_coefficient', 0.5)
        gini_score = 1.0 - gini
        
        # Max-min fairness
        max_min = metrics.get('max_min_fairness', 0.5)
        
        # Coefficient of variation (lower is better)
        cv = metrics.get('coefficient_of_variation', 0.5)
        cv_score = 1.0 / (1.0 + cv)
        
        # Combined fairness score
        fairness_score = (
            0.4 * gini_score +
            0.4 * max_min +
            0.2 * cv_score
        )
        
        return float(np.clip(fairness_score, 0.0, 1.0))
    
    def _calculate_robustness_score(self,
                                  solution: Dict[str, Any],
                                  metrics: Dict[str, float]) -> float:
        """Calculate robustness score (0-1)."""
        # System stability
        stability = metrics.get('stability_index', 0.5) / 10.0  # Normalize
        
        # Fault tolerance
        fault_tolerance = metrics.get('fault_tolerance', 0.5)
        
        # Recovery capability
        recovery_time = metrics.get('recovery_time', 50.0)
        recovery_score = max(0.0, 1.0 - recovery_time / 100.0)  # Normalize to 100 time units
        
        # Performance under stress
        stress_performance = metrics.get('performance_under_stress', 0.5)
        
        # Combined robustness score
        robustness_score = (
            0.3 * stability +
            0.3 * fault_tolerance +
            0.2 * recovery_score +
            0.2 * stress_performance
        )
        
        return float(np.clip(robustness_score, 0.0, 1.0))
    
    def _generate_weights_from_performance(self,
                                         metrics: Dict[str, float]) -> np.ndarray:
        """Generate weight vector based on current performance."""
        # Analyze performance gaps
        efficiency_gap = 1.0 - metrics.get('efficiency_score', 0.5)
        fairness_gap = 1.0 - metrics.get('fairness_score', 0.5)
        robustness_gap = 1.0 - metrics.get('robustness_score', 0.5)
        
        # Higher weights for objectives with larger gaps
        raw_weights = np.array([efficiency_gap, fairness_gap, robustness_gap])
        
        # Add small epsilon to avoid zero weights
        raw_weights = raw_weights + 0.01
        
        # Normalize
        weights = raw_weights / raw_weights.sum()
        
        # Apply bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / weights.sum()
        
        return weights
    
    def _non_dominated_sorting(self, points: List[ParetoPoint]) -> List[List[ParetoPoint]]:
        """Perform non-dominated sorting (NSGA-II style)."""
        # Reset dominance information
        for point in points:
            point.dominance_count = 0
            point.dominated_by = 0
            point.rank = 0
        
        # Calculate dominance relationships
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                dominates = self._dominates(points[i], points[j])
                if dominates == 1:
                    points[i].dominance_count += 1
                    points[j].dominated_by += 1
                elif dominates == -1:
                    points[j].dominance_count += 1
                    points[i].dominated_by += 1
        
        # Sort into fronts
        fronts = []
        remaining_points = points.copy()
        
        current_rank = 1
        while remaining_points:
            # Find non-dominated points (dominated_by == 0)
            front = [p for p in remaining_points if p.dominated_by == 0]
            
            if not front:
                break
            
            # Assign rank
            for point in front:
                point.rank = current_rank
            
            fronts.append(front)
            
            # Remove front from remaining points
            for point in front:
                remaining_points.remove(point)
                
                # Update dominance counts for remaining points
                for other in remaining_points:
                    if self._dominates(point, other) == 1:
                        other.dominated_by -= 1
            
            current_rank += 1
        
        return fronts
    
    def _dominates(self, point1: ParetoPoint, point2: ParetoPoint) -> int:
        """
        Check if point1 dominates point2.
        
        Returns:
            1 if point1 dominates point2
            -1 if point2 dominates point1
            0 if neither dominates
        """
        # Extract objective values
        obj1 = np.array([point1.efficiency, point1.fairness, point1.robustness])
        obj2 = np.array([point2.efficiency, point2.fairness, point2.robustness])
        
        # Check for dominance (maximization problem)
        better_in_all = np.all(obj1 >= obj2)
        strictly_better_in_some = np.any(obj1 > obj2)
        
        if better_in_all and strictly_better_in_some:
            return 1  # point1 dominates point2
        elif np.all(obj2 >= obj1) and np.any(obj2 > obj1):
            return -1  # point2 dominates point1
        else:
            return 0  # non-dominated
    
    def _select_new_frontier(self, 
                           ranked_points: List[List[ParetoPoint]]) -> List[ParetoPoint]:
        """Select new frontier from ranked points."""
        new_frontier = []
        
        # Add points from fronts until frontier is full
        for front in ranked_points:
            if len(new_frontier) + len(front) <= self.frontier_size:
                # Add entire front
                new_frontier.extend(front)
            else:
                # Need to select subset from this front
                remaining_slots = self.frontier_size - len(new_frontier)
                selected = self._select_from_front(front, remaining_slots)
                new_frontier.extend(selected)
                break
        
        return new_frontier
    
    def _select_from_front(self, 
                          front: List[ParetoPoint],
                          num_to_select: int) -> List[ParetoPoint]:
        """Select points from a front using crowding distance."""
        if len(front) <= num_to_select:
            return front
        
        # Calculate crowding distance
        self._calculate_crowding_distance(front)
        
        # Sort by crowding distance (descending)
        front_sorted = sorted(front, key=lambda p: p.crowding_distance, reverse=True)
        
        return front_sorted[:num_to_select]
    
    def _calculate_crowding_distance(self, front: List[ParetoPoint]):
        """Calculate crowding distance for points in a front."""
        n = len(front)
        if n == 0:
            return
        
        # Initialize crowding distances
        for point in front:
            point.crowding_distance = 0.0
        
        # For each objective
        objectives = ['efficiency', 'fairness', 'robustness']
        
        for obj in objectives:
            # Sort front by this objective
            front_sorted = sorted(front, key=lambda p: getattr(p, obj))
            
            # Set infinite distance for boundary points
            front_sorted[0].crowding_distance = float('inf')
            front_sorted[-1].crowding_distance = float('inf')
            
            # Get min and max values for normalization
            min_val = getattr(front_sorted[0], obj)
            max_val = getattr(front_sorted[-1], obj)
            value_range = max_val - min_val
            
            if value_range > 0:
                # Calculate crowding distance for interior points
                for i in range(1, n - 1):
                    prev_val = getattr(front_sorted[i - 1], obj)
                    next_val = getattr(front_sorted[i + 1], obj)
                    
                    distance = (next_val - prev_val) / value_range
                    front_sorted[i].crowding_distance += distance
    
    def _adapt_weights(self, performance_metrics: Dict[str, float]):
        """Adapt weights based on performance feedback."""
        if not self.frontier:
            return
        
        # Get current best point
        best_point = self._get_best_point(performance_metrics)
        
        # Generate new weight variations
        new_weights = self._evolve_weights(best_point.weights)
        
        # Create new points with evolved weights
        new_points = []
        for weights in new_weights:
            point = ParetoPoint(
                efficiency=best_point.efficiency,
                fairness=best_point.fairness,
                robustness=best_point.robustness,
                weights=weights,
                solution=best_point.solution
            )
            new_points.append(point)
        
        # Add to frontier for next update
        self.frontier.extend(new_points[:5])  # Add top 5 variations
    
    def _get_best_point(self, 
                       performance_metrics: Dict[str, float]) -> ParetoPoint:
        """Get the best point based on current performance."""
        if not self.frontier:
            # Return default point
            weights = np.array([0.4, 0.3, 0.3])  # Balanced weights
            return ParetoPoint(
                efficiency=0.5,
                fairness=0.5,
                robustness=0.5,
                weights=weights
            )
        
        # Score each point based on weighted sum
        scores = []
        for point in self.frontier:
            score = (
                point.weights[0] * point.efficiency +
                point.weights[1] * point.fairness +
                point.weights[2] * point.robustness
            )
            scores.append(score)
        
        # Return point with highest score
        best_idx = np.argmax(scores)
        return self.frontier[best_idx]
    
    def _evolve_weights(self, base_weights: np.ndarray) -> List[np.ndarray]:
        """Evolve weight vectors using genetic operators."""
        population = []
        
        # Create population
        for _ in range(self.population_size):
            if np.random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = population[np.random.randint(len(population))]
                parent2 = population[np.random.randint(len(population))]
                child = self._crossover_weights(parent1, parent2)
            else:
                # Mutation
                child = self._mutate_weights(base_weights.copy())
            
            # Ensure valid weights
            child = np.clip(child, self.min_weight, self.max_weight)
            child = child / child.sum()
            population.append(child)
        
        # Select elite weights
        elite_size = int(self.elitism_rate * len(population))
        if elite_size > 0:
            # Sort by similarity to base weights (closer is better for exploitation)
            similarities = [1.0 / (1.0 + np.linalg.norm(w - base_weights)) for w in population]
            elite_indices = np.argsort(similarities)[-elite_size:]
            elite = [population[i] for i in elite_indices]
        else:
            elite = []
        
        return elite
    
    def _mutate_weights(self, weights: np.ndarray) -> np.ndarray:
        """Mutate weight vector."""
        # Gaussian mutation
        mutation = np.random.normal(0, self.mutation_strength, size=weights.shape)
        mutated = weights + mutation
        
        # Ensure non-negative
        mutated = np.maximum(mutated, 0.0)
        
        return mutated
    
    def _crossover_weights(self, 
                          weights1: np.ndarray, 
                          weights2: np.ndarray) -> np.ndarray:
        """Crossover two weight vectors."""
        # Uniform crossover
        mask = np.random.random(size=weights1.shape) < 0.5
        child = np.where(mask, weights1, weights2)
        
        return child
    
    def get_recommended_weights(self,
                               performance_metrics: Dict[str, float]) -> np.ndarray:
        """Get recommended weight vector for current performance."""
        if not self.frontier:
            # Default balanced weights
            return np.array([0.4, 0.3, 0.3])
        
        # Get best point
        best_point = self._get_best_point(performance_metrics)
        
        return best_point.weights.copy()
    
    def get_frontier_metrics(self) -> FrontierMetrics:
        """Calculate metrics for current frontier."""
        if len(self.frontier) < 2:
            return FrontierMetrics(
                hypervolume=0.0,
                spread=0.0,
                convergence=0.0,
                uniformity=0.0,
                cardinality=len(self.frontier)
            )
        
        # Extract objective values
        objectives = np.array([[p.efficiency, p.fairness, p.robustness] 
                              for p in self.frontier])
        
        # Hypervolume calculation (simplified)
        hypervolume = self._calculate_hypervolume(objectives)
        
        # Spread/diversity
        spread = self._calculate_spread(objectives)
        
        # Convergence (distance to ideal point)
        convergence = self._calculate_convergence(objectives)
        
        # Uniformity of distribution
        uniformity = self._calculate_uniformity(objectives)
        
        return FrontierMetrics(
            hypervolume=hypervolume,
            spread=spread,
            convergence=convergence,
            uniformity=uniformity,
            cardinality=len(self.frontier)
        )
    
    def _calculate_hypervolume(self, objectives: np.ndarray) -> float:
        """Calculate hypervolume indicator (simplified)."""
        # For 3 objectives, approximate hypervolume
        # Normalize objectives to [0, 1]
        obj_norm = objectives.copy()
        for i in range(3):
            min_val = obj_norm[:, i].min()
            max_val = obj_norm[:, i].max()
            if max_val > min_val:
                obj_norm[:, i] = (obj_norm[:, i] - min_val) / (max_val - min_val)
        
        # Calculate volume dominated by each point
        volumes = []
        for point in obj_norm:
            # Volume of hyper-rectangle from origin to point
            volume = np.prod(point)
            volumes.append(volume)
        
        # Take maximum volume as approximation
        hypervolume = max(volumes) if volumes else 0.0
        
        return float(hypervolume)
    
    def _calculate_spread(self, objectives: np.ndarray) -> float:
        """Calculate spread/diversity metric."""
        if len(objectives) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = cdist(objectives, objectives, metric='euclidean')
        
        # Get upper triangular (excluding diagonal)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        
        if len(upper_tri) == 0:
            return 0.0
        
        # Spread is standard deviation of distances
        spread = np.std(upper_tri)
        
        return float(spread)
    
    def _calculate_convergence(self, objectives: np.ndarray) -> float:
        """Calculate convergence to ideal point."""
        # Ideal point (maximize all objectives)
        ideal_point = np.array([1.0, 1.0, 1.0])
        
        # Calculate distances to ideal point
        distances = np.linalg.norm(objectives - ideal_point, axis=1)
        
        # Convergence is inverse of average distance
        avg_distance = np.mean(distances)
        convergence = 1.0 / (1.0 + avg_distance)
        
        return float(convergence)
    
    def _calculate_uniformity(self, objectives: np.ndarray) -> float:
        """Calculate uniformity of distribution."""
        if len(objectives) < 3:
            return 0.0
        
        # Calculate nearest neighbor distances
        from scipy.spatial import KDTree
        tree = KDTree(objectives)
        distances, _ = tree.query(objectives, k=2)  # k=2 to exclude self
        
        # Get distances to nearest neighbor (second column)
        nn_distances = distances[:, 1]
        
        # Uniformity is inverse of coefficient of variation
        if np.mean(nn_distances) > 0:
            uniformity = 1.0 / (1.0 + np.std(nn_distances) / np.mean(nn_distances))
        else:
            uniformity = 0.0
        
        return float(uniformity)
    
    def get_frontier_points(self) -> List[Dict[str, Any]]:
        """Get frontier points for visualization."""
        points = []
        for point in self.frontier:
            points.append({
                'efficiency': point.efficiency,
                'fairness': point.fairness,
                'robustness': point.robustness,
                'weights': point.weights.tolist(),
                'rank': point.rank
            })
        
        return points
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get performance history for analysis."""
        if not self.performance_history:
            return {}
        
        # Extract metrics over time
        history = {
            'efficiency': [],
            'fairness': [],
            'robustness': [],
            'hypervolume': [],
            'spread': []
        }
        
        for metrics in self.performance_history:
            history['efficiency'].append(metrics.get('efficiency_score', 0.0))
            history['fairness'].append(metrics.get('fairness_score', 0.0))
            history['robustness'].append(metrics.get('robustness_score', 0.0))
            
            # Calculate frontier metrics for each step
            frontier_metrics = self.get_frontier_metrics()
            history['hypervolume'].append(frontier_metrics.hypervolume)
            history['spread'].append(frontier_metrics.spread)
        
        return history
    
    def save(self, path: str):
        """Save frontier state."""
        import pickle
        
        state = {
            'frontier': self.frontier,
            'archive': self.archive,
            'performance_history': self.performance_history,
            'weight_history': self.weight_history,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load frontier state."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.frontier = state['frontier']
        self.archive = state['archive']
        self.performance_history = state['performance_history']
        self.weight_history = state['weight_history']
        self.config = state['config']


class AdaptiveWeightController:
    """
    Adaptive weight controller for dynamic fairness-efficiency trade-off.
    
    This component adjusts the weights between objectives in real-time
    based on system performance and user preferences.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.config = config
        
        # Current weights
        self.weights = np.array([0.4, 0.3, 0.3])  # efficiency, fairness, robustness
        
        # Target performance levels
        self.targets = {
            'efficiency': config.get('target_efficiency', 0.7),
            'fairness': config.get('target_fairness', 0.6),
            'robustness': config.get('target_robustness', 0.5)
        }
        
        # Adaptation parameters
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.smoothing_factor = config.get('smoothing_factor', 0.9)
        
        # Performance history
        self.performance_buffer = []
        self.buffer_size = config.get('buffer_size', 10)
    
    def update_weights(self, 
                      current_performance: Dict[str, float],
                      user_preferences: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Update weights based on current performance and preferences."""
        # Update performance buffer
        self.performance_buffer.append(current_performance.copy())
        if len(self.performance_buffer) > self.buffer_size:
            self.performance_buffer.pop(0)
        
        # Calculate performance gaps
        gaps = {}
        for objective in ['efficiency', 'fairness', 'robustness']:
            current = current_performance.get(f'{objective}_score', 0.5)
            target = self.targets[objective]
            gap = target - current
            gaps[objective] = gap
        
        # Adjust weights based on gaps
        adjustments = np.zeros(3)
        
        # Higher weight for objectives with larger negative gaps (underperforming)
        for i, objective in enumerate(['efficiency', 'fairness', 'robustness']):
            gap = gaps[objective]
            if gap > 0:  # Underperforming
                adjustments[i] = self.adaptation_rate * gap
            else:  # Overperforming
                adjustments[i] = -self.adaptation_rate * abs(gap) * 0.5
        
        # Apply user preferences if provided
        if user_preferences:
            for i, objective in enumerate(['efficiency', 'fairness', 'robustness']):
                if objective in user_preferences:
                    preference = user_preferences[objective]
                    # Blend current weight with preference
                    adjustments[i] += self.adaptation_rate * (preference - self.weights[i])
        
        # Apply adjustments
        new_weights = self.weights + adjustments
        
        # Ensure valid weights
        new_weights = np.maximum(new_weights, 0.0)
        new_weights = new_weights / new_weights.sum()
        
        # Smooth update
        self.weights = (self.smoothing_factor * self.weights + 
                       (1 - self.smoothing_factor) * new_weights)
        
        # Normalize
        self.weights = self.weights / self.weights.sum()
        
        return self.weights.copy()
    
    def get_weights(self) -> np.ndarray:
        """Get current weights."""
        return self.weights.copy()
    
    def set_targets(self, targets: Dict[str, float]):
        """Set target performance levels."""
        for objective in ['efficiency', 'fairness', 'robustness']:
            if objective in targets:
                self.targets[objective] = targets[objective]
    
    def reset(self):
        """Reset to default weights."""
        self.weights = np.array([0.4, 0.3, 0.3])
        self.performance_buffer = []


# Integration with EGT-MARL
def integrate_frontier_with_egt_marl(frontier: DynamicParetoFrontier,
                                    egt_marl_system,
                                    performance_metrics: Dict[str, float]):
    """
    Integrate dynamic Pareto frontier with EGT-MARL system.
    
    This function:
    1. Gets recommended weights from frontier
    2. Updates EGT-MARL reward weights
    3. Adjusts agent strategies based on frontier
    """
    # Get recommended weights
    recommended_weights = frontier.get_recommended_weights(performance_metrics)
    
    # Update EGT-MARL reward weights
    if hasattr(egt_marl_system, 'reward_structure'):
        egt_marl_system.reward_structure.weights = {
            'efficiency': recommended_weights[0],
            'fairness': recommended_weights[1],
            'robustness': recommended_weights[2]
        }
    
    # Update agent exploration based on frontier diversity
    frontier_metrics = frontier.get_frontier_metrics()
    if frontier_metrics.spread < 0.1:  # Low diversity
        # Increase exploration
        for agent in getattr(egt_marl_system, 'agents', []):
            if hasattr(agent, 'epsilon'):
                agent.epsilon = min(1.0, agent.epsilon * 1.1)
    
    return recommended_weights


if __name__ == "__main__":
    # Example usage
    config = {
        'frontier_size': 50,
        'num_objectives': 3,
        'weight_adaptation_rate': 0.05,
        'mutation_strength': 0.1,
        'crossover_rate': 0.7
    }
    
    # Create dynamic Pareto frontier
    frontier = DynamicParetoFrontier(config)
    print(f"Created dynamic Pareto frontier with {frontier.frontier_size} points")
    
    # Create adaptive weight controller
    weight_controller = AdaptiveWeightController()
    
    # Example performance metrics
    performance = {
        'efficiency_score': 0.65,
        'fairness_score': 0.55,
        'robustness_score': 0.60,
        'total_survivors': 75,
        'mean_response_time': 45.0,
        'gini_coefficient': 0.35
    }
    
    # Update frontier
    frontier.update_frontier([], performance)
    
    # Get recommended weights
    weights = frontier.get_recommended_weights(performance)
    print(f"Recommended weights: Efficiency={weights[0]:.3f}, "
          f"Fairness={weights[1]:.3f}, Robustness={weights[2]:.3f}")
    
    # Get frontier metrics
    metrics = frontier.get_frontier_metrics()
    print(f"Frontier metrics: Hypervolume={metrics.hypervolume:.3f}, "
          f"Spread={metrics.spread:.3f}, Cardinality={metrics.cardinality}")