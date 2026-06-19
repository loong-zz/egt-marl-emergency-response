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
from collections import deque
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
    
    Enhanced with:
    - Phase-aware weight scheduling (exploration vs exploitation)
    - Context-aware adaptation (disaster severity/type)
    - Performance trend tracking (improvement rates)
    - Entropy-based diversity preservation
    - Adaptive mutation rate based on population diversity
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
        self.base_mutation_strength = self.mutation_strength
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.elitism_rate = config.get('elitism_rate', 0.1)
        self.population_size = config.get('population_size', 100)
        
        # Phase-aware scheduling
        self.training_phase = 'exploration'  # exploration, exploitation, refinement
        self.phase_config = {
            'exploration': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2, 'mutation_mult': 1.5},
            'exploitation': {'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3, 'mutation_mult': 1.0},
            'refinement': {'alpha': 0.2, 'beta': 0.3, 'gamma': 0.5, 'mutation_mult': 0.5},
        }
        
        # Context-aware parameters
        self.disaster_type = config.get('disaster_type', 'earthquake')
        self.disaster_severity = config.get('severity', 'medium')
        self.context_weights = self._compute_context_weights()
        
        # Performance trend tracking
        self.trend_window = config.get('trend_window', 10)
        self.objective_trends = {
            'efficiency': deque(maxlen=self.trend_window),
            'fairness': deque(maxlen=self.trend_window),
            'robustness': deque(maxlen=self.trend_window),
        }
        self.improvement_rates = {'efficiency': 0.0, 'fairness': 0.0, 'robustness': 0.0}
        
        # Entropy-based diversity
        self.diversity_threshold = config.get('diversity_threshold', 0.15)
        self.entropy_reg_weight = config.get('entropy_reg_weight', 0.01)
        
        # Reference point for hypervolume calculation
        self.reference_point = np.array([0.0, 0.0, 0.0])
        
        # Initialize frontier
        self.frontier: List[ParetoPoint] = []
        self.archive: List[ParetoPoint] = []  # Archive of all evaluated points
        self.best_frontier: List[ParetoPoint] = []  # Historical best frontier
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.weight_history: List[np.ndarray] = []
        self.phase_transitions: List[Dict[str, Any]] = []
        
        # Initialize with random weights
        self._initialize_frontier()
    
    def _compute_context_weights(self) -> np.ndarray:
        """
        Compute context-aware base weights based on disaster type and severity.
        
        Different disasters have different priorities:
        - Earthquake: rescue speed critical (efficiency heavy)
        - Flood: resource distribution critical (fairness heavy)
        - Fire: system robustness critical (robustness heavy)
        - High severity: efficiency priority
        - Low severity: robustness priority
        """
        # Base weights by disaster type
        type_weights = {
            'earthquake': np.array([0.45, 0.30, 0.25]),
            'flood': np.array([0.30, 0.45, 0.25]),
            'fire': np.array([0.30, 0.25, 0.45]),
            'hurricane': np.array([0.35, 0.35, 0.30]),
            'tsunami': np.array([0.40, 0.30, 0.30]),
        }
        
        # Severity adjustments
        severity_mult = {
            'low': np.array([1.0, 0.9, 1.1]),
            'medium': np.array([1.0, 1.0, 1.0]),
            'high': np.array([1.2, 0.9, 0.9]),
            'extreme': np.array([1.3, 0.8, 0.9]),
        }
        
        base = type_weights.get(self.disaster_type, np.array([0.4, 0.3, 0.3]))
        mult = severity_mult.get(self.disaster_severity, np.array([1.0, 1.0, 1.0]))
        
        weights = base * mult
        weights = weights / weights.sum()
        
        return weights
    
    def update_training_phase(self, episode: int, total_episodes: int):
        """
        Update training phase based on progress.
        
        Phase schedule:
        - 0-40%: exploration (focus on discovering diverse strategies)
        - 40-80%: exploitation (focus on efficiency)
        - 80-100%: refinement (focus on robustness and stability)
        """
        progress = episode / total_episodes if total_episodes > 0 else 0
        
        if progress < 0.4:
            new_phase = 'exploration'
        elif progress < 0.8:
            new_phase = 'exploitation'
        else:
            new_phase = 'refinement'
        
        if new_phase != self.training_phase:
            self.phase_transitions.append({
                'episode': episode,
                'from_phase': self.training_phase,
                'to_phase': new_phase,
                'progress': progress,
            })
            self.training_phase = new_phase
            self._on_phase_change()
    
    def _on_phase_change(self):
        """Handle phase transition."""
        phase_cfg = self.phase_config[self.training_phase]
        
        # Adjust mutation based on phase
        self.mutation_strength = self.base_mutation_strength * phase_cfg['mutation_mult']
        
        # Adjust elitism based on phase
        if self.training_phase == 'exploration':
            self.elitism_rate = 0.05  # Less elitism for more exploration
        elif self.training_phase == 'exploitation':
            self.elitism_rate = 0.15  # More elitism for convergence
        else:
            self.elitism_rate = 0.10  # Balanced
    
    def _update_trends(self, performance_metrics: Dict[str, float]):
        """Update performance trend tracking."""
        for obj in ['efficiency', 'fairness', 'robustness']:
            score = performance_metrics.get(f'{obj}_score', 0.5)
            self.objective_trends[obj].append(score)
            
            # Calculate improvement rate (slope of recent trend)
            if len(self.objective_trends[obj]) >= 3:
                values = list(self.objective_trends[obj])
                x = np.arange(len(values))
                if len(x) > 1 and np.std(x) > 0:
                    slope = np.polyfit(x, values, 1)[0]
                    self.improvement_rates[obj] = slope
    
    def _compute_diversity(self) -> float:
        """Compute population diversity via entropy of weight distribution."""
        if len(self.frontier) < 2:
            return 0.0
        
        all_weights = np.array([p.weights for p in self.frontier])
        # Average pairwise distance
        distances = cdist(all_weights, all_weights)
        avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
        return float(avg_distance)
    
    def _adapt_mutation_rate(self):
        """Adapt mutation rate based on population diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold:
            # Low diversity - increase mutation
            self.mutation_strength = min(
                self.base_mutation_strength * 2.0,
                self.mutation_strength * 1.2
            )
        elif diversity > self.diversity_threshold * 3:
            # High diversity - decrease mutation
            self.mutation_strength = max(
                self.base_mutation_strength * 0.5,
                self.mutation_strength * 0.9
            )
    
    def _apply_entropy_regularization(self):
        """Apply entropy regularization to maintain weight diversity."""
        if len(self.frontier) < 2:
            return
        
        all_weights = np.array([p.weights for p in self.frontier])
        # Compute entropy-like measure
        mean_weights = np.mean(all_weights, axis=0)
        entropy = -np.sum(mean_weights * np.log(mean_weights + 1e-8))
        
        # If entropy too low, nudge weights apart
        if entropy < 0.8:  # Low entropy threshold
            for point in self.frontier:
                # Add small noise proportional to regularization weight
                noise = np.random.normal(0, self.entropy_reg_weight, size=point.weights.shape)
                point.weights = point.weights + noise
                point.weights = np.clip(point.weights, self.min_weight, self.max_weight)
                point.weights = point.weights / point.weights.sum()
    
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
        # Update performance trends
        self._update_trends(performance_metrics)
        
        # Adapt mutation rate based on diversity
        self._adapt_mutation_rate()
        
        # Apply entropy regularization
        self._apply_entropy_regularization()
        
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
        new_frontier = self._select_new_frontier(ranked_points)
        
        # Update historical best frontier
        if len(new_frontier) > 0:
            # Check if new frontier has better hypervolume
            if len(self.best_frontier) == 0:
                self.best_frontier = [copy.deepcopy(p) for p in new_frontier]
            else:
                old_metrics = self._calculate_frontier_metrics(self.best_frontier)
                old_hv = old_metrics.hypervolume if old_metrics else 0.0
                new_metrics = self._calculate_frontier_metrics(new_frontier)
                new_hv = new_metrics.hypervolume if new_metrics else 0.0
                if new_hv > old_hv:
                    self.best_frontier = [copy.deepcopy(p) for p in new_frontier]
        
        self.frontier = new_frontier
        
        # Update weights based on performance
        self._adapt_weights(performance_metrics)
        
        # Track performance
        self.performance_history.append(performance_metrics.copy())
        if self.frontier:
            self.weight_history.append(self.frontier[0].weights.copy())
    
    def _calculate_frontier_metrics(self, frontier: List[ParetoPoint]) -> Optional[FrontierMetrics]:
        """Calculate metrics for a given frontier."""
        if len(frontier) < 2:
            return None
        
        objectives = np.array([[p.efficiency, p.fairness, p.robustness] for p in frontier])
        hv = self._calculate_hypervolume(objectives)
        spread = self._calculate_spread(objectives)
        conv = self._calculate_convergence(objectives)
        unif = self._calculate_uniformity(objectives)
        
        return FrontierMetrics(
            hypervolume=hv,
            spread=spread,
            convergence=conv,
            uniformity=unif,
            cardinality=len(frontier)
        )
    
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
        """Adapt weights based on performance feedback, phase, and context."""
        if not self.frontier:
            return
        
        # Get current best point
        best_point = self._get_best_point(performance_metrics)
        
        # Blend with context weights
        phase_cfg = self.phase_config[self.training_phase]
        blended = (
            phase_cfg['alpha'] * best_point.weights +
            phase_cfg['beta'] * self.context_weights +
            phase_cfg['gamma'] * self._generate_weights_from_performance(performance_metrics)
        )
        blended = np.clip(blended, self.min_weight, self.max_weight)
        blended = blended / blended.sum()
        
        # Generate new weight variations
        new_weights = self._evolve_weights(blended)
        
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
        
        # Blend with phase-aware context weights
        phase_cfg = self.phase_config[self.training_phase]
        blended = (
            phase_cfg['alpha'] * best_point.weights +
            phase_cfg['beta'] * self.context_weights
        )
        blended = np.clip(blended, self.min_weight, self.max_weight)
        blended = blended / blended.sum()
        
        return blended.copy()
    
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
            'best_frontier': self.best_frontier,
            'performance_history': self.performance_history,
            'weight_history': self.weight_history,
            'phase_transitions': self.phase_transitions,
            'training_phase': self.training_phase,
            'objective_trends': {k: list(v) for k, v in self.objective_trends.items()},
            'improvement_rates': self.improvement_rates,
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
        self.best_frontier = state.get('best_frontier', [])
        self.performance_history = state['performance_history']
        self.weight_history = state['weight_history']
        self.phase_transitions = state.get('phase_transitions', [])
        self.training_phase = state.get('training_phase', 'exploration')
        self.config = state['config']
        
        # Restore trend tracking
        if 'objective_trends' in state:
            for k, v in state['objective_trends'].items():
                self.objective_trends[k] = deque(v, maxlen=self.trend_window)
        if 'improvement_rates' in state:
            self.improvement_rates = state['improvement_rates']


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


    def update(self, batch: Dict[str, Any]) -> float:
        """
        Update dynamic frontier and compute frontier loss for integration with EGT-MARL.
        
        Args:
            batch: Experience batch containing performance metrics
            
        Returns:
            frontier_loss: Loss value to be fed back to EGT-MARL
        """
        try:
            # Extract performance metrics from batch
            performance_metrics = {}
            
            # Calculate efficiency score from rewards
            if 'rewards' in batch:
                rewards = batch['rewards']
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.detach().cpu().numpy()
                performance_metrics['efficiency_score'] = float(np.mean(rewards))
            
            # Calculate fairness score from Gini coefficient
            if 'gini_coefficient' in batch:
                gini = batch['gini_coefficient']
                if isinstance(gini, torch.Tensor):
                    gini = gini.detach().cpu().numpy()
                performance_metrics['fairness_score'] = float(1.0 - gini)
            
            # Default values
            if 'efficiency_score' not in performance_metrics:
                performance_metrics['efficiency_score'] = 0.5
            if 'fairness_score' not in performance_metrics:
                performance_metrics['fairness_score'] = 0.5
            if 'robustness_score' not in performance_metrics:
                performance_metrics['robustness_score'] = 0.5
            
            # Update frontier with empty solutions list (will use performance metrics)
            self.update_frontier([], performance_metrics)
            
            # Calculate frontier loss
            frontier_loss = self._calculate_frontier_loss(performance_metrics)
            
            return float(frontier_loss)
        
        except Exception as e:
            # Return 0.0 if update fails
            return 0.0
    
    def _calculate_frontier_loss(self, performance_metrics: Dict[str, float]) -> float:
        """
        Calculate frontier loss based on distance to Pareto optimal front.
        
        The loss encourages the system to move towards the Pareto frontier.
        """
        if not self.frontier:
            return 0.0
        
        # Get current performance as objective vector
        current_objectives = np.array([
            performance_metrics.get('efficiency_score', 0.5),
            performance_metrics.get('fairness_score', 0.5),
            performance_metrics.get('robustness_score', 0.5)
        ])
        
        # Find closest frontier point
        min_distance = float('inf')
        for point in self.frontier:
            frontier_objectives = np.array([point.efficiency, point.fairness, point.robustness])
            distance = np.linalg.norm(current_objectives - frontier_objectives)
            if distance < min_distance:
                min_distance = distance
        
        # Calculate loss as distance to frontier
        # Also include penalty for dominated solutions
        dominated_penalty = 0.0
        for point in self.frontier:
            frontier_objectives = np.array([point.efficiency, point.fairness, point.robustness])
            # Check if current is dominated by frontier point
            if np.all(frontier_objectives >= current_objectives) and np.any(frontier_objectives > current_objectives):
                dominated_penalty += np.sum(frontier_objectives - current_objectives)
        
        # Total loss = distance + dominated penalty
        total_loss = min_distance + dominated_penalty
        
        return float(total_loss)
    
    def get_frontier_weights(self) -> np.ndarray:
        """Get current frontier weights for integration with EGT-MARL."""
        # Get recommended weights based on current state
        dummy_metrics = {
            'efficiency_score': 0.5,
            'fairness_score': 0.5,
            'robustness_score': 0.5
        }
        return self.get_recommended_weights(dummy_metrics)


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


# Alias for backward compatibility
DynamicFrontier = DynamicParetoFrontier