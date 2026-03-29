"""
EGT Layer: Evolutionary Game Theory Layer
==========================================

Dynamic fairness-efficiency trade-off regulation using evolutionary game theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import copy


class EGTLayer(nn.Module):
    """
    Evolutionary Game Theory layer for dynamic fairness-efficiency trade-off.
    
    Implements:
    1. Replicator dynamics for strategy evolution
    2. Dynamic payoff matrix adaptation
    3. Fairness-efficiency trade-off optimization
    4. Convergence monitoring
    """
    
    def __init__(self, num_strategies: int = 4, payoff_matrix: Optional[torch.Tensor] = None,
                 learning_rate: float = 0.01, device: torch.device = torch.device("cpu")):
        super().__init__()
        
        self.num_strategies = num_strategies
        self.device = device
        
        # Payoff matrix (strategies x strategies)
        if payoff_matrix is None:
            # Initialize with random payoffs
            self.payoff_matrix = nn.Parameter(
                torch.randn(num_strategies, num_strategies, device=device)
            )
            # Make symmetric
            with torch.no_grad():
                self.payoff_matrix.data = (self.payoff_matrix.data + self.payoff_matrix.data.T) / 2
        else:
            self.payoff_matrix = nn.Parameter(payoff_matrix.to(device))
        
        # Strategy distribution (population shares)
        self.strategy_distribution = nn.Parameter(
            torch.ones(num_strategies, device=device) / num_strategies
        )
        
        # Historical strategy distributions for convergence analysis
        self.strategy_history = []
        self.max_history_length = 100
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.mutation_rate = 0.01
        self.selection_strength = 1.0
        
        # Strategy definitions
        self.strategy_names = [
            "Fairness-focused",
            "Efficiency-focused", 
            "Balanced",
            "Adaptive"
        ]
        
        # Convergence tracking
        self.convergence_threshold = 1e-4
        self.convergence_steps = 0
        self.is_converged = False
        
        # Performance metrics
        self.fitness_history = []
        self.diversity_history = []
    
    def get_strategy_distribution(self) -> torch.Tensor:
        """Get current strategy distribution."""
        return F.softmax(self.strategy_distribution, dim=0)
    
    def get_payoff_matrix(self) -> torch.Tensor:
        """Get current payoff matrix."""
        return self.payoff_matrix
    
    def calculate_fitness(self, strategy_idx: int, distribution: torch.Tensor) -> float:
        """
        Calculate fitness of a strategy given current population distribution.
        
        Args:
            strategy_idx: Index of strategy
            distribution: Current strategy distribution
            
        Returns:
            Fitness value
        """
        # Expected payoff against current population
        payoffs = self.payoff_matrix[strategy_idx]
        fitness = torch.sum(payoffs * distribution)
        
        return fitness.item()
    
    def replicator_dynamics_step(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Perform one step of replicator dynamics.
        
        Args:
            distribution: Current strategy distribution
            
        Returns:
            Updated distribution
        """
        # Calculate fitness for each strategy
        fitnesses = torch.zeros(self.num_strategies, device=self.device)
        for i in range(self.num_strategies):
            fitnesses[i] = self.calculate_fitness(i, distribution)
        
        # Average fitness
        avg_fitness = torch.sum(fitnesses * distribution)
        
        # Replicator dynamics equation: dx_i/dt = x_i * (f_i - f_avg)
        if avg_fitness > 0:
            growth_rates = (fitnesses - avg_fitness) / avg_fitness
        else:
            growth_rates = fitnesses - avg_fitness
        
        # Update distribution
        new_distribution = distribution * (1 + self.learning_rate * growth_rates)
        
        # Add mutation
        mutation = torch.ones_like(new_distribution) * self.mutation_rate / self.num_strategies
        new_distribution = (1 - self.mutation_rate) * new_distribution + mutation
        
        # Ensure non-negative and normalize
        new_distribution = torch.clamp(new_distribution, min=1e-8)
        new_distribution = new_distribution / torch.sum(new_distribution)
        
        return new_distribution
    
    def update_payoff_matrix(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update payoff matrix based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        with torch.no_grad():
            # Extract relevant metrics
            fairness_score = performance_metrics.get('fairness_score', 0.5)
            efficiency_score = performance_metrics.get('efficiency_score', 0.5)
            total_reward = performance_metrics.get('total_reward', 0.0)
            
            # Update payoffs based on strategy performance
            for i in range(self.num_strategies):
                for j in range(self.num_strategies):
                    # Base payoff
                    current_payoff = self.payoff_matrix[i, j].item()
                    
                    # Strategy-specific adjustments
                    if i == 0:  # Fairness-focused
                        adjustment = fairness_score - 0.5
                    elif i == 1:  # Efficiency-focused
                        adjustment = efficiency_score - 0.5
                    elif i == 2:  # Balanced
                        adjustment = (fairness_score + efficiency_score) / 2 - 0.5
                    else:  # Adaptive (i == 3)
                        adjustment = total_reward * 0.1
                    
                    # Update payoff
                    new_payoff = current_payoff + self.learning_rate * adjustment
                    self.payoff_matrix[i, j] = new_payoff
            
            # Maintain symmetry
            symmetric_matrix = (self.payoff_matrix + self.payoff_matrix.T) / 2
            self.payoff_matrix.data.copy_(symmetric_matrix)
    
    def evolve_strategies(self, performance_metrics: Dict[str, float], 
                         num_steps: int = 10) -> torch.Tensor:
        """
        Evolve strategy distribution based on performance.
        
        Args:
            performance_metrics: Performance metrics from environment
            num_steps: Number of evolution steps
            
        Returns:
            Updated strategy distribution
        """
        # Update payoff matrix based on performance
        self.update_payoff_matrix(performance_metrics)
        
        # Get current distribution
        current_distribution = self.get_strategy_distribution()
        
        # Perform multiple evolution steps
        for step in range(num_steps):
            current_distribution = self.replicator_dynamics_step(current_distribution)
        
        # Update strategy distribution
        with torch.no_grad():
            self.strategy_distribution.data.copy_(current_distribution)
        
        # Record history
        self.strategy_history.append(current_distribution.detach().cpu().numpy())
        if len(self.strategy_history) > self.max_history_length:
            self.strategy_history.pop(0)
        
        # Update convergence tracking
        self._update_convergence(current_distribution)
        
        # Update performance metrics
        self._update_performance_metrics(current_distribution, performance_metrics)
        
        return current_distribution
    
    def _update_convergence(self, distribution: torch.Tensor) -> None:
        """Update convergence tracking."""
        if len(self.strategy_history) < 2:
            return
        
        # Calculate change from previous distribution
        prev_dist = torch.tensor(self.strategy_history[-2], device=self.device)
        current_dist = distribution
        
        change = torch.norm(current_dist - prev_dist).item()
        
        if change < self.convergence_threshold:
            self.convergence_steps += 1
            if self.convergence_steps >= 10:
                self.is_converged = True
        else:
            self.convergence_steps = 0
            self.is_converged = False
    
    def _update_performance_metrics(self, distribution: torch.Tensor,
                                  performance_metrics: Dict[str, float]) -> None:
        """Update performance metrics history."""
        # Calculate fitness statistics
        fitnesses = []
        for i in range(self.num_strategies):
            fitness = self.calculate_fitness(i, distribution)
            fitnesses.append(fitness)
        
        avg_fitness = np.mean(fitnesses)
        self.fitness_history.append(avg_fitness)
        
        # Calculate diversity (entropy)
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-8)).item()
        self.diversity_history.append(entropy)
        
        # Trim history
        if len(self.fitness_history) > self.max_history_length:
            self.fitness_history.pop(0)
            self.diversity_history.pop(0)
    
    def get_fairness_efficiency_weights(self) -> Tuple[float, float]:
        """
        Get current fairness and efficiency weights from strategy distribution.
        
        Returns:
            Tuple of (fairness_weight, efficiency_weight)
        """
        distribution = self.get_strategy_distribution()
        
        # Fairness weight from fairness-focused strategy
        fairness_weight = distribution[0].item()
        
        # Efficiency weight from efficiency-focused strategy
        efficiency_weight = distribution[1].item()
        
        # Add contributions from balanced and adaptive strategies
        balanced_weight = distribution[2].item()
        adaptive_weight = distribution[3].item()
        
        # Balanced strategy contributes equally
        fairness_weight += balanced_weight * 0.5
        efficiency_weight += balanced_weight * 0.5
        
        # Adaptive strategy adjusts based on performance
        if len(self.fitness_history) > 0:
            recent_fitness = np.mean(self.fitness_history[-5:]) if len(self.fitness_history) >= 5 else 0.5
            # If performance is good, maintain current balance; if poor, shift toward efficiency
            if recent_fitness < 0.5:
                efficiency_weight += adaptive_weight * 0.7
                fairness_weight += adaptive_weight * 0.3
            else:
                efficiency_weight += adaptive_weight * 0.5
                fairness_weight += adaptive_weight * 0.5
        
        # Normalize
        total = fairness_weight + efficiency_weight
        if total > 0:
            fairness_weight /= total
            efficiency_weight /= total
        
        return fairness_weight, efficiency_weight
    
    def get_strategy_recommendation(self) -> Dict[str, Any]:
        """
        Get strategy recommendation based on current state.
        
        Returns:
            Dictionary with strategy recommendation and analysis
        """
        distribution = self.get_strategy_distribution()
        fairness_weight, efficiency_weight = self.get_fairness_efficiency_weights()
        
        # Determine dominant strategy
        dominant_idx = torch.argmax(distribution).item()
        dominant_strategy = self.strategy_names[dominant_idx]
        
        # Convergence status
        convergence_status = "Converged" if self.is_converged else f"Evolving ({self.convergence_steps}/10)"
        
        # Recommendation
        if fairness_weight > 0.7:
            recommendation = "Prioritize fairness: Ensure equitable resource distribution across all affected areas."
        elif efficiency_weight > 0.7:
            recommendation = "Prioritize efficiency: Focus resources on areas with highest survival probability gains."
        else:
            recommendation = "Balanced approach: Maintain trade-off between fairness and efficiency based on real-time conditions."
        
        return {
            'dominant_strategy': dominant_strategy,
            'strategy_distribution': distribution.detach().cpu().numpy().tolist(),
            'fairness_weight': fairness_weight,
            'efficiency_weight': efficiency_weight,
            'convergence_status': convergence_status,
            'recommendation': recommendation,
            'avg_fitness': np.mean(self.fitness_history[-5:]) if self.fitness_history else 0.0,
            'diversity': self.diversity_history[-1] if self.diversity_history else 0.0
        }
    
    def update(self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer,
              loss_fn: nn.Module) -> float:
        """
        Update EGT layer parameters.
        
        Args:
            batch: Experience batch
            optimizer: Optimizer
            loss_fn: Loss function
            
        Returns:
            Loss value
        """
        # Extract performance metrics from batch
        # In practice, would extract from environment info
        performance_metrics = {
            'fairness_score': 0.5,  # Placeholder
            'efficiency_score': 0.5,  # Placeholder
            'total_reward': batch.get('rewards', torch.tensor([0.0])).mean().item()
        }
        
        # Evolve strategies
        self.evolve_strategies(performance_metrics)
        
        # Calculate loss (encourage diversity and performance)
        distribution = self.get_strategy_distribution()
        
        # Diversity loss (encourage exploration)
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-8))
        diversity_loss = -entropy  # Maximize entropy
        
        # Performance loss (based on fitness)
        avg_fitness = np.mean(self.fitness_history[-5:]) if self.fitness_history else 0.5
        performance_loss = -avg_fitness  # Maximize fitness
        
        # Combined loss
        loss = 0.3 * diversity_loss + 0.7 * performance_loss
        
        # Optimize (though EGT typically doesn't use gradient descent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def reset_convergence(self) -> None:
        """Reset convergence tracking."""
        self.convergence_steps = 0
        self.is_converged = False
    
    def save(self, path: str) -> None:
        """Save EGT layer state."""
        torch.save({
            'payoff_matrix_state': self.payoff_matrix.state_dict(),
            'strategy_distribution_state': self.strategy_distribution.state_dict(),
            'strategy_history': self.strategy_history,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'convergence_steps': self.convergence_steps,
            'is_converged': self.is_converged,
            'config': {
                'num_strategies': self.num_strategies,
                'learning_rate': self.learning_rate,
                'mutation_rate': self.mutation_rate
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load EGT layer state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.payoff_matrix.load_state_dict(checkpoint['payoff_matrix_state'])
        self.strategy_distribution.load_state_dict(checkpoint['strategy_distribution_state'])
        
        self.strategy_history = checkpoint['strategy_history']
        self.fitness_history = checkpoint['fitness_history']
        self.diversity_history = checkpoint['diversity_history']
        
        self.convergence_steps = checkpoint['convergence_steps']
        self.is_converged = checkpoint['is_converged']