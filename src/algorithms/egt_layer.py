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
        """
        Args:
            num_strategies: 策略数量。**生产配置应显式传入 3** (与
                environments.config.constants.NUM_STRATEGIES 一致)。
                这里的默认值 4 是历史遗留,仅用于单元测试。
                推荐从 yaml 配置的 egt.num_strategies 字段读取。
        """
        super().__init__()

        self.num_strategies = num_strategies
        self.device = device
        
        # Payoff matrix (strategies x strategies) - theory-driven initialization
        # Based on the disaster resource allocation game structure:
        # Strategy 0: Fairness-focused - high payoff against fairness strategies, low against efficiency
        # Strategy 1: Efficiency-focused - high payoff against efficiency, moderate against balanced
        # Strategy 2: Balanced - moderate payoffs in most interactions
        # Strategy 3: Adaptive - payoff depends on context, but generally cooperative
        if payoff_matrix is None:
            self.payoff_matrix = nn.Parameter(
                self._init_theory_driven_payoff(num_strategies, device)
            )
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

        # Strategy definitions (must align with paper section 4.3)
        # Always length == num_strategies so that get_strategy_recommendation
        # can safely index it.  The first 4 follow the paper; any extras are
        # labelled generically.
        base_names = [
            "Fairness-focused",
            "Efficiency-focused",
            "Balanced",
            "Adaptive",
        ]
        if num_strategies <= len(base_names):
            self.strategy_names = base_names[:num_strategies]
        else:
            self.strategy_names = base_names + [
                f"Strategy-{i}" for i in range(len(base_names), num_strategies)
            ]

        # Convergence tracking
        self.convergence_threshold = 1e-4
        self.convergence_steps = 0
        self.is_converged = False

        # Performance metrics
        self.fitness_history = []
        self.diversity_history = []

        # Lambda parameter (fairness-efficiency balance, dynamically adjusted)
        # This is the core parameter passed to MARL layer for reward shaping
        self.lambda_param = 0.5  # 0=full efficiency, 1=full fairness

        # Fix audit Issue 2: optional phase-level anchor for lambda_param.
        # When the training script enters a new phase, it sets
        # `lambda_anchor` to the phase's target lambda. `_update_lambda()`
        # then BLENDS the evolved value (from replicator dynamics) with this
        # anchor instead of overwriting it. This preserves EGT's micro->macro
        # feedback signal across phase boundaries.
        #
        # `None` means "no anchor; trust pure replicator dynamics output".
        self.lambda_anchor: Optional[float] = None

        # Blend weight in [0, 1]: 0.0 = pure EGT, 1.0 = pure anchor.
        # A small bias toward the anchor keeps phases directionally aligned
        # without erasing the evolved value.
        self.lambda_anchor_blend: float = 0.3

    def _init_theory_driven_payoff(self, num_strategies: int,
                                    device: torch.device) -> torch.Tensor:
        """
        Initialize a theory-driven payoff matrix based on game-theoretic reasoning
        for disaster resource allocation.

        Structure (4 strategies):
        - Fairness-focused vs Fairness-focused: mutual cooperation (3.0)
        - Fairness-focused vs Efficiency-focused: tension (1.0 for F, 2.0 for E)
        - Fairness-focused vs Balanced: cooperation (2.5)
        - Fairness-focused vs Adaptive: moderate (2.0)

        - Efficiency-focused vs Efficiency-focused: competition (2.0)
        - Efficiency-focused vs Balanced: moderate (2.0)
        - Efficiency-focused vs Adaptive: moderate (2.0)

        - Balanced vs Balanced: cooperation (2.5)
        - Balanced vs Adaptive: cooperation (2.5)

        - Adaptive vs Adaptive: high cooperation (3.0)
        """
        if num_strategies == 4:
            # Paper-aligned 4-strategy payoff structure
            # Rows/cols: [Fairness, Efficiency, Balanced, Adaptive]
            base_payoff = torch.tensor([
                # F-vs-F, F-vs-E, F-vs-B, F-vs-A
                [3.0, 1.0, 2.5, 2.0],
                # E-vs-F, E-vs-E, E-vs-B, E-vs-A
                [2.0, 2.0, 2.0, 2.0],
                # B-vs-F, B-vs-E, B-vs-B, B-vs-A
                [2.5, 2.0, 2.5, 2.5],
                # A-vs-F, A-vs-E, A-vs-B, A-vs-A
                [2.0, 2.0, 2.5, 3.0],
            ], device=device)
        elif num_strategies == 3:
            # P1 fix + Issue 1: paper-aligned 3-strategy payoff structure
            # (deterministic AND strictly symmetric so the game stays a
            # symmetric potential game).
            #
            # Previous branch fell through to a random torch.eye + randn
            # initialisation, which let stochastic drift lock the replicator
            # dynamics onto a single strategy (manifesting as λ=1.0 forever).
            # Rows/cols: [Fairness, Efficiency, Balanced]
            #
            # Interpretation (strictly symmetric: A[i,j] == A[j,i]):
            #   - Fairness-Fairness cooperation: 3.0
            #   - Efficiency-Efficiency competition: 2.0  (slightly tense)
            #   - Balanced-Balanced cooperation: 2.5
            #   - Fairness-Efficiency tension: 1.5  (low — different goals)
            #   - Fairness-Balanced: 2.5
            #   - Efficiency-Balanced: 2.0
            #
            # Design rationale: by keeping each strategy competitive against
            # itself (3.0 / 2.0 / 2.5) and giving cross-strategy payoffs that
            # are neither trivially equal nor too extreme, the replicator
            # dynamics is no longer pulled into a single dominant strategy.
            base_payoff = torch.tensor([
                # F-vs-F, F-vs-E, F-vs-B
                [3.0, 1.5, 2.5],
                # E-vs-F, E-vs-E, E-vs-B  (must mirror row 0)
                [1.5, 2.0, 2.0],
                # B-vs-F, B-vs-E, B-vs-B  (must mirror rows 0,1)
                [2.5, 2.0, 2.5],
            ], device=device)
            # Sanity check: the matrix MUST be symmetric.  If a future edit
            # breaks symmetry, fail loudly here rather than silently
            # producing an asymmetric game.
            assert torch.allclose(base_payoff, base_payoff.T), \
                f"3-strategy payoff matrix must be symmetric, got {base_payoff.tolist()}"
        else:
            # Fallback for other num_strategies: identity + small noise
            base_payoff = torch.eye(num_strategies, device=device) * 2.5
            # Add small random perturbation
            base_payoff += torch.randn(num_strategies, num_strategies, device=device) * 0.1
            # Force symmetry
            base_payoff = (base_payoff + base_payoff.T) / 2

        # Make symmetric (payoffs are mutual in symmetric games)
        payoff = (base_payoff + base_payoff.T) / 2
        return payoff
    
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
        # Use abs(avg_fitness) for divisor so that negative avg_fitness
        # (rare but possible under poor-performance regimes) doesn't produce
        # NaN/Inf. The epsilon guard prevents division-by-near-zero.
        eps_guard = 1e-8
        if abs(avg_fitness) > eps_guard:
            growth_rates = (fitnesses - avg_fitness) / abs(avg_fitness)
        else:
            # Fallback: use absolute difference when avg_fitness is ~0
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
                        # Adaptive scaling: normalise total_reward by the recent
                        # reward standard deviation. This makes the adjustment
                        # invariant to reward magnitude (e.g. 0.1 raw scaling
                        # would otherwise produce wildly different adjustments
                        # for rewards in [-10, 10] vs rewards in [-1, 1]).
                        reward_window = list(self.fitness_history)[-100:]
                        if reward_window:
                            reward_std = max(float(np.std(reward_window)), 1e-6)
                            # Bound to [-1, 1] then scale by 0.3 to match the
                            # magnitude of other strategies' adjustments
                            # (which are in [-0.5, 0.5] range).
                            adjustment = float(total_reward) / reward_std * 0.3
                            # Clamp to [-0.5, 0.5] for safety
                            adjustment = max(-0.5, min(0.5, adjustment))
                        else:
                            # No history yet: fall back to simple scaled adjustment
                            adjustment = float(total_reward) * 0.1
                    
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

        Indexing policy (works for any num_strategies >= 2):
        - index 0           -> fairness-focused
        - index 1           -> efficiency-focused
        - index 2 (if any)  -> balanced (50/50)
        - index 3+          -> adaptive, distributed by recent fitness
        - any remaining     -> neutral, split 50/50

        Returns:
            Tuple of (fairness_weight, efficiency_weight)
        """
        distribution = self.get_strategy_distribution()
        n = int(distribution.shape[0])

        # Defensive fallback for the degenerate 1-strategy case.
        if n < 2:
            return 0.5, 0.5

        fairness_weight = float(distribution[0].item())
        efficiency_weight = float(distribution[1].item())

        if n >= 3:
            balanced_weight = float(distribution[2].item())
            fairness_weight += balanced_weight * 0.5
            efficiency_weight += balanced_weight * 0.5

        # Index 3 is "adaptive" when present, but the underlying semantic
        # (shift by recent fitness) applies to *all* remaining strategies as
        # well so we can support any num_strategies.
        # P10 fix: previously, when ``num_strategies=2`` (no balanced, no
        # adaptive strategy), the fitness-based adaptive branch was
        # *unreachable* because both ``n >= 3`` and ``n >= 4`` guards failed.
        # Apply the adaptive shift to the residual distribution mass (any
        # strategies beyond the first two) so the trade-off still reacts
        # to recent fitness regardless of how many strategies are present.
        adaptive_share = 0.0
        if n >= 4:
            adaptive_share += float(distribution[3].item())
        if n > 4:
            # Treat extra strategies as additional "adaptive" mass.
            adaptive_share += float(distribution[4:].sum().item())
        # n=2: use a small synthetic "adaptive" share driven by recent
        # fitness, so the trade-off still reacts even when no explicit
        # adaptive strategy exists.  Capped at 0.2 to avoid overpowering
        # the dominant 2-strategy semantics.
        if n == 2:
            adaptive_share = 0.2

        if adaptive_share > 0.0 and len(self.fitness_history) > 0:
            recent_fitness = np.mean(self.fitness_history[-5:]) if len(self.fitness_history) >= 5 else 0.5
            # If performance is good, maintain current balance; if poor, shift toward efficiency
            if recent_fitness < 0.5:
                efficiency_weight += adaptive_share * 0.7
                fairness_weight += adaptive_share * 0.3
            else:
                efficiency_weight += adaptive_share * 0.5
                fairness_weight += adaptive_share * 0.5

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
    
    def _extract_performance_metrics(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract real performance metrics from the experience batch.

        Looks for metrics in the batch under the following keys:
        - 'fairness_score' / 'gini_coefficient' (lower gini = higher fairness)
        - 'efficiency_score' / 'survival_rate' / 'rescue_rate'
        - 'total_reward' / 'rewards'
        - 'resource_utilization'
        - 'response_time'
        - 'queue_length'

        Falls back to derived metrics if explicit fields are missing.
        """
        metrics: Dict[str, float] = {}

        # Fairness: prefer explicit fairness_score, else derive from gini (1-gini)
        if 'fairness_score' in batch:
            metrics['fairness_score'] = float(batch['fairness_score'])
        elif 'gini_coefficient' in batch:
            gini = float(batch['gini_coefficient'])
            metrics['fairness_score'] = max(0.0, min(1.0, 1.0 - gini))
        elif 'fairness' in batch:
            metrics['fairness_score'] = float(batch['fairness'])
        else:
            metrics['fairness_score'] = 0.5  # neutral default

        # Efficiency: prefer explicit, else derive from survival/rescue rate
        if 'efficiency_score' in batch:
            metrics['efficiency_score'] = float(batch['efficiency_score'])
        elif 'survival_rate' in batch:
            metrics['efficiency_score'] = float(batch['survival_rate'])
        elif 'rescue_rate' in batch:
            metrics['efficiency_score'] = float(batch['rescue_rate'])
        else:
            metrics['efficiency_score'] = 0.5  # neutral default

        # Total reward (used by adaptive strategy)
        if 'rewards' in batch:
            rewards_tensor = batch['rewards']
            if torch.is_tensor(rewards_tensor):
                metrics['total_reward'] = rewards_tensor.mean().item()
            else:
                metrics['total_reward'] = float(np.mean(rewards_tensor))
        elif 'total_reward' in batch:
            metrics['total_reward'] = float(batch['total_reward'])
        else:
            metrics['total_reward'] = 0.0

        # Optional additional signals (kept for downstream use)
        if 'resource_utilization' in batch:
            metrics['resource_utilization'] = float(batch['resource_utilization'])
        if 'response_time' in batch:
            metrics['response_time'] = float(batch['response_time'])
        if 'queue_length' in batch:
            metrics['queue_length'] = float(batch['queue_length'])

        return metrics

    def update(self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer,
              loss_fn: nn.Module) -> float:
        """
        Update EGT layer parameters.

        Args:
            batch: Experience batch containing states, actions, rewards, and (optionally)
                   explicit performance metrics such as 'fairness_score'/'efficiency_score'.
            optimizer: Optimizer
            loss_fn: Loss function

        Returns:
            Loss value
        """
        # Extract real performance metrics from batch (no more hardcoded 0.5)
        performance_metrics = self._extract_performance_metrics(batch)

        # Evolve strategies based on real performance
        self.evolve_strategies(performance_metrics)

        # Update lambda parameter based on strategy distribution
        self._update_lambda()

        # Calculate loss (encourage diversity and performance)
        distribution = self.get_strategy_distribution()

        # Diversity loss (encourage exploration)
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-8))
        diversity_loss = -entropy  # Maximize entropy

        # Performance loss (based on fitness)
        # P9 fix: keep the scalar as a 0-dim tensor with the same device/dtype
        # as the distribution so that the final ``loss = 0.3 * dl + 0.7 * pl``
        # combines two tensors (avoids implicit float↔tensor promotion that
        # used to be relied on, which is brittle when EGT runs on GPU).
        avg_fitness = float(
            np.mean(self.fitness_history[-5:]) if self.fitness_history else 0.5
        )
        performance_loss = torch.tensor(
            -avg_fitness, dtype=diversity_loss.dtype, device=diversity_loss.device
        )  # Maximize fitness

        # Combined loss
        loss = 0.3 * diversity_loss + 0.7 * performance_loss

        # Optimize (though EGT typically doesn't use gradient descent)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def _update_lambda(self) -> None:
        """
        Update the lambda parameter (fairness-efficiency balance) based on the
        current strategy distribution. This implements the dynamic adjustment
        described in the paper.

        - lambda = 1.0: full fairness focus
        - lambda = 0.0: full efficiency focus
        """
        distribution = self.get_strategy_distribution()

        # Use the explicit fairness/efficiency weights from strategy distribution
        fairness_weight, efficiency_weight = self.get_fairness_efficiency_weights()

        # If system is under stress (low fitness), shift toward efficiency
        # If system is stable, shift toward fairness
        if self.fitness_history:
            recent_fitness = np.mean(self.fitness_history[-5:])
            # Adaptive shifting: stress reduces fairness priority
            if recent_fitness < 0.4:
                # Crisis mode: shift toward efficiency
                efficiency_weight = min(1.0, efficiency_weight * 1.2)
                fairness_weight = max(0.0, 1.0 - efficiency_weight)

        # lambda represents the fairness weight (0=efficiency, 1=fairness)
        evolved_lambda = float(fairness_weight)

        # Fix audit Issue 2: BLEND with phase anchor instead of overwriting.
        # `lambda_anchor` is set by the training script at each phase boundary.
        # Without this blend, a phase change would erase whatever the replicator
        # dynamics had evolved, breaking the EGT->MARL feedback loop.
        if self.lambda_anchor is not None:
            anchor = float(self.lambda_anchor)
            blend = float(self.lambda_anchor_blend)
            # Clamp blend to [0, 1] to be defensive.
            blend = max(0.0, min(1.0, blend))
            self.lambda_param = (1.0 - blend) * evolved_lambda + blend * anchor
        else:
            self.lambda_param = evolved_lambda

    def update_with_weights(self, batch: Dict[str, Any],
                            optimizer: torch.optim.Optimizer,
                            loss_fn: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Update EGT layer and return trade-off weights for the MARL layer.

        This method is the "handshake" between EGT (macro) and MARL (micro):
        it produces the weights that the MARL layer should use to shape rewards.

        Returns:
            (loss, weights_dict) where weights_dict contains:
            - 'fairness_weight': emphasis on fairness
            - 'efficiency_weight': emphasis on efficiency
            - 'lambda_param': scalar fairness-efficiency balance in [0, 1]
            - 'strategy_distribution': full distribution over strategies
        """
        # Run the standard update
        loss = self.update(batch, optimizer, loss_fn)

        # Compute weights from current strategy distribution
        distribution = self.get_strategy_distribution()
        fairness_weight, efficiency_weight = self.get_fairness_efficiency_weights()

        weights = {
            'fairness_weight': fairness_weight,
            'efficiency_weight': efficiency_weight,
            'lambda_param': self.lambda_param,
            'strategy_distribution': distribution.detach().cpu().numpy().tolist(),
        }

        return loss, weights
    
    def reset_convergence(self) -> None:
        """Reset convergence tracking."""
        self.convergence_steps = 0
        self.is_converged = False
    
    def save(self, path: str) -> None:
        """Save EGT layer state."""
        # Fix audit Finding N (long-standing, masked by other paths):
        # payoff_matrix and strategy_distribution are ``nn.Parameter``, not
        # ``nn.Module``, so they have no ``state_dict()`` method. Persist
        # them as detached tensors instead so save/load round-trips work.
        torch.save({
            'payoff_matrix_state': self.payoff_matrix.detach().clone(),
            'strategy_distribution_state': self.strategy_distribution.detach().clone(),
            'strategy_history': self.strategy_history,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'convergence_steps': self.convergence_steps,
            'is_converged': self.is_converged,
            # P5 fix: persist the runtime-computed lambda_param so that
            # loading a checkpoint doesn't reset it to the constructor
            # default (0.5) before the first update() recomputes it.
            'lambda_param': float(self.lambda_param),
            # Issue 2: also persist the phase anchor + blend so a checkpoint
            # resumed mid-phase continues blending with the right anchor.
            'lambda_anchor': (float(self.lambda_anchor)
                              if self.lambda_anchor is not None else None),
            'lambda_anchor_blend': float(self.lambda_anchor_blend),
            'config': {
                'num_strategies': self.num_strategies,
                'learning_rate': self.learning_rate,
                'mutation_rate': self.mutation_rate
            }
        }, path)

    def load(self, path: str) -> None:
        """Load EGT layer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # load_state_dict expects an nn.Module, but payoff_matrix /
        # strategy_distribution are nn.Parameter. Use .data.copy_() instead.
        with torch.no_grad():
            self.payoff_matrix.data.copy_(checkpoint['payoff_matrix_state'].to(self.device))
            self.strategy_distribution.data.copy_(
                checkpoint['strategy_distribution_state'].to(self.device)
            )

        self.strategy_history = checkpoint['strategy_history']
        self.fitness_history = checkpoint['fitness_history']
        self.diversity_history = checkpoint['diversity_history']

        self.convergence_steps = checkpoint['convergence_steps']
        self.is_converged = checkpoint['is_converged']
        # P5 fix: restore lambda_param if the checkpoint contains it
        # (older checkpoints may not, so fall back to default 0.5).
        if 'lambda_param' in checkpoint:
            self.lambda_param = float(checkpoint['lambda_param'])
        # Issue 2: restore phase anchor + blend weight.
        if 'lambda_anchor' in checkpoint and checkpoint['lambda_anchor'] is not None:
            self.lambda_anchor = float(checkpoint['lambda_anchor'])
        if 'lambda_anchor_blend' in checkpoint:
            self.lambda_anchor_blend = float(checkpoint['lambda_anchor_blend'])
        # P12 fix: restore learning_rate that was previously saved in
        # ``config`` but never read back.  This makes optimizer
        # reconstruction after a reload actually match the original run.
        cfg = checkpoint.get('config', {})
        if 'learning_rate' in cfg:
            self.learning_rate = float(cfg['learning_rate'])
        if 'mutation_rate' in cfg:
            self.mutation_rate = float(cfg['mutation_rate'])