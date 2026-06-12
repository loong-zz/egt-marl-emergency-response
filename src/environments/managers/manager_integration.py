"""
Manager Integration Module

This module integrates all the specialized managers into the training pipeline:
1. EGTManager - Dynamic fairness-efficiency weight adjustment
2. ReputationManager - Incentive-compatible reputation system
3. ParetoFrontierManager - Dynamic Pareto frontier
4. CommunicationManager - Agent-to-agent communication
5. CommunicationInterference - Communication interference model
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from .egt_manager import EGTManager
from .reputation_manager import ReputationManager
from .pareto_manager import ParetoFrontierManager
from .communication_manager import CommunicationManager
from .communication_interference import CommunicationInterference

logger = logging.getLogger(__name__)


class ManagerIntegration:
    """
    Manager Integration for EGT-MARL training pipeline.

    This class orchestrates all managers and integrates them into the training loop:
    1. Initializes all managers with appropriate configurations
    2. Provides callbacks for training loop integration
    3. Aggregates manager outputs for reward shaping and policy updates
    """

    def __init__(self, config: Dict = None):
        """
        Initialize all managers.

        Args:
            config: Configuration dictionary with manager parameters
        """
        # Initialize all managers
        self.egt_manager = EGTManager(config.get('egt', {}))
        self.reputation_manager = ReputationManager(config.get('reputation', {}))
        self.pareto_manager = ParetoFrontierManager(config.get('pareto', {}))
        self.communication_manager = CommunicationManager(config.get('communication', {}))
        self.communication_interference = CommunicationInterference(config.get('interference', {}))

        # Training state
        self.current_time_step = 0
        self.episode_rewards = []
        self.agent_fitness = {}

        logger.info("Manager Integration initialized with all managers")

    def on_episode_start(self, num_agents: int, num_regions: int):
        """
        Callback at the start of each episode.

        Args:
            num_agents: Number of agents in the simulation
            num_regions: Number of regions for fitness calculation
        """
        self.current_time_step = 0
        self.agent_fitness = {i: 0.0 for i in range(num_agents)}
        self.episode_rewards = []

        # Reset all managers
        self.egt_manager.reset()
        self.reputation_manager.reset()
        self.communication_manager.reset()
        self.communication_interference.reset()

        logger.debug(f"[INTEGRATION] Episode started - Agents: {num_agents}, Regions: {num_regions}")

    def on_step_start(self, hours_elapsed: float, aftershock_happening: bool = False):
        """
        Callback at the start of each step.

        Args:
            hours_elapsed: Hours since disaster start
            aftershock_happening: Whether an aftershock is happening
        """
        # Update communication interference model
        self.communication_interference.update_communication_quality(
            self.current_time_step,
            aftershock_happening
        )

        # Update Pareto weights based on disaster phase
        self.pareto_manager.update_weights(hours_elapsed)

        # Log manager states periodically
        if self.current_time_step % 100 == 0:
            logger.debug(f"[STEP {self.current_time_step}] "
                        f"Pareto: eff={self.pareto_manager.current_efficiency_weight:.2f}, "
                        f"fair={self.pareto_manager.current_fairness_weight:.2f}, "
                        f"EGT: λ={self.egt_manager.lambda_t:.4f}")

    def on_step_end(self, agent_states: Dict, agent_rewards: Dict, region_data: Dict):
        """
        Callback at the end of each step.

        Args:
            agent_states: Dictionary of agent states
            agent_rewards: Dictionary of agent rewards
            region_data: Dictionary of region fitness data
        """
        # Update time step
        self.current_time_step += 1

        # Calculate fitness for each agent and update reputation
        fitness_values = []
        for agent_id, state in agent_states.items():
            # Fitness based on survival rate and resource efficiency
            survival_rate = state.get('survival_rate', 0.0)
            resource_efficiency = state.get('resource_efficiency', 0.0)
            fitness = 0.7 * survival_rate + 0.3 * resource_efficiency
            self.agent_fitness[agent_id] = fitness
            fitness_values.append(fitness)
            
            # Update reputation based on agent performance
            reward = agent_rewards.get(agent_id, 0.0)
            # Positive reward = good performance = honest behavior
            is_honest = reward >= 0
            self.reputation_manager.update_reputation(agent_id, is_honest)

        # Update EGT lambda based on fitness distribution
        if len(fitness_values) > 0:
            new_lambda = self.egt_manager.update_lambda(fitness_values, self.current_time_step)

        # Store rewards
        self.episode_rewards.append(sum(agent_rewards.values()))

        # Log periodically
        if self.current_time_step % 100 == 0:
            gini = self.egt_manager.calculate_gini_coefficient(fitness_values)
            logger.debug(f"[STEP {self.current_time_step}] "
                        f"Gini={gini:.4f}, λ={new_lambda:.4f}")

    def on_resource_claim(self, agent_id: int, claimed_demand: float, actual_demand: float,
                         context: Dict = None) -> Tuple[bool, float]:
        """
        Callback when an agent claims resource demand.

        Args:
            agent_id: Agent making the claim
            claimed_demand: Amount claimed
            actual_demand: Actual amount needed
            context: Additional context (e.g., casualty severity, distance)

        Returns:
            (is_honest, reputation) tuple
        """
        is_honest = self.reputation_manager.verify_claim(
            agent_id, claimed_demand, actual_demand, context
        )

        # Get current reputation
        reputation = self.reputation_manager.get_reputation(agent_id)

        logger.debug(f"[REPUTATION] Agent{agent_id} - "
                    f"Claimed={claimed_demand:.2f}, Actual={actual_demand:.2f}, "
                    f"Honest={is_honest}, Reputation={reputation:.2f}")

        return (is_honest, reputation)

    def on_resource_allocation(self, agent_id: int, allocated_amount: float) -> float:
        """
        Callback when resources are allocated to an agent.

        Args:
            agent_id: Agent receiving resources
            allocated_amount: Amount allocated

        Returns:
            Allocation weight based on reputation
        """
        reputation = self.reputation_manager.get_reputation(agent_id)

        # Higher reputation = higher allocation priority
        allocation_weight = 0.5 + 0.5 * reputation

        logger.debug(f"[ALLOCATION] Agent{agent_id} allocated {allocated_amount:.2f} "
                    f"(reputation={reputation:.2f}, weight={allocation_weight:.2f})")

        return allocation_weight

    def check_communication(self, agent1_pos: Tuple[float, float],
                           agent2_pos: Tuple[float, float]) -> Tuple[bool, Optional[float]]:
        """
        Check if two agents can communicate.

        Args:
            agent1_pos: Position of first agent
            agent2_pos: Position of second agent

        Returns:
            (can_communicate, delay) tuple
        """
        # Calculate distance
        distance = np.sqrt((agent1_pos[0] - agent2_pos[0]) ** 2 +
                          (agent1_pos[1] - agent2_pos[1]) ** 2)

        # Check interference model
        return self.communication_interference.can_communicate(distance)

    def broadcast_casualties(self, agent_id: int, agent_pos: Tuple[float, float],
                            known_casualties: Dict[int, dict]):
        """
        Broadcast casualty information from an agent.

        Args:
            agent_id: Broadcasting agent ID
            agent_pos: Agent position
            known_casualties: Dictionary of known casualties
        """
        self.communication_manager.broadcast_casualties(
            agent_id, agent_pos, known_casualties, self.current_time_step
        )

    def receive_broadcasts(self, agent_id: int, agent_pos: Tuple[float, float],
                          nearby_agents: List[Tuple[int, Tuple[float, float]]]) -> Dict[int, dict]:
        """
        Receive casualty broadcasts for an agent.

        Args:
            agent_id: Receiving agent ID
            agent_pos: Agent position
            nearby_agents: List of (agent_id, position) tuples

        Returns:
            Dictionary of newly discovered casualties
        """
        return self.communication_manager.receive_broadcasts(
            agent_id, agent_pos, self.current_time_step, nearby_agents
        )

    def get_priority_score(self, casualty_severity: str, agent_reputation: float,
                           distance: float, resource_available: float) -> float:
        """
        Calculate priority score for task assignment.

        Args:
            casualty_severity: Severity level of the casualty
            agent_reputation: Reputation of the assigned agent
            distance: Distance to the casualty
            resource_available: Available resources

        Returns:
            Priority score
        """
        # Get Pareto weights
        efficiency_weight = self.pareto_manager.current_efficiency_weight
        fairness_weight = self.pareto_manager.current_fairness_weight

        # Efficiency component (based on distance and resources)
        efficiency_score = 1.0 / (1.0 + distance / 100.0) * resource_available

        # Fairness component (based on severity and reputation)
        severity_weights = {
            'CRITICAL': 1.0,
            'SEVERE': 0.7,
            'MODERATE': 0.4,
            'MILD': 0.2
        }
        severity_score = severity_weights.get(casualty_severity, 0.5) * agent_reputation

        # Combined score
        priority = efficiency_weight * efficiency_score + fairness_weight * severity_score

        return priority

    def get_shaped_reward(self, base_reward: float, agent_id: int,
                          action_type: str, context: Dict) -> float:
        """
        Shape the reward based on manager outputs.

        Args:
            base_reward: Original reward from environment
            agent_id: Agent receiving the reward
            action_type: Type of action taken
            context: Additional context

        Returns:
            Shaped reward
        """
        shaped_reward = base_reward

        # Apply EGT fairness bonus/penalty
        lambda_t = self.egt_manager.get_current_lambda()
        fairness_factor = 1.0 - lambda_t  # Higher lambda = more fairness focus

        # Apply Pareto efficiency/fairness adjustment
        efficiency_weight = self.pareto_manager.current_efficiency_weight
        fairness_weight = self.pareto_manager.current_fairness_weight

        # Reputation-based reward adjustment
        reputation = self.reputation_manager.get_reputation(agent_id)
        reputation_bonus = 0.1 * reputation

        # Combine adjustments
        shaped_reward = base_reward * (0.8 + 0.2 * reputation) + reputation_bonus

        logger.debug(f"[REWARD SHAPING] Agent{agent_id} - "
                    f"Base={base_reward:.2f}, Shaped={shaped_reward:.2f}, "
                    f"λ={lambda_t:.2f}, Reputation={reputation:.2f}")

        return shaped_reward

    def get_metrics(self) -> Dict:
        """Get aggregated metrics from all managers."""
        return {
            'egt': self.egt_manager.get_egt_metrics(),
            'pareto': self.pareto_manager.get_pareto_metrics(),
            'communication': self.communication_manager.get_communication_metrics(),
            'interference': self.communication_interference.get_interference_metrics(),
            'reputation': self.reputation_manager.get_reputation_metrics()
        }

    def on_episode_end(self) -> Dict:
        """
        Callback at the end of each episode.

        Returns:
            Episode summary dictionary
        """
        summary = {
            'final_lambda': self.egt_manager.lambda_t,
            'lambda_history_length': len(self.egt_manager.lambda_history),
            'total_communications': len(self.communication_manager.comm_history),
            'shared_casualties': len(self.communication_manager.shared_casualties),
            'avg_communication_delay': 0.0,
            'packet_loss_rate': 0.0
        }

        # Calculate average communication metrics
        interference_metrics = self.communication_interference.get_interference_metrics()
        if interference_metrics['total_packets'] > 0:
            summary['avg_communication_delay'] = interference_metrics['avg_delay']
            summary['packet_loss_rate'] = interference_metrics['loss_rate']

        logger.info(f"[EPISODE END] Summary: λ={summary['final_lambda']:.4f}, "
                   f"Communications={summary['total_communications']}, "
                   f"SharedCasualties={summary['shared_casualties']}")

        return summary


def create_manager_integration(config: Dict = None) -> ManagerIntegration:
    """
    Factory function to create ManagerIntegration.

    Args:
        config: Configuration dictionary

    Returns:
        ManagerIntegration instance
    """
    if config is None:
        config = {}

    return ManagerIntegration(config)
