"""
EGT-MARL: Evolutionary Game Theory - Multi-Agent Reinforcement Learning
=======================================================================

Main two-layer algorithm for dynamic medical resource allocation in disasters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import yaml
from pathlib import Path

from .marl_layer import MARLLayer
from .egt_layer import EGTLayer
from .anti_spoofing import AntiSpoofing
from .dynamic_frontier import DynamicFrontier


class EGTMARL:
    """
    Main EGT-MARL algorithm with two-layer architecture.
    
    Architecture:
    1. Micro-layer (MARL): Distributed decision execution using improved QMIX
    2. Macro-layer (EGT): Dynamic fairness-efficiency trade-off regulation
    """
    
    def __init__(self, env, config_path: Optional[str] = None):
        """
        Initialize EGT-MARL algorithm.
        
        Args:
            env: Disaster simulation environment
            config_path: Path to configuration file
        """
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "egt_marl.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # Metrics tracking
        self.metrics_history = {
            'total_rewards': [],
            'fairness_scores': [],
            'efficiency_scores': [],
            'pareto_frontier': [],
            'spoofing_detected': []
        }
    
    def _initialize_components(self) -> None:
        """Initialize all algorithm components."""
        # MARL layer for distributed decision execution
        self.marl_layer = MARLLayer(
            state_dim=self.config['marl']['state_dim'],
            action_dim=self.config['marl']['action_dim'],
            num_agents=self.config['marl']['num_agents'],
            hidden_dim=self.config['marl']['hidden_dim'],
            mixing_hidden_dim=self.config['marl']['mixing_hidden_dim'],
            device=self.device
        )
        
        # EGT layer for fairness-efficiency trade-off
        self.egt_layer = EGTLayer(
            num_strategies=self.config['egt']['num_strategies'],
            payoff_matrix=self._initialize_payoff_matrix(),
            learning_rate=self.config['egt']['learning_rate'],
            device=self.device
        )
        
        # Anti-spoofing mechanism
        self.anti_spoofing = AntiSpoofing(
            observation_dim=self.config['anti_spoofing']['observation_dim'],
            hidden_dim=self.config['anti_spoofing']['hidden_dim'],
            device=self.device
        )
        
        # Dynamic Pareto frontier
        self.dynamic_frontier = DynamicFrontier(
            num_objectives=self.config['dynamic_frontier']['num_objectives'],
            device=self.device
        )
        
        # Optimizers
        self.marl_optimizer = optim.Adam(
            self.marl_layer.parameters(),
            lr=self.config['training']['marl_lr']
        )
        self.egt_optimizer = optim.Adam(
            self.egt_layer.parameters(),
            lr=self.config['training']['egt_lr']
        )
        
        # Loss functions
        self.marl_loss_fn = nn.MSELoss()
        self.egt_loss_fn = nn.KLDivLoss()
    
    def _initialize_payoff_matrix(self) -> torch.Tensor:
        """Initialize payoff matrix for evolutionary game."""
        num_strategies = self.config['egt']['num_strategies']
        payoff_matrix = torch.randn(num_strategies, num_strategies)
        
        # Make symmetric for simplicity
        payoff_matrix = (payoff_matrix + payoff_matrix.T) / 2
        
        return payoff_matrix.to(self.device)
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        """
        Select actions for all agents.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            
        Returns:
            Dictionary of actions for each agent
        """
        # Get MARL actions (micro-level decisions)
        marl_actions = self.marl_layer.select_action(state, training)
        
        # Get EGT strategy distribution (macro-level regulation)
        strategy_distribution = self.egt_layer.get_strategy_distribution()
        
        # Apply anti-spoofing verification
        verified_actions = self.anti_spoofing.verify_actions(
            actions=marl_actions,
            state=state,
            strategy_distribution=strategy_distribution
        )
        
        # Adjust actions based on fairness-efficiency trade-off
        adjusted_actions = self._adjust_actions_with_egt(
            actions=verified_actions,
            strategy_distribution=strategy_distribution
        )
        
        return adjusted_actions
    
    def _adjust_actions_with_egt(self, actions: Dict[str, Any], 
                                strategy_distribution: torch.Tensor) -> Dict[str, Any]:
        """
        Adjust actions based on EGT strategy distribution.
        
        Args:
            actions: Original actions from MARL layer
            strategy_distribution: Current strategy distribution from EGT layer
            
        Returns:
            Adjusted actions
        """
        adjusted_actions = actions.copy()
        
        # Extract fairness and efficiency weights from strategy distribution
        fairness_weight = strategy_distribution[0].item()  # First strategy: fairness-focused
        efficiency_weight = strategy_distribution[1].item()  # Second strategy: efficiency-focused
        
        # Normalize weights
        total_weight = fairness_weight + efficiency_weight
        if total_weight > 0:
            fairness_weight /= total_weight
            efficiency_weight /= total_weight
        
        # Adjust resource allocation based on weights
        for agent_id, action in adjusted_actions.items():
            if 'resource_allocation' in action:
                original_allocation = action['resource_allocation']
                
                # Apply fairness adjustment (more equitable distribution)
                if fairness_weight > 0:
                    fairness_adjustment = self._calculate_fairness_adjustment(
                        agent_id, original_allocation
                    )
                    original_allocation = {
                        k: v * (1 + fairness_weight * fairness_adjustment.get(k, 0))
                        for k, v in original_allocation.items()
                    }
                
                # Apply efficiency adjustment (prioritize critical cases)
                if efficiency_weight > 0:
                    efficiency_adjustment = self._calculate_efficiency_adjustment(
                        agent_id, original_allocation
                    )
                    original_allocation = {
                        k: v * (1 + efficiency_weight * efficiency_adjustment.get(k, 0))
                        for k, v in original_allocation.items()
                    }
                
                # Ensure non-negative allocations
                action['resource_allocation'] = {
                    k: max(0, v) for k, v in original_allocation.items()
                }
        
        return adjusted_actions
    
    def _calculate_fairness_adjustment(self, agent_id: str, 
                                      allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate fairness-based adjustment for resource allocation."""
        # Simple proportional fairness adjustment
        total_resources = sum(allocation.values())
        if total_resources == 0:
            return {k: 0 for k in allocation.keys()}
        
        # Aim for more equal distribution
        target_share = 1.0 / len(allocation)
        current_shares = {k: v / total_resources for k, v in allocation.items()}
        
        adjustment = {}
        for resource_type, share in current_shares.items():
            # Positive adjustment if below target, negative if above
            adjustment[resource_type] = (target_share - share) * 0.5
        
        return adjustment
    
    def _calculate_efficiency_adjustment(self, agent_id: str, 
                                        allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency-based adjustment for resource allocation."""
        # Prioritize resources with higher urgency/impact
        adjustment = {}
        
        # Example: prioritize antibiotics for severe cases
        resource_priorities = {
            'broad_spectrum_antibiotics': 1.5,
            'pain_relievers': 1.2,
            'bandages': 1.0,
            'splints': 1.1,
            'blood_transfusion': 1.8
        }
        
        for resource_type in allocation.keys():
            priority = resource_priorities.get(resource_type, 1.0)
            # Higher priority gets positive adjustment
            adjustment[resource_type] = (priority - 1.0) * 0.3
        
        return adjustment
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update algorithm parameters from experience batch.
        
        Args:
            batch: Experience batch containing states, actions, rewards, next_states
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Update MARL layer
        marl_loss = self.marl_layer.update(batch, self.marl_optimizer, self.marl_loss_fn)
        losses['marl_loss'] = marl_loss
        
        # Update EGT layer
        egt_loss = self.egt_layer.update(batch, self.egt_optimizer, self.egt_loss_fn)
        losses['egt_loss'] = egt_loss
        
        # Update anti-spoofing mechanism
        spoofing_loss = self.anti_spoofing.update(batch)
        losses['spoofing_loss'] = spoofing_loss
        
        # Update dynamic Pareto frontier
        frontier_loss = self.dynamic_frontier.update(batch)
        losses['frontier_loss'] = frontier_loss
        
        # Update total steps
        self.total_steps += len(batch['states'])
        
        return losses
    
    def train_episode(self) -> Dict[str, Any]:
        """
        Train for one episode.
        
        Returns:
            Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        episode_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        while not done:
            # Select action
            action = self.select_action(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            episode_buffer['states'].append(state)
            episode_buffer['actions'].append(action)
            episode_buffer['rewards'].append(reward)
            episode_buffer['next_states'].append(next_state)
            episode_buffer['dones'].append(done)
            
            # Update
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Update if buffer is full
            if len(episode_buffer['states']) >= self.config['training']['batch_size']:
                batch = self._prepare_batch(episode_buffer)
                losses = self.update(batch)
                
                # Clear buffer
                for key in episode_buffer:
                    episode_buffer[key] = []
        
        # Final update with remaining experiences
        if len(episode_buffer['states']) > 0:
            batch = self._prepare_batch(episode_buffer)
            losses = self.update(batch)
        
        # Calculate metrics
        metrics = self._calculate_episode_metrics(episode_reward, info)
        
        # Update episode counter
        self.episode += 1
        
        return {
            'episode': self.episode,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'losses': losses,
            'metrics': metrics
        }
    
    def _prepare_batch(self, buffer: Dict[str, List]) -> Dict[str, Any]:
        """Prepare batch for training."""
        batch = {}
        for key, value in buffer.items():
            if key == 'actions':
                # Convert actions to tensor format
                batch[key] = self._actions_to_tensor(value)
            else:
                batch[key] = torch.tensor(value, device=self.device)
        return batch
    
    def _actions_to_tensor(self, actions: List[Dict]) -> torch.Tensor:
        """Convert list of action dictionaries to tensor."""
        # Simplified conversion - in practice would need proper handling
        action_tensors = []
        for action_dict in actions:
            # Flatten action dictionary
            flat_actions = []
            for agent_id, action in action_dict.items():
                if isinstance(action, dict):
                    # Handle resource allocation
                    for resource_type, amount in action.get('resource_allocation', {}).items():
                        flat_actions.append(amount)
                else:
                    flat_actions.append(action)
            action_tensors.append(flat_actions)
        
        return torch.tensor(action_tensors, device=self.device)
    
    def _calculate_episode_metrics(self, total_reward: float, 
                                  info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate episode metrics."""
        metrics = {
            'total_reward': total_reward,
            'fairness_score': info.get('fairness_score', 0.0),
            'efficiency_score': info.get('efficiency_score', 0.0),
            'pareto_score': self.dynamic_frontier.get_pareto_score(),
            'spoofing_rate': self.anti_spoofing.get_detection_rate()
        }
        
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'marl_layer_state': self.marl_layer.state_dict(),
            'egt_layer_state': self.egt_layer.state_dict(),
            'anti_spoofing_state': self.anti_spoofing.state_dict(),
            'dynamic_frontier_state': self.dynamic_frontier.state_dict(),
            'marl_optimizer_state': self.marl_optimizer.state_dict(),
            'egt_optimizer_state': self.egt_optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.best_reward = checkpoint['best_reward']
        
        self.marl_layer.load_state_dict(checkpoint['marl_layer_state'])
        self.egt_layer.load_state_dict(checkpoint['egt_layer_state'])
        self.anti_spoofing.load_state_dict(checkpoint['anti_spoofing_state'])
        self.dynamic_frontier.load_state_dict(checkpoint['dynamic_frontier_state'])
        
        self.marl_optimizer.load_state_dict(checkpoint['marl_optimizer_state'])
        self.egt_optimizer.load_state_dict(checkpoint['egt_optimizer_state'])
        
        self.metrics_history = checkpoint['metrics_history']
        
        print(f"Checkpoint loaded from {path}")
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get metrics history."""
        return self.metrics_history.copy()
    
    def run_episode(self, render: bool = False) -> Dict[str, Any]:
        """
        Run one episode without training.
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Episode results
        """
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        episode_actions = []
        episode_states = [state]
        
        while not done:
            # Select action (no exploration)
            action = self.select_action(state, training=False)
            episode_actions.append(action)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            if render:
                self.env.render()
            
            state = next_state
            episode_states.append(state)
            episode_reward += reward
            episode_steps += 1
        
        return {
            'total_reward': episode_reward,
            'steps': episode_steps,
            'states': episode_states,
            'actions': episode_actions,
            'info': info
        }