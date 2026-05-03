"""
MARL Layer: Multi-Agent Reinforcement Learning Layer
====================================================

Distributed decision execution using improved QMIX algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class MARLLayer(nn.Module):
    """
    Multi-Agent Reinforcement Learning layer using improved QMIX.
    
    Implements:
    1. Distributed Q-learning for individual agents
    2. Centralized mixing network for joint Q-value
    3. Target networks for stable learning
    4. Communication mechanism between agents
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_agents: int, 
                 hidden_dim: int = 64, communication_enabled: bool = True, 
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.communication_enabled = communication_enabled
        self.device = device
        
        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Learning parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.01    # Target network update rate
        
        # Initialize agent networks
        self.agent_networks = nn.ModuleList()
        self.target_agent_networks = nn.ModuleList()
        
        for _ in range(num_agents):
            # Agent Q-network (shared architecture, separate instances for exploration diversity)
            agent_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ).to(device)

            # Target network
            target_agent_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ).to(device)

            # Initialize target network with same weights
            target_agent_net.load_state_dict(agent_net.state_dict())

            self.agent_networks.append(agent_net)
            self.target_agent_networks.append(target_agent_net)

        # Shared agent network for batch processing (parameter sharing for efficiency)
        self.shared_agent_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

        # Shared target network
        self.shared_target_agent_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.shared_target_agent_net.load_state_dict(self.shared_agent_net.state_dict())
        
        # Mixing network for centralized Q-value
        self.mixing_network = nn.Sequential(
            nn.Linear(state_dim + num_agents * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        self.target_mixing_network = nn.Sequential(
            nn.Linear(state_dim + num_agents * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Initialize target mixing network
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Communication network (if enabled)
        if communication_enabled:
            self.communication_network = nn.Sequential(
                nn.Linear(state_dim * num_agents, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_agents * num_agents)
            ).to(device)
        else:
            self.communication_network = None
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass for MARL layer. Optimized batch processing using shared network."""
        batch_size, num_agents, state_dim = states.shape

        states_flat = states.view(batch_size * num_agents, state_dim)

        q_values_flat = self.shared_agent_net(states_flat)

        q_values = q_values_flat.view(batch_size, num_agents, self.action_dim)

        return q_values
    
    def select_actions(self, states: torch.Tensor, deterministic: bool = False, epsilon: float = None) -> torch.Tensor:
        """Select actions for all agents."""
        # Use provided epsilon or fall back to internal epsilon
        current_epsilon = epsilon if epsilon is not None else self.epsilon
        
        if deterministic or np.random.rand() > current_epsilon:
            with torch.no_grad():
                q_values = self.forward(states)
                # Q-values shape: (batch_size, num_agents, action_dim)
                actions = q_values.argmax(dim=2)
        else:
            actions = torch.randint(0, self.action_dim,
                                   (states.shape[0], self.num_agents),
                                   device=self.device)
        
        # Decay epsilon if not using external epsilon
        if not deterministic and epsilon is None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return actions
    
    def update(self, batch_states: torch.Tensor, batch_actions: torch.Tensor, 
               batch_rewards: torch.Tensor, batch_next_states: torch.Tensor, 
               batch_dones: torch.Tensor) -> float:
        """Update MARL layer using experience batch."""
        # Get current Q-values
        current_qs = self.forward(batch_states)
        
        # Get action indices and selected Q-values
        action_indices = batch_actions.long().unsqueeze(2)
        selected_qs = current_qs.gather(2, action_indices).squeeze(2)
        
        # Compute global state by averaging agent observations
        global_state = batch_states.mean(dim=1)  # Shape: (batch_size, state_dim)
        
        # Expand selected Q-values for mixing network input
        selected_qs_expanded = selected_qs.unsqueeze(2).repeat(1, 1, self.action_dim).flatten(start_dim=1)
        
        # Get current joint Q-value using mixing network
        current_mixing_input = torch.cat([global_state, selected_qs_expanded], dim=1)
        current_joint_q = self.mixing_network(current_mixing_input).squeeze()
        
        # Get next Q-values from target networks (optimized batch processing)
        with torch.no_grad():
            batch_size, num_agents, state_dim = batch_next_states.shape

            next_states_flat = batch_next_states.view(batch_size * num_agents, state_dim)

            next_qs_flat = self.shared_target_agent_net(next_states_flat)

            next_qs = next_qs_flat.view(batch_size, num_agents, self.action_dim)

            max_next_qs = next_qs.max(dim=2)[0]
            
            # Get target joint Q-value
            # Compute global state for next states
            next_global_state = batch_next_states.mean(dim=1)  # Shape: (batch_size, state_dim)
            
            # Expand max Q-values for mixing network input
            max_next_qs_expanded = max_next_qs.unsqueeze(2).repeat(1, 1, self.action_dim).flatten(start_dim=1)
            
            next_mixing_input = torch.cat([next_global_state, max_next_qs_expanded], dim=1)
            target_joint_q = self.target_mixing_network(next_mixing_input).squeeze()
            
            # Compute target
            target = batch_rewards + self.gamma * target_joint_q * (1 - batch_dones.float())
        
        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_joint_q, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target networks
        self._update_target_networks(self.tau)
        
        return loss.item()
    
    def _update_target_networks(self, tau: float) -> None:
        """Update target networks with soft update."""
        # Update agent target networks
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_agent_networks[i].parameters(),
                                         self.agent_networks[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update shared target network
        for target_param, param in zip(self.shared_target_agent_net.parameters(),
                                     self.shared_agent_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update mixing target network
        for target_param, param in zip(self.target_mixing_network.parameters(),
                                     self.mixing_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def _get_batch_agent_observations(self, batch_states: torch.Tensor, 
                                     agent_id: int) -> torch.Tensor:
        """Get batch of observations for a specific agent."""
        # This is a simplified version - in practice would need proper state processing
        batch_size = batch_states.shape[0]
        
        # For simplicity, assume states are already agent observations
        # In real implementation, would extract agent-specific features
        return batch_states
    
    def _actions_to_indices(self, batch_actions: torch.Tensor, 
                           agent_id: int) -> torch.Tensor:
        """Convert batch of actions to action indices for a specific agent."""
        batch_size = batch_actions.shape[0]
        
        # Simplified: assume actions are already indices
        # In real implementation, would need to convert from action dictionaries
        return batch_actions[:, agent_id].long()
    
    def get_communication_matrix(self) -> torch.Tensor:
        """Get communication matrix showing agent interactions."""
        if not self.communication_enabled or self.communication_network is None:
            return torch.zeros((self.num_agents, self.num_agents), device=self.device)
        
        # Create dummy input to extract communication patterns
        dummy_state = torch.randn(1, self.state_dim, device=self.device)
        
        with torch.no_grad():
            # Pass through communication network
            output = self.communication_network(dummy_state)
            
            # Reshape to get agent-wise communication
            # This is simplified - actual implementation would depend on network architecture
            comm_matrix = output.view(self.num_agents, -1)
        
        return comm_matrix
    
    def save(self, path: str) -> None:
        """Save MARL layer state."""
        torch.save({
            'agent_networks_state': [net.state_dict() for net in self.agent_networks],
            'target_agent_networks_state': [net.state_dict() for net in self.target_agent_networks],
            'mixing_network_state': self.mixing_network.state_dict(),
            'target_mixing_network_state': self.target_mixing_network.state_dict(),
            'communication_network_state': self.communication_network.state_dict() if self.communication_enabled else None,
            'epsilon': self.epsilon,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_agents': self.num_agents,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load MARL layer state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load agent networks
        for i, net in enumerate(self.agent_networks):
            net.load_state_dict(checkpoint['agent_networks_state'][i])
        
        # Load target networks
        for i, net in enumerate(self.target_agent_networks):
            net.load_state_dict(checkpoint['target_agent_networks_state'][i])
        
        # Load mixing networks
        self.mixing_network.load_state_dict(checkpoint['mixing_network_state'])
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network_state'])
        
        # Load communication network
        if self.communication_enabled and checkpoint['communication_network_state'] is not None:
            self.communication_network.load_state_dict(checkpoint['communication_network_state'])
        
        # Load epsilon
        self.epsilon = checkpoint['epsilon']