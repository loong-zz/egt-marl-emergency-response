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
            # Agent Q-network
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
        """Forward pass for MARL layer."""
        q_values = []
        for i, agent_net in enumerate(self.agent_networks):
            q_values.append(agent_net(states))
        return torch.stack(q_values, dim=1)
    
    def select_actions(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select actions for all agents."""
        if deterministic or np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.forward(states)
                actions = q_values.argmax(dim=2)
        else:
            # Random exploration
            actions = torch.randint(0, self.action_dim, 
                                   (states.shape[0], self.num_agents),
                                   device=self.device)
        
        # Decay epsilon
        if not deterministic:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return actions
    
    def update(self, batch_states: torch.Tensor, batch_actions: torch.Tensor, 
               batch_rewards: torch.Tensor, batch_next_states: torch.Tensor, 
               batch_dones: torch.Tensor) -> float:
        """Update MARL layer using experience batch."""
        # Get current Q-values
        current_qs = self.forward(batch_states)
        
        # Get action indices
        action_indices = batch_actions.long().unsqueeze(2)
        joint_q = current_qs.gather(2, action_indices).squeeze(2)
        
        # Get next Q-values from target networks
        with torch.no_grad():
            next_qs = []
            for i, target_net in enumerate(self.target_agent_networks):
                next_qs.append(target_net(batch_next_states))
            next_qs = torch.stack(next_qs, dim=1)
            
            # Get max next Q-values
            max_next_qs = next_qs.max(dim=2)[0]
            
            # Get target joint Q-value
            target_joint_q = self.target_mixing_network(torch.cat([batch_next_states, max_next_qs], dim=1))
            
            # Compute target
            target = batch_rewards + self.gamma * target_joint_q.squeeze() * (1 - batch_dones.float())
        
        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(joint_q, target)
        
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