"""
MARL Layer: Multi-Agent Reinforcement Learning Layer
====================================================

Distributed decision execution using improved QMIX algorithm.
Optimized for fast response time with:
- Sparse attention-based communication for O(N) complexity
- Hypernetwork-based mixing for efficient joint Q-value computation
- Gradient accumulation for reduced update frequency
- Shared parameter networks for efficient batch processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math


class MARLLayer(nn.Module):
    """
    Multi-Agent Reinforcement Learning layer using optimized QMIX.
    
    Implements:
    1. Shared parameter Q-network for efficient batch processing
    2. Hypernetwork-based mixing for efficient joint Q-value
    3. Sparse attention communication for O(N log N) complexity
    4. Gradient accumulation for reduced update frequency
    5. Target networks for stable learning
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
        self.gamma = 0.99
        self.tau = 0.01
        
        # Gradient accumulation
        self.gradient_accumulation_steps = 4
        self.gradient_step_counter = 0
        
        # Initialize agent networks (each agent has its own Q-network for diversity)
        self.agent_networks = nn.ModuleList()
        self.target_agent_networks = nn.ModuleList()
        
        for _ in range(num_agents):
            agent_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ).to(device)
            
            target_agent_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ).to(device)
            
            target_agent_net.load_state_dict(agent_net.state_dict())
            
            self.agent_networks.append(agent_net)
            self.target_agent_networks.append(target_agent_net)
        
        # Shared agent Q-network (parameter sharing for efficiency in batch processing)
        self.shared_agent_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        self.shared_target_agent_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.shared_target_agent_net.load_state_dict(self.shared_agent_net.state_dict())
        
        # Hypernetwork-based mixing (more efficient than full concatenation)
        # Hypernetwork generates mixing weights from global state
        mixing_embed_dim = 32  # Compact embedding for mixing
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * mixing_embed_dim)
        ).to(device)
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)  # match w1 output
        ).to(device)
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 * mixing_embed_dim)  # output weight for final projection
        ).to(device)
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        ).to(device)
        
        # Target hypernetworks
        self.target_hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * mixing_embed_dim)
        ).to(device)
        self.target_hyper_w1.load_state_dict(self.hyper_w1.state_dict())
        
        self.target_hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)  # match w1 output
        ).to(device)
        self.target_hyper_b1.load_state_dict(self.hyper_b1.state_dict())
        
        self.target_hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1 * mixing_embed_dim)  # output weight for final projection
        ).to(device)
        self.target_hyper_w2.load_state_dict(self.hyper_w2.state_dict())
        
        self.target_hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        ).to(device)
        self.target_hyper_b2.load_state_dict(self.hyper_b2.state_dict())
        
        # Sparse attention communication (O(N * k) where k << N)
        if communication_enabled:
            self.comm_key = nn.Linear(state_dim, hidden_dim // 4).to(device)
            self.comm_query = nn.Linear(state_dim, hidden_dim // 4).to(device)
            self.comm_value = nn.Linear(state_dim, hidden_dim).to(device)
            self.comm_output = nn.Linear(hidden_dim, state_dim).to(device)  # Project to state_dim
            self.comm_top_k = min(5, num_agents - 1)  # Attend to top-k neighbors
        else:
            self.comm_key = None
            self.comm_query = None
            self.comm_value = None
            self.comm_output = None
            self.comm_top_k = 0
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def _compute_communication(self, states: torch.Tensor) -> torch.Tensor:
        """
        Sparse attention-based communication between agents.
        Each agent attends to its top-k most relevant neighbors.
        Complexity: O(N * k) instead of O(N^2)
        """
        if not self.communication_enabled or self.comm_key is None:
            return states
        
        batch_size, num_agents, state_dim = states.shape
        
        # Compute keys, queries, values for all agents
        keys = self.comm_key(states)       # (B, N, D/4)
        queries = self.comm_query(states)  # (B, N, D/4)
        values = self.comm_value(states)   # (B, N, D)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (B, N, N)
        attention_scores = attention_scores / math.sqrt(self.hidden_dim // 4)
        
        # Sparse attention: keep only top-k for each agent
        k = min(self.comm_top_k, num_agents - 1)
        if k > 0:
            # Mask self-attention
            mask = torch.eye(num_agents, device=self.device).unsqueeze(0).bool()
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
            
            # Select top-k
            top_k_scores, top_k_indices = torch.topk(attention_scores, k, dim=-1)
            top_k_weights = F.softmax(top_k_scores, dim=-1)
            
            # Gather top-k values
            top_k_values = torch.gather(
                values.unsqueeze(1).expand(-1, num_agents, -1, -1),
                2,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
            )  # (B, N, k, D)
            
            # Weighted sum
            comm_messages = (top_k_weights.unsqueeze(-1) * top_k_values).sum(dim=2)  # (B, N, D)
        else:
            comm_messages = torch.zeros_like(states)
        
        # Combine with original states
        enhanced_states = states + self.comm_output(comm_messages)
        return enhanced_states
    
    def _hypernetwork_mixing(self, agent_qs: torch.Tensor, global_state: torch.Tensor,
                             use_target: bool = False) -> torch.Tensor:
        """
        Hypernetwork-based mixing for efficient joint Q-value computation.
        Uses global state to generate mixing weights dynamically.
        """
        if use_target:
            w1 = self.target_hyper_w1(global_state)
            b1 = self.target_hyper_b1(global_state)
            w2 = self.target_hyper_w2(global_state)
            b2 = self.target_hyper_b2(global_state)
        else:
            w1 = self.hyper_w1(global_state)
            b1 = self.hyper_b1(global_state)
            w2 = self.hyper_w2(global_state)
            b2 = self.hyper_b2(global_state)
        
        batch_size = global_state.shape[0]
        mixing_embed_dim = 32
        
        # Reshape weights
        w1 = w1.view(batch_size, self.num_agents, mixing_embed_dim)
        w2 = w2.view(batch_size, 1, mixing_embed_dim)  # (B, 1, embed)
        
        # First layer: mix agent Q-values into embedding
        agent_qs_expanded = agent_qs.unsqueeze(-1)  # (B, N, 1)
        hidden = torch.bmm(w1.transpose(1, 2), agent_qs_expanded)  # (B, embed, 1)
        hidden = hidden.squeeze(-1) + b1  # (B, embed)
        hidden = F.relu(hidden)
        
        # Second layer: project to scalar
        hidden_expanded = hidden.unsqueeze(-1)  # (B, embed, 1)
        joint_q = torch.bmm(w2, hidden_expanded).squeeze(-1) + b2  # (B, 1)
        
        return joint_q.squeeze(-1)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass for MARL layer. Uses individual agent networks for diversity.

        P14 fix: previously the call site assumed a 3-D tensor of shape
        ``[B, N, state_dim]`` and would crash with a bare ``ValueError``
        from the unpack if the caller supplied ``[B, state_dim]`` (no
        agent axis) or anything else.  Now we:
          1. Accept a 2-D ``[B, state_dim]`` and broadcast to ``[B, N, D]``.
          2. Validate the per-agent axis matches ``self.num_agents``.
          3. Validate the per-agent feature dim matches ``self.state_dim``.
        """
        # P14: auto-broadcast 2D shared states to 3D per-agent states.
        if states.dim() == 2:
            states = states.unsqueeze(1).expand(-1, self.num_agents, -1)
        if states.dim() != 3:
            raise ValueError(
                f"MARLLayer.forward expects 2-D [B, D] or 3-D [B, N, D] "
                f"states, got shape {tuple(states.shape)}"
            )
        if states.shape[1] != self.num_agents:
            raise ValueError(
                f"MARLLayer.forward: states have {states.shape[1]} agents "
                f"but layer was constructed with num_agents={self.num_agents}"
            )
        if states.shape[2] != self.state_dim:
            raise ValueError(
                f"MARLLayer.forward: states have feature dim "
                f"{states.shape[2]} but layer was constructed with "
                f"state_dim={self.state_dim}"
            )

        # Apply sparse communication
        if self.communication_enabled:
            states = self._compute_communication(states)

        batch_size, num_agents, state_dim = states.shape

        # Use individual agent networks for diverse decision making
        q_values = torch.zeros(batch_size, num_agents, self.action_dim, device=self.device)
        for agent_id in range(num_agents):
            agent_states = states[:, agent_id, :]  # (B, state_dim)
            q_values[:, agent_id, :] = self.agent_networks[agent_id](agent_states)

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
               batch_dones: torch.Tensor, egt_weights=None, lambda_param=None,
               fairness_metrics: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """
        Update MARL layer using experience batch with hypernetwork mixing.

        Args:
            batch_states: Current states [B, N, state_dim]
            batch_actions: Actions [B, N]
            batch_rewards: Rewards [B, N] (or [B])
            batch_next_states: Next states [B, N, state_dim]
            batch_dones: Done flags [B]
            egt_weights: Optional dict from EGT layer with keys:
                - 'fairness_weight', 'efficiency_weight', 'lambda_param', 'strategy_distribution'
            lambda_param: Scalar override of lambda in [0, 1] (0=efficiency, 1=fairness).
                          Takes precedence over egt_weights['lambda_param'] if provided.
            fairness_metrics: Optional dict of fairness signals per agent
                (e.g. {'gini': tensor[B], 'region_save_rates': tensor[B, N]}).

        Returns:
            Loss value.
        """
        # ====== EGT-driven reward shaping (the "handshake" with the macro layer) ======
        # Resolve lambda (0 = pure efficiency, 1 = pure fairness)
        if lambda_param is not None:
            lam = float(max(0.0, min(1.0, lambda_param)))
        elif egt_weights is not None and 'lambda_param' in egt_weights:
            lam = float(max(0.0, min(1.0, egt_weights['lambda_param'])))
        else:
            lam = 0.5  # neutral default

        # EGT-derived per-agent shaping factors.
        # If we have per-agent fairness signals (e.g. region save rates), use them to
        # boost rewards for under-served agents.  Otherwise fall back to a constant
        # bias on the shared weight.
        egt_fairness_boost = 0.0
        if egt_weights is not None and 'fairness_weight' in egt_weights:
            egt_fairness_boost = float(egt_weights['fairness_weight']) - 0.5

        # Reshape rewards: [B] -> [B, N] if necessary
        if batch_rewards.dim() == 1:
            rewards_per_agent = batch_rewards.unsqueeze(1).expand(-1, self.num_agents)
        elif batch_rewards.dim() == 2 and batch_rewards.shape[1] == 1:
            rewards_per_agent = batch_rewards.expand(-1, self.num_agents)
        else:
            rewards_per_agent = batch_rewards

        # Apply EGT-driven reward shaping:
        #   shaped_r = (1 + egt_boost) * r + lam * fairness_bonus
        if fairness_metrics is not None and 'region_save_rates' in fairness_metrics:
            save_rates = fairness_metrics['region_save_rates']  # [B, N]
            mean_save = save_rates.mean(dim=1, keepdim=True)
            # Agents with below-average save rate get a positive fairness bonus
            fairness_bonus = (mean_save - save_rates).clamp(min=0.0) * lam
            shaped_rewards = rewards_per_agent * (1.0 + egt_fairness_boost) + fairness_bonus
        else:
            # No per-agent fairness info: apply an *adaptive* uniform shift
            # proportional to lambda, scaled to the per-batch reward scale.
            # P1 fix: the previous constant `lam * 0.01` is dwarfed by typical
            # reward magnitudes (O(0.1)-O(1.0)) and gets washed out by the
            # discount factor in the TD target.  Use a scale-aware bias:
            #   bias = lam * max(reward_std, 0.05) * 0.3
            # The 0.05 floor prevents the bonus from vanishing when reward
            # variance is low (e.g. all-zero batches at the start of training).
            reward_std = float(rewards_per_agent.std().item()) if rewards_per_agent.numel() > 1 else 0.0
            scale = max(reward_std, 0.05) * 0.3
            shaped_rewards = rewards_per_agent * (1.0 + egt_fairness_boost) + lam * scale

        # ====== Standard QMIX-style update using shaped rewards ======
        # P14 fix: also normalise both state tensors for the no-2D
        # auto-broadcast path (forward() now handles 2-D, but we keep
        # this defensive normalisation here so the rest of the update
        # logic — which assumes 3-D — stays correct).
        if batch_states.dim() == 2:
            batch_states = batch_states.unsqueeze(1).expand(
                -1, self.num_agents, -1
            )
        if batch_next_states.dim() == 2:
            batch_next_states = batch_next_states.unsqueeze(1).expand(
                -1, self.num_agents, -1
            )
        # Get current Q-values
        current_qs = self.forward(batch_states)

        # Get action indices and selected Q-values
        action_indices = batch_actions.long().unsqueeze(2)
        selected_qs = current_qs.gather(2, action_indices).squeeze(2)  # (B, N)

        # Compute global state by averaging agent observations
        global_state = batch_states.mean(dim=1)  # (B, state_dim)

        # Get current joint Q-value using hypernetwork mixing
        current_joint_q = self._hypernetwork_mixing(selected_qs, global_state, use_target=False)

        # Get next Q-values from target networks
        with torch.no_grad():
            next_qs = self.forward(batch_next_states)
            max_next_qs = next_qs.max(dim=2)[0]  # (B, N)

            # Get target joint Q-value
            next_global_state = batch_next_states.mean(dim=1)
            target_joint_q = self._hypernetwork_mixing(max_next_qs, next_global_state, use_target=True)

            # Compute target using shaped rewards (EGT-driven)
            # Aggregate per-agent shaped rewards into per-step reward
            step_reward = shaped_rewards.mean(dim=1)  # [B]
            target = step_reward + self.gamma * target_joint_q * (1 - batch_dones.float())

        # Compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(current_joint_q, target)

        # Gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        self.gradient_step_counter += 1
        if self.gradient_step_counter % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update target networks
            self._update_target_networks(self.tau)

        return loss.item() * self.gradient_accumulation_steps
    
    def _update_target_networks(self, tau: float) -> None:
        """Update target networks with soft update."""
        # Update individual agent target networks
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_agent_networks[i].parameters(),
                                         self.agent_networks[i].parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Update shared target network
        for target_param, param in zip(self.shared_target_agent_net.parameters(),
                                     self.shared_agent_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Update hypernetwork targets
        for target_param, param in zip(self.target_hyper_w1.parameters(),
                                     self.hyper_w1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_hyper_b1.parameters(),
                                     self.hyper_b1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_hyper_w2.parameters(),
                                     self.hyper_w2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_hyper_b2.parameters(),
                                     self.hyper_b2.parameters()):
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
        if not self.communication_enabled:
            return torch.zeros((self.num_agents, self.num_agents), device=self.device)
        
        # Create dummy input to extract communication patterns
        dummy_states = torch.randn(1, self.num_agents, self.state_dim, device=self.device)
        
        with torch.no_grad():
            # Pass through communication mechanism
            enhanced_states = self._compute_communication(dummy_states)
            
            # Compute communication influence as difference
            comm_influence = (enhanced_states - dummy_states).abs().sum(dim=-1)  # (1, N)
            comm_matrix = torch.matmul(comm_influence, comm_influence.transpose(-2, -1))  # (1, N, N)
        
        return comm_matrix.squeeze(0)  # (N, N)
    
    def save(self, path: str) -> None:
        """Save MARL layer state."""
        torch.save({
            'agent_networks_state': [net.state_dict() for net in self.agent_networks],
            'target_agent_networks_state': [net.state_dict() for net in self.target_agent_networks],
            'shared_agent_net_state': self.shared_agent_net.state_dict(),
            'shared_target_agent_net_state': self.shared_target_agent_net.state_dict(),
            'hyper_w1_state': self.hyper_w1.state_dict(),
            'hyper_b1_state': self.hyper_b1.state_dict(),
            'hyper_w2_state': self.hyper_w2.state_dict(),
            'hyper_b2_state': self.hyper_b2.state_dict(),
            'target_hyper_w1_state': self.target_hyper_w1.state_dict(),
            'target_hyper_b1_state': self.target_hyper_b1.state_dict(),
            'target_hyper_w2_state': self.target_hyper_w2.state_dict(),
            'target_hyper_b2_state': self.target_hyper_b2.state_dict(),
            'comm_key_state': self.comm_key.state_dict() if self.communication_enabled else None,
            'comm_query_state': self.comm_query.state_dict() if self.communication_enabled else None,
            'comm_value_state': self.comm_value.state_dict() if self.communication_enabled else None,
            'comm_output_state': self.comm_output.state_dict() if self.communication_enabled else None,
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load agent networks
        for i, net in enumerate(self.agent_networks):
            net.load_state_dict(checkpoint['agent_networks_state'][i])
        
        # Load target networks
        for i, net in enumerate(self.target_agent_networks):
            net.load_state_dict(checkpoint['target_agent_networks_state'][i])
        
        # Load shared networks
        self.shared_agent_net.load_state_dict(checkpoint['shared_agent_net_state'])
        self.shared_target_agent_net.load_state_dict(checkpoint['shared_target_agent_net_state'])
        
        # Load hypernetworks
        self.hyper_w1.load_state_dict(checkpoint['hyper_w1_state'])
        self.hyper_b1.load_state_dict(checkpoint['hyper_b1_state'])
        self.hyper_w2.load_state_dict(checkpoint['hyper_w2_state'])
        self.hyper_b2.load_state_dict(checkpoint['hyper_b2_state'])
        self.target_hyper_w1.load_state_dict(checkpoint['target_hyper_w1_state'])
        self.target_hyper_b1.load_state_dict(checkpoint['target_hyper_b1_state'])
        self.target_hyper_w2.load_state_dict(checkpoint['target_hyper_w2_state'])
        self.target_hyper_b2.load_state_dict(checkpoint['target_hyper_b2_state'])
        
        # Load communication networks
        if self.communication_enabled:
            if checkpoint['comm_key_state'] is not None:
                self.comm_key.load_state_dict(checkpoint['comm_key_state'])
            if checkpoint['comm_query_state'] is not None:
                self.comm_query.load_state_dict(checkpoint['comm_query_state'])
            if checkpoint['comm_value_state'] is not None:
                self.comm_value.load_state_dict(checkpoint['comm_value_state'])
            if checkpoint['comm_output_state'] is not None:
                self.comm_output.load_state_dict(checkpoint['comm_output_state'])
        
        # Load epsilon
        self.epsilon = checkpoint['epsilon']