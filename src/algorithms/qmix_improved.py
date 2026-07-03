"""
Improved QMIX algorithm for EGT-MARL disaster resource allocation.

This module implements the enhanced QMIX algorithm with:
1. Hierarchical action space for different agent types
2. Enhanced reward structure with fairness and efficiency components
3. Attention mechanism for better credit assignment
4. Multi-objective optimization support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import math


class HierarchicalActionSpace:
    """
    Hierarchical action space for different agent types.
    
    Levels:
    1. Strategic level: High-level objectives (efficiency, fairness, robustness)
    2. Tactical level: Resource allocation and task selection
    3. Operational level: Movement and immediate actions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Define action dimensions for each level and agent type
        self.action_dims = {
            'drone': {
                'strategic': 4,  # Efficiency, Fairness, Reconnaissance, Communication
                'tactical': 8,   # Resource allocation, task prioritization
                'operational': 12  # Movement, sensor usage, communication actions
            },
            'vehicle': {
                'strategic': 3,  # Patient care, Transport, Resource management
                'tactical': 10,  # Patient triage, route planning, resource allocation
                'operational': 8   # Driving, loading/unloading, treatment
            },
            'personnel': {
                'strategic': 4,  # Treatment focus, Resource allocation, Coordination, Expansion
                'tactical': 12,  # Patient management, staff allocation, resource distribution
                'operational': 6   # Facility operations, communication, logistics
            }
        }
        
        # Action hierarchies
        self.hierarchies = {
            'drone': ['strategic', 'tactical', 'operational'],
            'vehicle': ['strategic', 'tactical', 'operational'],
            'personnel': ['strategic', 'tactical', 'operational']
        }
    
    def get_total_dim(self, agent_type: str) -> int:
        """Get total action dimension for agent type."""
        total = 0
        for level in self.hierarchies[agent_type]:
            total += self.action_dims[agent_type][level]
        return total
    
    def decode_action(self, agent_type: str, action_vector: np.ndarray) -> Dict[str, Any]:
        """Decode flat action vector into hierarchical actions."""
        actions = {}
        start_idx = 0
        
        for level in self.hierarchies[agent_type]:
            dim = self.action_dims[agent_type][level]
            level_actions = action_vector[start_idx:start_idx + dim]
            
            # Apply softmax to get probability distribution
            if dim > 1:
                probs = F.softmax(torch.FloatTensor(level_actions), dim=0).numpy()
                selected = np.argmax(probs)
            else:
                selected = 0
            
            actions[level] = {
                'vector': level_actions,
                'selected': selected,
                'probs': probs if dim > 1 else np.array([1.0])
            }
            
            start_idx += dim
        
        return actions
    
    def encode_action(self, agent_type: str, actions: Dict[str, Any]) -> np.ndarray:
        """Encode hierarchical actions into flat vector."""
        action_vector = []
        
        for level in self.hierarchies[agent_type]:
            if level in actions:
                # One-hot encode selected action
                dim = self.action_dims[agent_type][level]
                one_hot = np.zeros(dim)
                selected = actions[level].get('selected', 0)
                if 0 <= selected < dim:
                    one_hot[selected] = 1.0
                action_vector.extend(one_hot)
            else:
                # Default to zero vector
                dim = self.action_dims[agent_type][level]
                action_vector.extend(np.zeros(dim))
        
        return np.array(action_vector)


class EnhancedRewardStructure:
    """
    Enhanced reward structure with multiple components.
    
    Components:
    1. Efficiency reward: Based on survivors saved and response time
    2. Fairness reward: Based on equitable resource distribution
    3. Robustness reward: Based on system stability under stress
    4. Communication reward: Based on effective information sharing
    5. Resource efficiency reward: Based on optimal resource usage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Reward weights (can be dynamically adjusted)
        # Critical fix: add reputation + anti_spoofing_penalty to match
        # paper section 4.4 (Bayesian truthfulness + incentive compatibility).
        # Previously missing these 2 components, making anti-spoofing rewards
        # a no-op in the EnhancedRewardStructure.
        self.weights = {
            'efficiency': 0.30,
            'fairness': 0.20,
            'robustness': 0.10,
            'communication': 0.10,
            'resource_efficiency': 0.10,
            'reputation': 0.10,
            'anti_spoofing_penalty': 0.10
        }
        
        # Normalization factors
        self.normalization = {
            'survivors': 100.0,  # Normalize to 100 survivors
            'response_time': 60.0,  # Normalize to 60 minutes
            'gini_coefficient': 1.0,  # Already normalized
            'communication_overhead': 1000.0,  # Normalize to 1000 bytes
            'resource_utilization': 1.0  # Already normalized
        }
    
    def calculate_reward(self, 
                        metrics: Dict[str, Any],
                        previous_metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculate enhanced reward from metrics."""
        rewards = {}
        
        # 1. Efficiency reward
        efficiency_reward = self._calculate_efficiency_reward(metrics, previous_metrics)
        rewards['efficiency'] = efficiency_reward
        
        # 2. Fairness reward
        fairness_reward = self._calculate_fairness_reward(metrics, previous_metrics)
        rewards['fairness'] = fairness_reward
        
        # 3. Robustness reward
        robustness_reward = self._calculate_robustness_reward(metrics, previous_metrics)
        rewards['robustness'] = robustness_reward
        
        # 4. Communication reward
        communication_reward = self._calculate_communication_reward(metrics, previous_metrics)
        rewards['communication'] = communication_reward
        
        # 5. Resource efficiency reward
        resource_reward = self._calculate_resource_reward(metrics, previous_metrics)
        rewards['resource_efficiency'] = resource_reward

        # 6. Reputation reward (paper section 4.4)
        reputation_reward = self._calculate_reputation_reward(metrics, previous_metrics)
        rewards['reputation'] = reputation_reward

        # 7. Anti-spoofing penalty (paper section 4.4)
        anti_spoofing_penalty = self._calculate_anti_spoofing_penalty(metrics, previous_metrics)
        rewards['anti_spoofing_penalty'] = anti_spoofing_penalty
        
        # Weighted sum
        total_reward = sum(weight * rewards[component] 
                          for component, weight in self.weights.items())
        
        # Clip reward to reasonable range
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return total_reward, rewards
    
    def _calculate_efficiency_reward(self, 
                                   metrics: Dict[str, Any],
                                   previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate efficiency component of reward."""
        # Survivor-based reward
        survivors = metrics.get('total_survivors', 0)
        survivor_reward = survivors / self.normalization['survivors']
        
        # Response time penalty (lower is better)
        response_time = metrics.get('mean_response_time', 0)
        response_penalty = -response_time / self.normalization['response_time']
        
        # Task completion reward
        completion_rate = metrics.get('tasks_completion_rate', 0)
        completion_reward = completion_rate
        
        # Combine with weights
        efficiency_reward = (
            0.5 * survivor_reward +
            0.3 * response_penalty +
            0.2 * completion_reward
        )
        
        # Improvement bonus if previous metrics available
        if previous_metrics:
            prev_survivors = previous_metrics.get('total_survivors', 0)
            improvement = survivors - prev_survivors
            if improvement > 0:
                efficiency_reward += 0.1 * (improvement / self.normalization['survivors'])
        
        return efficiency_reward
    
    def _calculate_fairness_reward(self,
                                 metrics: Dict[str, Any],
                                 previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate fairness component of reward."""
        # Gini coefficient (0 is perfectly equal, 1 is perfectly unequal)
        gini = metrics.get('gini_coefficient', 0.5)
        gini_reward = 1.0 - gini  # Higher reward for lower inequality
        
        # Max-min fairness
        max_min = metrics.get('max_min_fairness', 0.5)
        
        # Coefficient of variation (lower is better)
        cv = metrics.get('coefficient_of_variation', 0.5)
        cv_reward = 1.0 / (1.0 + cv)
        
        # Combine with weights
        fairness_reward = (
            0.4 * gini_reward +
            0.4 * max_min +
            0.2 * cv_reward
        )
        
        return fairness_reward
    
    def _calculate_robustness_reward(self,
                                   metrics: Dict[str, Any],
                                   previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate robustness component of reward."""
        # System stability
        stability = metrics.get('stability_index', 0.5) / 10.0  # Normalize
        
        # Fault tolerance
        fault_tolerance = metrics.get('fault_tolerance', 0.5)
        
        # Performance under stress
        stress_performance = metrics.get('performance_under_stress', 0.5)
        
        # Recovery capability
        recovery = 1.0 - min(1.0, metrics.get('recovery_time', 0) / 100.0)
        
        # Combine with weights
        robustness_reward = (
            0.3 * stability +
            0.3 * fault_tolerance +
            0.2 * stress_performance +
            0.2 * recovery
        )
        
        return robustness_reward
    
    def _calculate_communication_reward(self,
                                      metrics: Dict[str, Any],
                                      previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate communication component of reward."""
        # Communication effectiveness (information quality)
        comm_effectiveness = metrics.get('communication_effectiveness', 0.5)
        
        # Overhead penalty (lower is better)
        overhead = metrics.get('communication_overhead', 0)
        overhead_penalty = -overhead / self.normalization['communication_overhead']
        
        # Latency penalty
        latency = metrics.get('communication_latency', 0)
        latency_penalty = -min(1.0, latency / 10.0)  # Normalize to 10 seconds
        
        # Combine with weights
        communication_reward = (
            0.5 * comm_effectiveness +
            0.3 * overhead_penalty +
            0.2 * latency_penalty
        )
        
        return communication_reward
    
    def _calculate_resource_reward(self,
                                 metrics: Dict[str, Any],
                                 previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate resource efficiency component of reward."""
        # Resource utilization (balanced, not too high or too low)
        utilization = metrics.get('overall_resource_utilization', 0.5)
        # Target utilization around 70%
        utilization_reward = 1.0 - abs(utilization - 0.7)
        
        # Survivors per resource (higher is better)
        survivors_per_resource = metrics.get('survivors_per_resource', 0)
        if survivors_per_resource > 0:
            resource_efficiency = min(1.0, survivors_per_resource / 10.0)  # Normalize
        else:
            resource_efficiency = 0.0
        
        # Waste penalty
        waste = metrics.get('resource_waste', 0)
        waste_penalty = -min(1.0, waste / 100.0)  # Normalize
        
        # Combine with weights
        resource_reward = (
            0.4 * utilization_reward +
            0.4 * resource_efficiency +
            0.2 * waste_penalty
        )
        
        return resource_reward

    def _calculate_reputation_reward(self,
                                     metrics: Dict[str, Any],
                                     previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate reputation component of reward (paper section 4.4).

        Higher average reputation -> higher reward, incentivizing honest behavior.
        Reputation is maintained by the anti-spoofing / reputation manager.
        """
        avg_reputation = metrics.get('avg_reputation', 0.5)
        # Linear mapping: reputation 0 -> reward 0, reputation 1 -> reward 1
        reputation_reward = float(avg_reputation)
        # Bonus for reputation improvement
        if previous_metrics:
            prev_rep = previous_metrics.get('avg_reputation', 0.5)
            improvement = avg_reputation - prev_rep
            if improvement > 0:
                reputation_reward += 0.1 * improvement
        return reputation_reward

    def _calculate_anti_spoofing_penalty(self,
                                          metrics: Dict[str, Any],
                                          previous_metrics: Optional[Dict[str, Any]]) -> float:
        """Calculate anti-spoofing penalty component (paper section 4.4).

        More detected strategic behavior -> more negative reward.
        This closes the incentive-compatibility loop: agents that try to
        game the system get penalized in reward, not just in internal state.
        """
        spoofing_rate = metrics.get('spoofing_rate', 0.0)
        num_strategic = metrics.get('num_strategic_agents', 0)
        # Penalty proportional to spoofing rate and count
        penalty = -1.0 * float(spoofing_rate) - 0.05 * float(num_strategic)
        return penalty

    def update_weights(self,
                      performance_feedback: Dict[str, float],
                      learning_rate: float = 0.01):
        """Dynamically update reward weights based on performance."""
        for component in self.weights:
            if component in performance_feedback:
                # Adjust weight based on performance
                # Higher performance -> increase weight slightly
                adjustment = learning_rate * performance_feedback[component]
                self.weights[component] += adjustment
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            for component in self.weights:
                self.weights[component] /= total


class AttentionMixingNetwork(nn.Module):
    """
    Attention-based mixing network for QMIX.
    
    Improvements over standard QMIX:
    1. Multi-head attention for better credit assignment
    2. Hierarchical mixing for different agent types
    3. Dynamic weight adjustment based on state
    """
    
    def __init__(self, 
                 state_dim: int,
                 num_agents: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Hypernetworks for mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Linear(hidden_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                agent_qs: torch.Tensor,  # [batch, num_agents]
                states: torch.Tensor) -> torch.Tensor:  # [batch, state_dim]
        """
        Compute mixed Q-value using attention mechanism.
        
        Args:
            agent_qs: Individual Q-values for each agent
            states: Global state information
            
        Returns:
            Mixed Q-value for the joint action
        """
        batch_size = agent_qs.shape[0]
        
        # Encode state
        state_emb = self.state_encoder(states)  # [batch, hidden_dim]
        
        # Prepare agent Q-values for attention
        agent_qs_reshaped = agent_qs.unsqueeze(-1)  # [batch, num_agents, 1]
        agent_qs_expanded = agent_qs_reshaped.expand(-1, -1, self.hidden_dim)
        
        # Apply attention layers
        x = agent_qs_expanded
        for attn_layer, layer_norm, ffn in zip(self.attention_layers, 
                                               self.layer_norms, 
                                               self.ffns):
            # Self-attention
            attn_output, _ = attn_layer(x, x, x)
            x = layer_norm(x + attn_output)
            
            # Feed-forward
            ff_output = ffn(x)
            x = layer_norm(x + ff_output)
        
        # Get mixing weights from hypernetworks
        w1 = torch.abs(self.hyper_w1(state_emb))
        w1 = w1.view(batch_size, self.hidden_dim, self.hidden_dim)
        
        b1 = self.hyper_b1(state_emb).view(batch_size, 1, self.hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(state_emb))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        b2 = self.hyper_b2(state_emb).view(batch_size, 1, 1)
        
        # First mixing layer
        hidden = F.elu(torch.bmm(x, w1) + b1)
        
        # Second mixing layer
        q_total = torch.bmm(hidden, w2) + b2
        
        # Sum over agents to get total Q value
        return q_total.sum(dim=1)
    
    def compute_attention_weights(self,
                                 agent_qs: torch.Tensor,
                                 states: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for interpretability."""
        batch_size = agent_qs.shape[0]
        
        # Encode state
        state_emb = self.state_encoder(states)
        
        # Prepare for attention
        agent_qs_reshaped = agent_qs.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        
        # Get attention weights from first layer
        _, attention_weights = self.attention_layers[0](
            agent_qs_reshaped, agent_qs_reshaped, agent_qs_reshaped
        )
        
        # Average attention weights across heads
        attention_weights = attention_weights.mean(dim=1)  # [batch, num_agents, num_agents]
        
        return attention_weights


class ImprovedQMIXAgent(nn.Module):
    """
    Improved QMIX agent with hierarchical action space and enhanced features.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 agent_type: str = 'drone',
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.agent_type = agent_type
        
        if config is None:
            config = {}
        self.config = config
        
        # Hierarchical action space
        self.action_space = HierarchicalActionSpace(config)
        
        # Enhanced reward structure
        self.reward_structure = EnhancedRewardStructure(config)
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Target Q-network
        self.target_q_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Hierarchical policy network
        self.hierarchical_policy = self._build_hierarchical_policy()
        
        # Value network for baseline
        self.value_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=config.get('learning_rate', 0.0001))
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
    def _build_hierarchical_policy(self) -> nn.ModuleDict:
        """Build hierarchical policy network for different action levels."""
        policy = nn.ModuleDict()
        
        # Strategic level policy
        policy['strategic'] = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 
                     self.action_space.action_dims[self.agent_type]['strategic'])
        )
        
        # Tactical level policy (conditioned on strategic action)
        tactical_input_dim = self.obs_dim + self.action_space.action_dims[self.agent_type]['strategic']
        policy['tactical'] = nn.Sequential(
            nn.Linear(tactical_input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2,
                     self.action_space.action_dims[self.agent_type]['tactical'])
        )
        
        # Operational level policy (conditioned on strategic and tactical actions)
        operational_input_dim = (self.obs_dim + 
                               self.action_space.action_dims[self.agent_type]['strategic'] +
                               self.action_space.action_dims[self.agent_type]['tactical'])
        policy['operational'] = nn.Sequential(
            nn.Linear(operational_input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2,
                     self.action_space.action_dims[self.agent_type]['operational'])
        )
        
        return policy
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        return self.q_network(obs)
    
    def get_action(self, 
                  obs: np.ndarray,
                  available_actions: Optional[List[int]] = None,
                  training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """Select action using epsilon-greedy policy with hierarchical structure."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random hierarchical action
            action_info = self._sample_random_hierarchical_action()
            # Convert hierarchical action to flat index
            flat_action = self._hierarchical_to_flat(action_info)
        else:
            # Exploitation: hierarchical policy
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                
                if available_actions is not None:
                    # Mask unavailable actions
                    mask = torch.ones_like(q_values) * -1e10
                    mask[0, available_actions] = 0
                    q_values = q_values + mask
                
                # Get hierarchical action
                action_info = self._get_hierarchical_action(obs_tensor)
                flat_action = torch.argmax(q_values).item()
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return flat_action, action_info
    
    def _sample_random_hierarchical_action(self) -> Dict[str, Any]:
        """Sample random hierarchical action."""
        action_info = {}
        
        for level in self.action_space.hierarchies[self.agent_type]:
            dim = self.action_space.action_dims[self.agent_type][level]
            selected = np.random.randint(dim)
            action_info[level] = {
                'selected': selected,
                'probs': np.ones(dim) / dim
            }
        
        return action_info
    
    def _get_hierarchical_action(self, obs: torch.Tensor) -> Dict[str, Any]:
        """Get hierarchical action using policy networks."""
        action_info = {}
        
        # Strategic level
        strategic_logits = self.hierarchical_policy['strategic'](obs)
        strategic_probs = F.softmax(strategic_logits, dim=-1)
        strategic_action = torch.multinomial(strategic_probs, 1).item()
        
        action_info['strategic'] = {
            'selected': strategic_action,
            'probs': strategic_probs.squeeze().detach().numpy()
        }
        
        # Tactical level (conditioned on strategic action)
        strategic_one_hot = F.one_hot(torch.tensor([strategic_action]), 
                                     self.action_space.action_dims[self.agent_type]['strategic']).float()
        tactical_input = torch.cat([obs, strategic_one_hot], dim=-1)
        
        tactical_logits = self.hierarchical_policy['tactical'](tactical_input)
        tactical_probs = F.softmax(tactical_logits, dim=-1)
        tactical_action = torch.multinomial(tactical_probs, 1).item()
        
        action_info['tactical'] = {
            'selected': tactical_action,
            'probs': tactical_probs.squeeze().detach().numpy()
        }
        
        # Operational level (conditioned on strategic and tactical actions)
        tactical_one_hot = F.one_hot(torch.tensor([tactical_action]),
                                    self.action_space.action_dims[self.agent_type]['tactical']).float()
        operational_input = torch.cat([obs, strategic_one_hot, tactical_one_hot], dim=-1)
        
        operational_logits = self.hierarchical_policy['operational'](operational_input)
        operational_probs = F.softmax(operational_logits, dim=-1)
        operational_action = torch.multinomial(operational_probs, 1).item()
        
        action_info['operational'] = {
            'selected': operational_action,
            'probs': operational_probs.squeeze().detach().numpy()
        }
        
        return action_info
    
    def _hierarchical_to_flat(self, action_info: Dict[str, Any]) -> int:
        """Convert hierarchical action to flat action index."""
        # Encode using action space
        flat_vector = self.action_space.encode_action(self.agent_type, action_info)
        
        # Find closest Q-value index (simplified)
        # In practice, would need mapping between hierarchical and flat actions
        return np.random.randint(self.action_dim)
    
    def update(self,
               batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
               mixing_network: AttentionMixingNetwork,
               target_mixing_network: AttentionMixingNetwork) -> Dict[str, float]:
        """Update agent using improved QMIX loss."""
        if len(batch) == 0:
            return {'q_loss': 0.0, 'value_loss': 0.0, 'epsilon': self.epsilon}

        # Unpack batch
        obs, actions, rewards, next_obs, dones = zip(*batch)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs))
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)
        rewards_tensor = torch.FloatTensor(rewards)
        next_obs_tensor = torch.FloatTensor(np.array(next_obs))
        dones_tensor = torch.FloatTensor(dones)

        # M4 audit fix: if ImprovedQMIX.update() stashed shaped rewards
        # computed via EnhancedRewardStructure, use them in place of the raw
        # rewards.  This makes the 5-component structure actually drive the
        # gradient.
        shaped = getattr(self, '_last_shaped_rewards', None)
        if shaped is not None and len(shaped) == len(rewards):
            rewards_tensor = torch.FloatTensor(shaped)

        # Current Q-values
        current_q_values = self.q_network(obs_tensor)
        current_q = current_q_values.gather(1, actions_tensor).squeeze()
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_obs_tensor)
            next_q = next_q_values.max(1)[0]
        
        # Expected Q-values
        expected_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
        
        # Q-loss (Huber loss for robustness)
        q_loss = F.smooth_l1_loss(current_q, expected_q)
        
        # Value loss (for baseline)
        values = self.value_network(obs_tensor).squeeze()
        value_loss = F.mse_loss(values, expected_q.detach())
        
        # Total loss
        total_loss = q_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
        
        return {
            'q_loss': q_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'epsilon': self.epsilon
        }

    def compute_auxiliary_loss(self,
                                batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]
                                ) -> Dict[str, float]:
        """
        Compute per-agent auxiliary losses (NO gradient / NO optimizer step).

        This is the monitoring-only counterpart of `update()`.  The real
        end-to-end QMIX update happens inside
        ``ImprovedQMIX._update_mixing_network``; this method exists purely
        so that per-agent statistics (q_loss, value_loss) can be logged.

        Args:
            batch: List of (obs, action, reward, next_obs, done) tuples.

        Returns:
            Dict with 'q_loss', 'value_loss', 'total_loss', 'epsilon'.
        """
        if len(batch) == 0:
            return {
                'q_loss': 0.0,
                'value_loss': 0.0,
                'total_loss': 0.0,
                'epsilon': float(self.epsilon),
            }
        try:
            obs, actions, rewards, next_obs, dones = zip(*batch)
            obs_tensor = torch.FloatTensor(np.array(obs))
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards)
            next_obs_tensor = torch.FloatTensor(np.array(next_obs))
            dones_tensor = torch.FloatTensor(dones)

            # Use shaped rewards if available
            shaped = getattr(self, '_last_shaped_rewards', None)
            if shaped is not None and len(shaped) == len(rewards):
                rewards_tensor = torch.FloatTensor(shaped)

            with torch.no_grad():
                current_q = self.q_network(obs_tensor).gather(
                    1, actions_tensor
                ).squeeze()
                next_q = self.target_q_network(next_obs_tensor).max(1)[0]
                expected_q = rewards_tensor + self.gamma * next_q * (1 - dones_tensor)
                q_loss = F.smooth_l1_loss(current_q, expected_q).item()
                values = self.value_network(obs_tensor).squeeze()
                value_loss = F.mse_loss(values, expected_q).item()
            return {
                'q_loss': q_loss,
                'value_loss': value_loss,
                'total_loss': q_loss + 0.5 * value_loss,
                'epsilon': float(self.epsilon),
            }
        except Exception:
            return {
                'q_loss': 0.0,
                'value_loss': 0.0,
                'total_loss': 0.0,
                'epsilon': float(self.epsilon),
            }

    def _soft_update_target_network(self):
        """Soft update target network parameters."""
        for target_param, param in zip(self.target_q_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'hierarchical_policy_state_dict': self.hierarchical_policy.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.hierarchical_policy.load_state_dict(checkpoint['hierarchical_policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.config = checkpoint['config']


class ImprovedQMIX:
    """
    Main improved QMIX algorithm class.

    Key improvements over standard QMIX:
    1. Hierarchical action space
    2. Enhanced multi-component reward structure
    3. Attention-based mixing network
    4. Dynamic reward weight adjustment
    5. Hierarchical policy learning
    """

    def __init__(self,
                 num_agents: int,
                 obs_dim: int,
                 state_dim: int,
                 action_dims: List[int],
                 agent_types: List[str],
                 config: Optional[Dict[str, Any]] = None):

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.agent_types = agent_types

        if config is None:
            config = {}
        self.config = config

        # Create agents
        self.agents = []
        for i, (action_dim, agent_type) in enumerate(zip(action_dims, agent_types)):
            agent = ImprovedQMIXAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                agent_type=agent_type,
                config=config
            )
            self.agents.append(agent)

        # Mixing networks
        self.mixing_network = AttentionMixingNetwork(
            state_dim=state_dim,
            num_agents=num_agents,
            hidden_dim=config.get('mixing_hidden_dim', 64),
            num_heads=config.get('attention_heads', 4),
            num_layers=config.get('num_layers', 2)
        )

        self.target_mixing_network = AttentionMixingNetwork(
            state_dim=state_dim,
            num_agents=num_agents,
            hidden_dim=config.get('mixing_hidden_dim', 64),
            num_heads=config.get('attention_heads', 4),
            num_layers=config.get('num_layers', 2)
        )

        # Copy weights
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        # Mixing network optimizer
        self.mixing_optimizer = torch.optim.Adam(
            self.mixing_network.parameters(),
            lr=config.get('learning_rate', 0.0001)
        )

        # Replay buffer
        self.replay_buffer = []
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 32)

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': [],
            'attention_weights': []
        }

        # Optimizer attribute for compatibility with EGTMARL wrapper
        # (use a dummy no-op optimizer that simply points to the per-agent
        # optimizers; EGTMARL may overwrite this, but per-agent optimizers
        # are still functional via the .agents[i].optimizer attribute)
        class _AggregateOptimizer:
            def __init__(self, optimizers):
                self._opts = optimizers
                self.param_groups = []
                for o in optimizers:
                    self.param_groups.extend(o.param_groups)
                self.state = {}
                self.defaults = {}
            def zero_grad(self, set_to_none=True):
                for o in self._opts:
                    o.zero_grad(set_to_none=set_to_none)
            def step(self, closure=None):
                for o in self._opts:
                    o.step(closure)
            def state_dict(self):
                return [o.state_dict() for o in self._opts]
            def load_state_dict(self, states):
                for o, s in zip(self._opts, states):
                    o.load_state_dict(s)
        self.optimizer = _AggregateOptimizer(
            [a.optimizer for a in self.agents] + [self.mixing_optimizer]
        )

    def parameters(self):
        """Aggregate parameters of all agents and the mixing network.

        This lets callers (e.g. EGTMARL) use a single optimizer for the
        whole ImprovedQMIX instance.
        """
        for agent in self.agents:
            for p in agent.parameters():
                yield p
        for p in self.mixing_network.parameters():
                yield p

    @property
    def epsilon(self) -> float:
        """Mean exploration epsilon across agents (used by checkpointing)."""
        if not self.agents:
            return 0.0
        return float(np.mean([float(a.epsilon) for a in self.agents]))

    @epsilon.setter
    def epsilon(self, value: float):
        """Broadcast epsilon to all agents (used by checkpointing)."""
        for a in self.agents:
            a.epsilon = float(value)

    def named_parameters(self):
        """Aggregate named parameters of all agents and the mixing network.

        Prefixed with ``agents.{i}.`` and ``mixing_network.`` so that
        ``state_dict()`` round-trips are unambiguous.
        """
        for i, agent in enumerate(self.agents):
            for name, p in agent.named_parameters():
                yield f"agents.{i}.{name}", p
        for name, p in self.mixing_network.named_parameters():
            yield f"mixing_network.{name}", p

    def state_dict(self):
        """Snapshot of all trainable parameters (agents + mixing network)."""
        return {name: p.detach().cpu().clone()
                for name, p in self.named_parameters()}

    def load_state_dict(self, state_dict, strict: bool = True):
        """Restore parameters produced by ``state_dict()``.

        Iterates over each submodule's own parameters in ``named_parameters``
        order, so the agent / mixing prefixes are stripped and the right
        tensor is matched to the right submodule.
        """
        agent_states = {}
        mixing_states = {}
        for name, tensor in state_dict.items():
            if name.startswith("agents."):
                # agents.<i>.<rest>
                _, idx_str, *rest = name.split(".", 2)
                rest_name = rest[0] if rest else ""
                agent_states.setdefault(idx_str, {})[rest_name] = tensor
            elif name.startswith("mixing_network."):
                mixing_states[name[len("mixing_network."):]] = tensor
            else:
                if strict:
                    raise KeyError(
                        f"Unexpected ImprovedQMIX state_dict key: {name}"
                    )

        ok = True
        for i, agent in enumerate(self.agents):
            sub = agent_states.get(str(i), {})
            if not sub:
                if strict:
                    raise KeyError(
                        f"Missing state for agent {i} in state_dict"
                    )
                continue
            missing, unexpected = agent.load_state_dict(sub, strict=False)
            if missing or unexpected:
                if strict:
                    raise RuntimeError(
                        f"agent {i}: missing={missing} unexpected={unexpected}"
                    )
                ok = False
        if mixing_states:
            m, u = self.mixing_network.load_state_dict(
                mixing_states, strict=False
            )
            if m or u:
                if strict:
                    raise RuntimeError(
                        f"mixing_network: missing={m} unexpected={u}"
                    )
                ok = False
        return ok
    
    def act(self, 
            observations: List[np.ndarray],
            states: np.ndarray,
            available_actions: Optional[List[List[int]]] = None,
            training: bool = True) -> Tuple[List[int], List[Dict[str, Any]]]:
        """Get actions for all agents."""
        actions = []
        action_infos = []
        
        for i, agent in enumerate(self.agents):
            obs = observations[i]
            avail_actions = available_actions[i] if available_actions else None
            
            action, action_info = agent.get_action(obs, avail_actions, training)
            actions.append(action)
            action_infos.append(action_info)
        
        return actions, action_infos
    
    def store_transition(self,
                        observations: List[np.ndarray],
                        actions: List[int],
                        rewards: List[float],
                        next_observations: List[np.ndarray],
                        state: np.ndarray,
                        next_state: np.ndarray,
                        dones: List[bool]):
        """Store transition in replay buffer."""
        for i in range(self.num_agents):
            transition = (
                observations[i].copy(),
                actions[i],
                rewards[i],
                next_observations[i].copy(),
                dones[i]
            )
            self.replay_buffer.append(transition)
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]
    
    def update(self) -> Dict[str, Any]:
        """Update all agents and mixing network end-to-end (full QMIX).

        The replay buffer stores per-agent transitions in groups of
        ``num_agents`` consecutive entries (one per agent, in agent order).
        A *timestep* corresponds to a group of length ``num_agents``; the
        QMIX mini-batch of size ``batch_size`` therefore needs
        ``batch_size * num_agents`` buffer entries.  This is the contract
        that ``_update_mixing_network`` enforces.

        Returns ``{'status': 'insufficient_data'}`` until the buffer
        contains at least one full mini-batch (otherwise we used to silently
        return ``mixing_loss=0.0`` and train nothing).
        """
        num_agents = self.num_agents
        # The QMIX mini-batch must consist of full per-timestep groups.
        if len(self.replay_buffer) < self.batch_size * num_agents:
            return {'status': 'insufficient_data'}

        # ---- 1. Sample complete per-timestep groups ----
        # Valid timestep starts are 0, N, 2N, ...  Pick ``batch_size``
        # distinct starts and reconstruct the per-timestep groups.
        max_starts = len(self.replay_buffer) // num_agents
        timestep_starts = np.random.choice(
            max_starts, size=self.batch_size, replace=False
        ) * num_agents
        timesteps = []
        for start in timestep_starts:
            group = self.replay_buffer[start:start + num_agents]
            if len(group) == num_agents:
                timesteps.append(group)

        if len(timesteps) == 0:
            return {'status': 'insufficient_data'}

        # ---- 2. Compute shaped rewards (5-component reward structure) ----
        # The 5-component reward only makes sense if we have an episode
        # metric to condition on.  When we don't, we fall back to the raw
        # per-agent reward.
        latest_metrics = self.training_stats.get('last_episode_metrics', {})
        if latest_metrics:
            for i, agent in enumerate(self.agents):
                shaped_rewards_per_agent = []
                for timestep in timesteps:
                    obs, action, reward, next_obs, done = timestep[i]
                    raw_metrics = dict(latest_metrics)
                    raw_metrics.setdefault('survivor_count', reward)
                    try:
                        shaped, _ = agent.reward_structure.calculate_reward(
                            raw_metrics
                        )
                    except Exception:
                        shaped = float(reward)
                    shaped_rewards_per_agent.append(float(shaped))
                agent._last_shaped_rewards = shaped_rewards_per_agent

        # ---- 3. End-to-end mixing-network update (Double-Q + soft target) ----
        # This is the only place where the agents are actually trained
        # against the joint return, so the gradient flows through both the
        # mixing network AND the per-agent Q-networks.  Without this the
        # system collapses to independent learners.
        mixing_loss = self._update_mixing_network(timesteps)

        # ---- 4. Per-agent auxiliary losses (no-grad, monitoring only) ----
        # Each agent's value baseline is recalibrated against the same
        # per-timestep group (one row per agent), purely for monitoring.
        agent_losses = []
        for i, agent in enumerate(self.agents):
            agent_batch = [timesteps[t][i] for t in range(len(timesteps))]
            try:
                loss_dict = agent.compute_auxiliary_loss(agent_batch)
            except AttributeError:
                # Older builds may not have compute_auxiliary_loss
                loss_dict = {
                    'q_loss': 0.0,
                    'value_loss': 0.0,
                    'total_loss': 0.0,
                    'epsilon': float(agent.epsilon),
                }
            agent_losses.append(loss_dict)

        # ---- 5. Collect statistics ----
        avg_q_loss = float(np.mean([loss['q_loss'] for loss in agent_losses]))
        avg_value_loss = float(np.mean([loss['value_loss'] for loss in agent_losses]))
        avg_total_loss = float(np.mean([loss['total_loss'] for loss in agent_losses]))
        avg_epsilon = float(np.mean([loss['epsilon'] for loss in agent_losses]))

        # Update training stats
        self.training_stats['losses'].append({
            'q_loss': avg_q_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss,
            'mixing_loss': mixing_loss,
        })
        self.training_stats['epsilon_values'].append(avg_epsilon)

        return {
            'q_loss': avg_q_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss,
            'mixing_loss': mixing_loss,
            'epsilon': avg_epsilon,
            'buffer_size': len(self.replay_buffer),
        }
    
    def _update_mixing_network(self, timesteps: List[List[Tuple]]) -> float:
        """
        End-to-end QMIX mixing-network update with double-Q learning.

        This implements the full QMIX training step (audit fix for the
        "placeholder mixing update" issue, risk-one in the audit report).
        For each timestep in the batch we:

          1. Compute per-agent Q-values for the chosen action:
                q_i = agent_i.q_network(obs_i).gather(a_i)
          2. Mix them through the online mixing network to get
                Q_tot = mixing_network(q_1..q_N, state)
          3. Compute the joint action value with the *target* mixing
             network on next-state Q-values (Double Q):
                Q'_tot = target_mixing_network(argmax_a' Q_i(s', a'),
                                               next_state)
          4. Form the TD target
                y = r_mean + gamma * Q'_tot * (1 - done)
          5. Optimise the MSE loss of (Q_tot, y) end-to-end, so the
             gradient flows through the mixing network AND the per-agent
             Q-networks.  This is the *only* place where the agents are
             actually trained to maximise the joint return (credit
             assignment), and where the mixing network learns to assign
             credit.  Without this, the system collapses to independent
             learners.

        Args:
            timesteps: A list of per-timestep groups.  Each group is a
                list of length ``num_agents`` of tuples
                ``(obs, action, reward, next_obs, done)`` – one per agent
                (in agent order).  ``update()`` reconstructs these groups
                from the flat replay buffer.

        Returns:
            mixing_loss (float).  0.0 if the input is empty or a malformed
            batch is supplied (this is the only fallback path; the caller
            guarantees the data shape so a real exception here is a bug).
        """
        if not timesteps:
            return 0.0
        try:
            num_agents = self.num_agents
            gamma = float(self.config.get('gamma', 0.99))
            time_batch = len(timesteps)

            # ---- 1. Stack per-timestep groups into tensors ----
            obs_tensor = torch.FloatTensor(np.stack([
                np.stack([np.asarray(t[i][0]) for i in range(num_agents)])
                for t in timesteps
            ]))  # [T, N, obs_dim]
            next_obs_tensor = torch.FloatTensor(np.stack([
                np.stack([np.asarray(t[i][3]) for i in range(num_agents)])
                for t in timesteps
            ]))  # [T, N, obs_dim]
            actions_tensor = torch.LongTensor([
                [int(t[i][1]) for i in range(num_agents)]
                for t in timesteps
            ])  # [T, N]

            # Per-agent rewards -> joint reward.  Prefer the shaped
            # 5-component reward computed in update() when it is
            # available; fall back to the per-step mean of the raw
            # per-agent reward.
            shaped_matrix = [
                agent._last_shaped_rewards
                for agent in self.agents
                if getattr(agent, '_last_shaped_rewards', None) is not None
            ]
            if (shaped_matrix
                    and len(shaped_matrix) == num_agents
                    and all(len(r) == time_batch for r in shaped_matrix)):
                rewards_tensor = torch.FloatTensor(
                    [float(np.mean([shaped_matrix[i][t]
                                    for i in range(num_agents)]))
                     for t in range(time_batch)]
                )
            else:
                rewards_tensor = torch.FloatTensor([
                    float(np.mean([t[i][2] for i in range(num_agents)]))
                    for t in timesteps
                ])
            done_tensor = torch.FloatTensor([
                1.0 if any(t[i][4] for i in range(num_agents)) else 0.0
                for t in timesteps
            ])  # [T]

            # State: average of per-agent observations (matches the
            # convention used in the rest of the codebase, e.g.
            # marl_layer.py).
            state_tensor = obs_tensor.mean(dim=1)  # [T, obs_dim]
            next_state_tensor = next_obs_tensor.mean(dim=1)  # [T, obs_dim]

            # ---- 2. Per-agent Q-values (chosen actions) ----
            per_agent_q = []
            for i, agent in enumerate(self.agents):
                q_i = agent.q_network(obs_tensor[:, i, :])  # [T, A_i]
                a_i = actions_tensor[:, i].clamp(0, q_i.shape[1] - 1).unsqueeze(1)
                per_agent_q.append(q_i.gather(1, a_i).squeeze(1))  # [T]
            q_per_agent = torch.stack(per_agent_q, dim=1)  # [T, N]

            # ---- 3. Joint Q via the online mixing network ----
            q_total = self.mixing_network(q_per_agent, state_tensor).squeeze(-1)
            # shape is [T]

            # ---- 4. Target joint Q via the target mixing network ----
            # Double Q: pick the action with the *online* network,
            # evaluate with the *target* network.
            with torch.no_grad():
                next_q_per_agent_target = []
                for i, agent in enumerate(self.agents):
                    nq_i = agent.target_q_network(next_obs_tensor[:, i, :])  # [T, A_i]
                    online_nq_i = agent.q_network(next_obs_tensor[:, i, :])
                    best_action = online_nq_i.argmax(dim=1, keepdim=True)  # [T, 1]
                    nq = nq_i.gather(
                        1, best_action.clamp(0, nq_i.shape[1] - 1)
                    ).squeeze(1)
                    next_q_per_agent_target.append(nq)
                next_q_per_agent_target = torch.stack(
                    next_q_per_agent_target, dim=1
                )  # [T, N]
                next_q_total = self.target_mixing_network(
                    next_q_per_agent_target, next_state_tensor
                ).squeeze(-1)  # [T]

                # ---- 5. TD target ----
                target = rewards_tensor + gamma * next_q_total * (1.0 - done_tensor)

            # ---- 6. End-to-end MSE loss ----
            mixing_loss = F.mse_loss(q_total, target.detach())

            # ---- 7. Optimise both the mixing network AND the per-agent
            #         Q-networks end-to-end.  We accumulate per-agent
            #         gradients and step each agent's optimizer.
            self.mixing_optimizer.zero_grad()
            for agent in self.agents:
                agent.optimizer.zero_grad()

            mixing_loss.backward()

            # Gradient clipping (separate for mixing vs agents)
            torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=10.0)
            for agent in self.agents:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)

            self.mixing_optimizer.step()
            for agent in self.agents:
                agent.optimizer.step()

            # ---- 8. Soft-update target mixing network ----
            with torch.no_grad():
                tau = float(self.config.get('tau', 0.005))
                for p, tp in zip(
                    self.mixing_network.parameters(),
                    self.target_mixing_network.parameters(),
                ):
                    tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

            return float(mixing_loss.item())

        except Exception as e:
            # Defensive: a malformed batch should never crash the trainer
            import logging
            logging.warning(f"QMIX._update_mixing_network failed: {type(e).__name__}: {e}")
            return 0.0

    def end_episode(self, 
                   episode_reward: float,
                   episode_length: int,
                   metrics: Dict[str, Any]):
        """Record episode statistics."""
        self.training_stats['episode_rewards'].append(episode_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        
        # Update reward structure weights based on episode performance
        for agent in self.agents:
            agent.reward_structure.update_weights(metrics, learning_rate=0.01)
    
    def get_attention_weights(self,
                            observations: List[np.ndarray],
                            state: np.ndarray) -> np.ndarray:
        """Get attention weights for interpretability."""
        # Get Q-values for all agents
        q_values = []
        for i, agent in enumerate(self.agents):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0)
            with torch.no_grad():
                q_val = agent.q_network(obs_tensor).max().item()
            q_values.append(q_val)
        
        q_tensor = torch.FloatTensor(q_values).unsqueeze(0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get attention weights
        with torch.no_grad():
            attention_weights = self.mixing_network.compute_attention_weights(
                q_tensor, state_tensor
            )
        
        return attention_weights.squeeze().numpy()
    
    def save(self, directory: str):
        """Save all agents and mixing network."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save agents
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(directory, f'agent_{i}.pt')
            agent.save(agent_path)
        
        # Save mixing network
        mixing_path = os.path.join(directory, 'mixing_network.pt')
        torch.save({
            'state_dict': self.mixing_network.state_dict(),
            'config': self.config
        }, mixing_path)
        
        # Save target mixing network
        target_mixing_path = os.path.join(directory, 'target_mixing_network.pt')
        torch.save({
            'state_dict': self.target_mixing_network.state_dict(),
            'config': self.config
        }, target_mixing_path)
        
        # Save training stats
        stats_path = os.path.join(directory, 'training_stats.pkl')
        import pickle
        with open(stats_path, 'wb') as f:
            pickle.dump(self.training_stats, f)
    
    def load(self, directory: str):
        """Load all agents and mixing network."""
        import os
        
        # Load agents
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(directory, f'agent_{i}.pt')
            if os.path.exists(agent_path):
                agent.load(agent_path)
        
        # Load mixing network
        mixing_path = os.path.join(directory, 'mixing_network.pt')
        if os.path.exists(mixing_path):
            checkpoint = torch.load(mixing_path, weights_only=False)
            self.mixing_network.load_state_dict(checkpoint['state_dict'])
        
        # Load target mixing network
        target_mixing_path = os.path.join(directory, 'target_mixing_network.pt')
        if os.path.exists(target_mixing_path):
            checkpoint = torch.load(target_mixing_path, weights_only=False)
            self.target_mixing_network.load_state_dict(checkpoint['state_dict'])
        
        # Load training stats
        stats_path = os.path.join(directory, 'training_stats.pkl')
        if os.path.exists(stats_path):
            import pickle
            with open(stats_path, 'rb') as f:
                self.training_stats = pickle.load(f)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()


# Utility functions for improved QMIX
def create_improved_qmix(config: Dict[str, Any]) -> ImprovedQMIX:
    """Factory function to create improved QMIX algorithm."""
    # Parse configuration
    num_agents = config.get('num_agents', 17)  # DRONE=num//10, VEHICLE=num//5, PERSONNEL=remaining
    obs_dim = config.get('obs_dim', 256)
    state_dim = config.get('state_dim', 512)
    
    # Define agent types and action dimensions (match disaster_sim.py's dynamic allocation)
    agent_types = []
    action_dims = []
    
    action_space = HierarchicalActionSpace(config)
    
    # Drones (num_agents // 10)
    n_drones = max(1, num_agents // 10)
    for _ in range(n_drones):
        agent_types.append('drone')
        action_dims.append(action_space.get_total_dim('drone'))
    
    # Vehicles (num_agents // 5)
    n_vehicles = max(1, num_agents // 5)
    for _ in range(n_vehicles):
        agent_types.append('vehicle')
        action_dims.append(action_space.get_total_dim('vehicle'))
    
    # Personnel (remaining)
    n_personnel = num_agents - n_drones - n_vehicles
    for _ in range(n_personnel):
        agent_types.append('personnel')
        action_dims.append(action_space.get_total_dim('personnel'))
    
    # Create improved QMIX
    qmix = ImprovedQMIX(
        num_agents=num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dims=action_dims,
        agent_types=agent_types,
        config=config
    )
    
    return qmix


def train_improved_qmix(qmix: ImprovedQMIX,
                       env,
                       num_episodes: int = 10000,
                       eval_frequency: int = 100,
                       save_frequency: int = 500,
                       save_dir: str = 'checkpoints'):
    """Training loop for improved QMIX."""
    import time
    from tqdm import tqdm
    
    print(f"Starting training for {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes)):
        # Reset environment
        observations = env.reset()
        state = env.get_state()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get actions
            actions, action_infos = qmix.act(observations, state, training=True)
            
            # Step environment
            next_observations, rewards, dones, info = env.step(actions)
            next_state = env.get_state()
            
            # Store transition
            qmix.store_transition(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                state=state,
                next_state=next_state,
                dones=dones
            )
            
            # Update
            if len(qmix.replay_buffer) >= qmix.batch_size:
                update_stats = qmix.update()
            
            # Update for next step
            observations = next_observations
            state = next_state
            episode_reward += sum(rewards)
            episode_length += 1
            
            # Check if episode is done
            done = all(dones) or episode_length >= env.max_steps
        
        # End episode
        metrics = env.get_metrics()
        qmix.end_episode(episode_reward, episode_length, metrics)
        
        # Evaluation
        if episode % eval_frequency == 0:
            eval_reward = evaluate_improved_qmix(qmix, env, num_eval_episodes=5)
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Eval Reward = {eval_reward:.2f}, "
                  f"Buffer Size = {len(qmix.replay_buffer)}")
        
        # Save checkpoint
        if episode % save_frequency == 0:
            checkpoint_dir = f"{save_dir}/episode_{episode}"
            qmix.save(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")
    
    print("Training completed!")
    return qmix


def evaluate_improved_qmix(qmix: ImprovedQMIX,
                          env,
                          num_eval_episodes: int = 10) -> float:
    """Evaluate improved QMIX algorithm."""
    total_rewards = []
    
    for _ in range(num_eval_episodes):
        observations = env.reset()
        state = env.get_state()
        done = False
        episode_reward = 0
        
        while not done:
            # Get actions (no exploration during evaluation)
            actions, _ = qmix.act(observations, state, training=False)
            
            # Step environment
            next_observations, rewards, dones, _ = env.step(actions)
            next_state = env.get_state()
            
            # Update
            observations = next_observations
            state = next_state
            episode_reward += sum(rewards)
            
            # Check if episode is done
            done = all(dones)
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


if __name__ == "__main__":
    # Example usage
    config = {
        'num_agents': 17,
        'obs_dim': 256,
        'state_dim': 512,
        'hidden_dim': 128,
        'mixing_hidden_dim': 64,
        'attention_heads': 4,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 10000,
        'batch_size': 32
    }
    
    # Create improved QMIX
    qmix = create_improved_qmix(config)
    print(f"Created improved QMIX with {qmix.num_agents} agents")
    
    # Note: Environment needs to be created separately
    # train_improved_qmix(qmix, env, num_episodes=1000)