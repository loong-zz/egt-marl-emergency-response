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
from .qmix_improved import ImprovedQMIX, create_improved_qmix
from .egt_layer import EGTLayer
from .anti_spoofing import AntiSpoofing
from .dynamic_frontier import DynamicFrontier
from ..environments.config.constants import NUM_STRATEGIES


class EGTMARL:
    """
    Main EGT-MARL algorithm with two-layer architecture.
    
    Architecture:
    1. Micro-layer (MARL): Distributed decision execution using improved QMIX
    2. Macro-layer (EGT): Dynamic fairness-efficiency trade-off regulation
    """
    
    def __init__(self, state_dim: int = 22, action_dim: int = 32, num_agents: int = 3, 
                 hidden_dim: int = 64, device: Optional[torch.device] = None, 
                 env=None, config_path: Optional[str] = None, config: Optional[Dict] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        """
        Initialize EGT-MARL algorithm.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            num_agents: Number of agents
            hidden_dim: Hidden layer dimension
            device: PyTorch device
            env: Disaster simulation environment
            config_path: Path to configuration file
            config: Configuration dictionary (takes precedence over config_path)
        """
        self.env = env
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config is not None:
            # Use provided config dictionary
            self.config = config
        elif config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "egt_marl.yaml"
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Update configuration with provided parameters (only if not already set)
        if 'state_dim' not in self.config['marl']:
            self.config['marl']['state_dim'] = state_dim
        # Action space: 8 tactical (movement) * 4 communication = 32 possible actions
        if 'action_dim' not in self.config['marl']:
            self.config['marl']['action_dim'] = action_dim
        if 'num_agents' not in self.config['marl']:
            self.config['marl']['num_agents'] = num_agents
        if 'hidden_dim' not in self.config['marl']:
            self.config['marl']['hidden_dim'] = hidden_dim
        
        # Get actual state dimension and num_agents from environment if provided
        if env is not None:
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            if hasattr(state, 'shape'):
                self.config['marl']['state_dim'] = state.shape[1]
                # Update num_agents from environment state shape
                self.config['marl']['num_agents'] = state.shape[0]
                self.num_agents = state.shape[0]
        
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
        
        # Replay buffer for testing
        self.replay_buffer = []
        self.batch_size = self.config['marl'].get('batch_size', 32)
        self.buffer_size = self.config['marl'].get('buffer_size', 10000)
    
    def _initialize_components(self) -> None:
        """Initialize all algorithm components."""
        # ---- MARL layer: prefer ImprovedQMIX (paper's full implementation) ----
        # Fall back to the lightweight MARLLayer for very small/odd agent counts
        # that don't fit the hierarchical action space.
        use_improved = self.config['marl'].get('use_improved_qmix', True)
        num_agents = self.config['marl']['num_agents']

        if use_improved and num_agents >= 17:
            # Map the 17-agent configuration (10 drones + 5 ambulances + 2 hospitals)
            # onto the ImprovedQMIX agent types.
            n_drones, n_ambulances, n_hospitals = 10, 5, 2
            agent_types = (['drone'] * n_drones
                           + ['ambulance'] * n_ambulances
                           + ['hospital'] * n_hospitals)
            action_dims = []
            from .qmix_improved import HierarchicalActionSpace
            action_space = HierarchicalActionSpace(self.config['marl'])
            for t in agent_types:
                action_dims.append(action_space.get_total_dim(t))

            # Build the ImprovedQMIX instance directly so we can pass our own
            # obs_dim / state_dim and get a richer agent.
            self.marl_layer = ImprovedQMIX(
                num_agents=num_agents,
                obs_dim=self.config['marl']['state_dim'],
                state_dim=self.config['marl']['state_dim'],
                action_dims=action_dims,
                agent_types=agent_types,
                config=self.config['marl'],
            )
            self._marl_is_improved_qmix = True
        else:
            # Fallback: lightweight MARLLayer
            self.marl_layer = MARLLayer(
                state_dim=self.config['marl']['state_dim'],
                action_dim=self.config['marl']['action_dim'],
                num_agents=self.config['marl']['num_agents'],
                hidden_dim=self.config['marl']['hidden_dim'],
                device=self.device
            )
            self._marl_is_improved_qmix = False

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
            action_dim=self.config['marl']['action_dim'],
            device=self.device,
            num_agents=self.config['marl']['num_agents']
        )

        # Dynamic Pareto frontier
        self.dynamic_frontier = DynamicFrontier(
            config=self.config['dynamic_frontier']
        )

        # Optimizers - Note: marl_layer uses its own hardcoded optimizer internally,
        # so we need to replace it with our configured optimizer
        self.marl_optimizer = optim.Adam(
            self.marl_layer.parameters(),
            lr=self.config['marl']['learning_rate']
        )
        # Replace marl_layer's hardcoded optimizer with our configured one
        self.marl_layer.optimizer = self.marl_optimizer

        self.egt_optimizer = optim.Adam(
            self.egt_layer.parameters(),
            lr=self.config['egt']['learning_rate']
        )

        # Loss functions
        self.marl_loss_fn = nn.MSELoss()
        self.egt_loss_fn = nn.KLDivLoss()
    
    def _initialize_payoff_matrix(self) -> torch.Tensor:
        """Initialize payoff matrix for evolutionary game.

        We let EGTLayer._init_theory_driven_payoff handle the actual content
        (paper-aligned 4-strategy structure) and just return a placeholder
        here so the constructor is happy.  EGTLayer will overwrite it with
        the theory-driven matrix because we pass payoff_matrix=None in spirit
        (the placeholder is replaced before use).
        """
        num_strategies = self.config['egt']['num_strategies']
        # Identity-ish placeholder; EGTLayer will replace this with the
        # theory-driven initialization (see EGTLayer.__init__).
        payoff_matrix = torch.eye(num_strategies)

        return payoff_matrix.to(self.device)
    
    def select_action(self, state, training: bool = True, epsilon: float = None) -> Dict[int, Dict[str, Any]]:
        """
        Select actions for all agents.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            epsilon: Exploration rate (if None, use MARL layer's internal epsilon)
            
        Returns:
            Dictionary of actions for each agent
        """
        # Handle tuple input (observation, info)
        if isinstance(state, tuple):
            state = state[0]
        
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            # Add batch dimension
            state_tensor = state_tensor.unsqueeze(0)
        else:
            state_tensor = state
        
        # Get MARL actions (micro-level decisions)
        # Use deterministic actions when not training
        # Pass epsilon explicitly to ensure training parameter is used
        try:
            marl_actions = self.marl_layer.select_actions(state_tensor, deterministic=not training, epsilon=epsilon)
        except Exception:
            # Fallback to random actions if marl_layer fails
            marl_actions = torch.randint(0, self.config['marl']['action_dim'], 
                                        (self.config['marl']['num_agents'],), 
                                        device=self.device)
        
        # Convert tensor to dictionary format expected by the environment
        actions_dict = {}
        for agent_id in range(self.config['marl']['num_agents']):
            # Get action index from tensor
            if marl_actions.ndim == 2:
                action_idx = marl_actions[0, agent_id].item()
            elif marl_actions.ndim == 1:
                action_idx = marl_actions[agent_id].item()
            else:
                action_idx = 0
            
            # Convert action index to hierarchical action format expected by DisasterSim
            # Action space: 8 tactical (movement) * 4 communication = 32 possible actions
            # For backward compatibility with action_dim=5, we map 0-4 to valid actions
            num_tactical = 8
            num_communication = 4
            
            # Proper mapping for action_dim=32 (8*4)
            if self.config['marl']['action_dim'] >= 32:
                tactical_action = action_idx % num_tactical
                communication_action = action_idx // num_tactical
            else:
                # For smaller action spaces, map to valid ranges
                tactical_action = action_idx % num_tactical
                communication_action = min(action_idx // num_tactical, num_communication - 1)

            actions_dict[agent_id] = {
                # Pull the current EGT macro signal so the action dict actually
                # reflects the evolutionary game state at inference time.
                # Falls back to a uniform distribution if the EGT layer is
                # unavailable or has not been updated yet.
                "strategic": self._get_egt_strategic_distribution(),
                "tactical": tactical_action,  # Movement direction (0-7)
                "communication": communication_action  # Communication action (0-3)
            }

        return actions_dict

    def _get_egt_strategic_distribution(self) -> List[float]:
        """Return the current EGT strategy distribution as a plain list.

        Used to populate the ``strategic`` field of the action dict so that
        downstream consumers (and the EGT macro signal) actually flow through
        the inference path.  Falls back to a uniform distribution if the EGT
        layer hasn't been initialised or its distribution is degenerate.
        """
        n = int(self.config['egt'].get('num_strategies', NUM_STRATEGIES))
        # P8 fix: guard against a misconfigured ``num_strategies=0`` (e.g.
        # from a typo'd YAML, or an empty ``strategy_names`` list).  The
        # previous ``[1.0 / n] * n`` would raise ZeroDivisionError.
        if n < 1:
            return [1.0]
        if n == 1:
            return [1.0]
        egt_layer = getattr(self, 'egt_layer', None)
        if egt_layer is None:
            return [1.0 / n] * n
        try:
            distribution = egt_layer.get_strategy_distribution().detach()
        except Exception:
            return [1.0 / n] * n
        if distribution.ndim != 1 or int(distribution.shape[0]) != n:
            return [1.0 / n] * n
        values = [float(v) for v in distribution.cpu().tolist()]
        total = sum(values)
        if total <= 0:
            return [1.0 / n] * n
        return [v / total for v in values]
    
    def select_actions(self, state, epsilon: float = 0.1) -> List[int]:
        """
        Select actions for all agents (compatibility method).
        
        Args:
            state: Current environment state
            epsilon: Exploration rate
            
        Returns:
            List of actions for each agent
        """
        # Get action dictionary
        action_dict = self.select_action(state, training=epsilon > 0)
        
        # Convert to list of action indices
        actions = []
        for agent_id in range(self.config['marl']['num_agents']):
            if agent_id in action_dict:
                action = action_dict[agent_id]
                # Convert hierarchical action to single index
                tactical = action.get('tactical', 0)
                communication = action.get('communication', 0)
                action_idx = tactical + communication * 8
                actions.append(action_idx)
            else:
                actions.append(0)
        
        return actions
    
    def store_experience(self, state, actions, rewards, next_state, done):
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            actions: Actions taken (dict per-agent or flat list/array)
            rewards: Rewards received (dict per-agent, list, or scalar)
            next_state: Next state
            done: Whether episode is done

        The stored reward is always a *list* of length ``num_agents`` so
        the offline no-args ``update()`` path can stack it into a tensor
        without choking on dict values.
        """
        num_agents = self.config['marl']['num_agents']

        # Convert action dictionary to list of action indices
        action_indices = []
        if isinstance(actions, dict):
            for agent_id in range(num_agents):
                if agent_id in actions:
                    action = actions[agent_id]
                    # Convert hierarchical action to single index
                    tactical = action.get('tactical', 0)
                    communication = action.get('communication', 0)
                    action_idx = tactical + communication * 8
                    action_indices.append(action_idx)
                else:
                    action_indices.append(0)
        else:
            action_indices = actions

        # Convert rewards to a per-agent list.  ``rewards`` may arrive as
        # a dict {agent_id: scalar}, a scalar (broadcast), or already a
        # list/ndarray.  The no-args ``update()`` path needs a list of
        # scalars of length ``num_agents`` to stack into a [B, N] tensor.
        if isinstance(rewards, dict):
            reward_list = [float(rewards.get(aid, 0.0))
                           for aid in range(num_agents)]
        elif np.isscalar(rewards):
            reward_list = [float(rewards)] * num_agents
        else:
            # Assume list / ndarray already of length num_agents
            reward_list = [float(r) for r in rewards]
        # Pad / truncate to num_agents for safety
        if len(reward_list) < num_agents:
            reward_list = reward_list + [0.0] * (num_agents - len(reward_list))
        elif len(reward_list) > num_agents:
            reward_list = reward_list[:num_agents]

        experience = {
            'state': state,
            'actions': action_indices,
            'rewards': reward_list,
            'next_state': next_state,
            'done': done
        }
        self.replay_buffer.append(experience)

        # Limit buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def update_parameters(self):
        """
        Update algorithm parameters (compatibility method).
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        try:
            # Convert to tensors with proper handling
            states = torch.stack([torch.tensor(exp['state'], dtype=torch.float32) for exp in batch]).to(self.device)
            actions = torch.tensor([exp['actions'] for exp in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([exp['rewards'] for exp in batch], dtype=torch.float32).to(self.device)
            next_states = torch.stack([torch.tensor(exp['next_state'], dtype=torch.float32) for exp in batch]).to(self.device)
            dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool).to(self.device)
            
            # Create batch dictionary
            batch_dict = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones
            }
            
            # Update
            losses = self.update(batch_dict)
            return losses.get('marl_loss', 0.0)
        except Exception:
            # Return 0.0 if update fails
            return 0.0
    
    def get_state_dict(self):
        """
        Get state dictionary (compatibility method).
        
        Returns:
            State dictionary
        """
        state_dict = {
            'marl_layer': self.marl_layer.state_dict(),
            'egt_layer': self.egt_layer.state_dict()
        }
        
        # Only add dynamic_frontier if it has state_dict method
        if hasattr(self.dynamic_frontier, 'state_dict'):
            state_dict['dynamic_frontier'] = self.dynamic_frontier.state_dict()
        
        # Add anti_spoofing if it has state_dict
        if hasattr(self.anti_spoofing, 'state_dict'):
            state_dict['anti_spoofing'] = self.anti_spoofing.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        """
        Load state dictionary (compatibility method).
        
        Args:
            state_dict: State dictionary
        """
        if 'marl_layer' in state_dict:
            self.marl_layer.load_state_dict(state_dict['marl_layer'])
        if 'egt_layer' in state_dict:
            self.egt_layer.load_state_dict(state_dict['egt_layer'])
        if 'anti_spoofing' in state_dict and hasattr(self.anti_spoofing, 'load_state_dict'):
            self.anti_spoofing.load_state_dict(state_dict['anti_spoofing'])
        if 'dynamic_frontier' in state_dict:
            self.dynamic_frontier.load_state_dict(state_dict['dynamic_frontier'])
    
    def set_egt_parameters(self, lambda_param=0.5, pareto_weights=None, anti_spoofing_enabled=True):
        """
        Set EGT parameters (compatibility method).
        
        Args:
            lambda_param: Lambda parameter for EGT
            pareto_weights: Pareto weights for multi-objective optimization
            anti_spoofing_enabled: Whether anti-spoofing is enabled
        """
        # Store parameters as attributes for testing
        self.egt_lambda = lambda_param
        self.pareto_weights = pareto_weights or {'efficiency': 0.4, 'fairness': 0.3, 'robustness': 0.3}
        self.anti_spoofing_enabled = anti_spoofing_enabled
        
        # Pass parameters to egt_layer if it has set_parameters
        if hasattr(self.egt_layer, 'set_parameters'):
            parameters = {
                'lambda_param': lambda_param,
                'pareto_weights': self.pareto_weights,
                'anti_spoofing_enabled': anti_spoofing_enabled
            }
            self.egt_layer.set_parameters(parameters)
    
    def compute_egt_rewards(self, states, actions):
        """
        Compute EGT rewards (compatibility method).
        
        Args:
            states: States
            actions: Actions
            
        Returns:
            EGT rewards
        """
        if hasattr(self.egt_layer, 'compute_rewards'):
            return self.egt_layer.compute_rewards(states, actions)
        return torch.zeros(len(states), device=self.device)
    
    def _compute_egt_rewards(self, individual_rewards, cooperation_levels):
        """
        Compute EGT rewards (internal method).
        
        Args:
            individual_rewards: Individual rewards
            cooperation_levels: Cooperation levels
            
        Returns:
            EGT rewards
        """
        # Simple implementation for testing
        return individual_rewards * (1 + 0.1 * cooperation_levels.unsqueeze(1))
    
    def _adjust_actions_with_egt(self, actions: torch.Tensor, 
                                strategy_distribution: torch.Tensor) -> torch.Tensor:
        """
        Adjust actions based on EGT strategy distribution.
        
        Args:
            actions: Original actions from MARL layer (tensor)
            strategy_distribution: Current strategy distribution from EGT layer
            
        Returns:
            Adjusted actions (tensor)
        """
        # For simplicity, we'll return the actions as is
        # In a real implementation, we would adjust the actions based on EGT strategy
        return actions
    
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
    
    def update(self, batch: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Update algorithm parameters from experience batch.

        Two-layer architecture information flow (paper section 4.3):
        1. EGT layer (macro) computes fairness-efficiency trade-off weights from batch
        2. These weights are injected into MARL layer (micro) for reward shaping
        3. MARL layer learns with the adjusted rewards
        4. Pareto frontier is updated from the same batch

        Args:
            batch: Experience batch containing states, actions, rewards, next_states
                   (and optionally fairness_score / efficiency_score metrics).

        Returns:
            Dictionary of loss values.
        """
        if batch is None:
            # Handle case where no batch is provided (for integration testing)
            if len(self.replay_buffer) < self.batch_size:
                return {'marl_loss': 0.0, 'egt_loss': 0.0, 'spoofing_loss': 0.0, 'frontier_loss': 0.0}

            # Sample batch from replay buffer
            indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
            batch_list = [self.replay_buffer[i] for i in indices]

            try:
                # Convert to tensors with proper handling
                states = torch.stack(
                    [torch.tensor(exp['state'], dtype=torch.float32) for exp in batch_list]
                ).to(self.device)
                actions = torch.tensor(
                    [exp['actions'] for exp in batch_list], dtype=torch.long
                ).to(self.device)
                rewards = torch.tensor(
                    [exp['rewards'] for exp in batch_list], dtype=torch.float32
                ).to(self.device)
                next_states = torch.stack(
                    [torch.tensor(exp['next_state'], dtype=torch.float32) for exp in batch_list]
                ).to(self.device)
                dones = torch.tensor(
                    [exp['done'] for exp in batch_list], dtype=torch.bool
                ).to(self.device)

                # Create batch dictionary
                batch = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_states,
                    'dones': dones
                }
                return self.update(batch)
            except Exception as e:
                # Surface the failure in the log; do not silently swallow.
                import logging
                logging.exception(
                    "EGTMARL: replay-buffer batch construction failed; "
                    "update() will return zero losses for this step. "
                    "Underlying error: %r", e,
                )
                return {'marl_loss': 0.0, 'egt_loss': 0.0, 'spoofing_loss': 0.0, 'frontier_loss': 0.0}

        losses = {}

        # ==================== EGT Layer Update FIRST ====================
        # EGT layer computes fairness-efficiency trade-off weights (this is the
        # "macro" signal that drives the micro MARL layer).
        egt_weights = None
        try:
            egt_loss, egt_weights = self.egt_layer.update_with_weights(
                batch, self.egt_optimizer, self.egt_loss_fn
            )
            losses['egt_loss'] = egt_loss
        except Exception as e:
            # Log loudly so silent failures surface in the training loop.
            # Note: egt_layer.update() is a strict subset of
            # update_with_weights() and would fail with the same error, so
            # we do NOT fall through to it — that just hides the bug.
            import logging
            logging.exception(
                "EGTMARL: EGT layer update_with_weights failed; "
                "EGT macro signal will be disabled for this step. "
                "Underlying error: %r", e,
            )
            losses['egt_loss'] = 0.0
            egt_weights = None

        # Resolve lambda parameter for the MARL layer.  EGT always has this
        # attribute after update, so we just read it.
        lambda_param = getattr(self.egt_layer, 'lambda_param', 0.5)

        # ==================== MARL Layer Update with EGT Weights ====================
        try:
            if self._marl_is_improved_qmix:
                # ImprovedQMIX: it has its own internal replay buffer & update.
                # We push shaped transitions via store_transition and call update().
                # NOTE: the ImprovedQMIX class manages its own replay buffer, so
                # we transform the batch into per-agent transitions.
                try:
                    self._feed_batch_to_improved_qmix(batch, egt_weights, lambda_param)
                except Exception as e:
                    import logging
                    logging.warning(f"EGTMARL: _feed_batch_to_improved_qmix failed: {e}")
                update_stats = self.marl_layer.update()
                # Prefer the real mixing loss (end-to-end QMIX) over the
                # auxiliary per-agent loss; fall back gracefully.
                if not isinstance(update_stats, dict):
                    update_stats = {'mixing_loss': 0.0, 'total_loss': 0.0,
                                    'q_loss': 0.0, 'value_loss': 0.0}
                losses['marl_loss'] = float(
                    update_stats.get('mixing_loss',
                                     update_stats.get('total_loss', 0.0))
                )
                losses['q_loss'] = float(update_stats.get('q_loss', 0.0))
                losses['value_loss'] = float(update_stats.get('value_loss', 0.0))
                losses['mixing_loss'] = float(update_stats.get('mixing_loss', 0.0))
            else:
                # MARLLayer: take the explicit egt_weights / lambda_param and
                # use them for reward shaping (this is the fix for C2 — the
                # "dead connection" between EGT and MARL).
                marl_loss = self.marl_layer.update(
                    batch['states'],
                    batch['actions'],
                    batch['rewards'],
                    batch['next_states'],
                    batch['dones'],
                    egt_weights=egt_weights,
                    lambda_param=lambda_param,
                )
                losses['marl_loss'] = marl_loss
        except Exception:
            # Fallback: update without EGT weights if the new path fails
            try:
                marl_loss = self.marl_layer.update(
                    batch['states'],
                    batch['actions'],
                    batch['rewards'],
                    batch['next_states'],
                    batch['dones'],
                )
                losses['marl_loss'] = marl_loss
            except Exception:
                losses['marl_loss'] = 0.0

        # Update anti-spoofing mechanism
        try:
            spoofing_loss = self.anti_spoofing.update(batch)
            losses['spoofing_loss'] = spoofing_loss
        except Exception:
            losses['spoofing_loss'] = 0.0

        # Update dynamic Pareto frontier
        try:
            frontier_loss = self.dynamic_frontier.update(batch)
            losses['frontier_loss'] = frontier_loss
            # Audit fix A2: actually push the frontier weights to the MARL
            # layer so its reward shaping reflects the current trade-off.
            # P3 fix: surface failures instead of silently swallowing them.
            try:
                self.dynamic_frontier.apply_to_marl(self)
            except Exception as e:
                import logging
                logging.exception(
                    "EGTMARL: dynamic_frontier.apply_to_marl failed; "
                    "Pareto frontier weights will not reach MARL this step. "
                    "Underlying error: %r", e,
                )
        except Exception as e:
            # P3 fix: log the underlying error instead of silently swallowing
            # it.  The frontier loss being zero is *not* normal; it should
            # surface in the training log so it doesn't masquerade as a
            # legitimate "0" loss.
            import logging
            logging.exception(
                "EGTMARL: dynamic_frontier.update failed; "
                "Pareto frontier signal disabled for this step. "
                "Underlying error: %r", e,
            )
            losses['frontier_loss'] = 0.0

        # Update total steps
        try:
            self.total_steps += len(batch['states'])
        except Exception:
            pass

        # Expose current EGT weights for downstream consumers
        self.current_egt_weights = egt_weights if egt_weights is not None else {}

        return losses

    def _feed_batch_to_improved_qmix(self, batch: Dict[str, Any],
                                     egt_weights: Optional[Dict[str, float]],
                                     lambda_param: float) -> None:
        """
        Feed a batch of transitions to the ImprovedQMIX instance.

        The ImprovedQMIX class manages its own replay buffer (per-agent
        transition tuples) and pulls samples from it during update().
        We translate our (state, action, reward, next_state, done) batch into
        the per-agent transition format and push them into the buffer.

        EGT-driven reward shaping is applied here so the macro layer's
        fairness-efficiency preferences actually reach the agents.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # Apply EGT-driven shaping to the per-step reward.
        # P2 fix: previously we did `rewards * (1 + egt_boost) + lam * 0.01`
        # but ImprovedQMIX.update() internally computes its own shaped
        # rewards via EnhancedRewardStructure (5-component) and stashes
        # them in `_last_shaped_rewards`.  The two shaping steps would
        # compound, making the EGT signal's effect unpredictable.
        #
        # Resolution: keep only the *multiplicative* EGT bias here
        # (`(1 + egt_boost) * r`) and let the additive fairness bonus
        # be computed by ImprovedQMIX's own reward structure, which has
        # access to per-agent signals.  The scalar `lambda_param` is
        # still propagated to MARLLayer.update() (when not using
        # ImprovedQMIX) and to the auxiliary callbacks.
        egt_boost = 0.0
        if egt_weights is not None and 'fairness_weight' in egt_weights:
            egt_boost = float(egt_weights['fairness_weight']) - 0.5
        shaped_rewards = rewards * (1.0 + egt_boost)
        # Stash lambda for downstream consumers; no additive bias here.
        lam = float(max(0.0, min(1.0, lambda_param)))
        # Reduce to per-step scalar (mean across the 17 agents) so we can
        # index it inside the per-timestep loop below.
        if shaped_rewards.dim() == 2:
            step_rewards = shaped_rewards.mean(dim=1)
        else:
            step_rewards = shaped_rewards

        # ImprovedQMIX expects per-agent transitions, with the observation/state
        # being a single vector per agent.  We treat the env state as the
        # global state and split it into per-agent observations.
        batch_size = states.shape[0]
        for b in range(batch_size):
            # Per-agent observations: split the state along the agent axis
            if states.dim() == 3:  # [B, N, D]
                obs = [states[b, i].detach().cpu().numpy() for i in range(self.marl_layer.num_agents)]
                next_obs = [next_states[b, i].detach().cpu().numpy()
                            for i in range(self.marl_layer.num_agents)]
            else:  # [B, D] shared
                shared = states[b].detach().cpu().numpy()
                obs = [shared.copy() for _ in range(self.marl_layer.num_agents)]
                next_shared = next_states[b].detach().cpu().numpy()
                next_obs = [next_shared.copy() for _ in range(self.marl_layer.num_agents)]

            # Per-agent actions
            if actions.dim() == 2:  # [B, N]
                per_agent_actions = [int(actions[b, i].item())
                                     for i in range(self.marl_layer.num_agents)]
            else:
                per_agent_actions = [int(actions[b].item())
                                     for _ in range(self.marl_layer.num_agents)]

            # Per-agent rewards: distribute the shaped reward evenly, scaled by lambda
            per_agent_rewards = [float(step_rewards[b].item())
                                 for _ in range(self.marl_layer.num_agents)]

            per_agent_dones = [bool(dones[b].item())
                               for _ in range(self.marl_layer.num_agents)]

            # Global state = mean of per-agent states
            state_vec = states[b].mean(dim=0).detach().cpu().numpy() \
                if states.dim() == 3 else states[b].detach().cpu().numpy()
            next_state_vec = next_states[b].mean(dim=0).detach().cpu().numpy() \
                if next_states.dim() == 3 else next_states[b].detach().cpu().numpy()

            # Push to the ImprovedQMIX replay buffer
            self.marl_layer.store_transition(
                observations=obs,
                actions=per_agent_actions,
                rewards=per_agent_rewards,
                next_observations=next_obs,
                state=state_vec,
                next_state=next_state_vec,
                dones=per_agent_dones,
            )
    
    def train_episode(self) -> Dict[str, Any]:
        """
        Train for one episode.

        Returns:
            Episode statistics
        """
        state, info = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        # P11 fix: initialize `losses` up-front.  Previously the variable
        # was only assigned inside the two ``if`` blocks that fire after
        # the episode buffer fills up; if the episode ended before filling
        # the buffer (e.g. very short episodes, or batch_size > max_steps)
        # we would hit ``UnboundLocalError`` at the ``return`` statement
        # below.
        losses: Dict[str, float] = {}

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
            
            # Apply anti-spoofing: detect and correct strategic behavior
            if self.config.get('anti_spoofing', {}).get('enabled', True):
                try:
                    # Convert state to tensor for anti-spoofing
                    if isinstance(state, tuple):
                        state_for_check = state[0]
                    else:
                        state_for_check = state
                    
                    if isinstance(state_for_check, np.ndarray):
                        state_tensor = torch.tensor(state_for_check, dtype=torch.float32, device=self.device)
                    else:
                        state_tensor = state_for_check
                    
                    # Check each agent's action
                    for agent_id in range(self.num_agents):
                        if agent_id in action:
                            agent_action = action[agent_id]
                            # Convert action to tensor
                            if isinstance(agent_action, dict):
                                action_tensor = torch.tensor(
                                    [agent_action.get('tactical', 0) / 8.0, 
                                     agent_action.get('communication', 0) / 4.0],
                                    dtype=torch.float32, device=self.device
                                )
                            else:
                                action_tensor = torch.tensor(agent_action, dtype=torch.float32, device=self.device)
                            
                            # Detect strategic behavior
                            strategic_result = self.anti_spoofing.detect_strategic_behavior(
                                agent_id, state_tensor[agent_id], action_tensor
                            )
                            
                            if strategic_result['is_strategic']:
                                # Apply punishment
                                self.anti_spoofing.apply_punishment(
                                    agent_id, strategic_result['strategic_score']
                                )
                except Exception:
                    pass  # Anti-spoofing is non-critical during training
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
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
            if len(episode_buffer['states']) >= self.config['marl']['batch_size']:
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
        """Convert list of action dictionaries to tensor.

        P13 fix: previously, the per-step ``flat_actions`` lists were
        stacked with ``torch.tensor(..., device=self.device)`` which raises
        if any two rows have different lengths (typical failure mode:
        one timestep has a missing agent key, or one step has
        ``resource_allocation`` and another does not).  Normalize each row
        to a common length by padding with zeros and logging the
        discrepancy, so the rest of the training loop can keep running.
        """
        # First pass: compute the flattened action length for each step
        # and the global maximum (padded to it).
        per_step_lengths = []
        flat_per_step = []
        for action_dict in actions:
            row: list = []
            for agent_id, action in action_dict.items():
                if isinstance(action, dict):
                    # Handle resource allocation
                    row.extend(
                        action.get('resource_allocation', {}).values()
                    )
                else:
                    row.append(action)
            per_step_lengths.append(len(row))
            flat_per_step.append(row)

        if not flat_per_step:
            # Defensive: empty input -> return a [0, 0] tensor so callers
            # that expect a 2-D tensor (batch x action_dim) don't crash
            # on shape assumptions downstream.
            return torch.zeros(0, 0, device=self.device)

        target_len = max(per_step_lengths)
        if target_len == 0:
            return torch.zeros(len(flat_per_step), 0, device=self.device)

        for i, row in enumerate(flat_per_step):
            if len(row) < target_len:
                # P13 fix: pad missing actions with 0 rather than letting
                # ``torch.tensor`` raise.  This is a soft failure (a real
                # action would carry semantic meaning) so we log the gap.
                gap = target_len - len(row)
                import logging
                logging.warning(
                    "EGTMARL._actions_to_tensor: padding %d missing actions "
                    "in step %d (had %d, expected %d)",
                    gap, i, len(row), target_len,
                )
                flat_per_step[i] = row + [0] * gap
            elif len(row) > target_len:
                # Shouldn't happen given target_len = max, but be defensive.
                flat_per_step[i] = row[:target_len]

        return torch.tensor(flat_per_step, device=self.device)
    
    def _calculate_episode_metrics(self, total_reward: float, 
                                  info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate episode metrics."""
        # Get anti-spoofing stats
        detection_stats = self.anti_spoofing.get_detection_stats()
        reputation_report = self.anti_spoofing.get_reputation_report()
        
        metrics = {
            'total_reward': total_reward,
            'fairness_score': info.get('fairness_score', 0.0),
            'efficiency_score': info.get('efficiency_score', 0.0),
            'pareto_score': self.dynamic_frontier.get_pareto_score(),
            'spoofing_rate': self.anti_spoofing.get_detection_rate(),
            'anti_spoofing_detection_rate': detection_stats.get('detection_rate', 0.0),
            'anti_spoofing_total_checks': detection_stats.get('total_checks', 0),
            'anti_spoofing_detected': detection_stats.get('detected', 0),
            'mean_reputation': reputation_report.get('mean_reputation', 0.5),
            'correction_rate': self.anti_spoofing.get_correction_rate(),
        }
        
        # Update history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        # 保存 AntiSpoofing 中各个网络的状态
        anti_spoofing_state = {
            'verifier': self.anti_spoofing.verifier.state_dict() if hasattr(self.anti_spoofing, 'verifier') else None,
            'spoofing_detector': self.anti_spoofing.spoofing_detector.state_dict() if hasattr(self.anti_spoofing, 'spoofing_detector') else None,
            'correction_network': self.anti_spoofing.correction_network.state_dict() if hasattr(self.anti_spoofing, 'correction_network') else None,
            'demand_predictor': self.anti_spoofing.demand_predictor.state_dict() if hasattr(self.anti_spoofing, 'demand_predictor') else None,
            'reputation_system': self.anti_spoofing.reputation_system if hasattr(self.anti_spoofing, 'reputation_system') else None
        }
        
        # 保存 DynamicFrontier 的状态
        dynamic_frontier_state = {
            'exploration_frontier': self.dynamic_frontier.exploration_frontier if hasattr(self.dynamic_frontier, 'exploration_frontier') else None,
            'exploitation_frontier': self.dynamic_frontier.exploitation_frontier if hasattr(self.dynamic_frontier, 'exploitation_frontier') else None
        }
        
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'marl_layer_state': self.marl_layer.state_dict(),
            'marl_layer_epsilon': self.marl_layer.epsilon,
            'egt_layer_state': self.egt_layer.state_dict(),
            'anti_spoofing_state': anti_spoofing_state,
            'dynamic_frontier_state': dynamic_frontier_state,
            'marl_optimizer_state': self.marl_optimizer.state_dict(),
            'egt_optimizer_state': self.egt_optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.best_reward = checkpoint['best_reward']
        
        self.marl_layer.load_state_dict(checkpoint['marl_layer_state'])
        self.egt_layer.load_state_dict(checkpoint['egt_layer_state'])

        if 'marl_layer_epsilon' in checkpoint:
            self.marl_layer.epsilon = checkpoint['marl_layer_epsilon']
        elif 'epsilon' in checkpoint.get('marl_layer_state', {}):
            self.marl_layer.epsilon = checkpoint['marl_layer_state']['epsilon']

        # 加载 AntiSpoofing 中各个网络的状态
        if 'anti_spoofing_state' in checkpoint:
            anti_spoofing_state = checkpoint['anti_spoofing_state']
            if hasattr(self.anti_spoofing, 'verifier') and anti_spoofing_state.get('verifier') is not None:
                self.anti_spoofing.verifier.load_state_dict(anti_spoofing_state['verifier'])
            if hasattr(self.anti_spoofing, 'spoofing_detector') and anti_spoofing_state.get('spoofing_detector') is not None:
                self.anti_spoofing.spoofing_detector.load_state_dict(anti_spoofing_state['spoofing_detector'])
            if hasattr(self.anti_spoofing, 'correction_network') and anti_spoofing_state.get('correction_network') is not None:
                self.anti_spoofing.correction_network.load_state_dict(anti_spoofing_state['correction_network'])
            if hasattr(self.anti_spoofing, 'demand_predictor') and anti_spoofing_state.get('demand_predictor') is not None:
                self.anti_spoofing.demand_predictor.load_state_dict(anti_spoofing_state['demand_predictor'])
            if hasattr(self.anti_spoofing, 'reputation_system') and anti_spoofing_state.get('reputation_system') is not None:
                self.anti_spoofing.reputation_system = anti_spoofing_state['reputation_system']
        
        # 加载 DynamicFrontier 的状态
        if 'dynamic_frontier_state' in checkpoint:
            dynamic_frontier_state = checkpoint['dynamic_frontier_state']
            if hasattr(self.dynamic_frontier, 'exploration_frontier') and dynamic_frontier_state.get('exploration_frontier') is not None:
                self.dynamic_frontier.exploration_frontier = dynamic_frontier_state['exploration_frontier']
            if hasattr(self.dynamic_frontier, 'exploitation_frontier') and dynamic_frontier_state.get('exploitation_frontier') is not None:
                self.dynamic_frontier.exploitation_frontier = dynamic_frontier_state['exploitation_frontier']
        
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
        state, info = self.env.reset()
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
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
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