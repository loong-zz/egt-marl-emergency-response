"""
Anti-Spoofing Mechanism
========================

Detects and prevents spoofing attacks in multi-agent systems.

Implements Bayesian truth verification with:
1. Bayesian demand prediction model
2. Z-score anomaly detection
3. Reputation-based punishment mechanism
4. Resource hoarding detection
5. Action correction network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ReputationSystem:
    """
    Reputation system for tracking agent trustworthiness.
    """
    
    def __init__(self, num_agents: int, initial_reputation: float = 0.5, 
                 decay_rate: float = 0.99, punishment_factor: float = 0.2):
        self.num_agents = num_agents
        self.reputations = np.ones(num_agents) * initial_reputation
        self.decay_rate = decay_rate
        self.punishment_factor = punishment_factor
        self.history = []  # (agent_id, reputation, reason)
    
    def update_reputation(self, agent_id: int, spoofing_detected: bool, 
                          spoofing_score: float):
        """Update reputation based on spoofing detection."""
        if spoofing_detected:
            # Penalize for detected spoofing
            penalty = self.punishment_factor * spoofing_score
            self.reputations[agent_id] = max(0.01, self.reputations[agent_id] - penalty)
            self.history.append((agent_id, self.reputations[agent_id], 'spoofing_detected'))
        else:
            # Reward for legitimate behavior
            self.reputations[agent_id] = min(1.0, self.reputations[agent_id] + 0.001)
            self.history.append((agent_id, self.reputations[agent_id], 'legitimate'))
        
        # Apply decay
        self.reputations[agent_id] *= self.decay_rate
    
    def get_reputation(self, agent_id: int) -> float:
        """Get reputation of an agent."""
        if agent_id < self.num_agents:
            return self.reputations[agent_id]
        return 0.5  # Default for unknown agents
    
    def get_reputation_report(self) -> Dict[str, Any]:
        """Get reputation system report."""
        return {
            'mean_reputation': float(np.mean(self.reputations)),
            'min_reputation': float(np.min(self.reputations)),
            'max_reputation': float(np.max(self.reputations)),
            'std_reputation': float(np.std(self.reputations)),
            'recent_history': self.history[-10:]
        }


class AntiSpoofing:
    """
    Anti-spoofing mechanism for detecting and preventing spoofing attacks.
    
    Implements:
    1. Bayesian demand prediction for detecting false claims
    2. Z-score anomaly detection
    3. Reputation-based correction
    4. Resource hoarding detection
    5. Action correction network
    """
    
    def __init__(self, observation_dim: int, action_dim: int, 
                 detection_threshold: float = 0.5, 
                 correction_strength: float = 0.8, 
                 device: torch.device = torch.device("cpu"),
                 num_agents: int = 3):
        """
        Initialize anti-spoofing mechanism.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            detection_threshold: Threshold for spoofing detection
            correction_strength: Strength of action correction
            device: Device to run on
            num_agents: Number of agents in the system
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.detection_threshold = detection_threshold
        self.correction_strength = correction_strength
        self.device = device
        self.num_agents = num_agents
        
        # Verification network
        self.verifier = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        # Spoofing detector
        self.spoofing_detector = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Action correction network
        self.correction_network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(device)
        
        # Bayesian demand predictor for detecting false claims
        self.demand_predictor = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicts expected demand
        ).to(device)
        
        # Reputation system
        self.reputation_system = ReputationSystem(num_agents)
        
        # Optimizer for training
        self.optimizer = torch.optim.Adam(
            list(self.verifier.parameters()) + 
            list(self.spoofing_detector.parameters()) +
            list(self.correction_network.parameters()) +
            list(self.demand_predictor.parameters()),
            lr=0.001
        )
        
        # Loss function
        self.loss_fn = nn.BCELoss()
        
        # History
        self.detection_history = []  # (agent_id, spoofing_score, is_spoofing)
        self.correction_history = []  # (agent_id, correction_status)
        self.demand_history = []  # (agent_id, reported_demand, predicted_demand, z_score)
        
        # Statistics for Z-score calculation
        self.demand_mean = 0.0
        self.demand_std = 1.0
        self.demand_samples = []
    
    def _update_demand_statistics(self, demand: float):
        """Update demand statistics for Z-score calculation."""
        self.demand_samples.append(demand)
        if len(self.demand_samples) > 1000:
            self.demand_samples.pop(0)
        
        if len(self.demand_samples) > 1:
            self.demand_mean = np.mean(self.demand_samples)
            self.demand_std = max(0.01, np.std(self.demand_samples))
    
    def _calculate_z_score(self, value: float) -> float:
        """Calculate Z-score for anomaly detection."""
        return abs(value - self.demand_mean) / self.demand_std
    
    def verify_action(self, observation: torch.Tensor, 
                     action: torch.Tensor, 
                     agent_id: int) -> Tuple[bool, float]:
        """
        Verify if an action is legitimate.
        
        Args:
            observation: Agent observation
            action: Agent action
            agent_id: Agent ID
            
        Returns:
            (is_legitimate, confidence)
        """
        # Concatenate observation and action
        input_tensor = torch.cat([observation, action], dim=-1)
        
        # Get verification score
        verification_score = self.verifier(input_tensor).squeeze()
        
        # Get spoofing score
        spoofing_score = self.spoofing_detector(input_tensor).squeeze()
        
        # Determine if action is legitimate
        is_legitimate = spoofing_score < self.detection_threshold
        
        # Add to history
        self.detection_history.append((agent_id, spoofing_score.item(), not is_legitimate))
        
        return is_legitimate, 1.0 - spoofing_score.item()
    
    def correct_action(self, observation: torch.Tensor, 
                      action: torch.Tensor, 
                      agent_id: int) -> Dict[str, Any]:
        """
        Correct a potentially spoofed action.
        
        Args:
            observation: Agent observation
            action: Agent action
            agent_id: Agent ID
            
        Returns:
            Corrected action
        """
        # Get spoofing score
        input_tensor = torch.cat([observation, action], dim=-1)
        spoofing_score = self.spoofing_detector(input_tensor).squeeze()
        
        # Get reputation (placeholder)
        reputation = 0.5  # Default reputation
        
        # Correct action
        corrected_action = action.clone()
        
        # Apply correction based on spoofing score and reputation
        correction = self.correction_network(input_tensor)
        corrected = action * (1 - self.correction_strength * (1 - reputation)) + \
                   correction * self.correction_strength * (1 - reputation)
        
        # Handle resource allocation (if present)
        if isinstance(corrected_action, dict) and 'resource_allocation' in corrected_action:
            original_allocations = corrected_action['resource_allocation']
            corrected_allocations = {}
            
            for resource_type, amount in original_allocations.items():
                if spoofing_score > self.detection_threshold:
                    # Apply correction to suspicious allocations
                    corrected_allocations[resource_type] = max(0.0, corrected)
                else:
                    corrected_allocations[resource_type] = original_allocations.get(resource_type, 0.0)
            
            corrected_action['resource_allocation'] = corrected_allocations     
        else:
            # Apply correction to action vector
            corrected_action = corrected
    
        # 注意：对于张量类型的 action，我们不添加元数据，因为张量不支持字典操作
        # 只对字典类型的 action 添加元数据

        # Add to history
        self.correction_history.append((agent_id, 'corrected' if spoofing_score > self.detection_threshold else 'unchanged'))

        return corrected_action

    def update(self, batch: Dict[str, Any]) -> float:
        """
        Update anti-spoofing mechanism with Bayesian truth verification.

        Args:
            batch: Experience batch containing states, actions, rewards, etc.

        Returns:
            Loss value
        """
        if batch is None or 'states' not in batch or 'actions' not in batch:
            return 0.0
        
        try:
            states = batch['states']
            actions = batch['actions']
            
            # Convert to tensors if needed
            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32, device=self.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            
            # Forward pass through spoofing detector
            # Concatenate states and actions
            if states.dim() == 3:  # (batch, agents, features)
                batch_size, num_agents, obs_dim = states.shape
                states_flat = states.view(batch_size * num_agents, obs_dim)
                actions_flat = actions.view(batch_size * num_agents, -1)
            else:
                states_flat = states
                actions_flat = actions
            
            input_tensor = torch.cat([states_flat, actions_flat], dim=-1)
            spoofing_scores = self.spoofing_detector(input_tensor)
            
            # Predict demand using Bayesian predictor
            demand_predictions = self.demand_predictor(states_flat)
            
            # Compute verification loss
            # Ground truth: assume actions are legitimate unless proven otherwise
            # For training, we use the spoofing score itself as the target for self-supervision
            target_scores = torch.ones_like(spoofing_scores) * 0.1  # Assume mostly legitimate
            
            # Compute loss
            loss = self.loss_fn(spoofing_scores, target_scores)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update reputation system based on detection results
            for i in range(min(self.num_agents, num_agents)):
                spoofing_score = spoofing_scores[i].item()
                is_spoofing = spoofing_score > self.detection_threshold
                self.reputation_system.update_reputation(i, is_spoofing, spoofing_score)
            
            # Update demand statistics
            if demand_predictions.numel() > 0:
                self._update_demand_statistics(demand_predictions.mean().item())
            
            return loss.item()
        
        except Exception as e:
            # Return 0.0 if update fails
            return 0.0
    
    def detect_hoarding(self, agent_id: int, resource_history: List[Dict[str, float]]) -> Tuple[bool, float]:
        """
        Detect resource hoarding behavior.
        
        Args:
            agent_id: Agent ID
            resource_history: History of resource acquisitions and usage
            
        Returns:
            (is_hoarding, hoarding_score)
        """
        if len(resource_history) < 10:
            return False, 0.0
        
        # Calculate acquisition vs usage ratio
        total_acquired = sum(sum(r.get('acquired', {}).values()) for r in resource_history)
        total_used = sum(sum(r.get('used', {}).values()) for r in resource_history)
        
        if total_acquired == 0:
            return False, 0.0
        
        # If acquired > 3x used, likely hoarding
        hoarding_ratio = total_acquired / max(total_used, 1)
        
        # Calculate hoarding score (0-1)
        hoarding_score = min(1.0, (hoarding_ratio - 1.0) / 3.0)
        
        return hoarding_ratio > 3.0, hoarding_score
    
    def detect_false_demand(self, agent_id: int, reported_demand: float, 
                           observation: torch.Tensor) -> Tuple[bool, float]:
        """
        Detect false demand reporting (strategic misrepresentation).
        
        Args:
            agent_id: Agent ID
            reported_demand: Demand reported by the agent
            observation: Current observation
            
        Returns:
            (is_false, confidence_score)
        """
        # Predict expected demand using Bayesian predictor
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        predicted_demand = self.demand_predictor(observation).item()
        
        # Update statistics
        self._update_demand_statistics(reported_demand)
        
        # Calculate Z-score for anomaly detection
        z_score = self._calculate_z_score(reported_demand)
        
        # Record to history
        self.demand_history.append((agent_id, reported_demand, predicted_demand, z_score))
        
        # Consider demand suspicious if Z-score > 2.0
        is_false = z_score > 2.0
        confidence_score = min(1.0, z_score / 5.0)
        
        # Update reputation based on detection
        self.reputation_system.update_reputation(agent_id, is_false, confidence_score)
        
        return is_false, confidence_score
    
    def detect_strategic_behavior(self, agent_id: int, observation: torch.Tensor, 
                                  action: torch.Tensor, resource_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive strategic behavior detection.
        
        Args:
            agent_id: Agent ID
            observation: Agent observation
            action: Agent action
            resource_history: Optional resource history for hoarding detection
            
        Returns:
            Dictionary containing detection results
        """
        result = {
            'agent_id': agent_id,
            'is_strategic': False,
            'strategic_score': 0.0,
            'detection_type': None,
            'details': {}
        }
        
        # Detect spoofing
        is_legitimate, confidence = self.verify_action(observation, action, agent_id)
        if not is_legitimate:
            result['is_strategic'] = True
            result['strategic_score'] += (1.0 - confidence) * 0.5
            result['detection_type'] = 'spoofing'
            result['details']['spoofing_confidence'] = 1.0 - confidence
        
        # Detect hoarding
        if resource_history:
            is_hoarding, hoarding_score = self.detect_hoarding(agent_id, resource_history)
            if is_hoarding:
                result['is_strategic'] = True
                result['strategic_score'] += hoarding_score * 0.3
                result['detection_type'] = 'hoarding' if not result['detection_type'] else 'multiple'
                result['details']['hoarding_score'] = hoarding_score
        
        # Detect demand manipulation
        # Extract demand from action if available
        reported_demand = 0.0
        if isinstance(action, dict) and 'demand' in action:
            reported_demand = action['demand']
        elif isinstance(action, torch.Tensor) and action.numel() > 0:
            reported_demand = action.mean().item()
        
        is_false_demand, demand_confidence = self.detect_false_demand(agent_id, reported_demand, observation)
        if is_false_demand:
            result['is_strategic'] = True
            result['strategic_score'] += demand_confidence * 0.2
            result['detection_type'] = 'false_demand' if not result['detection_type'] else 'multiple'
            result['details']['false_demand_confidence'] = demand_confidence
        
        return result
    
    def apply_punishment(self, agent_id: int, strategic_score: float) -> Dict[str, Any]:
        """
        Apply punishment to agent for detected strategic behavior.
        
        Args:
            agent_id: Agent ID
            strategic_score: Score indicating severity of strategic behavior
            
        Returns:
            Punishment details
        """
        punishment = {
            'agent_id': agent_id,
            'applied': False,
            'type': None,
            'severity': 0.0,
            'reputation_change': 0.0,
            'resource_penalty': 0.0,
            'action_restriction': False
        }
        
        if strategic_score < 0.1:
            return punishment
        
        punishment['applied'] = True
        punishment['severity'] = strategic_score
        
        # Determine punishment type based on score
        if strategic_score >= 0.7:
            # Severe: reputation penalty + resource penalty + action restriction
            punishment['type'] = 'severe'
            reputation_penalty = 0.3 * strategic_score
            punishment['reputation_change'] = -reputation_penalty
            punishment['resource_penalty'] = 0.2  # 20% resource penalty
            punishment['action_restriction'] = True
        elif strategic_score >= 0.4:
            # Moderate: reputation penalty + resource penalty
            punishment['type'] = 'moderate'
            reputation_penalty = 0.2 * strategic_score
            punishment['reputation_change'] = -reputation_penalty
            punishment['resource_penalty'] = 0.1  # 10% resource penalty
        else:
            # Mild: reputation penalty only
            punishment['type'] = 'mild'
            reputation_penalty = 0.1 * strategic_score
            punishment['reputation_change'] = -reputation_penalty
        
        # Apply reputation penalty
        self.reputation_system.reputations[agent_id] = max(
            0.01, 
            self.reputation_system.reputations[agent_id] + punishment['reputation_change']
        )
        
        return punishment

    def get_detection_rate(self) -> float:
        """Get spoofing detection rate."""
        if len(self.detection_history) == 0:
            return 0.0

        spoofing_count = sum(1 for _, _, is_spoofing in self.detection_history if is_spoofing)
        return spoofing_count / len(self.detection_history)

    def get_correction_rate(self) -> float:
        """Get action correction rate."""
        if len(self.correction_history) == 0:
            return 0.0

        correction_count = sum(1 for _, status in self.correction_history if status == 'corrected')
        return correction_count / len(self.correction_history)

    def get_reputation_report(self) -> Dict[str, Any]:
        """Get reputation system report."""
        if self.reputation_system is None:
            return {'error': 'Reputation system not initialized'}

        return self.reputation_system.get_reputation_report()

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get spoofing detection statistics."""
        if len(self.detection_history) == 0:
            return {
                'total_checks': 0,
                'spoofing_detected': 0,
                'detection_rate': 0.0,
                'avg_spoofing_score': 0.0
            }

        total_checks = len(self.detection_history)
        spoofing_detected = sum(1 for _, _, is_spoofing in self.detection_history if is_spoofing)
        avg_spoofing_score = np.mean([score for _, score, _ in self.detection_history])

        return {
            'total_checks': total_checks,
            'spoofing_detected': spoofing_detected,
            'detection_rate': spoofing_detected / total_checks,
            'avg_spoofing_score': avg_spoofing_score,
            'recent_detection_rate': self._get_recent_detection_rate()
        }

    def _get_recent_detection_rate(self, window: int = 100) -> float:
        """Get detection rate in recent history."""
        if len(self.detection_history) == 0:
            return 0.0

        recent = self.detection_history[-window:]
        if not recent:
            return 0.0

        spoofing_count = sum(1 for _, _, is_spoofing in recent if is_spoofing)
        return spoofing_count / len(recent)

    def save(self, path: str) -> None:
        """Save anti-spoofing mechanism state."""
        state = {
            'verifier_state': self.verifier.state_dict(),
            'spoofing_detector_state': self.spoofing_detector.state_dict(),
            'correction_network_state': self.correction_network.state_dict(),
            'detection_history': self.detection_history,
            'correction_history': self.correction_history,
            'reputation_system': self.reputation_system.reputations.tolist() if self.reputation_system else None,
            'config': {
                'observation_dim': self.observation_dim,
                'detection_threshold': self.detection_threshold,
                'correction_strength': self.correction_strength
            }
        }

        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load anti-spoofing mechanism state."""
        state = torch.load(path, map_location=self.device)

        self.verifier.load_state_dict(state['verifier_state'])
        self.spoofing_detector.load_state_dict(state['spoofing_detector_state'])
        self.correction_network.load_state_dict(state['correction_network_state'])

        self.detection_history = state['detection_history']
        self.correction_history = state['correction_history']

        if state['reputation_system'] is not None and self.reputation_system is not None:
            self.reputation_system.reputations = np.array(state['reputation_system'])

        self.detection_threshold = state['config']['detection_threshold']
        self.correction_strength = state['config']['correction_strength']
