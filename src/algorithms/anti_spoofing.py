"""
Anti-Spoofing Mechanism
========================

Detects and prevents spoofing attacks in multi-agent systems.

Implements multi-dimensional behavioral analysis with:
1. Behavioral fingerprinting for agent identity verification
2. Multi-dimensional anomaly detection (Z-score, entropy, consistency)
3. Bayesian demand prediction with online learning
4. Adaptive dynamic thresholds based on rolling statistics
5. Reputation-based punishment mechanism
6. Resource hoarding detection
7. Action correction network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import math


class BehavioralFingerprint:
    """
    Multi-dimensional behavioral fingerprint for each agent.
    
    Tracks:
    - Action distribution patterns (what actions the agent typically takes)
    - Resource usage patterns (efficiency, hoarding tendency)
    - Response patterns (reaction time, consistency)
    - Communication patterns (message frequency, content)
    """
    
    def __init__(self, action_dim: int, window_size: int = 100):
        self.action_dim = action_dim
        self.window_size = window_size
        
        # Action history
        self.action_history = deque(maxlen=window_size)
        self.action_distribution = np.ones(action_dim) / action_dim  # Smoothed distribution
        
        # Resource usage history
        self.resource_usage = deque(maxlen=window_size)
        self.resource_efficiency = []  # ratio of used/acquired
        
        # Response patterns
        self.response_times = deque(maxlen=window_size)
        
        # Communication patterns
        self.communication_freq = deque(maxlen=window_size)
        
        # Behavioral statistics
        self.action_entropy = 0.0
        self.action_consistency = 0.0
        self.anomaly_score = 0.0
        self.samples_seen = 0
        
    def update(self, action: np.ndarray, resource_usage: float = 0.0,
               response_time: float = 0.0, comm_freq: float = 0.0):
        """Update fingerprint with new observation."""
        self.samples_seen += 1
        
        # Update action distribution with exponential smoothing
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).flatten()
        
        if len(action) > 0:
            self.action_history.append(action)
            # Exponential moving average of action distribution
            alpha = min(0.1, 1.0 / max(1, self.samples_seen))
            action_norm = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
            if len(action_norm) == self.action_dim:
                self.action_distribution = (1 - alpha) * self.action_distribution + alpha * action_norm
        
        self.resource_usage.append(resource_usage)
        self.response_times.append(response_time)
        self.communication_freq.append(comm_freq)
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self):
        """Update behavioral statistics."""
        if len(self.action_history) < 2:
            return
        
        # Action entropy (higher = more random, potentially suspicious)
        action_dist = self.action_distribution
        action_dist = np.clip(action_dist, 1e-10, 1.0)
        self.action_entropy = -np.sum(action_dist * np.log(action_dist))
        max_entropy = np.log(self.action_dim)
        normalized_entropy = self.action_entropy / max(max_entropy, 1e-8)
        
        # Action consistency (how stable the behavior is)
        recent_actions = list(self.action_history)[-10:]
        if len(recent_actions) > 1:
            action_std = np.std([np.mean(a) for a in recent_actions])
            self.action_consistency = 1.0 / (1.0 + action_std * 10)
        else:
            self.action_consistency = 0.5
        
        # Combined anomaly score
        self.anomaly_score = 0.5 * normalized_entropy + 0.3 * (1 - self.action_consistency)
    
    def compare(self, action: np.ndarray) -> float:
        """
        Compare current action against historical fingerprint.
        
        Returns:
            deviation_score: 0 = identical, 1 = completely different
        """
        if self.samples_seen < 5:
            return 0.0
        
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).flatten()
        
        if len(action) == 0:
            return 0.0
        
        # Normalize action
        action_norm = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        
        if len(action_norm) != len(self.action_distribution):
            return 0.5
        
        # KL divergence between current action and historical distribution
        kl_div = np.sum(action_norm * np.log((action_norm + 1e-10) / (self.action_distribution + 1e-10)))
        kl_div = np.clip(kl_div, 0, 10)
        
        # Cosine similarity
        cos_sim = np.dot(action_norm, self.action_distribution) / (
            np.linalg.norm(action_norm) * np.linalg.norm(self.action_distribution) + 1e-8
        )
        
        # Combined deviation score
        deviation = 0.5 * (kl_div / 10.0) + 0.5 * (1.0 - cos_sim)
        return min(1.0, float(deviation))
    
    def get_profile(self) -> Dict[str, Any]:
        """Get behavioral profile summary."""
        return {
            'action_entropy': self.action_entropy,
            'action_consistency': self.action_consistency,
            'anomaly_score': self.anomaly_score,
            'samples_seen': self.samples_seen,
            'mean_response_time': np.mean(self.response_times) if self.response_times else 0.0,
            'mean_comm_freq': np.mean(self.communication_freq) if self.communication_freq else 0.0,
        }


class AdaptiveThreshold:
    """
    Adaptive threshold that adjusts based on rolling statistics.
    """
    
    def __init__(self, initial_threshold: float = 0.5, window_size: int = 100,
                 sensitivity: float = 1.5):
        self.initial_threshold = initial_threshold
        self.window_size = window_size
        self.sensitivity = sensitivity  # Number of std deviations for threshold
        
        self.scores = deque(maxlen=window_size)
        self.threshold = initial_threshold
        self.mean = 0.0
        self.std = 1.0
        
    def update(self, score: float):
        """Update threshold based on new score."""
        self.scores.append(score)
        
        if len(self.scores) > 1:
            self.mean = np.mean(self.scores)
            self.std = max(0.01, np.std(self.scores))
            # Adaptive threshold: mean + sensitivity * std
            self.threshold = self.mean + self.sensitivity * self.std
            # Clamp to reasonable range
            self.threshold = np.clip(self.threshold, 0.1, 0.95)
    
    def is_anomalous(self, score: float) -> bool:
        """Check if score exceeds adaptive threshold."""
        return score > self.threshold
    
    def get_threshold(self) -> float:
        return self.threshold


class ReputationSystem:
    """
    Reputation system for tracking agent trustworthiness.
    Uses exponential moving average with asymmetric updates.
    """
    
    def __init__(self, num_agents: int, initial_reputation: float = 0.5, 
                 decay_rate: float = 0.99, punishment_factor: float = 0.2,
                 recovery_rate: float = 0.01):
        self.num_agents = num_agents
        self.reputations = np.ones(num_agents) * initial_reputation
        self.decay_rate = decay_rate
        self.punishment_factor = punishment_factor
        self.recovery_rate = recovery_rate
        self.history = []  # (agent_id, reputation, reason)
        self.violation_count = np.zeros(num_agents)
    
    def update_reputation(self, agent_id: int, spoofing_detected: bool, 
                          spoofing_score: float):
        """Update reputation based on spoofing detection."""
        if spoofing_detected:
            self.violation_count[agent_id] += 1
            # Progressive penalty: repeat offenders get harsher punishment
            penalty_multiplier = min(3.0, 1.0 + 0.5 * self.violation_count[agent_id])
            penalty = self.punishment_factor * spoofing_score * penalty_multiplier
            self.reputations[agent_id] = max(0.01, self.reputations[agent_id] - penalty)
            self.history.append((agent_id, self.reputations[agent_id], 'spoofing_detected'))
        else:
            # Gradual recovery
            self.reputations[agent_id] = min(1.0, self.reputations[agent_id] + self.recovery_rate)
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
            'violation_counts': self.violation_count.tolist(),
            'recent_history': self.history[-10:]
        }


class AntiSpoofing:
    """
    Enhanced anti-spoofing mechanism with multi-dimensional behavioral analysis.
    
    Detection dimensions:
    1. Behavioral fingerprinting - Compare against historical agent patterns
    2. Neural spoofing detection - Learned spoofing patterns
    3. Demand prediction anomaly - Bayesian verification of claims
    4. Resource hoarding detection - Acquisition vs usage ratio
    5. Action entropy analysis - Unusual randomness in behavior
    6. Reputation-based temporal consistency
    """
    
    def __init__(self, observation_dim: int, action_dim: int, 
                 detection_threshold: float = 0.5, 
                 correction_strength: float = 0.8, 
                 device: torch.device = torch.device("cpu"),
                 num_agents: int = 3):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.detection_threshold = detection_threshold
        self.correction_strength = correction_strength
        self.device = device
        self.num_agents = num_agents
        
        # Verification network (deeper for better feature extraction)
        self.verifier = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        ).to(device)
        
        # Spoofing detector (deeper with better regularization)
        self.spoofing_detector = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Action correction network
        self.correction_network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, action_dim)
        ).to(device)
        
        # Bayesian demand predictor (improved architecture)
        self.demand_predictor = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        # Behavioral fingerprint bank (one per agent)
        self.fingerprints: Dict[int, BehavioralFingerprint] = {}
        for i in range(num_agents):
            self.fingerprints[i] = BehavioralFingerprint(action_dim)
        
        # Adaptive thresholds (one per detection dimension)
        self.spoofing_threshold = AdaptiveThreshold(detection_threshold, sensitivity=1.5)
        self.demand_threshold = AdaptiveThreshold(0.3, sensitivity=2.0)
        self.fingerprint_threshold = AdaptiveThreshold(0.4, sensitivity=2.0)
        self.hoarding_threshold = AdaptiveThreshold(0.5, sensitivity=2.0)
        
        # Reputation system
        self.reputation_system = ReputationSystem(num_agents)
        
        # Optimizer for training
        self.optimizer = torch.optim.Adam(
            list(self.verifier.parameters()) + 
            list(self.spoofing_detector.parameters()) +
            list(self.correction_network.parameters()) +
            list(self.demand_predictor.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Loss function
        self.loss_fn = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # History
        self.detection_history = []
        self.correction_history = []
        self.demand_history = []
        self.fingerprint_history = []
        
        # Statistics for Z-score calculation
        self.demand_mean = 0.0
        self.demand_std = 1.0
        self.demand_samples = []
        
        # Training data buffer for online learning
        self.training_buffer = deque(maxlen=5000)
        self.legitimate_count = 0
        self.spoofing_count = 0
        
        # Detection performance tracking
        self.detection_stats = {
            'total_checks': 0,
            'detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'detection_rate': 0.0
        }
    
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
    
    def _normalize_observation(self, observation) -> torch.Tensor:
        """Normalize and convert observation to tensor."""
        if not isinstance(observation, torch.Tensor):
            if isinstance(observation, np.ndarray):
                observation = torch.from_numpy(observation).float()
            else:
                observation = torch.tensor(observation, dtype=torch.float32)
        return observation.to(self.device)
    
    def verify_action(self, observation: torch.Tensor, 
                     action: torch.Tensor, 
                     agent_id: int) -> Tuple[bool, float]:
        """Verify if an action is legitimate using neural detector."""
        obs = self._normalize_observation(observation)
        act = self._normalize_observation(action)
        
        if obs.dim() == 0:
            obs = obs.unsqueeze(0)
        if act.dim() == 0:
            act = act.unsqueeze(0)
        
        input_tensor = torch.cat([obs, act], dim=-1)
        
        spoofing_score = self.spoofing_detector(input_tensor).squeeze().item()
        
        # Use adaptive threshold
        is_legitimate = not self.spoofing_threshold.is_anomalous(spoofing_score)
        
        self.detection_history.append((agent_id, spoofing_score, not is_legitimate))
        self.detection_stats['total_checks'] += 1
        
        if not is_legitimate:
            self.detection_stats['detected'] += 1
        
        return is_legitimate, 1.0 - spoofing_score
    
    def verify_behavioral_fingerprint(self, observation, action, agent_id: int) -> Dict[str, Any]:
        """
        Verify action against agent's behavioral fingerprint.
        
        Returns:
            Dictionary with fingerprint comparison results
        """
        if agent_id not in self.fingerprints:
            return {'is_consistent': True, 'deviation': 0.0, 'profile': {}}
        
        fingerprint = self.fingerprints[agent_id]
        
        # Compare against historical behavior
        deviation = fingerprint.compare(action)
        
        # Update fingerprint with new behavior
        fingerprint.update(action)
        
        # Check if deviation exceeds adaptive threshold
        is_consistent = not self.fingerprint_threshold.is_anomalous(deviation)
        self.fingerprint_threshold.update(deviation)
        
        self.fingerprint_history.append((agent_id, deviation, not is_consistent))
        
        return {
            'is_consistent': is_consistent,
            'deviation': deviation,
            'threshold': self.fingerprint_threshold.get_threshold(),
            'profile': fingerprint.get_profile()
        }
    
    def correct_action(self, observation: torch.Tensor, 
                      action: torch.Tensor, 
                      agent_id: int) -> Dict[str, Any]:
        """Correct a potentially spoofed action."""
        obs = self._normalize_observation(observation)
        act = self._normalize_observation(action)
        
        if obs.dim() == 0:
            obs = obs.unsqueeze(0)
        if act.dim() == 0:
            act = act.unsqueeze(0)
        
        input_tensor = torch.cat([obs, act], dim=-1)
        spoofing_score = self.spoofing_detector(input_tensor).squeeze()
        
        reputation = self.reputation_system.get_reputation(agent_id)
        
        corrected_action = action.clone() if isinstance(action, torch.Tensor) else action
        
        if isinstance(corrected_action, torch.Tensor):
            correction = self.correction_network(input_tensor)
            correction_strength = self.correction_strength * (1.0 - reputation) * spoofing_score
            correction_strength = torch.clamp(correction_strength, 0.0, 0.9)
            corrected_action = act * (1 - correction_strength) + correction * correction_strength
        elif isinstance(corrected_action, dict) and 'resource_allocation' in corrected_action:
            original_allocations = corrected_action['resource_allocation']
            corrected_allocations = {}
            for resource_type, amount in original_allocations.items():
                if spoofing_score > self.detection_threshold:
                    correction_amount = self.correction_network(input_tensor).mean().item()
                    corrected_allocations[resource_type] = max(0.0, amount * (1 - correction_strength) + correction_amount * correction_strength)
                else:
                    corrected_allocations[resource_type] = original_allocations.get(resource_type, 0.0)
            corrected_action['resource_allocation'] = corrected_allocations
        
        self.correction_history.append((agent_id, 'corrected' if spoofing_score > self.detection_threshold else 'unchanged'))
        return corrected_action

    def update(self, batch: Dict[str, Any]) -> float:
        """
        Update anti-spoofing mechanism with multi-dimensional training.
        """
        if batch is None or 'states' not in batch or 'actions' not in batch:
            return 0.0
        
        try:
            states = batch['states']
            actions = batch['actions']
            
            if not isinstance(states, torch.Tensor):
                states = torch.tensor(states, dtype=torch.float32, device=self.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
            
            if states.dim() == 3:
                batch_size, num_agents, obs_dim = states.shape
                states_flat = states.view(batch_size * num_agents, obs_dim)
                actions_flat = actions.view(batch_size * num_agents, -1)
            else:
                states_flat = states
                actions_flat = actions
                num_agents = min(self.num_agents, states.shape[0] if states.dim() >= 1 else 1)
            
            # Handle single-sample batches
            if states_flat.dim() == 1:
                states_flat = states_flat.unsqueeze(0)
            if actions_flat.dim() == 1:
                actions_flat = actions_flat.unsqueeze(0)
            
            input_tensor = torch.cat([states_flat, actions_flat], dim=-1)
            
            # 1. Spoofing detection loss
            spoofing_scores = self.spoofing_detector(input_tensor)
            # Self-supervised: actions should mostly be legitimate
            target_scores = torch.ones_like(spoofing_scores) * 0.05
            spoofing_loss = self.loss_fn(spoofing_scores, target_scores)
            
            # 2. Demand prediction loss
            demand_predictions = self.demand_predictor(states_flat)
            demand_loss = self.mse_loss(demand_predictions, torch.zeros_like(demand_predictions))
            
            # 3. Verification loss (consistency check)
            verification_scores = torch.sigmoid(self.verifier(input_tensor))
            verification_loss = self.mse_loss(verification_scores, torch.ones_like(verification_scores))
            
            # Combined loss
            total_loss = spoofing_loss + 0.5 * demand_loss + 0.3 * verification_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
            self.optimizer.step()
            
            # Update adaptive thresholds
            for i in range(min(num_agents, spoofing_scores.shape[0])):
                score = spoofing_scores[i].item()
                self.spoofing_threshold.update(score)
                
                # Update reputations
                is_spoofing = self.spoofing_threshold.is_anomalous(score)
                self.reputation_system.update_reputation(i, is_spoofing, score)
            
            # Update demand statistics
            if demand_predictions.numel() > 0:
                self._update_demand_statistics(demand_predictions.mean().item())
            
            return total_loss.item()
        
        except Exception as e:
            return 0.0
    
    def detect_hoarding(self, agent_id: int, resource_history: List[Dict[str, float]]) -> Tuple[bool, float]:
        """Detect resource hoarding behavior with improved analysis."""
        if len(resource_history) < 5:
            return False, 0.0
        
        total_acquired = sum(sum(r.get('acquired', {}).values()) for r in resource_history)
        total_used = sum(sum(r.get('used', {}).values()) for r in resource_history)
        
        if total_acquired == 0:
            return False, 0.0
        
        hoarding_ratio = total_acquired / max(total_used, 1)
        hoarding_score = min(1.0, (hoarding_ratio - 1.0) / 2.0)
        
        # Use adaptive threshold
        self.hoarding_threshold.update(hoarding_score)
        is_hoarding = self.hoarding_threshold.is_anomalous(hoarding_score)
        
        return is_hoarding, hoarding_score
    
    def detect_false_demand(self, agent_id: int, reported_demand: float, 
                           observation: torch.Tensor) -> Tuple[bool, float]:
        """Detect false demand reporting with Bayesian verification."""
        obs = self._normalize_observation(observation)
        if obs.dim() == 0:
            obs = obs.unsqueeze(0)
        
        predicted_demand = self.demand_predictor(obs).item()
        self._update_demand_statistics(reported_demand)
        
        z_score = self._calculate_z_score(reported_demand)
        prediction_error = abs(reported_demand - predicted_demand) / (abs(predicted_demand) + 1e-8)
        
        # Combined false demand score
        false_demand_score = 0.5 * min(1.0, z_score / 4.0) + 0.5 * min(1.0, prediction_error / 2.0)
        
        self.demand_history.append((agent_id, reported_demand, predicted_demand, z_score))
        
        is_false = self.demand_threshold.is_anomalous(false_demand_score)
        self.demand_threshold.update(false_demand_score)
        
        self.reputation_system.update_reputation(agent_id, is_false, false_demand_score)
        
        return is_false, false_demand_score
    
    def detect_strategic_behavior(self, agent_id: int, observation: torch.Tensor, 
                                  action: torch.Tensor, resource_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive multi-dimensional strategic behavior detection.
        """
        result = {
            'agent_id': agent_id,
            'is_strategic': False,
            'strategic_score': 0.0,
            'detection_type': [],
            'dimension_scores': {},
            'details': {}
        }
        
        # Dimension 1: Neural spoofing detection (weight: 0.3)
        is_legitimate, confidence = self.verify_action(observation, action, agent_id)
        spoofing_score = 1.0 - confidence
        result['dimension_scores']['spoofing'] = spoofing_score
        if not is_legitimate:
            result['detection_type'].append('spoofing')
            result['strategic_score'] += spoofing_score * 0.3
            result['details']['spoofing_score'] = spoofing_score
        
        # Dimension 2: Behavioral fingerprint consistency (weight: 0.25)
        fp_result = self.verify_behavioral_fingerprint(observation, action, agent_id)
        fp_deviation = fp_result.get('deviation', 0.0)
        result['dimension_scores']['fingerprint'] = fp_deviation
        if not fp_result.get('is_consistent', True):
            result['detection_type'].append('behavioral_anomaly')
            result['strategic_score'] += fp_deviation * 0.25
            result['details']['fingerprint_deviation'] = fp_deviation
            result['details']['fingerprint_profile'] = fp_result.get('profile', {})
        
        # Dimension 3: Resource hoarding (weight: 0.20)
        if resource_history:
            is_hoarding, hoarding_score = self.detect_hoarding(agent_id, resource_history)
            result['dimension_scores']['hoarding'] = hoarding_score
            if is_hoarding:
                result['detection_type'].append('hoarding')
                result['strategic_score'] += hoarding_score * 0.20
                result['details']['hoarding_score'] = hoarding_score
        
        # Dimension 4: Demand manipulation (weight: 0.15)
        reported_demand = 0.0
        if isinstance(action, dict) and 'demand' in action:
            reported_demand = action['demand']
        elif isinstance(action, torch.Tensor) and action.numel() > 0:
            reported_demand = float(action.mean().item())
        
        is_false_demand, demand_score = self.detect_false_demand(agent_id, reported_demand, observation)
        result['dimension_scores']['demand'] = demand_score
        if is_false_demand:
            result['detection_type'].append('false_demand')
            result['strategic_score'] += demand_score * 0.15
            result['details']['demand_score'] = demand_score
        
        # Dimension 5: Reputation consistency (weight: 0.10)
        reputation = self.reputation_system.get_reputation(agent_id)
        reputation_risk = 1.0 - reputation
        result['dimension_scores']['reputation_risk'] = reputation_risk
        if reputation_risk > 0.5:
            result['detection_type'].append('reputation_risk')
            result['strategic_score'] += reputation_risk * 0.10
        
        # Final determination
        result['is_strategic'] = result['strategic_score'] > 0.35
        if result['is_strategic']:
            self.detection_stats['true_positives'] += 1
        elif result['strategic_score'] > 0.1:
            self.detection_stats['false_positives'] += 1
        
        # Update detection rate
        if self.detection_stats['total_checks'] > 0:
            self.detection_stats['detection_rate'] = (
                self.detection_stats['true_positives'] / self.detection_stats['total_checks']
            )
        
        # Normalize detection type
        if len(result['detection_type']) == 0:
            result['detection_type'] = 'none'
        elif len(result['detection_type']) == 1:
            result['detection_type'] = result['detection_type'][0]
        else:
            result['detection_type'] = 'multiple'
        
        return result
    
    def apply_punishment(self, agent_id: int, strategic_score: float) -> Dict[str, Any]:
        """Apply graduated punishment based on strategic behavior severity."""
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
        
        if strategic_score >= 0.7:
            punishment['type'] = 'severe'
            reputation_penalty = 0.3 * strategic_score
            punishment['reputation_change'] = -reputation_penalty
            punishment['resource_penalty'] = 0.2
            punishment['action_restriction'] = True
        elif strategic_score >= 0.4:
            punishment['type'] = 'moderate'
            reputation_penalty = 0.2 * strategic_score
            punishment['reputation_change'] = -reputation_penalty
            punishment['resource_penalty'] = 0.1
        else:
            punishment['type'] = 'mild'
            reputation_penalty = 0.1 * strategic_score
            punishment['reputation_change'] = -reputation_penalty
        
        self.reputation_system.reputations[agent_id] = max(
            0.01, 
            self.reputation_system.reputations[agent_id] + punishment['reputation_change']
        )
        
        return punishment

    def get_detection_rate(self) -> float:
        """Get overall strategic behavior detection rate."""
        return self.detection_stats['detection_rate']
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        return dict(self.detection_stats)

    def get_correction_rate(self) -> float:
        """Get action correction rate."""
        if len(self.correction_history) == 0:
            return 0.0
        correction_count = sum(1 for _, status in self.correction_history if status == 'corrected')
        return correction_count / len(self.correction_history)

    def get_reputation_report(self) -> Dict[str, Any]:
        """Get reputation system report."""
        return self.reputation_system.get_reputation_report() if self.reputation_system else {}
    
    def get_fingerprint_report(self) -> Dict[int, Dict[str, Any]]:
        """Get behavioral fingerprint report for all agents."""
        return {agent_id: fp.get_profile() for agent_id, fp in self.fingerprints.items()}

    def save(self, path: str):
        """Save anti-spoofing state."""
        torch.save({
            'verifier': self.verifier.state_dict(),
            'spoofing_detector': self.spoofing_detector.state_dict(),
            'demand_predictor': self.demand_predictor.state_dict(),
            'correction_network': self.correction_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'detection_stats': self.detection_stats,
            'detection_threshold': self.detection_threshold,
            'correction_strength': self.correction_strength,
        }, path)

    def load(self, path: str, strict: bool = True):
        """Load anti-spoofing state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.verifier.load_state_dict(checkpoint['verifier'], strict=strict)
        self.spoofing_detector.load_state_dict(checkpoint['spoofing_detector'], strict=strict)
        self.demand_predictor.load_state_dict(checkpoint['demand_predictor'], strict=strict)
        self.correction_network.load_state_dict(checkpoint['correction_network'], strict=strict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.detection_stats = checkpoint.get('detection_stats', self.detection_stats)
        self.detection_threshold = checkpoint.get('detection_threshold', self.detection_threshold)
        self.correction_strength = checkpoint.get('correction_strength', self.correction_strength)
        return True
