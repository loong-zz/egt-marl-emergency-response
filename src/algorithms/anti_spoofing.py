"""
Anti-Spoofing Mechanism
========================

Detects and prevents spoofing attacks in multi-agent systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class AntiSpoofing:
    """
    Anti-spoofing mechanism for detecting and preventing spoofing attacks.
    
    Implements:
    1. Action verification using MLPs
    2. Spoofing detection using anomaly detection
    3. Reputation-based correction
    4. Action correction network
    """
    
    def __init__(self, observation_dim: int, action_dim: int, 
                 detection_threshold: float = 0.5, 
                 correction_strength: float = 0.8, 
                 device: torch.device = torch.device("cpu")):
        """
        Initialize anti-spoofing mechanism.
        
        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            detection_threshold: Threshold for spoofing detection
            correction_strength: Strength of action correction
            device: Device to run on
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.detection_threshold = detection_threshold
        self.correction_strength = correction_strength
        self.device = device
        
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
        
        # Reputation system (placeholder)
        self.reputation_system = None
        
        # History
        self.detection_history = []  # (agent_id, spoofing_score, is_spoofing)
        self.correction_history = []  # (agent_id, correction_status)
    
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
        Update anti-spoofing mechanism.

        Args:
            batch: Experience batch

        Returns:
            Loss value
        """
        # In practice, would update based on verification outcomes
        # For now, return placeholder loss
        return 0.0

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
