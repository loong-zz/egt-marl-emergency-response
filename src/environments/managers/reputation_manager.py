"""
Reputation Manager for incentive-compatible mechanism.

This module implements:
1. Bayesian truthfulness verification using Gaussian Process Regression
2. Dynamic reputation system with forgetting factor
3. Anomaly detection using z-score method
"""

import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReputationManager:
    """
    Reputation Manager for anti-spoofing incentive-compatible mechanism.
    
    This class implements:
    1. Bayesian demand prediction using historical data
    2. Z-score based anomaly detection for suspicious claims
    3. Dynamic reputation update with forgetting factor
    4. Resource allocation penalty for low-reputation agents
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the reputation manager.
        
        Args:
            config: Configuration dictionary with reputation parameters
        """
        # Reputation parameters (from paper section 4.6.4)
        self.forgetting_factor = config.get('forgetting_factor', 0.95)  # alpha
        self.penalty_factor = config.get('penalty_factor', 0.7)          # beta
        self.anomaly_threshold = config.get('anomaly_threshold', 2.0)    # z_threshold
        self.reputation_update_frequency = config.get('update_frequency', 10)
        
        # Agent reputations: {agent_id: reputation_score}
        self.reputations: Dict[int, float] = {}
        
        # Historical claims: {agent_id: [(claimed_demand, actual_demand, timestamp)]}
        self.claim_history: Dict[int, List[tuple]] = {}
        
        # Prediction model parameters (Gaussian Process approximation)
        self.prediction_models: Dict[int, Dict] = {}
        
        # Statistics for z-score calculation
        self.claim_statistics: Dict[int, Dict] = {}  # {agent_id: {'mean': ..., 'std': ...}}
        
    def get_reputation(self, agent_id: int) -> float:
        """
        Get the reputation score for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Reputation score in [0, 1], default is 1.0 if not found
        """
        return self.reputations.get(agent_id, 1.0)
    
    def update_reputation(self, agent_id: int, is_honest: bool):
        """
        Update reputation based on whether the agent was honest.
        
        Reputation update rule (from paper section 4.4.2):
        r_i(t+1) = alpha * r_i(t) + (1-alpha) * I{honest}
        
        Args:
            agent_id: ID of the agent
            is_honest: True if the agent was honest, False otherwise
        """
        current_rep = self.reputations.get(agent_id, 1.0)
        
        if is_honest:
            # Honest behavior - increase reputation
            new_rep = self.forgetting_factor * current_rep + (1 - self.forgetting_factor) * 1.0
        else:
            # Dishonest behavior - penalize reputation
            new_rep = self.penalty_factor * current_rep
            
        # Clamp to [0, 1]
        new_rep = max(0.0, min(1.0, new_rep))
        
        self.reputations[agent_id] = new_rep
        logger.debug(f"[REPUTATION] Agent{agent_id} reputation updated: {current_rep:.4f} -> {new_rep:.4f} (honest={is_honest})")
    
    def predict_demand(self, agent_id: int, context: Dict) -> tuple:
        """
        Predict expected demand using historical data.
        
        This is a simplified implementation of Gaussian Process Regression
        that maintains mean and variance of past claims.
        
        Args:
            agent_id: ID of the agent
            context: Context features (time, location, etc.)
            
        Returns:
            (predicted_demand, prediction_std): Predicted demand and uncertainty
        """
        if agent_id not in self.claim_history or len(self.claim_history[agent_id]) == 0:
            # No history - return default prediction
            return (10.0, 5.0)  # Default: mean=10, std=5
        
        # Get historical claims
        claims = [claim[0] for claim in self.claim_history[agent_id]]
        
        if len(claims) == 1:
            return (claims[0], 3.0)
        
        # Calculate statistics
        mean = np.mean(claims)
        std = np.std(claims) if len(claims) > 1 else 3.0
        
        # Store statistics for z-score calculation
        self.claim_statistics[agent_id] = {
            'mean': mean,
            'std': max(std, 0.1),  # Avoid division by zero
            'count': len(claims)
        }
        
        return (mean, std)
    
    def detect_anomaly(self, agent_id: int, claimed_demand: float) -> tuple:
        """
        Detect anomalous demand claims using z-score method.
        
        Args:
            agent_id: ID of the agent
            claimed_demand: Demand claimed by the agent
            
        Returns:
            (is_anomalous, z_score): Whether the claim is anomalous and its z-score
        """
        if agent_id not in self.claim_statistics:
            # Insufficient data - cannot detect anomaly
            return (False, 0.0)
        
        stats = self.claim_statistics[agent_id]
        mean = stats['mean']
        std = stats['std']
        
        # Calculate z-score
        z_score = abs(claimed_demand - mean) / std
        
        is_anomalous = z_score > self.anomaly_threshold
        
        if is_anomalous:
            logger.debug(f"[ANOMALY] Agent{agent_id} claim={claimed_demand:.2f} is anomalous (z={z_score:.2f}, threshold={self.anomaly_threshold})")
        
        return (is_anomalous, z_score)
    
    def verify_claim(self, agent_id: int, claimed_demand: float, 
                     actual_demand: float, context: Dict = None) -> bool:
        """
        Verify if an agent's claim is truthful.
        
        Args:
            agent_id: ID of the agent
            claimed_demand: Demand claimed by the agent
            actual_demand: True/actual demand
            context: Optional context features
            
        Returns:
            True if the claim is considered truthful, False otherwise
        """
        # Store the claim
        if agent_id not in self.claim_history:
            self.claim_history[agent_id] = []
        self.claim_history[agent_id].append((claimed_demand, actual_demand, len(self.claim_history[agent_id])))
        
        # Update prediction statistics
        self.predict_demand(agent_id, context or {})
        
        # Detect anomaly
        is_anomalous, z_score = self.detect_anomaly(agent_id, claimed_demand)
        
        # Calculate accuracy (relative error)
        if actual_demand > 0:
            relative_error = abs(claimed_demand - actual_demand) / actual_demand
        else:
            relative_error = float('inf') if claimed_demand != 0 else 0.0
        
        # Determine if honest (conservative threshold)
        is_honest = not is_anomalous and relative_error < 0.3  # 30% tolerance
        
        # Update reputation
        self.update_reputation(agent_id, is_honest)
        
        return is_honest
    
    def adjust_resource_allocation(self, agent_id: int, requested_amount: float) -> float:
        """
        Adjust resource allocation based on agent reputation.
        
        Low reputation agents get reduced allocations.
        
        Args:
            agent_id: ID of the agent
            requested_amount: Amount requested by the agent
            
        Returns:
            Adjusted amount (reputation * requested_amount)
        """
        reputation = self.get_reputation(agent_id)
        adjusted = reputation * requested_amount
        
        if reputation < 0.5:
            logger.debug(f"[ALLOCATION] Agent{agent_id} reputation={reputation:.4f}, request={requested_amount:.2f} -> adjusted={adjusted:.2f}")
        
        return adjusted
    
    def get_reputation_metrics(self) -> Dict:
        """Get current reputation metrics for logging/monitoring."""
        if not self.reputations:
            return {
                'avg_reputation': 1.0,
                'min_reputation': 1.0,
                'max_reputation': 1.0,
                'agent_count': 0,
                'parameters': {
                    'forgetting_factor': self.forgetting_factor,
                    'penalty_factor': self.penalty_factor,
                    'anomaly_threshold': self.anomaly_threshold
                }
            }
        
        reps = list(self.reputations.values())
        return {
            'avg_reputation': np.mean(reps),
            'min_reputation': min(reps),
            'max_reputation': max(reps),
            'agent_count': len(reps),
            'parameters': {
                'forgetting_factor': self.forgetting_factor,
                'penalty_factor': self.penalty_factor,
                'anomaly_threshold': self.anomaly_threshold
            }
        }
    
    def reset(self):
        """Reset the reputation manager to initial state."""
        self.reputations = {}
        self.claim_history = {}
        self.prediction_models = {}
        self.claim_statistics = {}
        logger.debug("[REPUTATION] Manager reset")