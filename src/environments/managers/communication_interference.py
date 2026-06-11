"""
Communication Interference Model for disaster scenarios.

This module implements:
1. Communication delay simulation (exponential distribution)
2. Packet loss simulation (probabilistic)
3. Communication interruption during aftershocks
4. Time-varying communication quality
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class CommunicationInterference:
    """
    Communication Interference Model for disaster scenarios.
    
    This class implements realistic communication disturbances:
    1. Delay: exponential distribution with mean 0.5-2 seconds
    2. Packet loss: 5-20% probability
    3. Interruption: during aftershocks, 10% chance of complete interruption
    4. Time-varying quality: improves over time
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the communication interference model.
        
        Args:
            config: Configuration dictionary with interference parameters
        """
        # Delay parameters (from paper section 5.1.5)
        self.min_delay_mean = config.get('min_delay_mean', 0.5)  # seconds
        self.max_delay_mean = config.get('max_delay_mean', 2.0)  # seconds
        
        # Packet loss parameters
        self.min_packet_loss = config.get('min_packet_loss', 0.05)  # 5%
        self.max_packet_loss = config.get('max_packet_loss', 0.20)  # 20%
        
        # Interruption parameters
        self.interruption_probability = config.get('interruption_probability', 0.10)  # 10%
        self.interruption_duration_mean = config.get('interruption_duration_mean', 3.0)  # seconds
        
        # Time improvement factor
        self.improvement_rate = config.get('improvement_rate', 0.001)  # per step
        
        # Random number generator
        self.rng = np.random.RandomState(42)
        
        # Current state
        self.current_time = 0
        self.current_delay_mean = self.max_delay_mean
        self.current_packet_loss = self.max_packet_loss
        self.is_interrupted = False
        self.interruption_end_time = 0
        
        # Statistics
        self.total_packets = 0
        self.lost_packets = 0
        self.total_delay = 0.0
        
    def update_communication_quality(self, time_step: int, aftershock_happening: bool = False):
        """
        Update communication quality based on time and aftershock status.
        
        Args:
            time_step: Current simulation time step
            aftershock_happening: Whether an aftershock is happening
        """
        self.current_time = time_step
        
        # Communication quality improves over time
        progress = min(time_step / 3600.0, 1.0)  # Normalize to 1 hour
        improvement = 1.0 - self.improvement_rate * time_step
        
        # Update delay mean (decreases over time)
        self.current_delay_mean = self.max_delay_mean - (self.max_delay_mean - self.min_delay_mean) * progress
        self.current_delay_mean = max(self.min_delay_mean, self.current_delay_mean)
        
        # Update packet loss (decreases over time)
        self.current_packet_loss = self.max_packet_loss - (self.max_packet_loss - self.min_packet_loss) * progress
        self.current_packet_loss = max(self.min_packet_loss, self.current_packet_loss)
        
        # Check for interruption due to aftershock
        if aftershock_happening and not self.is_interrupted:
            if self.rng.random() < self.interruption_probability:
                duration = self.rng.exponential(self.interruption_duration_mean)
                self.is_interrupted = True
                self.interruption_end_time = time_step + int(duration / 0.1)  # Assuming 0.1s per step
                logger.debug(f"[COMM-INT] Communication interrupted for {duration:.1f}s at step {time_step}")
        
        # Check if interruption has ended
        if self.is_interrupted and time_step >= self.interruption_end_time:
            self.is_interrupted = False
            logger.debug(f"[COMM-INT] Communication restored at step {time_step}")
    
    def get_delay(self) -> float:
        """
        Generate communication delay using exponential distribution.
        
        Returns:
            Delay in seconds
        """
        if self.is_interrupted:
            return float('inf')  # Infinite delay when interrupted
        
        # Exponential distribution with current mean
        delay = self.rng.exponential(self.current_delay_mean)
        self.total_delay += delay
        
        logger.debug(f"[COMM-INT] Generated delay: {delay:.3f}s (mean={self.current_delay_mean:.2f}s)")
        return delay
    
    def is_packet_lost(self) -> bool:
        """
        Determine if a packet is lost based on current loss probability.
        
        Returns:
            True if packet is lost, False otherwise
        """
        if self.is_interrupted:
            return True  # All packets lost when interrupted
        
        self.total_packets += 1
        is_lost = self.rng.random() < self.current_packet_loss
        
        if is_lost:
            self.lost_packets += 1
            logger.debug(f"[COMM-INT] Packet lost (loss rate={self.current_packet_loss:.2%})")
        
        return is_lost
    
    def can_communicate(self, distance: float) -> Tuple[bool, Optional[float]]:
        """
        Check if communication is possible and return delay.
        
        Args:
            distance: Distance between agents
            
        Returns:
            (can_communicate, delay) tuple
        """
        if self.is_interrupted:
            return (False, None)
        
        # Distance affects communication quality (simplified)
        if distance > 100.0:  # Beyond effective range
            if self.rng.random() < 0.3:  # 30% chance of failure at long distance
                return (False, None)
        
        delay = self.get_delay()
        is_lost = self.is_packet_lost()
        
        if is_lost:
            return (False, delay)
        
        return (True, delay)
    
    def get_interference_metrics(self) -> Dict:
        """Get current interference metrics for logging/monitoring."""
        loss_rate = self.lost_packets / max(self.total_packets, 1)
        avg_delay = self.total_delay / max(self.total_packets - self.lost_packets, 1)
        
        return {
            'current_delay_mean': self.current_delay_mean,
            'current_packet_loss': self.current_packet_loss,
            'is_interrupted': self.is_interrupted,
            'interruption_end_time': self.interruption_end_time,
            'total_packets': self.total_packets,
            'lost_packets': self.lost_packets,
            'loss_rate': loss_rate,
            'avg_delay': avg_delay,
            'total_delay': self.total_delay,
            'parameters': {
                'min_delay_mean': self.min_delay_mean,
                'max_delay_mean': self.max_delay_mean,
                'min_packet_loss': self.min_packet_loss,
                'max_packet_loss': self.max_packet_loss,
                'interruption_probability': self.interruption_probability,
                'improvement_rate': self.improvement_rate
            }
        }
    
    def reset(self):
        """Reset the communication interference model to initial state."""
        self.current_time = 0
        self.current_delay_mean = self.max_delay_mean
        self.current_packet_loss = self.max_packet_loss
        self.is_interrupted = False
        self.interruption_end_time = 0
        self.total_packets = 0
        self.lost_packets = 0
        self.total_delay = 0.0
        self.rng = np.random.RandomState(42)
        logger.debug("[COMM-INT] Model reset")