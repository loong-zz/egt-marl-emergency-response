"""
Evolutionary Game Theory Manager for dynamic fairness-efficiency trade-off.

This module implements the EGT layer that dynamically adjusts the fairness-efficiency
weight lambda(t) based on the Gini coefficient of agent fitness distribution.
"""

import numpy as np
import logging

from typing import Dict, List

logger = logging.getLogger(__name__)


class EGTManager:
    """
    Evolutionary Game Theory Manager for dynamic fairness-efficiency trade-off.
    
    This class implements the evolutionary game theory meta-controller that:
    1. Monitors the fitness distribution across regions
    2. Calculates the Gini coefficient to measure inequality
    3. Dynamically adjusts the fairness-efficiency weight lambda(t)
    4. Implements threshold decay: tau(t) = tau_0 * exp(-nu * t)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the EGT manager.
        
        Args:
            config: Configuration dictionary with EGT parameters
        """
        # EGT parameters (from paper section 4.6.3)
        self.kappa = config.get('kappa', 0.01)      # Adjustment rate
        self.tau_0 = config.get('tau_0', 0.3)       # Initial threshold
        self.nu = config.get('nu', 0.01)            # Decay rate
        self.lambda_min = config.get('lambda_min', 0.0)
        self.lambda_max = config.get('lambda_max', 1.0)
        self.delta = config.get('delta', 0.02)       # Hysteresis threshold
        
        # Current fairness-efficiency weight
        self.lambda_t = config.get('initial_lambda', 0.5)
        
        # Historical data for monitoring
        self.fitness_history = []
        self.lambda_history = []
        
    def calculate_gini_coefficient(self, values: List[float]) -> float:
        """
        Calculate the Gini coefficient for a list of values.
        
        The Gini coefficient measures inequality:
        - 0 = perfect equality
        - 1 = maximum inequality
        
        Args:
            values: List of fitness values (e.g., survival rates)
            
        Returns:
            Gini coefficient in [0, 1]
        """
        if len(values) == 0:
            return 0.0
            
        n = len(values)
        if n == 1:
            return 0.0
            
        # Sort values
        sorted_values = sorted(values)
        
        # Calculate Gini coefficient
        numerator = 0.0
        for i in range(n):
            numerator += (2 * i - n + 1) * sorted_values[i]
        
        denominator = n * sum(sorted_values)
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
    
    def calculate_region_fitness(self, region_data: Dict) -> float:
        """
        Calculate fitness for a region based on survival rate and unrest.
        
        Fitness formula (from paper section 4.3.2):
        f_j,t = saved_j,t / n_j,0 - eta * unrest_j,t
        
        Args:
            region_data: Dictionary with 'saved', 'initial_casualties', and 'unrest'
            
        Returns:
            Fitness value for the region
        """
        saved = region_data.get('saved', 0)
        initial = region_data.get('initial_casualties', 1)
        unrest = region_data.get('unrest', 0.0)
        eta = 0.1  # Weight for unrest penalty
        
        survival_rate = saved / initial if initial > 0 else 0.0
        return survival_rate - eta * unrest
    
    def get_target_threshold(self, time_step: int) -> float:
        """
        Calculate the target inequality threshold at time t.
        
        Threshold decay formula (from paper section 4.3.2):
        tau(t) = tau_0 * exp(-nu * t)
        
        Args:
            time_step: Current simulation time step
            
        Returns:
            Target inequality threshold
        """
        return self.tau_0 * np.exp(-self.nu * time_step)
    
    def update_lambda(self, fitness_values: List[float], time_step: int) -> float:
        """
        Update the fairness-efficiency weight lambda(t).
        
        The weight is adjusted based on:
        - Current Gini coefficient of fitness distribution
        - Target threshold that decays over time
        - Hysteresis to prevent oscillation
        
        Args:
            fitness_values: List of fitness values for each region
            time_step: Current simulation time step
            
        Returns:
            Updated lambda(t) value
        """
        # Calculate current inequality
        gini = self.calculate_gini_coefficient(fitness_values)
        
        # Get target threshold
        tau = self.get_target_threshold(time_step)
        
        # Update lambda based on inequality
        if gini > tau + self.delta:
            # Inequality too high - increase fairness weight
            self.lambda_t = min(self.lambda_max, self.lambda_t + self.kappa)
            logger.debug(f"[EGT] Gini={gini:.4f} > tau={tau:.4f}+delta, increasing lambda to {self.lambda_t:.4f}")
        elif gini < tau - self.delta:
            # Inequality too low - increase efficiency weight
            self.lambda_t = max(self.lambda_min, self.lambda_t - self.kappa)
            logger.debug(f"[EGT] Gini={gini:.4f} < tau={tau:.4f}-delta, decreasing lambda to {self.lambda_t:.4f}")
        # else: keep lambda unchanged (hysteresis)
        
        # Store history
        self.lambda_history.append(self.lambda_t)
        self.fitness_history.append(fitness_values.copy())
        
        return self.lambda_t
    
    def get_current_lambda(self) -> float:
        """Get the current fairness-efficiency weight."""
        return self.lambda_t
    
    def get_egt_metrics(self) -> Dict:
        """Get current EGT metrics for logging/monitoring."""
        return {
            'lambda_t': self.lambda_t,
            'kappa': self.kappa,
            'tau_0': self.tau_0,
            'nu': self.nu,
            'history_length': len(self.lambda_history)
        }
    
    def reset(self):
        """Reset the EGT manager to initial state."""
        self.lambda_t = 0.5
        self.fitness_history = []
        self.lambda_history = []
        logger.debug("[EGT] Manager reset")