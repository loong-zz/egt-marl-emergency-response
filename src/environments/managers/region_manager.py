"""
Region Manager for spatial partitioning and regional fitness calculation.

This module implements regional adaptation for EGT-MARL by:
1. Dividing the map into regions based on NUM_REGIONS
2. Tracking casualties, rescues, and agents per region
3. Calculating regional fitness (survival rate)
4. Computing cross-region Gini coefficient for fairness monitoring
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RegionManager:
    """
    Manages spatial regions and their fitness metrics.
    
    Divides the disaster map into a grid of regions and tracks:
    - Initial casualties per region
    - Rescued casualties per region
    - Active agents per region
    - Regional fitness (survival rate)
    """
    
    def __init__(self, map_size: Tuple[float, float], num_regions: int = 4):
        """
        Initialize the region manager.
        
        Args:
            map_size: (width, height) of the disaster map
            num_regions: Number of regions to divide the map into (must be a perfect square, e.g., 4, 9, 16)
        """
        self.map_size = map_size
        self.num_regions = num_regions
        
        # Calculate grid dimensions (e.g., 4 regions -> 2x2 grid)
        self.grid_size = int(np.sqrt(num_regions))
        if self.grid_size * self.grid_size != num_regions:
            logger.warning(f"num_regions={num_regions} is not a perfect square, rounding to {self.grid_size**2}")
            self.num_regions = self.grid_size ** 2
        
        # Region boundaries
        self.region_width = map_size[0] / self.grid_size
        self.region_height = map_size[1] / self.grid_size
        
        # Region statistics: {region_id: {'initial': int, 'saved': int, 'agents': set(), 'fitness': float}}
        self.region_stats = {i: {'initial': 0, 'saved': 0, 'agents': set(), 'fitness': 0.0} 
                            for i in range(self.num_regions)}
        
        # Tracking for agent/casualty to region mapping
        self.agent_regions = {}  # {agent_id: region_id}
        self.casualty_regions = {}  # {casualty_id: region_id}
        
        # History for monitoring
        self.gini_history = []
        self.fitness_history = []
        
        logger.info(f"RegionManager initialized: {self.grid_size}x{self.grid_size} grid, "
                   f"{self.num_regions} regions, map_size={map_size}")
    
    def get_region_id(self, position: np.ndarray) -> int:
        """
        Get the region ID for a given position.
        
        Args:
            position: [x, y] coordinates
            
        Returns:
            Region ID (0 to num_regions-1)
        """
        x, y = position[0], position[1]
        
        # Clamp to map boundaries
        x = max(0, min(x, self.map_size[0] - 1e-6))
        y = max(0, min(y, self.map_size[1] - 1e-6))
        
        # Calculate grid coordinates
        col = int(x / self.region_width)
        row = int(y / self.region_height)
        
        # Clamp to valid range
        col = min(col, self.grid_size - 1)
        row = min(row, self.grid_size - 1)
        
        # Convert to region ID
        region_id = row * self.grid_size + col
        return region_id
    
    def get_region_bounds(self, region_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get the boundary coordinates for a region.
        
        Args:
            region_id: Region ID
            
        Returns:
            ((min_x, min_y), (max_x, max_y)) boundary coordinates
        """
        row = region_id // self.grid_size
        col = region_id % self.grid_size
        
        min_x = col * self.region_width
        max_x = (col + 1) * self.region_width
        min_y = row * self.region_height
        max_y = (row + 1) * self.region_height
        
        return ((min_x, min_y), (max_x, max_y))
    
    def register_casualty(self, casualty_id: int, position: np.ndarray) -> int:
        """
        Register a casualty's initial position.
        
        Args:
            casualty_id: Unique casualty identifier
            position: [x, y] coordinates
            
        Returns:
            Region ID where casualty is located
        """
        region_id = self.get_region_id(position)
        self.casualty_regions[casualty_id] = region_id
        self.region_stats[region_id]['initial'] += 1
        return region_id
    
    def register_agent(self, agent_id: int, position: np.ndarray) -> int:
        """
        Register an agent's current position.
        
        Args:
            agent_id: Unique agent identifier
            position: [x, y] coordinates
            
        Returns:
            Region ID where agent is located
        """
        region_id = self.get_region_id(position)
        
        # Remove from old region if exists
        if agent_id in self.agent_regions:
            old_region = self.agent_regions[agent_id]
            self.region_stats[old_region]['agents'].discard(agent_id)
        
        # Add to new region
        self.agent_regions[agent_id] = region_id
        self.region_stats[region_id]['agents'].add(agent_id)
        
        return region_id
    
    def update_agent_position(self, agent_id: int, position: np.ndarray):
        """
        Update agent position and reassign region if needed.
        
        Args:
            agent_id: Agent identifier
            position: New [x, y] coordinates
        """
        new_region = self.get_region_id(position)
        
        if agent_id in self.agent_regions:
            old_region = self.agent_regions[agent_id]
            if old_region != new_region:
                # Agent moved to a new region
                self.region_stats[old_region]['agents'].discard(agent_id)
                self.region_stats[new_region]['agents'].add(agent_id)
                self.agent_regions[agent_id] = new_region
                logger.debug(f"Agent {agent_id} moved from region {old_region} to {new_region}")
        else:
            # New agent
            self.agent_regions[agent_id] = new_region
            self.region_stats[new_region]['agents'].add(agent_id)
    
    def record_rescue(self, casualty_id: int):
        """
        Record a casualty rescue in the appropriate region.
        
        Args:
            casualty_id: Identifier of rescued casualty
        """
        if casualty_id in self.casualty_regions:
            region_id = self.casualty_regions[casualty_id]
            self.region_stats[region_id]['saved'] += 1
            logger.debug(f"Casualty {casualty_id} rescued in region {region_id}")
        else:
            logger.warning(f"Casualty {casualty_id} not found in region registry")
    
    def calculate_region_fitness(self, region_id: int) -> float:
        """
        Calculate fitness for a specific region.
        
        Fitness = saved / initial (survival rate)
        
        Args:
            region_id: Region identifier
            
        Returns:
            Fitness value [0, 1]
        """
        stats = self.region_stats[region_id]
        initial = stats['initial']
        saved = stats['saved']
        
        if initial == 0:
            return 0.0
        
        fitness = saved / initial
        self.region_stats[region_id]['fitness'] = fitness
        return fitness
    
    def get_all_fitness_values(self) -> List[float]:
        """
        Get fitness values for all regions.
        
        Returns:
            List of fitness values (one per region)
        """
        return [self.calculate_region_fitness(i) for i in range(self.num_regions)]
    
    def calculate_gini_coefficient(self) -> float:
        """
        Calculate Gini coefficient across regions based on fitness.
        
        Returns:
            Gini coefficient [0, 1]
        """
        fitness_values = self.get_all_fitness_values()
        
        if len(fitness_values) == 0 or sum(fitness_values) == 0:
            return 0.0
        
        n = len(fitness_values)
        if n == 1:
            return 0.0
        
        # Sort values
        sorted_values = sorted(fitness_values)
        
        # Calculate Gini coefficient
        numerator = 0.0
        for i in range(n):
            numerator += (2 * i - n + 1) * sorted_values[i]
        
        denominator = n * sum(sorted_values)
        
        if denominator == 0:
            return 0.0
        
        gini = numerator / denominator
        self.gini_history.append(gini)
        return gini
    
    def calculate_theil_index(self) -> float:
        """
        Calculate Theil index (another inequality measure).
        
        Theil index is more sensitive to differences in the upper tail.
        
        Returns:
            Theil index [0, inf]
        """
        fitness_values = self.get_all_fitness_values()
        
        # Filter out zero values for Theil calculation
        positive_values = [v for v in fitness_values if v > 0]
        
        if len(positive_values) == 0:
            return 0.0
        
        mean_fitness = np.mean(positive_values)
        if mean_fitness == 0:
            return 0.0
        
        # Theil index formula
        theil = 0.0
        for v in positive_values:
            theil += (v / mean_fitness) * np.log(v / mean_fitness)
        
        theil /= len(positive_values)
        return theil
    
    def get_region_summary(self) -> Dict:
        """
        Get a summary of all regions.
        
        Returns:
            Dictionary with region statistics
        """
        summary = {}
        for region_id in range(self.num_regions):
            stats = self.region_stats[region_id]
            bounds = self.get_region_bounds(region_id)
            summary[region_id] = {
                'bounds': bounds,
                'initial_casualties': stats['initial'],
                'saved_casualties': stats['saved'],
                'active_agents': len(stats['agents']),
                'fitness': self.calculate_region_fitness(region_id)
            }
        
        summary['cross_region_gini'] = self.calculate_gini_coefficient()
        summary['cross_region_theil'] = self.calculate_theil_index()
        
        return summary
    
    def get_metrics(self) -> Dict:
        """Get current region metrics for logging/monitoring."""
        fitness_values = self.get_all_fitness_values()
        return {
            'num_regions': self.num_regions,
            'grid_size': self.grid_size,
            'fitness_values': fitness_values,
            'gini_coefficient': self.calculate_gini_coefficient(),
            'theil_index': self.calculate_theil_index(),
            'avg_fitness': np.mean(fitness_values) if fitness_values else 0.0,
            'min_fitness': min(fitness_values) if fitness_values else 0.0,
            'max_fitness': max(fitness_values) if fitness_values else 0.0,
        }
    
    def reset(self):
        """Reset the region manager to initial state."""
        self.region_stats = {i: {'initial': 0, 'saved': 0, 'agents': set(), 'fitness': 0.0} 
                            for i in range(self.num_regions)}
        self.agent_regions = {}
        self.casualty_regions = {}
        self.gini_history = []
        self.fitness_history = []
        logger.debug("[REGION] Manager reset")
