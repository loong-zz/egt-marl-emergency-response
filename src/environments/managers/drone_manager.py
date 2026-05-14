"""
DroneManager class for disaster simulation.

Handles drone-specific operations including resource delivery, casualty search,
and depot resupply.
"""

import logging
from typing import Dict, Optional
import numpy as np
from ..config.constants import (
    AgentType, ResourceType, SimulationConfig, RESOURCE_ABBR
)
from ..entities.agent import RescueAgent
from ..entities.casualty import Casualty
from ..entities.depot import ResourceDepot

logger = logging.getLogger(__name__)


class DroneManager:
    """Manages drone-specific behavior and operations."""
    
    def __init__(self, config: SimulationConfig, env=None):
        self.config = config
        self.env = env
    
    def find_needy_agent(
        self,
        drone: RescueAgent,
        agents: Dict[int, RescueAgent]
    ) -> Optional[RescueAgent]:
        """
        Find the nearest agent in need of resources.
        
        Args:
            drone: Drone looking for needy agents
            agents: All agents in the simulation
            
        Returns:
            Nearest needy agent or None
        """
        needy_agents = []
        
        for agent in agents.values():
            if agent.id == drone.id or agent.agent_type == AgentType.DRONE:
                continue
            
            # Check if agent needs resources
            total = sum(agent.capacity.values())
            max_total = sum(agent.max_capacity.values())
            
            if total < max_total * 0.5:  # 50% threshold
                dist = np.linalg.norm(drone.position - agent.position)
                needy_agents.append((dist, agent))
        
        if needy_agents:
            needy_agents.sort(key=lambda x: x[0])
            return needy_agents[0][1]
        
        return None
    
    def find_undiscovered_casualty(
        self,
        drone: RescueAgent,
        casualties: Dict[int, Casualty]
    ) -> Optional[Casualty]:
        """
        Find the nearest undiscovered casualty.
        
        Args:
            drone: Drone searching for casualties
            casualties: All casualties in the simulation
            
        Returns:
            Nearest undiscovered casualty or None
        """
        undiscovered = []
        
        for casualty in casualties.values():
            if casualty.discovered_by is None:
                dist = np.linalg.norm(drone.position - casualty.position)
                undiscovered.append((dist, casualty))
        
        if undiscovered:
            undiscovered.sort(key=lambda x: x[0])
            return undiscovered[0][1]
        
        return None
    
    def find_nearest_depot(
        self,
        drone: RescueAgent,
        depots: Dict[int, ResourceDepot]
    ) -> Optional[ResourceDepot]:
        """
        Find the nearest resource depot.
        
        Args:
            drone: Drone looking for depot
            depots: All depots in the simulation
            
        Returns:
            Nearest depot or None
        """
        nearest_dist = float('inf')
        nearest_depot = None
        
        for depot in depots.values():
            dist = np.linalg.norm(drone.position - depot.position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_depot = depot
        
        return nearest_depot
    
    def move_to_target(
        self,
        drone: RescueAgent,
        target_position: np.ndarray
    ) -> bool:
        """
        Move drone towards a target position.
        
        Args:
            drone: Drone to move
            target_position: Target position
            
        Returns:
            True if reached target, False otherwise
        """
        direction = target_position - drone.position
        distance = np.linalg.norm(direction)
        
        if distance <= 1.0:
            return True
        
        direction = direction / distance
        max_speed = drone.get_max_speed()
        drone.position += direction * max_speed * self.config.time_step
        
        # Clip position to map bounds
        map_size = self.config.map_size[0]
        drone.position = np.clip(drone.position, 0, map_size)
        
        return False
    
    def deliver_resources(
        self,
        drone: RescueAgent,
        target_agent: RescueAgent
    ) -> bool:
        """
        Deliver resources from drone to target agent.
        
        Args:
            drone: Drone delivering resources
            target_agent: Agent receiving resources
            
        Returns:
            True if delivery completed, False otherwise
        """
        distance = np.linalg.norm(drone.position - target_agent.position)
        
        if distance > self.config.drone_delivery_range:
            # Move towards target
            self.move_to_target(drone, target_agent.position)
            
            if self.env.step_count % 50 == 0:
                new_dist = np.linalg.norm(drone.position - target_agent.position)
                logger.debug(
                    f"[DRONE {drone.id}] Status=DELIVERING->Agent{target_agent.id} | "
                    f"Position={drone.position[0]:.1f},{drone.position[1]:.1f} | "
                    f"Distance={new_dist:.1f}m"
                )
            return False
        
        # Deliver resources
        any_transfer = False
        transferred_resources = []
        
        for resource_type in ResourceType:
            if target_agent.capacity[resource_type] < target_agent.max_capacity[resource_type]:
                needed = target_agent.max_capacity[resource_type] - target_agent.capacity[resource_type]
                available = drone.capacity.get(resource_type, 0.0)
                transfer = min(needed, available)
                
                if transfer > 0:
                    drone.capacity[resource_type] -= transfer
                    target_agent.capacity[resource_type] += transfer
                    abbr = RESOURCE_ABBR.get(resource_type.name, resource_type.name[:4])
                    transferred_resources.append(f"{abbr}+{transfer:.2f}")
                    any_transfer = True
        
        if any_transfer:
            logger.info(
                f"[DRONE RESUPPLY] Drone{drone.id} -> Agent{target_agent.id} | "
                f"Resources={','.join(transferred_resources)} | "
                f"Remaining={drone.format_resource_log()}"
            )
        
        return any_transfer
    
    def return_to_depot(
        self,
        drone: RescueAgent,
        depots: Dict[int, ResourceDepot]
    ) -> bool:
        """
        Return drone to nearest depot for resupply.
        
        Args:
            drone: Drone returning to depot
            depots: All depots
            
        Returns:
            True if reached depot and refilled, False otherwise
        """
        nearest_depot = self.find_nearest_depot(drone, depots)
        
        if not nearest_depot:
            return False
        
        distance = np.linalg.norm(drone.position - nearest_depot.position)
        
        if distance > self.config.drone_delivery_range:
            # Move towards depot
            self.move_to_target(drone, nearest_depot.position)
            drone.current_mission = f"go_to_depot_{nearest_depot.id}"
            return False
        
        # Record state before refilling
        pos_str = f"[{drone.position[0]:.1f},{drone.position[1]:.1f}]"
        before_str = ", ".join(
            f"{RESOURCE_ABBR.get(rt.name, rt.name[:4])}:{drone.capacity[rt]:.2f}"
            for rt in ResourceType
        )
        
        # Refill at depot
        for resource_type in ResourceType:
            drone.capacity[resource_type] = drone.max_capacity[resource_type]
        
        logger.info(
            f"[DRONE RESUPPLY] Drone{drone.id} refilled at depot {nearest_depot.id} | "
            f"Position={pos_str} | Distance={distance:.1f}m | Before={before_str}"
        )
        drone.current_mission = None
        
        return True
    
    def search_casualties(
        self,
        drone: RescueAgent,
        casualties: Dict[int, Casualty]
    ) -> None:
        """
        Search for undiscovered casualties.
        
        Args:
            drone: Drone searching
            casualties: All casualties
        """
        target_casualty = self.find_undiscovered_casualty(drone, casualties)
        
        if not target_casualty:
            return
        
        distance = np.linalg.norm(drone.position - target_casualty.position)
        
        if distance > 5.0:
            # Move towards casualty
            self.move_to_target(drone, target_casualty.position)
            drone.current_mission = f"searching_casualty_{target_casualty.id}"
            
            if self.config.max_steps % 50 == 0:
                new_dist = np.linalg.norm(drone.position - target_casualty.position)
                logger.debug(
                    f"[DRONE {drone.id}] Status=SEARCHING->Casualty{target_casualty.id} | "
                    f"Position={drone.position[0]:.1f},{drone.position[1]:.1f} | "
                    f"Distance={new_dist:.1f}m"
                )
        else:
            # Discover casualty
            target_casualty.discover(drone.id, self.config.time_step * 0)
            drone.current_mission = None
            
            if self.config.max_steps % 50 == 0:
                logger.debug(
                    f"[DRONE {drone.id}] Discovered Casualty{target_casualty.id} "
                    f"at position {target_casualty.position}"
                )
