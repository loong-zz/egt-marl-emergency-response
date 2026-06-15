"""
ResourceManager class for disaster simulation.

Handles all resource-related operations including allocation, transfer, and monitoring.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
from ..config.constants import ResourceType, AgentType, SimulationConfig, RESOURCE_ABBR
from ..entities.agent import RescueAgent
from ..entities.depot import ResourceDepot

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages resource allocation, transfer, and monitoring."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._logged_insufficient = set()  # Track logged (agent_id, casualty_id) pairs
    
    def transfer_resources(
        self,
        source: RescueAgent,
        target: RescueAgent,
        resource_type: ResourceType,
        amount: float
    ) -> float:
        """
        Transfer resources from one agent to another.
        
        Args:
            source: Agent transferring resources
            target: Agent receiving resources
            resource_type: Type of resource to transfer
            amount: Amount to transfer
            
        Returns:
            The actual amount transferred
        """
        available = source.capacity.get(resource_type, 0.0)
        needed = target.max_capacity[resource_type] - target.capacity[resource_type]
        
        transfer_amount = min(amount, available, needed)
        
        if transfer_amount > 0:
            source.capacity[resource_type] -= transfer_amount
            target.capacity[resource_type] += transfer_amount
            
            abbr = RESOURCE_ABBR.get(resource_type.name, resource_type.name[:4])
            logger.debug(
                f"[RESOURCE TRANSFER] Agent{source.id} -> Agent{target.id} | "
                f"{abbr}: {transfer_amount:.2f}"
            )
        
        return transfer_amount
    
    def refill_from_depot(self, agent: RescueAgent, depot: ResourceDepot) -> bool:
        """
        Refill agent resources from a depot.
        
        Args:
            agent: Agent to refill
            depot: Depot to refill from
            
        Returns:
            True if refilling occurred, False otherwise
        """
        distance = np.linalg.norm(agent.position - depot.position)
        
        if distance > 10.0:
            return False
        
        any_refilled = False
        transferred = []
        
        # Record state before refilling
        before_str = agent.format_resource_log()
        
        for resource_type in ResourceType:
            needed = agent.max_capacity[resource_type] - agent.capacity[resource_type]
            if needed > 0:
                # 调用depot.consume()获取实际可补充的资源量（可能少于需求量）
                actual = depot.consume(resource_type, needed)
                agent.capacity[resource_type] += actual
                any_refilled = True
                abbr = RESOURCE_ABBR.get(resource_type.name, resource_type.name[:4])
                transferred.append(f"{abbr}+{actual:.2f}")
        
        if any_refilled:
            logger.info(
                f"[AGENT RESUPPLY] Agent{agent.id} refilled at depot {depot.id} | "
                f"Pos={agent.format_position_log()} | Dist={distance:.1f}m | "
                f"Before={before_str} | Resources={','.join(transferred)}"
            )
            agent._has_refilled = True
        
        return any_refilled
    
    def check_resource_sufficiency(
        self,
        agent: RescueAgent,
        required_resources: Dict[ResourceType, float],
        casualty_id: Optional[int] = None
    ) -> Tuple[bool, Dict[ResourceType, float]]:
        """
        Check if agent has sufficient resources for a task.
        
        Args:
            agent: Agent to check
            required_resources: Dict of resource types to required amounts
            casualty_id: Optional casualty ID for logging
            
        Returns:
            Tuple of (is_sufficient, deficit_dict)
        """
        deficit = {}
        sufficient = True
        
        for resource_type, required in required_resources.items():
            available = agent.capacity.get(resource_type, 0.0)
            if available < required:
                deficit[resource_type] = required - available
                sufficient = False
        
        if not sufficient and casualty_id is not None:
            log_key = (agent.id, casualty_id)
            if log_key not in self._logged_insufficient:
                self._logged_insufficient.add(log_key)
                deficit_str = ", ".join(
                    f"{RESOURCE_ABBR.get(rt.name, rt.name[:4])}:{d:.2f}"
                    for rt, d in deficit.items()
                )
                logger.warning(
                    f"[RESOURCE INSUFFICIENT] Agent{agent.id} for Casualty{casualty_id} | "
                    f"Deficit={deficit_str}"
                )
        
        return sufficient, deficit
    
    def calculate_total_resources(self, agent: RescueAgent) -> float:
        """Calculate total resources carried by an agent."""
        return sum(agent.capacity.values())
    
    def is_agent_resource_low(self, agent: RescueAgent, threshold: float = 0.2) -> bool:
        """Check if agent's total resources are below threshold."""
        total = self.calculate_total_resources(agent)
        max_total = sum(agent.max_capacity.values())
        return total < max_total * threshold
    
    def find_needy_agents(
        self,
        agents: Dict[int, RescueAgent],
        drone_id: int,
        threshold: float = 0.5
    ) -> list:
        """
        Find agents in need of resources.
        
        Args:
            agents: All agents in the simulation
            drone_id: ID of drone looking for needy agents
            threshold: Resource percentage threshold
            
        Returns:
            List of (distance, agent) tuples for needy agents
        """
        needy_agents = []
        
        for agent in agents.values():
            if agent.id == drone_id or agent.agent_type == AgentType.DRONE:
                continue
            
            total = self.calculate_total_resources(agent)
            max_total = sum(agent.max_capacity.values())
            
            if total < max_total * threshold:
                needy_agents.append((total / max_total, agent))
        
        # Sort by resource percentage (most needy first)
        needy_agents.sort(key=lambda x: x[0])
        
        return [agent for _, agent in needy_agents]
    
    def distribute_resources(
        self,
        donor: RescueAgent,
        recipients: list,
        resource_type: Optional[ResourceType] = None
    ) -> None:
        """
        Distribute resources from a donor to multiple recipients.
        
        Args:
            donor: Agent donating resources
            recipients: List of agents receiving resources
            resource_type: Specific resource type to distribute (None for all types)
        """
        if not recipients:
            return
        
        types_to_distribute = [resource_type] if resource_type else list(ResourceType)
        
        for rt in types_to_distribute:
            available = donor.capacity.get(rt, 0.0)
            if available <= 0:
                continue
            
            # Calculate total needed across all recipients
            total_needed = sum(
                recipient.max_capacity[rt] - recipient.capacity[rt]
                for recipient in recipients
            )
            
            if total_needed <= 0:
                continue
            
            # Distribute proportionally
            for recipient in recipients:
                needed = recipient.max_capacity[rt] - recipient.capacity[rt]
                if needed <= 0:
                    continue
                
                share = (needed / total_needed) * available
                transfer = min(share, needed, available)
                
                if transfer > 0:
                    donor.capacity[rt] -= transfer
                    recipient.capacity[rt] += transfer
                    available -= transfer
                    
                    if available <= 0:
                        break
