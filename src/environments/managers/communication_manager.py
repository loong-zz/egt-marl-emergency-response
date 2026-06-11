"""
Communication Manager for agent information sharing.

This module implements:
1. Agent-to-agent communication for casualty location sharing
2. Communication range checking
3. Broadcast protocol for discovered casualties
4. Information synchronization across agents
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


class CommunicationManager:
    """
    Communication Manager for agent information sharing.
    
    This class implements:
    1. Communication range checking between agents
    2. Casualty information broadcast protocol
    3. Information synchronization across agents
    4. Shared knowledge base for discovered casualties
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the communication manager.
        
        Args:
            config: Configuration dictionary with communication parameters
        """
        # Communication range (meters)
        self.communication_range = config.get('communication_range', 50.0)
        
        # Broadcast frequency (steps)
        self.broadcast_frequency = config.get('broadcast_frequency', 5)
        
        # Maximum message size (number of casualties per broadcast)
        self.max_broadcast_size = config.get('max_broadcast_size', 10)
        
        # Shared knowledge base: {casualty_id: {agent_id: (timestamp, position, severity)}}
        self.shared_casualties: Dict[int, Dict[int, tuple]] = {}
        
        # Agent last broadcast time
        self.last_broadcast_time: Dict[int, int] = {}
        
        # Communication history for debugging
        self.comm_history: List[Dict] = []
        
    def can_communicate(self, agent1_pos: Tuple[float, float], 
                        agent2_pos: Tuple[float, float]) -> bool:
        """
        Check if two agents can communicate based on distance.
        
        Args:
            agent1_pos: Position of first agent (x, y)
            agent2_pos: Position of second agent (x, y)
            
        Returns:
            True if agents are within communication range
        """
        distance = np.sqrt((agent1_pos[0] - agent2_pos[0]) ** 2 + 
                          (agent1_pos[1] - agent2_pos[1]) ** 2)
        return distance <= self.communication_range
    
    def broadcast_casualties(self, agent_id: int, agent_pos: Tuple[float, float],
                             known_casualties: Dict[int, dict], timestamp: int):
        """
        Broadcast casualty information to nearby agents.
        
        Args:
            agent_id: ID of broadcasting agent
            agent_pos: Position of broadcasting agent
            known_casualties: Dictionary of known casualties
            timestamp: Current simulation time step
        """
        # Check broadcast frequency
        if agent_id in self.last_broadcast_time:
            if timestamp - self.last_broadcast_time[agent_id] < self.broadcast_frequency:
                return  # Too soon to broadcast again
        
        # Update last broadcast time
        self.last_broadcast_time[agent_id] = timestamp
        
        # Prepare broadcast message (limit size)
        broadcast_list = list(known_casualties.items())[:self.max_broadcast_size]
        
        # Update shared knowledge base
        for casualty_id, casualty_info in broadcast_list:
            if casualty_id not in self.shared_casualties:
                self.shared_casualties[casualty_id] = {}
            
            # Store information with timestamp
            position = casualty_info.get('position', (0.0, 0.0))
            severity = casualty_info.get('severity', 'UNKNOWN')
            
            self.shared_casualties[casualty_id][agent_id] = (timestamp, position, severity)
            
            logger.debug(f"[COMM] Agent{agent_id} broadcast Casualty{casualty_id} (Severity={severity})")
        
        # Record communication history
        self.comm_history.append({
            'agent_id': agent_id,
            'timestamp': timestamp,
            'num_casualties': len(broadcast_list),
            'position': agent_pos
        })
    
    def receive_broadcasts(self, agent_id: int, agent_pos: Tuple[float, float],
                          timestamp: int, nearby_agents: List[Tuple[int, Tuple[float, float]]]
                          ) -> Dict[int, dict]:
        """
        Receive broadcasted information from nearby agents.
        
        Args:
            agent_id: ID of receiving agent
            agent_pos: Position of receiving agent
            timestamp: Current simulation time step
            nearby_agents: List of (agent_id, position) tuples for nearby agents
            
        Returns:
            Dictionary of newly discovered casualties from broadcasts
        """
        new_casualties = {}
        
        # Check shared knowledge base for new information
        for casualty_id, agent_reports in self.shared_casualties.items():
            # Check if any nearby agent has reported this casualty recently
            for reporter_id, (report_time, position, severity) in agent_reports.items():
                # Skip own reports
                if reporter_id == agent_id:
                    continue
                
                # Check if reporter is nearby
                reporter_pos = None
                for aid, pos in nearby_agents:
                    if aid == reporter_id:
                        reporter_pos = pos
                        break
                
                # If reporter is nearby or if we don't have position info, trust the report
                if reporter_pos is None or self.can_communicate(agent_pos, reporter_pos):
                    # Only add if not already known or if newer information
                    if casualty_id not in new_casualties:
                        new_casualties[casualty_id] = {
                            'position': position,
                            'severity': severity,
                            'discovered_by': reporter_id,
                            'timestamp': report_time
                        }
                    else:
                        # Update if newer
                        if report_time > new_casualties[casualty_id]['timestamp']:
                            new_casualties[casualty_id].update({
                                'position': position,
                                'severity': severity,
                                'discovered_by': reporter_id,
                                'timestamp': report_time
                            })
        
        if new_casualties:
            logger.debug(f"[COMM] Agent{agent_id} received {len(new_casualties)} casualty reports")
        
        return new_casualties
    
    def get_shared_casualties(self) -> Dict[int, dict]:
        """
        Get all shared casualty information.
        
        Returns:
            Dictionary of all casualties in the shared knowledge base
        """
        result = {}
        for casualty_id, agent_reports in self.shared_casualties.items():
            # Get the most recent report
            latest_report = None
            latest_time = -1
            
            for reporter_id, (report_time, position, severity) in agent_reports.items():
                if report_time > latest_time:
                    latest_time = report_time
                    latest_report = {
                        'position': position,
                        'severity': severity,
                        'discovered_by': reporter_id,
                        'timestamp': report_time,
                        'reported_by': list(agent_reports.keys())
                    }
            
            if latest_report:
                result[casualty_id] = latest_report
        
        return result
    
    def remove_stale_information(self, max_age: int = 100):
        """
        Remove stale information from the shared knowledge base.
        
        Args:
            max_age: Maximum age (in steps) of information to keep
        """
        current_time = max(self.last_broadcast_time.values(), default=0)
        
        casualties_to_remove = []
        for casualty_id, agent_reports in self.shared_casualties.items():
            # Check if all reports are stale
            all_stale = True
            for reporter_id, (report_time, _, _) in agent_reports.items():
                if current_time - report_time <= max_age:
                    all_stale = False
                    break
            
            if all_stale:
                casualties_to_remove.append(casualty_id)
        
        for casualty_id in casualties_to_remove:
            del self.shared_casualties[casualty_id]
            logger.debug(f"[COMM] Removed stale casualty {casualty_id}")
    
    def get_communication_metrics(self) -> Dict:
        """Get current communication metrics for logging/monitoring."""
        shared_count = len(self.shared_casualties)
        total_reports = sum(len(reports) for reports in self.shared_casualties.values())
        
        return {
            'communication_range': self.communication_range,
            'broadcast_frequency': self.broadcast_frequency,
            'max_broadcast_size': self.max_broadcast_size,
            'shared_casualties_count': shared_count,
            'total_reports': total_reports,
            'communication_events': len(self.comm_history),
            'agent_broadcast_count': len(self.last_broadcast_time)
        }
    
    def reset(self):
        """Reset the communication manager to initial state."""
        self.shared_casualties = {}
        self.last_broadcast_time = {}
        self.comm_history = []
        logger.debug("[COMM] Manager reset")