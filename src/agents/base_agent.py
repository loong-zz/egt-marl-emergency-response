"""
Base Agent class for EGT-MARL disaster resource allocation system.

This module defines the abstract base class for all agents in the disaster simulation,
providing common interfaces and functionality for both rescue and malicious agents.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Enumeration of agent types in the disaster simulation."""
    DRONE = "drone"           # Fast reconnaissance, small-scale rescue
    AMBULANCE = "ambulance"   # Medium-scale rescue, transportation
    HOSPITAL = "hospital"     # Large-scale rescue, on-site treatment
    MALICIOUS = "malicious"   # Malicious agent for robustness testing


class AgentStatus(Enum):
    """Enumeration of agent operational status."""
    IDLE = "idle"             # Available for assignment
    ASSIGNED = "assigned"     # Task assigned, en route
    BUSY = "busy"             # Currently performing task
    BROKEN = "broken"         # Equipment failure
    OUT_OF_SERVICE = "out_of_service"  # No longer operational


@dataclass
class AgentState:
    """Data class representing the state of an agent."""
    position: np.ndarray  # [x, y] coordinates
    velocity: np.ndarray  # [vx, vy] velocity vector
    resources: Dict[str, float]  # Available resources (medical supplies, fuel, etc.)
    capacity: Dict[str, float]  # Maximum capacity for each resource
    status: AgentStatus  # Current operational status
    assigned_task: Optional[str] = None  # ID of assigned task
    task_progress: float = 0.0  # Progress of current task (0.0 to 1.0)
    health: float = 1.0  # Agent health (0.0 to 1.0)
    battery: float = 1.0  # Battery/energy level (0.0 to 1.0)
    reputation: float = 1.0  # Reputation score for anti-spoofing


@dataclass  
class AgentAction:
    """Data class representing an agent's action."""
    movement: np.ndarray  # Movement vector [dx, dy]
    resource_allocation: Dict[str, float]  # Resources to allocate
    communication: Dict[str, Any]  # Communication messages to send
    task_selection: Optional[str] = None  # Selected task ID


class BaseAgent(ABC, nn.Module):
    """
    Abstract base class for all agents in the disaster simulation.
    
    This class defines the common interface and functionality that all agents
    must implement, including state management, action selection, and learning.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        initial_position: np.ndarray,
        initial_resources: Dict[str, float],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (drone, ambulance, hospital, malicious)
            initial_position: Initial [x, y] position
            initial_resources: Initial resources dictionary
            config: Configuration dictionary for agent parameters
        """
        super().__init__()
        
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or self._get_default_config()
        
        # Initialize state
        self.state = AgentState(
            position=initial_position.copy(),
            velocity=np.zeros(2),
            resources=initial_resources.copy(),
            capacity=self._get_default_capacity(),
            status=AgentStatus.IDLE,
            health=1.0,
            battery=1.0,
            reputation=1.0
        )
        
        # Initialize neural networks
        self._initialize_networks()
        
        # History tracking
        self.action_history: List[AgentAction] = []
        self.state_history: List[AgentState] = []
        self.reward_history: List[float] = []
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'resources_delivered': 0.0,
            'distance_traveled': 0.0,
            'survivors_rescued': 0,
            'communication_count': 0,
            'malicious_actions': 0 if agent_type != AgentType.MALICIOUS else 0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the agent."""
        return {
            'max_speed': 10.0,  # Maximum movement speed
            'acceleration': 2.0,  # Acceleration rate
            'deceleration': 2.0,  # Deceleration rate
            'battery_drain_rate': 0.001,  # Battery drain per step
            'health_decay_rate': 0.0001,  # Health decay per step
            'communication_range': 100.0,  # Maximum communication range
            'sensor_range': 50.0,  # Sensor detection range
            'learning_rate': 0.001,  # Learning rate for policy updates
            'discount_factor': 0.99,  # Discount factor for future rewards
            'exploration_rate': 0.1,  # Initial exploration rate
            'exploration_decay': 0.995,  # Exploration rate decay
            'batch_size': 32,  # Batch size for learning
            'memory_capacity': 10000,  # Experience replay memory capacity
        }
    
    def _get_default_capacity(self) -> Dict[str, float]:
        """Get default resource capacity based on agent type."""
        if self.agent_type == AgentType.DRONE:
            return {
                'medical_kits': 5.0,
                'water': 10.0,
                'food': 5.0,
                'fuel': 20.0,
                'battery': 100.0
            }
        elif self.agent_type == AgentType.AMBULANCE:
            return {
                'medical_kits': 20.0,
                'water': 50.0,
                'food': 30.0,
                'fuel': 100.0,
                'patient_capacity': 4.0
            }
        elif self.agent_type == AgentType.HOSPITAL:
            return {
                'medical_kits': 100.0,
                'water': 500.0,
                'food': 300.0,
                'fuel': 500.0,
                'patient_capacity': 50.0,
                'surgical_capacity': 10.0
            }
        else:  # MALICIOUS
            return {
                'disruption_power': 10.0,
                'stealth': 100.0,
                'battery': 100.0
            }
    
    @abstractmethod
    def _initialize_networks(self) -> None:
        """
        Initialize neural networks for the agent.
        
        This method should be implemented by subclasses to create
        policy networks, value networks, or other learning components.
        """
        pass
    
    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        available_actions: Optional[List[Any]] = None,
        training: bool = True
    ) -> AgentAction:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            available_actions: List of available actions (if action space is constrained)
            training: Whether the agent is in training mode (affects exploration)
            
        Returns:
            Selected action as an AgentAction object
        """
        pass
    
    @abstractmethod
    def update_policy(
        self,
        experiences: List[Tuple[np.ndarray, AgentAction, float, np.ndarray, bool]]
    ) -> Dict[str, float]:
        """
        Update the agent's policy based on collected experiences.
        
        Args:
            experiences: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            Dictionary of training metrics (losses, gradients, etc.)
        """
        pass
    
    def update_state(
        self,
        new_position: np.ndarray,
        new_velocity: np.ndarray,
        resource_changes: Dict[str, float],
        status_update: Optional[AgentStatus] = None
    ) -> None:
        """
        Update the agent's state based on environment feedback.
        
        Args:
            new_position: New position [x, y]
            new_velocity: New velocity [vx, vy]
            resource_changes: Changes to resources (positive for gain, negative for consumption)
            status_update: New status if changed
        """
        # Update position and velocity
        self.state.position = new_position.copy()
        self.state.velocity = new_velocity.copy()
        
        # Update resources
        for resource, change in resource_changes.items():
            if resource in self.state.resources:
                self.state.resources[resource] = max(0.0, min(
                    self.state.resources[resource] + change,
                    self.state.capacity.get(resource, float('inf'))
                ))
        
        # Update status if provided
        if status_update is not None:
            self.state.status = status_update
        
        # Update battery and health
        self.state.battery = max(0.0, self.state.battery - self.config['battery_drain_rate'])
        self.state.health = max(0.0, self.state.health - self.config['health_decay_rate'])
        
        # Save state to history
        self.state_history.append(self._clone_state())
    
    def get_observation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Process environment state to create agent-specific observation.
        
        Args:
            environment_state: Raw environment state
            
        Returns:
            Processed observation vector
        """
        # Base implementation: combine agent state with relevant environment information
        observation_parts = []
        
        # Agent's own state
        observation_parts.extend(self.state.position)
        observation_parts.extend(self.state.velocity)
        observation_parts.extend([self.state.resources.get(r, 0.0) for r in sorted(self.state.resources.keys())])
        observation_parts.append(self.state.health)
        observation_parts.append(self.state.battery)
        observation_parts.append(self.state.reputation)
        
        # Nearby agents (within sensor range)
        nearby_agents = self._get_nearby_agents(environment_state)
        for agent_info in nearby_agents:
            observation_parts.extend(agent_info['position'])
            observation_parts.append(agent_info['distance'])
            observation_parts.append(agent_info['type_encoding'])
        
        # Nearby tasks/patients
        nearby_tasks = self._get_nearby_tasks(environment_state)
        for task_info in nearby_tasks:
            observation_parts.extend(task_info['position'])
            observation_parts.append(task_info['distance'])
            observation_parts.append(task_info['urgency'])
            observation_parts.append(task_info['resource_need'])
        
        return np.array(observation_parts, dtype=np.float32)
    
    def _get_nearby_agents(self, environment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get information about nearby agents within sensor range."""
        nearby_agents = []
        sensor_range = self.config['sensor_range']
        
        if 'agents' in environment_state:
            for agent_id, agent_data in environment_state['agents'].items():
                if agent_id == self.agent_id:
                    continue  # Skip self
                
                # Calculate distance
                other_pos = np.array(agent_data.get('position', [0, 0]))
                distance = np.linalg.norm(self.state.position - other_pos)
                
                if distance <= sensor_range:
                    nearby_agents.append({
                        'id': agent_id,
                        'position': other_pos,
                        'distance': distance,
                        'type': agent_data.get('type', 'unknown'),
                        'type_encoding': self._encode_agent_type(agent_data.get('type', 'unknown'))
                    })
        
        return nearby_agents
    
    def _get_nearby_tasks(self, environment_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get information about nearby tasks/patients within sensor range."""
        nearby_tasks = []
        sensor_range = self.config['sensor_range']
        
        if 'tasks' in environment_state:
            for task_id, task_data in environment_state['tasks'].items():
                task_pos = np.array(task_data.get('position', [0, 0]))
                distance = np.linalg.norm(self.state.position - task_pos)
                
                if distance <= sensor_range:
                    nearby_tasks.append({
                        'id': task_id,
                        'position': task_pos,
                        'distance': distance,
                        'urgency': task_data.get('urgency', 0.0),
                        'resource_need': task_data.get('resource_need', 0.0),
                        'type': task_data.get('type', 'rescue')
                    })
        
        return nearby_tasks
    
    def _encode_agent_type(self, agent_type: str) -> float:
        """Encode agent type as a numerical value."""
        type_mapping = {
            'drone': 0.0,
            'ambulance': 0.33,
            'hospital': 0.66,
            'malicious': 1.0,
            'unknown': 0.5
        }
        return type_mapping.get(agent_type.lower(), 0.5)
    
    def receive_communication(self, message: Dict[str, Any], sender_id: str) -> None:
        """
        Process incoming communication from another agent.
        
        Args:
            message: Communication message
            sender_id: ID of the sending agent
        """
        self.metrics['communication_count'] += 1
        
        # Base implementation: just log the communication
        # Subclasses can override to implement specific communication protocols
        if 'log_communications' in self.config and self.config['log_communications']:
            print(f"Agent {self.agent_id} received message from {sender_id}: {message}")
    
    def send_communication(self, target_agent_id: str, message_type: str, 
                         content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a communication message to send to another agent.
        
        Args:
            target_agent_id: ID of the target agent
            message_type: Type of message (request, response, alert, etc.)
            content: Message content
            
        Returns:
            Formatted communication message
        """
        self.metrics['communication_count'] += 1
        
        message = {
            'sender': self.agent_id,
            'receiver': target_agent_id,
            'type': message_type,
            'content': content,
            'timestamp': len(self.action_history),
            'sender_position': self.state.position.tolist(),
            'sender_type': self.agent_type.value,
            'sender_reputation': self.state.reputation
        }
        
        return message
    
    def calculate_reward(
        self,
        action: AgentAction,
        next_state: AgentState,
        task_completed: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate reward for the taken action.
        
        Args:
            action: Action that was taken
            next_state: Resulting state after the action
            task_completed: Information about completed task (if any)
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        
        # Distance penalty (encourage efficient movement)
        movement_magnitude = np.linalg.norm(action.movement)
        reward -= 0.01 * movement_magnitude
        
        # Resource usage penalty
        total_resource_used = sum(abs(v) for v in action.resource_allocation.values())
        reward -= 0.001 * total_resource_used
        
        # Task completion bonus
        if task_completed is not None:
            reward += task_completed.get('reward', 0.0)
            self.metrics['tasks_completed'] += 1
            
            # Track resources delivered
            if 'resources_delivered' in task_completed:
                self.metrics['resources_delivered'] += task_completed['resources_delivered']
            
            # Track survivors rescued
            if 'survivors_rescued' in task_completed:
                self.metrics['survivors_rescued'] += task_completed['survivors_rescued']
        
        # Battery conservation bonus
        if self.state.battery > 0.5:
            reward += 0.001
        
        # Health conservation bonus
        if self.state.health > 0.7:
            reward += 0.001
        
        # Reputation bonus/penalty
        reward += 0.01 * self.state.reputation
        
        return reward
    
    def _clone_state(self) -> AgentState:
        """Create a deep copy of the current state."""
        return AgentState(
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(),
            resources=self.state.resources.copy(),
            capacity=self.state.capacity.copy(),
            status=self.state.status,
            assigned_task=self.state.assigned_task,
            task_progress=self.state.task_progress,
            health=self.state.health,
            battery=self.state.battery,
            reputation=self.state.reputation
        )
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get agent state as a dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'state': {
                'position': self.state.position.tolist(),
                'velocity': self.state.velocity.tolist(),
                'resources': self.state.resources,
                'capacity': self.state.capacity,
                'status': self.state.status.value,
                'assigned_task': self.state.assigned_task,
                'task_progress': self.state.task_progress,
                'health': self.state.health,
                'battery': self.state.battery,
                'reputation': self.state.reputation
            },
            'metrics': self.metrics,
            'config': self.config
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load agent state from a dictionary."""
        # Verify agent ID matches
        if state_dict['agent_id'] != self.agent_id:
            raise ValueError(f"Agent ID mismatch: {self.agent_id} != {state_dict['agent_id']}")
        
        # Load state
        state_data = state_dict['state']
        self.state = AgentState(
            position=np.array(state_data['position']),
            velocity=np.array(state_data['velocity']),
            resources=state_data['resources'],
            capacity=state_data['capacity'],
            status=AgentStatus(state_data['status']),
            assigned_task=state_data['assigned_task'],
            task_progress=state_data['task_progress'],
            health=state_data['health'],
            battery=state_data['battery'],
            reputation=state_data['reputation']
        )
        
        # Load metrics
        self.metrics.update(state_dict.get('metrics', {}))
        
        # Update config
        self.config.update(state_dict.get('config', {}))
    
    def reset(self) -> None:
        """Reset the agent to initial state (keeping configuration)."""
        # Reset state
        self.state = AgentState(
            position=self.state.position.copy(),  # Keep position
            velocity=np.zeros(2),
            resources={k: self.state.capacity.get(k, 0.0) for k in self.state.resources.keys()},
            capacity=self.state.capacity.copy(),
            status=AgentStatus.IDLE,
            health=1.0,
            battery=1.0,
            reputation=1.0
        )
        
        # Clear history
        self.action_history.clear()
        self.state_history.clear()
        self.reward_history.clear()
        
        # Reset metrics (keep cumulative stats if needed)
        self.metrics = {
            'tasks_completed': 0,
            'resources_delivered': 0.0,
            'distance_traveled': 0.0,
            'survivors_rescued': 0,
            'communication_count': 0,
            'malicious_actions': 0 if self.agent_type != AgentType.MALICIOUS else 0
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return (f"BaseAgent(id={self.agent_id}, type={self.agent_type.value}, "
                f"position={self.state.position}, status={self.state.status.value}, "
                f"health={self.state.health:.2f}, battery={self.state.battery:.2f})")
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (f"BaseAgent(\n"
                f"  agent_id={self.agent_id},\n"
                f"  agent_type={self.agent_type},\n"
                f"  state={self.state},\n"
                f"  metrics={self.metrics},\n"
                f"  config={self.config}\n"
                f")")


# Example concrete implementation for testing
class SimpleAgent(BaseAgent):
    """Simple implementation of BaseAgent for testing purposes."""
    
    def _initialize_networks(self) -> None:
        """Initialize a simple policy network."""
        self.policy_network = nn.Sequential(
            nn.Linear(50, 128),  # Input size will be adjusted based on observation
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Output: action probabilities
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: state value
        )
    
    def select_action(
        self,
        observation: np.ndarray,
        available_actions: Optional[List[Any]] = None,
        training: bool = True
    ) -> AgentAction:
        """Select action using a simple policy."""
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = torch.softmax(self.policy_network(obs_tensor), dim=-1)
        
        # Select action (explore during training)
        if training and np.random.random() < self.config['exploration_rate']:
            action_idx = np.random.randint(action_probs.shape[-1])
        else:
            action_idx = torch.argmax(action_probs).item()
        
        # Decode action index to AgentAction
        # This is simplified - real implementation would map to meaningful actions
        action = AgentAction(
            movement=np.array([np.cos(action_idx * 0.1), np.sin(action_idx * 0.1)]),
            resource_allocation={'medical_kits': 0.1},
            communication={},
            task_selection=None
        )
        
        # Save action to history
        self.action_history.append(action)
        
        return action
    
    def update_policy(
        self,
        experiences: List[Tuple[np.ndarray, AgentAction, float, np.ndarray, bool]]
    ) -> Dict[str, float]:
        """Simple policy update (placeholder implementation)."""
        # This is a placeholder - real implementation would use RL algorithms
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'grad_norm': 0.0
        }
        
        # Decay exploration rate
        self.config['exploration_rate'] *= self.config['exploration_decay']
        
        return metrics
