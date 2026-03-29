"""
Rescue Agent implementations for EGT-MARL disaster resource allocation system.

This module contains specialized rescue agent classes:
- DroneAgent: Fast reconnaissance and small-scale rescue
- AmbulanceAgent: Medium-scale rescue and transportation  
- HospitalAgent: Large-scale rescue and on-site treatment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentType, AgentStatus, AgentState, AgentAction


@dataclass
class RescueTask:
    """Data class representing a rescue task."""
    task_id: str
    position: np.ndarray
    urgency: float  # 0.0 to 1.0
    required_resources: Dict[str, float]
    estimated_duration: float  # in time steps
    survivors: int
    task_type: str  # 'reconnaissance', 'first_aid', 'transport', 'treatment'


class DroneAgent(BaseAgent):
    """
    Drone agent for fast reconnaissance and small-scale rescue.
    
    Characteristics:
    - High speed, low capacity
    - Excellent for scouting and initial assessment
    - Can deliver small medical supplies
    - Limited by battery life
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_position: np.ndarray,
        initial_resources: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize drone agent."""
        if initial_resources is None:
            initial_resources = {
                'medical_kits': 3.0,
                'water': 5.0,
                'food': 3.0,
                'fuel': 15.0,
                'battery': 100.0
            }
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.DRONE,
            initial_position=initial_position,
            initial_resources=initial_resources,
            config=config
        )
        
        # Drone-specific configuration
        self.config.update({
            'max_speed': 30.0,  # Drones are fast
            'acceleration': 5.0,
            'sensor_range': 100.0,  # Good sensors
            'camera_quality': 0.9,
            'reconnaissance_efficiency': 0.8,
            'battery_drain_rate': 0.002,  # Higher drain due to flight
        })
    
    def _initialize_networks(self) -> None:
        """Initialize neural networks for drone agent."""
        # Policy network for action selection
        self.policy_network = nn.Sequential(
            nn.Linear(self._get_observation_dim(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self._get_action_dim())
        )
        
        # Value network for state evaluation
        self.value_network = nn.Sequential(
            nn.Linear(self._get_observation_dim(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Communication network for multi-agent coordination
        self.communication_encoder = nn.Sequential(
            nn.Linear(self._get_observation_dim(), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Task prioritization network
        self.task_network = nn.Sequential(
            nn.Linear(10, 64),  # Task features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Task priority score
        )
    
    def _get_observation_dim(self) -> int:
        """Get observation dimension for drone."""
        # Base: position(2) + velocity(2) + resources(5) + health + battery + reputation = 11
        # Nearby agents: up to 10 agents × 6 features = 60
        # Nearby tasks: up to 10 tasks × 6 features = 60
        # Total: 131, round to 150 for network input
        return 150
    
    def _get_action_dim(self) -> int:
        """Get action dimension for drone."""
        # Movement direction (8 discrete directions) + speed (3 levels) + 
        # resource allocation (5 resources × 3 levels) + communication (4 types)
        return 8 + 3 + 15 + 4
    
    def select_action(
        self,
        observation: np.ndarray,
        available_actions: Optional[List[Any]] = None,
        training: bool = True
    ) -> AgentAction:
        """
        Select action for drone agent.
        
        Drone actions focus on:
        1. Efficient reconnaissance and scouting
        2. Quick delivery of small medical supplies
        3. Communication of discovered information
        """
        # Ensure observation has correct dimension
        obs_padded = self._pad_observation(observation)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0)
        
        # Get action logits from policy network
        with torch.no_grad():
            action_logits = self.policy_network(obs_tensor)
        
        # Apply exploration during training
        if training and np.random.random() < self.config['exploration_rate']:
            # Epsilon-greedy exploration
            if np.random.random() < 0.3:
                # Random action
                action_idx = np.random.randint(action_logits.shape[-1])
            else:
                # Soft exploration with temperature
                temperature = 2.0
                action_probs = F.softmax(action_logits / temperature, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()
        else:
            # Greedy action
            action_idx = torch.argmax(action_logits).item()
        
        # Decode action index to AgentAction
        action = self._decode_action(action_idx, observation)
        
        # Save action to history
        self.action_history.append(action)
        
        return action
    
    def _pad_observation(self, observation: np.ndarray) -> np.ndarray:
        """Pad observation to expected dimension."""
        target_dim = self._get_observation_dim()
        if len(observation) >= target_dim:
            return observation[:target_dim]
        else:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:len(observation)] = observation
            return padded
    
    def _decode_action(self, action_idx: int, observation: np.ndarray) -> AgentAction:
        """Decode action index to AgentAction object."""
        # Decode movement direction (0-7: 8 directions)
        if action_idx < 8:
            angle = action_idx * (2 * np.pi / 8)
            movement = np.array([np.cos(angle), np.sin(angle)]) * self.config['max_speed']
        # Decode speed level (8-10: 3 levels)
        elif action_idx < 11:
            speed_level = action_idx - 8  # 0, 1, 2
            speed = self.config['max_speed'] * (0.3 + 0.35 * speed_level)
            movement = self.state.velocity / (np.linalg.norm(self.state.velocity) + 1e-6) * speed
        # Decode resource allocation (11-25: 5 resources × 3 levels)
        elif action_idx < 26:
            resource_idx = (action_idx - 11) // 3
            allocation_level = (action_idx - 11) % 3  # 0: none, 1: some, 2: all
            
            resource_names = ['medical_kits', 'water', 'food', 'fuel', 'battery']
            if resource_idx < len(resource_names):
                resource = resource_names[resource_idx]
                allocation = allocation_level * 0.5  # 0, 0.5, 1.0
                resource_allocation = {resource: allocation}
            else:
                resource_allocation = {}
            
            movement = np.zeros(2)
        # Decode communication type (26-29: 4 types)
        else:
            comm_type_idx = action_idx - 26
            comm_types = ['recon_report', 'resource_request', 'assistance_request', 'task_update']
            if comm_type_idx < len(comm_types):
                communication = {'type': comm_types[comm_type_idx], 'content': {}}
            else:
                communication = {}
            
            movement = np.zeros(2)
            resource_allocation = {}
        
        # Create AgentAction
        action = AgentAction(
            movement=movement,
            resource_allocation=resource_allocation if 'resource_allocation' in locals() else {},
            communication=communication if 'communication' in locals() else {},
            task_selection=None
        )
        
        return action
    
    def update_policy(
        self,
        experiences: List[Tuple[np.ndarray, AgentAction, float, np.ndarray, bool]]
    ) -> Dict[str, float]:
        """Update drone agent policy using PPO."""
        if len(experiences) < self.config['batch_size']:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Convert experiences to tensors
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Pad states to consistent dimension
        states_padded = [self._pad_observation(s) for s in states]
        next_states_padded = [self._pad_observation(s) for s in next_states]
        
        states_tensor = torch.FloatTensor(states_padded)
        next_states_tensor = torch.FloatTensor(next_states_padded)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        
        # Calculate advantages (simplified)
        with torch.no_grad():
            current_values = self.value_network(states_tensor).squeeze()
            next_values = self.value_network(next_states_tensor).squeeze()
            targets = rewards_tensor + self.config['discount_factor'] * next_values * (1 - dones_tensor)
            advantages = targets - current_values
        
        # Get action indices (simplified - would need proper encoding)
        action_indices = torch.randint(0, self._get_action_dim(), (len(experiences),))
        
        # Calculate policy loss
        action_logits = self.policy_network(states_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        selected_action_probs = action_probs[torch.arange(len(experiences)), action_indices]
        
        # PPO loss
        old_probs = selected_action_probs.detach()
        ratio = selected_action_probs / (old_probs + 1e-8)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value loss
        value_loss = F.mse_loss(current_values, targets)
        
        # Entropy bonus
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Optimize
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Decay exploration rate
        self.config['exploration_rate'] *= self.config['exploration_decay']
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'exploration_rate': self.config['exploration_rate']
        }
    
    def prioritize_tasks(self, tasks: List[RescueTask]) -> List[Tuple[str, float]]:
        """Prioritize tasks based on drone capabilities."""
        prioritized = []
        
        for task in tasks:
            # Calculate distance
            distance = np.linalg.norm(self.state.position - task.position)
            
            # Calculate urgency score
            urgency_score = task.urgency
            
            # Calculate resource match score
            resource_match = 0.0
            for resource, need in task.required_resources.items():
                if resource in self.state.resources and self.state.resources[resource] >= need:
                    resource_match += 1.0
            resource_match /= max(len(task.required_resources), 1)
            
            # Calculate efficiency score (drones are good for reconnaissance)
            if task.task_type == 'reconnaissance':
                efficiency = self.config['reconnaissance_efficiency']
            else:
                efficiency = 0.5
            
            # Combine scores
            priority_score = (
                0.4 * urgency_score +
                0.3 * (1 - distance / 1000) +  # Normalized distance
                0.2 * resource_match +
                0.1 * efficiency
            )
            
            prioritized.append((task.task_id, priority_score))
        
        # Sort by priority
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized
    
    def generate_reconnaissance_report(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reconnaissance report based on observed data."""
        report = {
            'agent_id': self.agent_id,
            'timestamp': len(self.action_history),
            'position': self.state.position.tolist(),
            'battery_level': self.state.battery,
            'camera_quality': self.config['camera_quality'],
            'observations': []
        }
        
        if 'tasks' in observed_data:
            for task_id, task_data in observed_data['tasks'].items():
                if 'position' in task_data:
                    distance = np.linalg.norm(self.state.position - np.array(task_data['position']))
                    if distance <= self.config['sensor_range']:
                        report['observations'].append({
                            'type': 'task',
                            'id': task_id,
                            'position': task_data['position'],
                            'urgency': task_data.get('urgency', 0.0),
                            'distance': distance,
                            'confidence': min(1.0, self.config['camera_quality'] * (1 - distance/self.config['sensor_range']))
                        })
        
        if 'agents' in observed_data:
            for agent_id, agent_data in observed_data['agents'].items():
                if agent_id != self.agent_id and 'position' in agent_data:
                    distance = np.linalg.norm(self.state.position - np.array(agent_data['position']))
                    if distance <= self.config['sensor_range']:
                        report['observations'].append({
                            'type': 'agent',
                            'id': agent_id,
                            'position': agent_data['position'],
                            'status': agent_data.get('status', 'unknown'),
                            'distance': distance,
                            'confidence': min(1.0, 0.8 * (1 - distance/self.config['sensor_range']))
                        })
        
        return report


class AmbulanceAgent(BaseAgent):
    """
    Ambulance agent for medium-scale rescue and transportation.
    
    Characteristics:
    - Moderate speed, medium capacity
    - Can transport multiple patients
    - Equipped for basic medical treatment
    - Requires road access
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_position: np.ndarray,
        initial_resources: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ambulance agent."""
        if initial_resources is None:
            initial_resources = {
                'medical_kits': 15.0,
                'water': 30.0,
                'food': 20.0,
                'fuel': 80.0,
                'patient_capacity': 4.0
            }
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.AMBULANCE,
            initial_position=initial_position,
            initial_resources=initial_resources,
            config=config
        )
        
        # Ambulance-specific configuration
        self.config.update({
            'max_speed': 15.0,  # Limited by road conditions
            'acceleration': 2.0,
            'patient_treatment_rate': 0.1,  # Patients treated per time step
            'road_dependency': 0.8,  # How dependent on roads
            'medical_expertise': 0.7,  # Medical treatment capability
        })
        
        # Current patients
        self.patients: List[Dict[str, Any]] = []
    
    def _initialize_networks(self) -> None:
        """Initialize neural networks for ambulance agent."""
        # Similar to DroneAgent but with different architecture
        self.policy_network = nn.Sequential(
            nn.Linear(self._get_observation_dim(), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self._get_action_dim())
        )
        
        # Treatment decision network
        self.treatment_network = nn.Sequential(
            nn.Linear(20, 64),  # Patient state features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Treatment decisions
        )
    
    def _get_observation_dim(self) -> int:
        """Get observation dimension for ambulance."""
        return 200  # Larger than drone due to patient information
    
    def _get_action_dim(self) -> int:
        """Get action dimension for ambulance."""
        return 50  # More complex actions including patient care
    
    def select_action(
        self,
        observation: np.ndarray,
        available_actions: Optional[List[Any]] = None,
        training: bool = True
    ) -> AgentAction:
        """Select action for ambulance agent."""
        # Similar implementation to DroneAgent but with ambulance-specific logic
        obs_padded = self._pad_observation(observation)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.policy_network(obs_tensor)
        
        if training and np.random.random() < self.config['exploration_rate']:
            action_idx = np.random.randint(action_logits.shape[-1])
        else:
            action_idx = torch.argmax(action_logits).item()
        
        # Decode to ambulance-specific action
        action = self._decode_ambulance_action(action_idx)
        
        self.action_history.append(action)
        return action
    
    def _decode_ambulance_action(self, action_idx: int) -> AgentAction:
        """Decode action for ambulance."""
        # Simplified decoding - would be more complex in real implementation
        if action_idx < 8:  # Movement
            angle = action_idx * (2 * np.pi / 8)
            movement = np.array([np.cos(angle), np.sin(angle)]) * self.config['max_speed']
        elif action_idx < 13:  # Patient loading/unloading
            movement = np.zeros(2)
            # Patient management actions
        elif action_idx < 23:  # Treatment actions
            movement = np.zeros(2)
            # Medical treatment actions
        else:  # Communication and coordination
            movement = np.zeros(2)
        
        return AgentAction(
            movement=movement,
            resource_allocation={},
            communication={},
            task_selection=None
        )
    
    def load_patient(self, patient_info: Dict[str, Any]) -> bool:
        """Load a patient into the ambulance."""
        if len(self.patients) >= self.state.resources.get('patient_capacity', 0):
            return False
        
        self.patients.append(patient_info)
        return True
    
    def unload_patient(self, patient_index: int) -> Optional[Dict[str, Any]]:
        """Unload a patient from the ambulance."""
        if 0 <= patient_index < len(self.patients):
            return self.patients.pop(patient_index)
        return None
    
    def provide_treatment(self) -> Dict[str, float]:
        """Provide medical treatment to loaded patients."""
        treatment_results = {
            'patients_treated': 0,
            'medical_kits_used': 0.0,
            'health_improvement': 0.0
        }
        
        treatment_rate = self.config['medical_expertise']
        
        for patient in self.patients:
            if self.state.resources.get('medical_kits', 0) > 0:
                # Use medical kits
                kits_used = min(1.0, self.state.resources['medical_kits'])
                self.state.resources['medical_kits'] -= kits_used
                
                # Improve patient health
                health_improvement = kits_used * treatment_rate
                if 'health' in patient:
                    patient['health'] = min(1.0, patient.get('health', 0) + health_improvement)
                
                treatment_results['patients_treated'] += 1
                treatment_results['medical_kits_used'] += kits_used
                treatment_results['health_improvement'] += health_improvement
        
        return treatment_results


class HospitalAgent(BaseAgent):
    """
    Mobile hospital agent for large-scale rescue and on-site treatment.
    
    Characteristics:
    - Low speed, high capacity
    - Full medical treatment capabilities
    - Can perform surgeries
    - Requires significant resources
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_position: np.ndarray,
        initial_resources: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize hospital agent."""
        if initial_resources is None:
            initial_resources = {
                'medical_kits': 80.0,
                'water': 400.0,
                'food': 250.0,
                'fuel': 400.0,
                'patient_capacity': 40.0,
                'surgical_capacity': 8.0,
                'blood_supply': 20.0,
                'medication': 50.0
            }
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HOSPITAL,
            initial_position=initial_position,
            initial_resources=initial_resources,
            config=config
        )
        
        # Hospital-specific configuration
        self.config.update({
            'max_speed': 5.0,  # Very slow due to size
            'acceleration': 0.5,
            'treatment_capacity': 10.0,  # Patients treated per time step
            'surgical_capacity': 2.0,  # Surgeries per time step
            'resource_consumption_rate': 0.05,  # Base resource consumption
            'medical_expertise': 0.9,  # High medical capability
        })
        
        # Hospital state
        self.admitted_patients: List[Dict[str, Any]] = []
        self.surgery_queue: List[Dict[str, Any]] = []
        self.discharged_patients: List[Dict[str, Any]] = []
    
    def _initialize_networks(self) -> None:
        """Initialize neural networks for hospital agent."""
        # Complex network for hospital management
        self.policy_network = nn.Sequential(
            nn.Linear(self._get_observation_dim(), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self._get_action_dim())
        )
        
        # Resource allocation network
        self.resource_network = nn.Sequential(
            nn.Linear(50, 128),  # Resource and patient state
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20)  # Resource allocation decisions
        )
        
        # Triage network
        self.triage_network = nn.Sequential(
            nn.Linear(15, 64),  # Patient assessment features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Triage categories
        )
    
    def _get_observation_dim(self) -> int:
        """Get observation dimension for hospital."""
        return 300  # Very large due to complex state
    
    def _get_action_dim(self) -> int:
        """Get action dimension for hospital."""
        return 100  # Most complex action space
    
    def select_action(
        self,
        observation: np.ndarray,
        available_actions: Optional[List[Any]] = None,
        training: bool = True
    ) -> AgentAction:
        """Select action for hospital agent."""
        obs_padded = self._pad_observation(observation)
        obs_tensor = torch.FloatTensor(obs_padded).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.policy_network(obs_tensor)
        
        if training and np.random.random() < self.config['exploration_rate']:
            # Boltzmann exploration for complex action space
            temperature = 1.5
            action_probs = F.softmax(action_logits / temperature, dim=-1)
            action_idx = torch.multinomial(action_probs, 1).item()
        else:
            action_idx = torch.argmax(action_logits).item()
        
        action = self._decode_hospital_action(action_idx)
        self.action_history.append(action)
        return action
    
    def _decode_hospital_action(self, action_idx: int) -> AgentAction:
        """Decode action for hospital."""
        # Hospital has very complex action space including:
        # - Movement and positioning
        # - Patient admission and discharge
        # - Resource allocation
        # - Staff assignment
        # - Communication with other hospitals and ambulances
        
        # Simplified implementation
        if action_idx < 5:  # Basic movement
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]  # N, E, S, W, stay
            dx, dy = directions[action_idx]
            movement = np.array([dx, dy]) * self.config['max_speed']
        elif action_idx < 15:  # Patient management
            movement = np.zeros(2)
            # Various patient care actions
        elif action_idx < 30:  # Resource management
            movement = np.zeros(2)
            # Resource allocation actions
        elif action_idx < 50:  # Medical procedures
            movement = np.zeros(2)
            # Treatment and surgery actions
        else:  # Coordination and communication
            movement = np.zeros(2)
            # Complex coordination actions
        
        return AgentAction(
            movement=movement,
            resource_allocation={},
            communication={},
            task_selection=None
        )
    
    def admit_patient(self, patient_info: Dict[str, Any]) -> str:
        """Admit a patient to the hospital."""
        if len(self.admitted_patients) >= self.state.resources.get('patient_capacity', 0):
            return "rejected"  # No capacity
        
        # Perform triage
        triage_category = self.perform_triage(patient_info)
        patient_info['triage'] = triage_category
        patient_info['admission_time'] = len(self.action_history)
        patient_info['treatment_priority'] = self._calculate_treatment_priority(patient_info)
        
        self.admitted_patients.append(patient_info)
        
        # Queue for surgery if needed
        if patient_info.get('requires_surgery', False):
            self.surgery_queue.append(patient_info)
        
        return triage_category
    
    def perform_triage(self, patient_info: Dict[str, Any]) -> str:
        """Perform triage to categorize patient urgency."""
        # Extract features for triage network
        features = self._extract_triage_features(patient_info)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            triage_scores = self.triage_network(features_tensor)
            triage_idx = torch.argmax(triage_scores).item()
        
        categories = ['immediate', 'delayed', 'minimal', 'expectant']
        return categories[triage_idx] if triage_idx < len(categories) else 'delayed'
    
    def _extract_triage_features(self, patient_info: Dict[str, Any]) -> List[float]:
        """Extract features for triage decision."""
        features = [
            patient_info.get('age', 30) / 100.0,
            patient_info.get('injury_severity', 0.5),
            patient_info.get('vital_signs', {}).get('heart_rate', 80) / 200.0,
            patient_info.get('vital_signs', {}).get('blood_pressure_systolic', 120) / 200.0,
            patient_info.get('vital_signs', {}).get('respiratory_rate', 16) / 60.0,
            patient_info.get('consciousness_level', 1.0),
            patient_info.get('bleeding_level', 0.0),
            patient_info.get('burn_percentage', 0.0) / 100.0,
            patient_info.get('fractures', 0),
            patient_info.get('time_since_injury', 0) / 24.0,  # Normalized to days
            patient_info.get('pre_existing_conditions', 0),
            1.0 if patient_info.get('pregnant', False) else 0.0,
            1.0 if patient_info.get('pediatric', False) else 0.0,
            patient_info.get('pain_level', 0.0),
            patient_info.get('mobility', 1.0)
        ]
        
        # Pad to expected length
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
    
    def _calculate_treatment_priority(self, patient_info: Dict[str, Any]) -> float:
        """Calculate treatment priority score."""
        base_priority = 0.0
        
        # Triage category weights
        triage_weights = {
            'immediate': 1.0,
            'delayed': 0.7,
            'minimal': 0.4,
            'expectant': 0.1
        }
        
        triage = patient_info.get('triage', 'delayed')
        base_priority += triage_weights.get(triage, 0.5)
        
        # Injury severity
        base_priority += patient_info.get('injury_severity', 0.5) * 0.5
        
        # Time since admission (earlier patients get priority)
        admission_time = patient_info.get('admission_time', 0)
        time_penalty = min(1.0, admission_time / 1000.0)  # Normalized
        base_priority += (1 - time_penalty) * 0.3
        
        return min(1.0, base_priority)
    
    def provide_medical_care(self) -> Dict[str, Any]:
        """Provide medical care to admitted patients."""
        care_results = {
            'patients_treated': 0,
            'surgeries_performed': 0,
            'patients_discharged': 0,
            'resources_consumed': {},
            'mortality': 0
        }
        
        treatment_capacity = self.config['treatment_capacity']
        surgical_capacity = self.config['surgical_capacity']
        
        # Sort patients by priority
        self.admitted_patients.sort(
            key=lambda p: p.get('treatment_priority', 0.5),
            reverse=True
        )
        
        # Provide treatment
        patients_treated = 0
        for patient in self.admitted_patients[:int(treatment_capacity)]:
            if self._provide_patient_treatment(patient):
                patients_treated += 1
        
        care_results['patients_treated'] = patients_treated
        
        # Perform surgeries
        surgeries_performed = 0
        for patient in self.surgery_queue[:int(surgical_capacity)]:
            if self._perform_surgery(patient):
                surgeries_performed += 1
                self.surgery_queue.remove(patient)
        
        care_results['surgeries_performed'] = surgeries_performed
        
        # Check for discharges
        discharged = []
        for patient in self.admitted_patients:
            if patient.get('health', 0) >= 0.8:  # Healthy enough for discharge
                discharged.append(patient)
                self.discharged_patients.append(patient)
        
        for patient in discharged:
            self.admitted_patients.remove(patient)
        
        care_results['patients_discharged'] = len(discharged)
        
        # Check for mortality
        mortality = 0
        for patient in self.admitted_patients[:]:  # Copy for safe removal
            if patient.get('health', 0) <= 0.1:  # Very low health
                mortality += 1
                self.admitted_patients.remove(patient)
        
        care_results['mortality'] = mortality
        
        return care_results
    
    def _provide_patient_treatment(self, patient: Dict[str, Any]) -> bool:
        """Provide treatment to a single patient."""
        # Check resource availability
        required_resources = {
            'medical_kits': 0.5,
            'medication': 0.2,
            'blood_supply': 0.1 if patient.get('requires_blood', False) else 0.0
        }
        
        # Check if we have enough resources
        for resource, amount in required_resources.items():
            if self.state.resources.get(resource, 0) < amount:
                return False
        
        # Consume resources
        for resource, amount in required_resources.items():
            self.state.resources[resource] = max(0, self.state.resources.get(resource, 0) - amount)
        
        # Improve patient health
        treatment_effectiveness = self.config['medical_expertise']
        health_improvement = treatment_effectiveness * 0.3  # Base improvement
        
        # Adjust based on injury severity
        injury_severity = patient.get('injury_severity', 0.5)
        health_improvement *= (1 - injury_severity * 0.5)
        
        patient['health'] = min(1.0, patient.get('health', 0) + health_improvement)
        
        return True
    
    def _perform_surgery(self, patient: Dict[str, Any]) -> bool:
        """Perform surgery on a patient."""
        # Check surgical resource availability
        required_resources = {
            'medical_kits': 2.0,
            'blood_supply': 1.0,
            'medication': 0.5
        }
        
        for resource, amount in required_resources.items():
            if self.state.resources.get(resource, 0) < amount:
                return False
        
        # Consume resources
        for resource, amount in required_resources.items():
            self.state.resources[resource] = max(0, self.state.resources.get(resource, 0) - amount)
        
        # Perform surgery
        surgical_success_rate = self.config['medical_expertise'] * 0.8
        if np.random.random() < surgical_success_rate:
            # Successful surgery
            patient['health'] = min(1.0, patient.get('health', 0) + 0.6)
            patient['requires_surgery'] = False
            return True
        else:
            # Failed surgery
            patient['health'] = max(0.0, patient.get('health', 0) - 0.2)
            return False
    
    def coordinate_with_ambulances(self, ambulance_agents: List[Any]) -> Dict[str, Any]:
        """Coordinate with ambulance agents for patient transfer."""
        coordination = {
            'patients_transferred': 0,
            'ambulances_assigned': 0,
            'transfer_requests': []
        }
        
        # Check which patients are ready for transfer to ambulances
        transfer_candidates = []
        for patient in self.admitted_patients:
            if patient.get('health', 0) >= 0.6 and not patient.get('requires_surgery', False):
                transfer_candidates.append(patient)
        
        # Assign ambulances
        for ambulance in ambulance_agents:
            if not hasattr(ambulance, 'load_patient'):
                continue
            
            ambulance_capacity = ambulance.state.resources.get('patient_capacity', 0)
            current_patients = len(getattr(ambulance, 'patients', []))
            available_slots = ambulance_capacity - current_patients
            
            for _ in range(min(available_slots, len(transfer_candidates))):
                if transfer_candidates:
                    patient = transfer_candidates.pop(0)
                    transfer_request = {
                        'patient_id': patient.get('id', 'unknown'),
                        'destination': 'field_hospital' if patient.get('health', 0) < 0.8 else 'permanent_hospital',
                        'priority': patient.get('treatment_priority', 0.5)
                    }
                    coordination['transfer_requests'].append(transfer_request)
                    coordination['patients_transferred'] += 1
        
        coordination['ambulances_assigned'] = len(coordination['transfer_requests'])
        return coordination
    
    def get_hospital_status(self) -> Dict[str, Any]:
        """Get current hospital status report."""
        return {
            'agent_id': self.agent_id,
            'position': self.state.position.tolist(),
            'status': self.state.status.value,
            'resources': self.state.resources,
            'patient_count': len(self.admitted_patients),
            'surgery_queue': len(self.surgery_queue),
            'discharged_count': len(self.discharged_patients),
            'capacity_utilization': len(self.admitted_patients) / max(1, self.state.resources.get('patient_capacity', 1)),
            'resource_adequacy': self._calculate_resource_adequacy(),
            'average_patient_health': self._calculate_average_health(),
            'mortality_rate': self._calculate_mortality_rate()
        }
    
    def _calculate_resource_adequacy(self) -> float:
        """Calculate how adequate current resources are."""
        critical_resources = ['medical_kits', 'blood_supply', 'medication', 'fuel']
        adequacy_scores = []
        
        for resource in critical_resources:
            current = self.state.resources.get(resource, 0)
            capacity = self.state.capacity.get(resource, 1)
            adequacy = min(1.0, current / max(1, capacity * 0.3))  # Need at least 30% capacity
            adequacy_scores.append(adequacy)
        
        return sum(adequacy_scores) / len(adequacy_scores) if adequacy_scores else 0.0
    
    def _calculate_average_health(self) -> float:
        """Calculate average health of admitted patients."""
        if not self.admitted_patients:
            return 0.0
        
        total_health = sum(p.get('health', 0) for p in self.admitted_patients)
        return total_health / len(self.admitted_patients)
    
    def _calculate_mortality_rate(self) -> float:
        """Calculate hospital mortality rate."""
        total_patients = len(self.admitted_patients) + len(self.discharged_patients)
        if total_patients == 0:
            return 0.0
        
        # Estimate mortality based on patient health
        critical_patients = sum(1 for p in self.admitted_patients if p.get('health', 0) < 0.3)
        estimated_mortality = critical_patients * 0.1  # 10% mortality risk for critical patients
        
        return min(1.0, estimated_mortality / max(1, total_patients))


# Factory function for creating rescue agents
def create_rescue_agent(
    agent_type: str,
    agent_id: str,
    initial_position: np.ndarray,
    initial_resources: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseAgent:
    """
    Factory function to create rescue agents by type.
    
    Args:
        agent_type: Type of agent ('drone', 'ambulance', 'hospital')
        agent_id: Unique identifier for the agent
        initial_position: Initial [x, y] position
        initial_resources: Initial resources dictionary
        config: Configuration dictionary
        
    Returns:
        Created rescue agent instance
    """
    agent_type = agent_type.lower()
    
    if agent_type == 'drone':
        return DroneAgent(agent_id, initial_position, initial_resources, config)
    elif agent_type == 'ambulance':
        return AmbulanceAgent(agent_id, initial_position, initial_resources, config)
    elif agent_type == 'hospital':
        return HospitalAgent(agent_id, initial_position, initial_resources, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                         f"Supported types: 'drone', 'ambulance', 'hospital'")