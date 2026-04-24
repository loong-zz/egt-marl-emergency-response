"""
Predefined disaster scenarios for DisasterSim-2026.

This module contains high-fidelity disaster scenarios for testing and evaluation:
1. Earthquake scenarios (various magnitudes and locations)
2. Flood scenarios (river floods, flash floods, coastal floods)
3. Hurricane/typhoon scenarios
4. Industrial accident scenarios
5. Pandemic scenarios
6. Compound disaster scenarios
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import random
from enum import Enum


class DisasterType(Enum):
    """Types of disasters."""
    EARTHQUAKE = "earthquake"
    FLOOD = "flood"
    HURRICANE = "hurricane"
    INDUSTRIAL_ACCIDENT = "industrial_accident"
    PANDEMIC = "pandemic"
    COMPOUND = "compound"


class SeverityLevel(Enum):
    """Disaster severity levels."""
    MILD = "mild"        # Limited impact, localized
    MODERATE = "moderate"  # Significant impact, regional
    SEVERE = "severe"    # Major impact, widespread
    CATASTROPHIC = "catastrophic"  # Devastating impact, national


@dataclass
class DisasterScenario:
    """Base class for disaster scenarios."""
    name: str
    disaster_type: DisasterType
    severity: SeverityLevel
    epicenter: Tuple[float, float]  # (x, y) coordinates
    radius: float  # Affected radius in meters
    duration: float  # Duration in hours
    
    # Impact parameters
    population_density: float  # People per square km
    infrastructure_damage: float  # 0.0 to 1.0
    medical_needs_multiplier: float  # Multiplier for medical needs
    
    # Resource constraints
    resource_availability: Dict[str, float]  # Resource type -> availability factor
    
    # Environmental conditions
    weather_conditions: Dict[str, Any]
    time_of_day: str  # "day", "night", "dawn", "dusk"
    season: str  # "spring", "summer", "autumn", "winter"
    
    # Dynamic parameters
    progression_rate: float  # How quickly the disaster evolves
    secondary_hazards: List[str]  # e.g., ["aftershocks", "landslides", "fires"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary."""
        return {
            'name': self.name,
            'disaster_type': self.disaster_type.value,
            'severity': self.severity.value,
            'epicenter': list(self.epicenter),
            'radius': self.radius,
            'duration': self.duration,
            'population_density': self.population_density,
            'infrastructure_damage': self.infrastructure_damage,
            'medical_needs_multiplier': self.medical_needs_multiplier,
            'resource_availability': self.resource_availability,
            'weather_conditions': self.weather_conditions,
            'time_of_day': self.time_of_day,
            'season': self.season,
            'progression_rate': self.progression_rate,
            'secondary_hazards': self.secondary_hazards
        }


class EarthquakeScenario(DisasterScenario):
    """Earthquake disaster scenario."""
    
    def __init__(self,
                 name: str,
                 severity: SeverityLevel,
                 epicenter: Tuple[float, float],
                 magnitude: float,  # Richter scale
                 depth: float,  # Depth in km
                 fault_type: str,  # "strike-slip", "reverse", "normal"
                 **kwargs):
        
        # Calculate radius based on magnitude
        radius = self._calculate_radius(magnitude)
        
        # Calculate infrastructure damage
        infrastructure_damage = self._calculate_damage(magnitude, depth)
        
        # Calculate medical needs multiplier
        medical_needs = self._calculate_medical_needs(magnitude, severity)
        
        # Set earthquake-specific parameters
        earthquake_params = {
            'magnitude': magnitude,
            'depth': depth,
            'fault_type': fault_type,
            'aftershock_probability': self._calculate_aftershock_prob(magnitude),
            'liquefaction_risk': self._calculate_liquefaction_risk(magnitude, depth),
            'tsunami_risk': self._calculate_tsunami_risk(magnitude, fault_type)
        }
        
        # Merge with weather conditions
        weather_conditions = kwargs.get('weather_conditions', {})
        weather_conditions.update(earthquake_params)
        
        # Call parent constructor
        super().__init__(
            name=name,
            disaster_type=DisasterType.EARTHQUAKE,
            severity=severity,
            epicenter=epicenter,
            radius=radius,
            duration=self._calculate_duration(magnitude),
            population_density=kwargs.get('population_density', 5000.0),
            infrastructure_damage=infrastructure_damage,
            medical_needs_multiplier=medical_needs,
            resource_availability=kwargs.get('resource_availability', {
                'medical_kits': 0.3,
                'water': 0.2,
                'food': 0.4,
                'fuel': 0.5
            }),
            weather_conditions=weather_conditions,
            time_of_day=kwargs.get('time_of_day', 'day'),
            season=kwargs.get('season', 'any'),
            progression_rate=kwargs.get('progression_rate', 0.1),
            secondary_hazards=self._get_secondary_hazards(magnitude)
        )
        
        # Earthquake-specific attributes
        self.magnitude = magnitude
        self.depth = depth
        self.fault_type = fault_type
    
    def _calculate_radius(self, magnitude: float) -> float:
        """Calculate affected radius based on magnitude."""
        # Simplified formula: radius (km) = 10^(0.5*magnitude - 1.5)
        radius_km = 10 ** (0.5 * magnitude - 1.5)
        return radius_km * 1000  # Convert to meters
    
    def _calculate_damage(self, magnitude: float, depth: float) -> float:
        """Calculate infrastructure damage factor."""
        # Base damage from magnitude
        base_damage = min(1.0, (magnitude - 4.0) / 6.0)
        
        # Adjust for depth (shallower = more damage)
        depth_factor = max(0.1, 1.0 - depth / 100.0)
        
        return min(1.0, base_damage * depth_factor)
    
    def _calculate_medical_needs(self, magnitude: float, severity: SeverityLevel) -> float:
        """Calculate medical needs multiplier."""
        severity_factors = {
            SeverityLevel.MILD: 1.0,
            SeverityLevel.MODERATE: 2.0,
            SeverityLevel.SEVERE: 4.0,
            SeverityLevel.CATASTROPHIC: 8.0
        }
        
        magnitude_factor = 1.0 + (magnitude - 5.0) * 0.5
        severity_factor = severity_factors.get(severity, 1.0)
        
        return magnitude_factor * severity_factor
    
    def _calculate_duration(self, magnitude: float) -> float:
        """Calculate disaster duration in hours."""
        # Larger earthquakes have longer recovery periods
        return 24.0 + (magnitude - 5.0) * 12.0
    
    def _calculate_aftershock_prob(self, magnitude: float) -> float:
        """Calculate aftershock probability."""
        return min(0.9, 0.1 + (magnitude - 5.0) * 0.1)
    
    def _calculate_liquefaction_risk(self, magnitude: float, depth: float) -> float:
        """Calculate liquefaction risk."""
        # Higher magnitude and shallower depth increase risk
        magnitude_risk = max(0.0, (magnitude - 5.0) / 3.0)
        depth_risk = max(0.1, 1.0 - depth / 50.0)
        
        return min(1.0, magnitude_risk * depth_risk)
    
    def _calculate_tsunami_risk(self, magnitude: float, fault_type: str) -> float:
        """Calculate tsunami risk."""
        # Higher magnitude and certain fault types increase risk
        magnitude_risk = max(0.0, (magnitude - 6.0) / 3.0)
        
        fault_risk = {
            'reverse': 0.8,  # Thrust faults generate tsunamis
            'strike-slip': 0.2,
            'normal': 0.1
        }.get(fault_type, 0.3)
        
        return min(1.0, magnitude_risk * fault_risk)
    
    def _get_secondary_hazards(self, magnitude: float) -> List[str]:
        """Get secondary hazards based on magnitude."""
        hazards = ['aftershocks']
        
        if magnitude >= 6.0:
            hazards.append('landslides')
        if magnitude >= 6.5:
            hazards.append('fires')
        if magnitude >= 7.0:
            hazards.append('tsunami')
        if magnitude >= 7.5:
            hazards.append('liquefaction')
        
        return hazards


class FloodScenario(DisasterScenario):
    """Flood disaster scenario."""
    
    def __init__(self,
                 name: str,
                 severity: SeverityLevel,
                 epicenter: Tuple[float, float],
                 flood_type: str,  # "river", "flash", "coastal", "urban"
                 water_depth: float,  # Meters
                 flow_velocity: float,  # m/s
                 **kwargs):
        
        # Calculate radius based on flood type
        radius = self._calculate_radius(flood_type, water_depth)
        
        # Calculate infrastructure damage
        infrastructure_damage = self._calculate_damage(water_depth, flow_velocity)
        
        # Calculate medical needs multiplier
        medical_needs = self._calculate_medical_needs(water_depth, severity)
        
        # Set flood-specific parameters
        flood_params = {
            'flood_type': flood_type,
            'water_depth': water_depth,
            'flow_velocity': flow_velocity,
            'contamination_risk': self._calculate_contamination_risk(flood_type),
            'evacuation_difficulty': self._calculate_evacuation_difficulty(water_depth, flow_velocity)
        }
        
        # Merge with weather conditions
        weather_conditions = kwargs.get('weather_conditions', {})
        weather_conditions.update(flood_params)
        
        # Call parent constructor
        super().__init__(
            name=name,
            disaster_type=DisasterType.FLOOD,
            severity=severity,
            epicenter=epicenter,
            radius=radius,
            duration=self._calculate_duration(flood_type, water_depth),
            population_density=kwargs.get('population_density', 3000.0),
            infrastructure_damage=infrastructure_damage,
            medical_needs_multiplier=medical_needs,
            resource_availability=kwargs.get('resource_availability', {
                'medical_kits': 0.4,
                'water': 0.1,  # Water contaminated
                'food': 0.3,
                'fuel': 0.6,
                'boats': 0.5
            }),
            weather_conditions=weather_conditions,
            time_of_day=kwargs.get('time_of_day', 'day'),
            season=kwargs.get('season', 'rainy'),
            progression_rate=kwargs.get('progression_rate', 0.2),
            secondary_hazards=self._get_secondary_hazards(flood_type)
        )
        
        # Flood-specific attributes
        self.flood_type = flood_type
        self.water_depth = water_depth
        self.flow_velocity = flow_velocity
    
    def _calculate_radius(self, flood_type: str, water_depth: float) -> float:
        """Calculate affected radius."""
        radius_factors = {
            'river': 5000.0,  # 5km for river floods
            'flash': 2000.0,  # 2km for flash floods
            'coastal': 10000.0,  # 10km for coastal floods
            'urban': 3000.0   # 3km for urban floods
        }
        
        base_radius = radius_factors.get(flood_type, 3000.0)
        depth_factor = 1.0 + water_depth * 0.2
        
        return base_radius * depth_factor
    
    def _calculate_damage(self, water_depth: float, flow_velocity: float) -> float:
        """Calculate infrastructure damage factor."""
        depth_damage = min(1.0, water_depth / 5.0)  # 5m water causes max damage
        velocity_damage = min(1.0, flow_velocity / 3.0)  # 3m/s flow causes max damage
        
        return min(1.0, 0.6 * depth_damage + 0.4 * velocity_damage)
    
    def _calculate_medical_needs(self, water_depth: float, severity: SeverityLevel) -> float:
        """Calculate medical needs multiplier."""
        severity_factors = {
            SeverityLevel.MILD: 1.0,
            SeverityLevel.MODERATE: 1.5,
            SeverityLevel.SEVERE: 2.5,
            SeverityLevel.CATASTROPHIC: 4.0
        }
        
        depth_factor = 1.0 + water_depth * 0.3
        severity_factor = severity_factors.get(severity, 1.0)
        
        return depth_factor * severity_factor
    
    def _calculate_duration(self, flood_type: str, water_depth: float) -> float:
        """Calculate flood duration in hours."""
        duration_factors = {
            'river': 72.0,  # River floods last days
            'flash': 12.0,  # Flash floods are short but intense
            'coastal': 48.0,  # Coastal floods depend on tides
            'urban': 24.0   # Urban floods
        }
        
        base_duration = duration_factors.get(flood_type, 24.0)
        depth_factor = 1.0 + water_depth * 0.1
        
        return base_duration * depth_factor
    
    def _calculate_contamination_risk(self, flood_type: str) -> float:
        """Calculate water contamination risk."""
        contamination_risks = {
            'urban': 0.8,  # Urban floods often contaminated
            'river': 0.6,
            'flash': 0.4,
            'coastal': 0.3  # Saltwater contamination
        }
        
        return contamination_risks.get(flood_type, 0.5)
    
    def _calculate_evacuation_difficulty(self, water_depth: float, flow_velocity: float) -> float:
        """Calculate evacuation difficulty."""
        depth_difficulty = min(1.0, water_depth / 2.0)  # >2m water makes evacuation hard
        velocity_difficulty = min(1.0, flow_velocity / 2.0)  # >2m/s flow makes evacuation hard
        
        return min(1.0, 0.7 * depth_difficulty + 0.3 * velocity_difficulty)
    
    def _get_secondary_hazards(self, flood_type: str) -> List[str]:
        """Get secondary hazards."""
        hazards = ['waterborne_diseases', 'mold']
        
        if flood_type in ['flash', 'river']:
            hazards.append('landslides')
        if flood_type == 'urban':
            hazards.append('infrastructure_collapse')
        if flood_type == 'coastal':
            hazards.append('saltwater_intrusion')
        
        return hazards


class HurricaneScenario(DisasterScenario):
    """Hurricane/typhoon disaster scenario."""
    
    def __init__(self,
                 name: str,
                 severity: SeverityLevel,
                 epicenter: Tuple[float, float],
                 wind_speed: float,  # m/s
                 rainfall: float,  # mm
                 storm_surge: float,  # meters
                 **kwargs):
        
        # Calculate radius based on wind speed
        radius = self._calculate_radius(wind_speed)
        
        # Calculate infrastructure damage
        infrastructure_damage = self._calculate_damage(wind_speed, storm_surge)
        
        # Calculate medical needs multiplier
        medical_needs = self._calculate_medical_needs(wind_speed, severity)
        
        # Set hurricane-specific parameters
        hurricane_params = {
            'wind_speed': wind_speed,
            'rainfall': rainfall,
            'storm_surge': storm_surge,
            'category': self._calculate_category(wind_speed),
            'eye_diameter': self._calculate_eye_diameter(wind_speed),
            'forward_speed': kwargs.get('forward_speed', 5.0)  # m/s
        }
        
        # Merge with weather conditions
        weather_conditions = kwargs.get('weather_conditions', {})
        weather_conditions.update(hurricane_params)
        
        # Call parent constructor
        super().__init__(
            name=name,
            disaster_type=DisasterType.HURRICANE,
            severity=severity,
            epicenter=epicenter,
            radius=radius,
            duration=self._calculate_duration(wind_speed),
            population_density=kwargs.get('population_density', 2000.0),
            infrastructure_damage=infrastructure_damage,
            medical_needs_multiplier=medical_needs,
            resource_availability=kwargs.get('resource_availability', {
                'medical_kits': 0.5,
                'water': 0.4,
                'food': 0.3,
                'fuel': 0.2,
                'generators': 0.4
            }),
            weather_conditions=weather_conditions
        )


class DisasterScenarioFactory:
    """Disaster scenario factory for creating predefined scenarios."""
    
    def __init__(self):
        """Initialize the scenario factory."""
        self.predefined_scenarios = {
            'earthquake_standard': {
                'disaster_type': 'earthquake',
                'severity': 'medium',
                'map_size': (50, 50)
            },
            'flood_standard': {
                'disaster_type': 'flood',
                'severity': 'medium',
                'map_size': (50, 50)
            },
            'hurricane_standard': {
                'disaster_type': 'hurricane',
                'severity': 'medium',
                'map_size': (50, 50)
            }
        }
    
    def create_scenario(self, disaster_type: str, severity: str, map_size: Tuple[int, int]):
        """Create a disaster scenario."""
        class Scenario:
            def __init__(self, disaster_type, severity, map_size):
                self.disaster_type = disaster_type
                self.severity = severity
                self.map_size = map_size
                self.params = {
                    'epicenter': (map_size[0]//2, map_size[1]//2),
                    'radius': min(map_size) // 4,
                    'intensity': self._get_intensity(severity)
                }
            
            def _get_intensity(self, severity):
                severity_map = {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8,
                    'critical': 0.95
                }
                return severity_map.get(severity, 0.5)
        
        return Scenario(disaster_type, severity, map_size)
    
    def get_predefined_scenarios(self):
        """Get predefined scenarios."""
        return self.predefined_scenarios
