"""
Metrics calculation for EGT-MARL disaster resource allocation system.

This module provides functions for calculating various performance metrics:
- Efficiency metrics: Total survivors, response time, resource utilization
- Fairness metrics: Gini coefficient, max-min fairness, Theil index
- Robustness metrics: Performance under attacks, system recovery
- Practicality metrics: Decision time, communication overhead
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
import time


@dataclass
class EfficiencyMetrics:
    """Data class for efficiency metrics."""
    total_survivors: int
    mean_response_time: float  # in minutes
    median_response_time: float
    response_time_90th_percentile: float
    resource_utilization: Dict[str, float]  # per resource type
    overall_resource_utilization: float
    tasks_completed: int
    tasks_completion_rate: float
    survivors_per_resource: float  # efficiency metric


@dataclass
class FairnessMetrics:
    """Data class for fairness metrics."""
    gini_coefficient: float
    theil_index: float
    max_min_fairness: float
    atkinson_index: float  # Inequality aversion parameter
    coefficient_of_variation: float
    fairness_score: float  # Combined fairness score (0-1)


@dataclass
class RobustnessMetrics:
    """Data class for robustness metrics."""
    performance_under_attack: Dict[str, float]  # Performance at different attack levels
    recovery_time: float  # Time to recover to 90% of normal performance
    degradation_rate: float  # Performance degradation per attack unit
    stability_index: float  # System stability measure
    fault_tolerance: float  # Ability to handle failures


@dataclass
class PracticalityMetrics:
    """Data class for practicality metrics."""
    decision_time_mean: float  # Average decision time in milliseconds
    decision_time_std: float  # Decision time standard deviation
    communication_overhead: float  # Bytes transmitted per decision
    computational_complexity: float  # O-notation estimate
    memory_usage: float  # Memory usage in MB
    scalability_score: float  # How well it scales with problem size


def calculate_efficiency_metrics(
    simulation_results: Dict[str, Any],
    time_units_per_minute: float = 1.0
) -> EfficiencyMetrics:
    """
    Calculate efficiency metrics from simulation results.
    
    Args:
        simulation_results: Dictionary containing simulation results
        time_units_per_minute: Conversion factor from simulation time units to minutes
        
    Returns:
        EfficiencyMetrics object with calculated metrics
    """
    # Extract basic data
    survivors = simulation_results.get('survivors', [])
    response_times = simulation_results.get('response_times', [])
    resource_usage = simulation_results.get('resource_usage', {})
    resource_capacity = simulation_results.get('resource_capacity', {})
    tasks = simulation_results.get('tasks', [])
    
    # Total survivors
    total_survivors = len(survivors)
    
    # Response time statistics
    if response_times:
        response_times_minutes = [t * time_units_per_minute for t in response_times]
        mean_response_time = np.mean(response_times_minutes)
        median_response_time = np.median(response_times_minutes)
        response_time_90th = np.percentile(response_times_minutes, 90)
    else:
        mean_response_time = median_response_time = response_time_90th = 0.0
    
    # Resource utilization
    resource_utilization = {}
    for resource, usage in resource_usage.items():
        capacity = resource_capacity.get(resource, 1.0)
        if capacity > 0:
            utilization = usage / capacity
        else:
            utilization = 0.0
        resource_utilization[resource] = min(1.0, utilization)
    
    # Overall resource utilization (weighted average)
    if resource_utilization:
        overall_resource_utilization = np.mean(list(resource_utilization.values()))
    else:
        overall_resource_utilization = 0.0
    
    # Task completion
    if tasks:
        completed_tasks = sum(1 for task in tasks if task.get('completed', False))
        tasks_completed = completed_tasks
        tasks_completion_rate = completed_tasks / len(tasks)
    else:
        tasks_completed = 0
        tasks_completion_rate = 0.0
    
    # Survivors per resource (efficiency)
    total_resources_used = sum(resource_usage.values())
    if total_resources_used > 0:
        survivors_per_resource = total_survivors / total_resources_used
    else:
        survivors_per_resource = 0.0
    
    return EfficiencyMetrics(
        total_survivors=total_survivors,
        mean_response_time=mean_response_time,
        median_response_time=median_response_time,
        response_time_90th_percentile=response_time_90th,
        resource_utilization=resource_utilization,
        overall_resource_utilization=overall_resource_utilization,
        tasks_completed=tasks_completed,
        tasks_completion_rate=tasks_completion_rate,
        survivors_per_resource=survivors_per_resource
    )


def calculate_fairness_metrics(
    allocation_results: Dict[str, Any],
    population_data: Optional[Dict[str, Any]] = None
) -> FairnessMetrics:
    """
    Calculate fairness metrics from allocation results.
    
    Args:
        allocation_results: Dictionary containing allocation data
        population_data: Optional population data for weighted fairness
        
    Returns:
        FairnessMetrics object with calculated metrics
    """
    # Extract allocation data
    allocations = allocation_results.get('allocations', [])
    if not allocations:
        # Return default values if no data
        return FairnessMetrics(
            gini_coefficient=0.0,
            theil_index=0.0,
            max_min_fairness=1.0,
            atkinson_index=0.0,
            coefficient_of_variation=0.0,
            fairness_score=1.0
        )
    
    # Convert to numpy array
    allocations_array = np.array(allocations)
    
    # Gini coefficient
    gini_coefficient = _calculate_gini(allocations_array)
    
    # Theil index (generalized entropy index with α=1)
    theil_index = _calculate_theil_index(allocations_array)
    
    # Max-min fairness
    max_min_fairness = _calculate_max_min_fairness(allocations_array)
    
    # Atkinson index (with inequality aversion parameter ε=0.5)
    atkinson_index = _calculate_atkinson_index(allocations_array, epsilon=0.5)
    
    # Coefficient of variation
    mean_allocation = np.mean(allocations_array)
    std_allocation = np.std(allocations_array)
    if mean_allocation > 0:
        coefficient_of_variation = std_allocation / mean_allocation
    else:
        coefficient_of_variation = 0.0
    
    # Combined fairness score (higher is better)
    fairness_score = _calculate_combined_fairness_score(
        gini_coefficient,
        theil_index,
        max_min_fairness,
        atkinson_index,
        coefficient_of_variation
    )
    
    return FairnessMetrics(
        gini_coefficient=gini_coefficient,
        theil_index=theil_index,
        max_min_fairness=max_min_fairness,
        atkinson_index=atkinson_index,
        coefficient_of_variation=coefficient_of_variation,
        fairness_score=fairness_score
    )


def calculate_robustness_metrics(
    performance_data: Dict[str, List[float]],
    attack_levels: List[float]
) -> RobustnessMetrics:
    """
    Calculate robustness metrics from performance under attacks.
    
    Args:
        performance_data: Dictionary mapping metric names to performance values at different attack levels
        attack_levels: List of attack levels (e.g., [0.0, 0.1, 0.2, 0.3] for 0%, 10%, 20%, 30% malicious agents)
        
    Returns:
        RobustnessMetrics object with calculated metrics
    """
    if not performance_data or not attack_levels:
        return RobustnessMetrics(
            performance_under_attack={},
            recovery_time=0.0,
            degradation_rate=0.0,
            stability_index=1.0,
            fault_tolerance=1.0
        )
    
    # Performance under attack
    performance_under_attack = {}
    for metric_name, values in performance_data.items():
        if len(values) == len(attack_levels):
            performance_under_attack[metric_name] = dict(zip(attack_levels, values))
    
    # Calculate degradation rate (slope of performance vs attack level)
    degradation_rates = {}
    for metric_name, values in performance_data.items():
        if len(values) >= 2:
            # Linear regression of performance on attack level
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                attack_levels[:len(values)], values
            )
            degradation_rates[metric_name] = slope
    
    # Average degradation rate
    if degradation_rates:
        avg_degradation_rate = np.mean(list(degradation_rates.values()))
    else:
        avg_degradation_rate = 0.0
    
    # Recovery time (simplified estimation)
    # In real implementation, would measure time to recover after attack stops
    recovery_time = _estimate_recovery_time(performance_data, attack_levels)
    
    # Stability index (inverse of performance variance)
    stability_values = []
    for values in performance_data.values():
        if len(values) > 1:
            # Normalize values to [0, 1] range for comparison
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
            stability = 1.0 / (np.std(normalized) + 1e-8)
            stability_values.append(min(stability, 10.0))  # Cap at 10
    
    stability_index = np.mean(stability_values) if stability_values else 1.0
    
    # Fault tolerance (performance at max attack level relative to baseline)
    fault_tolerance = 1.0
    for metric_name, values in performance_data.items():
        if len(values) > 1:
            baseline = values[0]  # Performance at 0% attack
            max_attack_performance = values[-1]  # Performance at max attack level
            if baseline > 0:
                tolerance = max_attack_performance / baseline
                fault_tolerance = min(fault_tolerance, tolerance)
    
    return RobustnessMetrics(
        performance_under_attack=performance_under_attack,
        recovery_time=recovery_time,
        degradation_rate=avg_degradation_rate,
        stability_index=stability_index,
        fault_tolerance=max(0.0, fault_tolerance)
    )


def calculate_practicality_metrics(
    system_measurements: Dict[str, List[float]],
    problem_size: Dict[str, float]
) -> PracticalityMetrics:
    """
    Calculate practicality metrics from system measurements.
    
    Args:
        system_measurements: Dictionary containing timing and resource measurements
        problem_size: Dictionary describing problem size (agents, tasks, etc.)
        
    Returns:
        PracticalityMetrics object with calculated metrics
    """
    # Decision time statistics
    decision_times = system_measurements.get('decision_times', [])
    if decision_times:
        decision_time_mean = np.mean(decision_times)
        decision_time_std = np.std(decision_times)
    else:
        decision_time_mean = decision_time_std = 0.0
    
    # Communication overhead
    communication_data = system_measurements.get('communication_data', [])
    if communication_data:
        communication_overhead = np.mean(communication_data)
    else:
        communication_overhead = 0.0
    
    # Computational complexity estimation
    computational_complexity = _estimate_computational_complexity(
        system_measurements, problem_size
    )
    
    # Memory usage
    memory_usage = system_measurements.get('memory_usage', [0.0])
    avg_memory_usage = np.mean(memory_usage) if memory_usage else 0.0
    
    # Scalability score
    scalability_score = _calculate_scalability_score(
        system_measurements, problem_size
    )
    
    return PracticalityMetrics(
        decision_time_mean=decision_time_mean,
        decision_time_std=decision_time_std,
        communication_overhead=communication_overhead,
        computational_complexity=computational_complexity,
        memory_usage=avg_memory_usage,
        scalability_score=scalability_score
    )


def _calculate_gini(x: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    gini = 0.5 * rmad
    return float(gini)


def _calculate_theil_index(x: np.ndarray) -> float:
    """Calculate Theil index (generalized entropy index with α=1)."""
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0.0
    
    # Avoid division by zero and log(0)
    x_normalized = x / mean_x
    x_normalized = np.where(x_normalized > 0, x_normalized, 1e-10)
    
    theil = np.mean(x_normalized * np.log(x_normalized))
    return float(theil)


def _calculate_max_min_fairness(x: np.ndarray) -> float:
    """Calculate max-min fairness ratio."""
    if len(x) == 0:
        return 1.0
    
    min_val = np.min(x)
    max_val = np.max(x)
    
    if max_val > 0:
        return min_val / max_val
    else:
        return 1.0


def _calculate_atkinson_index(x: np.ndarray, epsilon: float = 0.5) -> float:
    """Calculate Atkinson index with inequality aversion parameter epsilon."""
    if len(x) == 0:
        return 0.0
    
    mean_x = np.mean(x)
    
    if epsilon == 1:
        # Special case for epsilon = 1
        geometric_mean = np.exp(np.mean(np.log(np.where(x > 0, x, 1e-10))))
        atkinson = 1 - geometric_mean / mean_x
    else:
        # General case
        term = np.mean((x / mean_x) ** (1 - epsilon))
        atkinson = 1 - term ** (1 / (1 - epsilon))
    
    return float(atkinson)


def _calculate_combined_fairness_score(
    gini: float,
    theil: float,
    max_min: float,
    atkinson: float,
    cv: float
) -> float:
    """Calculate combined fairness score from multiple metrics."""
    # Normalize metrics to [0, 1] range where 1 is perfectly fair
    # Gini: 0 is perfectly equal, 1 is perfectly unequal
    gini_score = 1 - gini
    
    # Theil: 0 is perfectly equal, higher values indicate inequality
    theil_normalized = min(theil, 1.0)  # Cap at 1
    theil_score = 1 - theil_normalized
    
    # Max-min: 1 is perfectly fair, 0 is perfectly unfair
    max_min_score = max_min
    
    # Atkinson: 0 is perfectly equal, 1 is perfectly unequal
    atkinson_score = 1 - atkinson
    
    # Coefficient of variation: lower is better
    cv_score = 1 / (1 + cv)
    
    # Weighted average (adjust weights based on importance)
    weights = {
        'gini': 0.3,
        'theil': 0.2,
        'max_min': 0.25,
        'atkinson': 0.15,
        'cv': 0.1
    }
    
    combined_score = (
        weights['gini'] * gini_score +
        weights['theil'] * theil_score +
        weights['max_min'] * max_min_score +
        weights['atkinson'] * atkinson_score +
        weights['cv'] * cv_score
    )
    
    return float(combined_score)


def _estimate_recovery_time(
    performance_data: Dict[str, List[float]],
    attack_levels: List[float]
) -> float:
    """Estimate recovery time from performance data."""
    # Simplified estimation: time to reach 90% of baseline performance
    # after attack stops. In real implementation, would measure actual recovery.
    
    if not performance_data or len(attack_levels) < 2:
        return 0.0
    
    # Find the steepest drop in performance
    max_drop = 0.0
    for values in performance_data.values():
        if len(values) >= 2:
            baseline = values[0]
            worst_performance = min(values)
            if baseline > 0:
                drop = (baseline - worst_performance) / baseline
                max_drop = max(max_drop, drop)
    
    # Estimate recovery time based on drop magnitude
    # Assuming linear recovery: time = drop * recovery_factor
    recovery_factor = 10.0  # Time units per unit of performance drop
    recovery_time = max_drop * recovery_factor
    
    return float(recovery_time)


def _estimate_computational_complexity(
    system_measurements: Dict[str, List[float]],
    problem_size: Dict[str, float]
) -> float:
    """Estimate computational complexity from measurements."""
    # This is a simplified estimation
    # In practice, would perform complexity analysis
    
    # Extract relevant measurements
    decision_times = system_measurements.get('decision_times', [])
    if not decision_times:
        return 1.0
    
    # Get problem size metrics
    n_agents = problem_size.get('agents', 1)
    n_tasks = problem_size.get('tasks', 1)
    
    # Simple complexity estimation based on decision time scaling
    avg_decision_time = np.mean(decision_times)
    
    # Estimate O-notation complexity
    # This is a very rough estimation
    if n_agents * n_tasks > 1000 and avg_decision_time > 100:
        complexity = 3.0  # Likely O(n³) or worse
    elif n_agents * n_tasks > 100 and avg_decision_time > 10:
        complexity = 2.0  # Likely O(n²)
    elif n_agents * n_tasks > 10 and avg_decision_time > 1:
        complexity = 1.5  # Likely O(n log n)
    else:
        complexity = 1.0  # Likely O(n) or better
    
    return float(complexity)


def _calculate_scalability_score(
    system_measurements: Dict[str, List[float]],
    problem_size: Dict[str, float]
) -> float:
    """Calculate scalability score based on performance scaling."""
    # Simplified scalability assessment
    # In practice, would measure performance at different problem sizes
    
    decision_times = system_measurements.get('decision_times', [])
    if not decision_times or len(decision_times) < 2:
        return 0.5  # Default moderate scalability
    
    avg_decision_time = np.mean(decision_times)
    std_decision_time = np.std(decision_times)
    
    # Get problem size
    n_agents = problem_size.get('agents', 1)
    n_tasks = problem_size.get('tasks', 1)
    total_size = n_agents * n_tasks
    
    # Calculate time per unit problem size
    if total_size > 0 and avg_decision_time > 0:
        time_per_unit = avg_decision_time / total_size
        
        # Lower time per unit indicates better scalability
        # Normalize to [0, 1] range (inverse relationship)
        if time_per_unit < 0.1:
            scalability = 0.9  # Excellent scalability
        elif time_per_unit < 1.0:
            scalability = 0.7  # Good scalability
        elif time_per_unit < 10.0:
            scalability = 0.5  # Moderate scalability
        elif time_per_unit < 100.0:
            scalability = 0.3  # Poor scalability
        else:
            scalability = 0.1  # Very poor scalability
    else:
        scalability = 0.5
    
    # Adjust based on variance (consistent performance is better)
    if std_decision_time > avg_decision_time * 0.5:
        scalability *= 0.8  # Penalize high variance
    
    return float(scalability)


def calculate_comprehensive_metrics(
    simulation_results: Dict[str, Any],
    allocation_results: Dict[str, Any],
    performance_data: Dict[str, List[float]],
    attack_levels: List[float],
    system_measurements: Dict[str, List[float]],
    problem_size: Dict[str, float],
    time_units_per_minute: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate all metrics comprehensively.
    
    Args:
        simulation_results: Simulation efficiency data
        allocation_results: Allocation fairness data
        performance_data: Robustness performance data
        attack_levels: Attack levels for robustness testing
        system_measurements: Practicality measurement data
        problem_size: Problem size description
        time_units_per_minute: Time conversion factor
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Calculate efficiency metrics
    efficiency = calculate_efficiency_metrics(simulation_results, time_units_per_minute)
    metrics['efficiency'] = efficiency.__dict__
    
    # Calculate fairness metrics
    fairness = calculate_fairness_metrics(allocation_results)
    metrics['fairness'] = fairness.__dict__
    
    # Calculate robustness metrics
    robustness = calculate_robustness_metrics(performance_data, attack_levels)
    metrics['robustness'] = robustness.__dict__
    
    # Calculate practicality metrics
    practicality = calculate_practicality_metrics(system_measurements, problem_size)
    metrics['practicality'] = practicality.__dict__
    
    # Calculate overall score (weighted combination)
    overall_score = _calculate_overall_score(efficiency, fairness, robustness, practicality)
    metrics['overall_score'] = overall_score
    
    return metrics


def _calculate_overall_score(
    efficiency: EfficiencyMetrics,
    fairness: FairnessMetrics,
    robustness: RobustnessMetrics,
    practicality: PracticalityMetrics
) -> float:
    """Calculate overall performance score from all metrics."""
    # Normalize individual scores to [0, 1] range
    
    # Efficiency score (weight survivors and response time)
    efficiency_score = (
        0.6 * (efficiency.total_survivors / 100) +  # Normalize to 100 survivors
        0.3 * (1 - efficiency.mean_response_time / 60) +  # Normalize to 60 minutes
        0.1 * efficiency.overall_resource_utilization
    )
    efficiency_score = max(0.0, min(1.0, efficiency_score))
    
    # Fairness score (already normalized)
    fairness_score = fairness.fairness_score
    
    # Robustness score (combine fault tolerance and stability)
    robustness_score = (
        0.6 * robustness.fault_tolerance +
        0.4 * (robustness.stability_index / 10)  # Normalize stability index
    )
    robustness_score = max(0.0, min(1.0, robustness_score))
    
    # Practicality score (focus on decision time and scalability)
    practicality_score = (
        0.5 * (1 - practicality.decision_time_mean / 1000) +  # Normalize to 1000ms
        0.3 * practicality.scalability_score +
        0.2 * (1 - practicality.memory_usage / 1000)  # Normalize to 1000MB
    )
    practicality_score = max(0.0, min(1.0, practicality_score))
    
    # Weighted overall score
    weights = {
        'efficiency': 0.35,
        'fairness': 0.25,
        'robustness': 0.20,
        'practicality': 0.20
    }
    
    overall_score = (
        weights['efficiency'] * efficiency_score +
        weights['fairness'] * fairness_score +
        weights['robustness'] * robustness_score +
        weights['practicality'] * practicality_score
    )
    
    return float(overall_score)