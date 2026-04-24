"""
Data visualization utilities for EGT-MARL disaster resource allocation system.

This module provides functions for visualizing:
- Training curves
- Algorithm comparisons
- Ablation studies
- Disaster scenarios
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_training_curves(
    training_data: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training curves for rewards and losses.
    
    Args:
        training_data: Dictionary containing training metrics
        save_path: Path to save the plot
        show: Whether to show the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    if 'rewards' in training_data:
        axes[0].plot(training_data['rewards'], label='Rewards')
        axes[0].set_title('Training Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
    
    # Plot losses
    if 'losses' in training_data:
        axes[1].plot(training_data['losses'], label='Losses')
        axes[1].set_title('Training Losses')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_algorithm_comparison(
    algorithm_results: Dict[str, Dict[str, List[float]]],
    metric: str = 'total_reward',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot comparison of different algorithms.
    
    Args:
        algorithm_results: Dictionary of algorithm results
        metric: Metric to compare
        save_path: Path to save the plot
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algorithm, results in algorithm_results.items():
        if metric in results:
            ax.plot(results[metric], label=algorithm)
    
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_study(
    ablation_results: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot ablation study results.
    
    Args:
        ablation_results: Dictionary of ablation results
        save_path: Path to save the plot
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    components = list(ablation_results.keys())
    scores = list(ablation_results.values())
    
    ax.bar(components, scores)
    ax.set_title('Ablation Study Results')
    ax.set_xlabel('Component')
    ax.set_ylabel('Performance Score')
    ax.set_xticklabels(components, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_disaster_scenario(
    scenario_data: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize disaster scenario and resource allocation.
    
    Args:
        scenario_data: Dictionary containing scenario information
        save_path: Path to save the plot
        show: Whether to show the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot affected areas
    if 'affected_areas' in scenario_data:
        for area in scenario_data['affected_areas']:
            x, y = area.get('coordinates', (0, 0))
            severity = area.get('severity', 1)
            ax.scatter(x, y, s=50 * severity, alpha=0.6, label=f'Area {area.get("id", "")}')
    
    # Plot resource depots
    if 'resource_depots' in scenario_data:
        for depot in scenario_data['resource_depots']:
            x, y = depot.get('coordinates', (0, 0))
            ax.scatter(x, y, s=100, marker='^', color='green', label=f'Depot {depot.get("id", "")}')
    
    # Plot rescue agents
    if 'rescue_agents' in scenario_data:
        for agent in scenario_data['rescue_agents']:
            x, y = agent.get('position', (0, 0))
            agent_type = agent.get('type', 'unknown')
            if agent_type == 'drone':
                color = 'blue'
            elif agent_type == 'ambulance':
                color = 'red'
            else:
                color = 'purple'
            ax.scatter(x, y, s=80, marker='*', color=color, label=f'{agent_type} {agent.get("id", "")}')
    
    ax.set_title('Disaster Scenario Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
