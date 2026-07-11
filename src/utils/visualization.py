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
import logging
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")

logger = logging.getLogger(__name__)


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


def plot_egt_strategy_evolution(
    strategy_history: List[Dict[str, Any]],
    strategy_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot EGT strategy distribution evolution over training episodes.

    Generates two subplots:
    1. Stacked area chart of strategy distribution over time
    2. Fairness vs Efficiency weight evolution line chart

    Args:
        strategy_history: List of dicts with keys 'episode' and 'strategy_distribution'
                         (and optionally 'fairness_weight', 'efficiency_weight')
        strategy_names: Names of the strategies (e.g. ['Fairness', 'Efficiency', 'Balanced'])
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not strategy_history:
        logger.warning("plot_egt_strategy_evolution: empty strategy_history, skipping")
        return

    episodes = [s['episode'] for s in strategy_history]
    distributions = np.array([s['strategy_distribution'] for s in strategy_history])

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图1：策略分布堆叠面积图
    ax1 = axes[0]
    ax1.stackplot(episodes, distributions.T, labels=strategy_names, alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Strategy Distribution')
    ax1.set_title('EGT Strategy Distribution Evolution')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 子图2：公平-效率权重演化
    ax2 = axes[1]
    if 'fairness_weight' in strategy_history[0]:
        fairness_weights = [s['fairness_weight'] for s in strategy_history]
        efficiency_weights = [s['efficiency_weight'] for s in strategy_history]
        ax2.plot(episodes, fairness_weights, 'b-o', label='Fairness Weight', markersize=4)
        ax2.plot(episodes, efficiency_weights, 'r-s', label='Efficiency Weight', markersize=4)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balance Line')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Weight')
        ax2.set_title('Fairness-Efficiency Trade-off Evolution')
        ax2.legend(loc='best')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No fairness/efficiency weight data',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Fairness-Efficiency Trade-off (N/A)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy evolution plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_egt_strategy_recommendation(
    recommendation: Dict[str, Any],
    strategy_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a single EGT strategy recommendation as a pie chart with annotation.

    Args:
        recommendation: Output of EGTLayer.get_strategy_recommendation()
        strategy_names: Names of strategies
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    distribution = recommendation.get('strategy_distribution', [])
    if not distribution:
        logger.warning("plot_egt_strategy_recommendation: empty distribution, skipping")
        return

    # Pad with zeros if distribution length < strategy_names length
    if len(distribution) < len(strategy_names):
        distribution = list(distribution) + [0.0] * (len(strategy_names) - len(distribution))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1：策略分布饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
    wedges, texts, autotexts = ax1.pie(
        distribution, labels=strategy_names, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    ax1.set_title(f"Strategy Distribution\n(Dominant: {recommendation.get('dominant_strategy', 'N/A')})")

    # 子图2：公平-效率权重条形图
    fairness_w = recommendation.get('fairness_weight', 0.0)
    efficiency_w = recommendation.get('efficiency_weight', 0.0)
    ax2.barh(['Fairness', 'Efficiency'], [fairness_w, efficiency_w],
             color=['#4A90E2', '#E24A4A'])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Weight')
    ax2.set_title('Fairness-Efficiency Trade-off')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate([fairness_w, efficiency_w]):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center')

    # 添加推荐文本
    rec_text = recommendation.get('recommendation', '')
    if rec_text:
        fig.suptitle(f"Recommendation: {rec_text}", fontsize=10, y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy recommendation plot saved to {save_path}")

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
            elif agent_type == 'vehicle' or agent_type == 'ambulance':
                color = 'red'
            elif agent_type == 'personnel' or agent_type == 'hospital':
                color = 'green'
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
