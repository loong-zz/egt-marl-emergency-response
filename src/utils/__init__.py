"""
Utilities module for EGT-MARL disaster resource allocation system.

This module contains utility functions for:
- Metrics calculation and evaluation
- Fairness measurement and analysis
- Data visualization and plotting
- Data processing and transformation
"""

from .metrics import (
    calculate_efficiency_metrics,
    calculate_fairness_metrics,
    calculate_robustness_metrics,
    calculate_practicality_metrics,
)

from .fairness import (
    gini_coefficient,
    theil_index,
    max_min_fairness,
    calculate_fairness_efficiency_tradeoff,
)

from .visualization import (
    plot_training_curves,
    plot_algorithm_comparison,
    plot_ablation_study,
    visualize_disaster_scenario,
)

from .data_processing import (
    collect_experiment_data,
    analyze_results,
    convert_data_format,
    save_results,
)

__all__ = [
    # Metrics
    "calculate_efficiency_metrics",
    "calculate_fairness_metrics", 
    "calculate_robustness_metrics",
    "calculate_practicality_metrics",
    
    # Fairness
    "gini_coefficient",
    "theil_index",
    "max_min_fairness",
    "calculate_fairness_efficiency_tradeoff",
    
    # Visualization
    "plot_training_curves",
    "plot_algorithm_comparison",
    "plot_ablation_study",
    "visualize_disaster_scenario",
    
    # Data processing
    "collect_experiment_data",
    "analyze_results",
    "convert_data_format",
    "save_results",
]

__version__ = "1.0.0"