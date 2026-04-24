"""
Data processing utilities for EGT-MARL disaster resource allocation system.

This module provides functions for:
- Collecting experiment data
- Analyzing results
- Converting data formats
- Saving and loading results
"""

import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import os


def collect_experiment_data(
    experiment_dir: str,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Collect experiment data from multiple runs.
    
    Args:
        experiment_dir: Directory containing experiment results
        metrics: List of metrics to collect
        
    Returns:
        Dictionary containing collected data
    """
    if metrics is None:
        metrics = ['total_reward', 'fairness_score', 'efficiency_score']
    
    collected_data = {metric: [] for metric in metrics}
    
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        return collected_data
    
    # Collect data from each run
    for run_dir in experiment_path.iterdir():
        if run_dir.is_dir():
            results_file = run_dir / 'results.json'
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    for metric in metrics:
                        if metric in results:
                            collected_data[metric].append(results[metric])
                except Exception as e:
                    print(f"Error reading {results_file}: {e}")
    
    return collected_data


def analyze_results(
    results_data: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze experiment results and calculate statistics.
    
    Args:
        results_data: Dictionary of collected results
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {}
    
    for metric, values in results_data.items():
        if values:
            analysis[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        else:
            analysis[metric] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
    
    return analysis


def convert_data_format(
    data: Dict[str, Any],
    target_format: str = 'pandas'
) -> Any:
    """
    Convert data to different formats.
    
    Args:
        data: Input data
        target_format: Target format ('pandas', 'numpy', 'json')
        
    Returns:
        Data in the target format
    """
    if target_format == 'pandas':
        # Convert to DataFrame
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    
    elif target_format == 'numpy':
        # Convert to numpy array
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return {k: np.array(v) for k, v in data.items()}
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.array([data])
    
    elif target_format == 'json':
        # Convert to JSON string
        return json.dumps(data, indent=2)
    
    else:
        return data


def save_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = 'json'
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results to save
        save_path: Path to save file
        format: File format ('json', 'pickle', 'csv')
    """
    save_path = Path(save_path)
    
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == 'csv':
        # Convert to DataFrame first
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
