"""
EGT-MARL 实验脚本包

包含训练、评估、基准测试和消融研究的实验脚本。
"""

from .train_egt_marl import train_egt_marl
from .evaluate_baselines import evaluate_baselines
from .ablation_study import run_ablation_study
from .robustness_test import run_robustness_test

__all__ = [
    'train_egt_marl',
    'evaluate_baselines',
    'run_ablation_study',
    'run_robustness_test'
]

__version__ = '1.0.0'