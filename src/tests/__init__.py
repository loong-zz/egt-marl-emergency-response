"""
EGT-MARL 测试套件

包含环境测试、算法测试和指标测试。
"""

from .test_environment import TestDisasterSim
from .test_algorithms import TestEGTMARL
from .test_metrics import TestMetricsCollector

__all__ = [
    'TestDisasterSim',
    'TestEGTMARL',
    'TestMetricsCollector'
]

__version__ = '1.0.0'