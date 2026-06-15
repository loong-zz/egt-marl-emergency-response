import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FairnessMetricsManager:
    """
    公平性指标监控管理器
    负责计算和监控各种公平性指标，包括基尼系数、泰尔指数等，并提供可视化支持
    """
    
    def __init__(self):
        # 存储各时间步的公平性指标
        self.metrics_history = []  # List of {step: int, metrics: dict}
        
        # 存储各agent的累计指标
        self.agent_metrics = defaultdict(lambda: {
            'rescues': 0,
            'resources_used': 0.0,
            'response_time': 0.0,
            'survival_rate': 0.0
        })
        
        # 存储区域指标
        self.region_metrics = {}  # {region_id: {fitness, rescued, initial}}
        
        # 可视化数据缓存
        self.visualization_cache = {}
        
    def record_agent_metrics(self, agent_id: int, rescues: int = 0, 
                            resources_used: float = 0.0, response_time: float = 0.0,
                            survival_rate: float = 0.0):
        """
        记录单个agent的指标
        
        Args:
            agent_id: Agent ID
            rescues: 救援次数
            resources_used: 使用的资源量
            response_time: 响应时间
            survival_rate: 生存率
        """
        self.agent_metrics[agent_id]['rescues'] += rescues
        self.agent_metrics[agent_id]['resources_used'] += resources_used
        self.agent_metrics[agent_id]['response_time'] += response_time
        self.agent_metrics[agent_id]['survival_rate'] = survival_rate  # 最新值
    
    def update_region_metrics(self, region_id: int, fitness: float, 
                             rescued: int, initial: int):
        """
        更新区域指标
        
        Args:
            region_id: 区域ID
            fitness: 区域适应度
            rescued: 已救援人数
            initial: 初始伤员数
        """
        self.region_metrics[region_id] = {
            'fitness': fitness,
            'rescued': rescued,
            'initial': initial
        }
    
    def calculate_gini_coefficient(self, values: List[float]) -> float:
        """
        计算基尼系数
        
        Args:
            values: 数值列表（如各agent的救援次数、资源使用量等）
        
        Returns:
            基尼系数（0表示完全公平，1表示完全不公平）
        """
        if len(values) == 0 or sum(values) == 0:
            return 0.0
        
        n = len(values)
        if n == 1:
            return 0.0
        
        sorted_values = sorted(values)
        numerator = 0.0
        for i in range(n):
            numerator += (2 * i - n + 1) * sorted_values[i]
        
        denominator = n * sum(values)
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_theil_index(self, values: List[float]) -> float:
        """
        计算泰尔指数
        
        Args:
            values: 数值列表
        
        Returns:
            泰尔指数（0表示完全公平，值越大越不公平）
        """
        if len(values) == 0 or sum(values) == 0:
            return 0.0
        
        n = len(values)
        total = sum(values)
        theil = 0.0
        
        for v in values:
            if v > 0:
                theil += (v / total) * np.log((v / total) / (1.0 / n))
        
        return theil
    
    def calculate_coefficient_of_variation(self, values: List[float]) -> float:
        """
        计算变异系数（标准差/均值）
        
        Args:
            values: 数值列表
        
        Returns:
            变异系数
        """
        if len(values) == 0:
            return 0.0
        
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        
        std = np.std(values)
        return std / mean
    
    def get_overall_fairness_metrics(self) -> Dict:
        """
        获取整体公平性指标
        
        Returns:
            包含各种公平性指标的字典
        """
        # 获取所有agent的救援次数
        rescue_counts = [metrics['rescues'] for metrics in self.agent_metrics.values()]
        
        # 获取所有agent的资源使用量
        resource_usages = [metrics['resources_used'] for metrics in self.agent_metrics.values()]
        
        # 获取所有agent的生存率
        survival_rates = [metrics['survival_rate'] for metrics in self.agent_metrics.values()]
        
        # 获取所有区域的适应度
        region_fitnesses = [metrics['fitness'] for metrics in self.region_metrics.values()]
        
        metrics = {
            # Agent层面的公平性指标
            'agent_rescue_gini': self.calculate_gini_coefficient(rescue_counts),
            'agent_rescue_theil': self.calculate_theil_index(rescue_counts),
            'agent_rescue_cv': self.calculate_coefficient_of_variation(rescue_counts),
            
            'agent_resource_gini': self.calculate_gini_coefficient(resource_usages),
            'agent_resource_theil': self.calculate_theil_index(resource_usages),
            'agent_resource_cv': self.calculate_coefficient_of_variation(resource_usages),
            
            'agent_survival_gini': self.calculate_gini_coefficient(survival_rates),
            'agent_survival_theil': self.calculate_theil_index(survival_rates),
            
            # 区域层面的公平性指标
            'region_fitness_gini': self.calculate_gini_coefficient(region_fitnesses),
            'region_fitness_theil': self.calculate_theil_index(region_fitnesses),
            'region_fitness_cv': self.calculate_coefficient_of_variation(region_fitnesses),
            
            # 基本统计信息
            'num_agents': len(self.agent_metrics),
            'num_regions': len(self.region_metrics),
            'total_rescues': sum(rescue_counts),
            'avg_rescues_per_agent': np.mean(rescue_counts) if rescue_counts else 0.0
        }
        
        return metrics
    
    def record_step_metrics(self, step: int):
        """
        记录当前时间步的公平性指标
        
        Args:
            step: 当前时间步
        """
        metrics = self.get_overall_fairness_metrics()
        self.metrics_history.append({
            'step': step,
            'metrics': metrics
        })
        
        # 保持历史记录在合理范围内（最多1000步）
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
    
    def get_metrics_history(self) -> List[Dict]:
        """
        获取历史指标记录
        
        Returns:
            历史指标列表
        """
        return self.metrics_history
    
    def generate_visualization_data(self) -> Dict:
        """
        生成可视化数据
        
        Returns:
            可视化数据字典，包含图表所需的数据
        """
        if not self.metrics_history:
            return {}
        
        # 提取时间序列数据
        steps = [entry['step'] for entry in self.metrics_history]
        
        # 提取各种指标的时间序列
        agent_rescue_gini = [entry['metrics']['agent_rescue_gini'] for entry in self.metrics_history]
        agent_resource_gini = [entry['metrics']['agent_resource_gini'] for entry in self.metrics_history]
        region_fitness_gini = [entry['metrics']['region_fitness_gini'] for entry in self.metrics_history]
        
        agent_rescue_theil = [entry['metrics']['agent_rescue_theil'] for entry in self.metrics_history]
        region_fitness_theil = [entry['metrics']['region_fitness_theil'] for entry in self.metrics_history]
        
        # 生成可视化数据结构
        visualization_data = {
            'time_series': {
                'steps': steps,
                'gini_coefficients': {
                    'agent_rescue': agent_rescue_gini,
                    'agent_resource': agent_resource_gini,
                    'region_fitness': region_fitness_gini
                },
                'theil_indices': {
                    'agent_rescue': agent_rescue_theil,
                    'region_fitness': region_fitness_theil
                }
            },
            'current_state': self.get_overall_fairness_metrics(),
            'agent_distribution': {
                'agent_ids': list(self.agent_metrics.keys()),
                'rescue_counts': [m['rescues'] for m in self.agent_metrics.values()],
                'resource_usages': [m['resources_used'] for m in self.agent_metrics.values()]
            },
            'region_distribution': {
                'region_ids': list(self.region_metrics.keys()),
                'fitness_values': [m['fitness'] for m in self.region_metrics.values()],
                'rescued_counts': [m['rescued'] for m in self.region_metrics.values()],
                'initial_counts': [m['initial'] for m in self.region_metrics.values()]
            }
        }
        
        return visualization_data
    
    def generate_summary_report(self) -> str:
        """
        生成公平性指标汇总报告
        
        Returns:
            汇总报告字符串
        """
        metrics = self.get_overall_fairness_metrics()
        
        report = [
            "=" * 60,
            "公平性指标监控报告",
            "=" * 60,
            "",
            "【Agent层面公平性】",
            f"  救援分布基尼系数: {metrics['agent_rescue_gini']:.4f}",
            f"  救援分布泰尔指数: {metrics['agent_rescue_theil']:.4f}",
            f"  救援分布变异系数: {metrics['agent_rescue_cv']:.4f}",
            "",
            f"  资源使用基尼系数: {metrics['agent_resource_gini']:.4f}",
            f"  资源使用泰尔指数: {metrics['agent_resource_theil']:.4f}",
            f"  资源使用变异系数: {metrics['agent_resource_cv']:.4f}",
            "",
            f"  生存率基尼系数: {metrics['agent_survival_gini']:.4f}",
            f"  生存率泰尔指数: {metrics['agent_survival_theil']:.4f}",
            "",
            "【区域层面公平性】",
            f"  区域适应度基尼系数: {metrics['region_fitness_gini']:.4f}",
            f"  区域适应度泰尔指数: {metrics['region_fitness_theil']:.4f}",
            f"  区域适应度变异系数: {metrics['region_fitness_cv']:.4f}",
            "",
            "【统计信息】",
            f"  Agent数量: {metrics['num_agents']}",
            f"  区域数量: {metrics['num_regions']}",
            f"  总救援次数: {metrics['total_rescues']}",
            f"  平均救援次数/Agent: {metrics['avg_rescues_per_agent']:.2f}",
            "",
            "=" * 60
        ]
        
        return "\n".join(report)
    
    def reset(self):
        """重置所有指标"""
        self.metrics_history = []
        self.agent_metrics.clear()
        self.region_metrics.clear()
        self.visualization_cache.clear()
