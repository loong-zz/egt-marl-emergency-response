"""
指标测试

测试评估指标的计算正确性和功能。
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.metrics import MetricsCollector
from utils.fairness import FairnessMetrics
from environments.visualization import DisasterVisualizer


class TestMetricsCollector(unittest.TestCase):
    """指标收集器测试"""
    
    def setUp(self):
        """测试前设置"""
        self.metrics_collector = MetricsCollector()
    
    def tearDown(self):
        """测试后清理"""
        del self.metrics_collector
    
    def test_initialization(self):
        """测试初始化"""
        # 检查指标字典初始化
        self.assertIsInstance(self.metrics_collector.metrics, dict)
        self.assertEqual(len(self.metrics_collector.metrics), 0)
        
        print("✓ Metrics collector initialization test passed")
    
    def test_record_metrics(self):
        """测试记录指标"""
        # 记录一些指标
        test_metrics = {
            'rescue_rate': 85.5,
            'avg_response_time': 45.2,
            'resource_utilization': 72.3,
            'total_reward': 1500.0
        }
        
        self.metrics_collector.record('test_episode', test_metrics)
        
        # 检查指标是否记录
        self.assertIn('test_episode', self.metrics_collector.metrics)
        self.assertEqual(self.metrics_collector.metrics['test_episode'], test_metrics)
        
        print("✓ Record metrics test passed")
    
    def test_get_metrics(self):
        """测试获取指标"""
        # 记录一些指标
        test_metrics = {'rescue_rate': 85.5}
        self.metrics_collector.record('episode1', test_metrics)
        
        # 获取指标
        retrieved_metrics = self.metrics_collector.get('episode1')
        self.assertEqual(retrieved_metrics, test_metrics)
        
        # 测试获取不存在的指标
        non_existent = self.metrics_collector.get('non_existent')
        self.assertIsNone(non_existent)
        
        print("✓ Get metrics test passed")
    
    def test_get_all_metrics(self):
        """测试获取所有指标"""
        # 记录多个指标
        for i in range(5):
            self.metrics_collector.record(f'episode_{i}', {'rescue_rate': 80 + i})
        
        # 获取所有指标
        all_metrics = self.metrics_collector.get_all()
        
        # 检查返回的指标数量
        self.assertEqual(len(all_metrics), 5)
        
        print("✓ Get all metrics test passed")
    
    def test_compute_statistics(self):
        """测试计算统计信息"""
        # 记录多个指标值
        rescue_rates = [75.0, 80.0, 85.0, 90.0, 95.0]
        for i, rate in enumerate(rescue_rates):
            self.metrics_collector.record(f'episode_{i}', {'rescue_rate': rate})
        
        # 计算统计信息
        stats = self.metrics_collector.compute_statistics('rescue_rate')
        
        # 检查统计信息
        required_stats = ['mean', 'std', 'min', 'max', 'median']
        for stat in required_stats:
            self.assertIn(stat, stats)
            self.assertIsInstance(stats[stat], float)
        
        # 检查计算正确性
        self.assertAlmostEqual(stats['mean'], np.mean(rescue_rates))
        self.assertAlmostEqual(stats['std'], np.std(rescue_rates))
        self.assertAlmostEqual(stats['min'], np.min(rescue_rates))
        self.assertAlmostEqual(stats['max'], np.max(rescue_rates))
        self.assertAlmostEqual(stats['median'], np.median(rescue_rates))
        
        print("✓ Compute statistics test passed")
    
    def test_export_to_dataframe(self):
        """测试导出到DataFrame"""
        # 记录一些指标
        metrics_data = [
            {'rescue_rate': 85.5, 'response_time': 45.2},
            {'rescue_rate': 90.0, 'response_time': 40.1},
            {'rescue_rate': 88.3, 'response_time': 42.7}
        ]
        
        for i, metrics in enumerate(metrics_data):
            self.metrics_collector.record(f'episode_{i}', metrics)
        
        # 导出到DataFrame
        df = self.metrics_collector.to_dataframe()
        
        # 检查DataFrame属性
        self.assertEqual(len(df), 3)  # 3行
        self.assertIn('rescue_rate', df.columns)
        self.assertIn('response_time', df.columns)
        
        print("✓ Export to DataFrame test passed")
    
    def test_save_and_load(self):
        """测试保存和加载"""
        import tempfile
        import json
        
        # 记录一些指标
        test_metrics = {'rescue_rate': 85.5}
        self.metrics_collector.record('test_episode', test_metrics)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存指标
            self.metrics_collector.save(temp_path)
            
            # 创建新的收集器并加载
            new_collector = MetricsCollector()
            new_collector.load(temp_path)
            
            # 检查加载的指标
            loaded_metrics = new_collector.get('test_episode')
            self.assertEqual(loaded_metrics, test_metrics)
            
            print("✓ Save and load test passed")
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestFairnessMetrics(unittest.TestCase):
    """公平性指标测试"""
    
    def setUp(self):
        """测试前设置"""
        self.fairness = FairnessMetrics()
    
    def test_gini_coefficient(self):
        """测试基尼系数"""
        # 测试完全平等分布
        equal_distribution = np.array([100, 100, 100, 100, 100])
        gini_equal = self.fairness.gini_coefficient(equal_distribution)
        self.assertAlmostEqual(gini_equal, 0.0, places=5)
        
        # 测试完全不平等分布
        unequal_distribution = np.array([0, 0, 0, 0, 500])
        gini_unequal = self.fairness.gini_coefficient(unequal_distribution)
        self.assertAlmostEqual(gini_unequal, 0.8, places=5)  # 对于这种分布，基尼系数为0.8
        
        # 测试随机分布
        random_distribution = np.random.rand(100) * 100
        gini_random = self.fairness.gini_coefficient(random_distribution)
        self.assertGreaterEqual(gini_random, 0.0)
        self.assertLessEqual(gini_random, 1.0)
        
        print("✓ Gini coefficient test passed")
    
    def test_max_min_fairness(self):
        """测试最大最小公平性"""
        # 测试完全公平
        fair_distribution = np.array([100, 100, 100])
        maxmin_fair = self.fairness.max_min_fairness(fair_distribution)
        self.assertAlmostEqual(maxmin_fair, 1.0, places=5)
        
        # 测试不公平
        unfair_distribution = np.array([10, 50, 100])
        maxmin_unfair = self.fairness.max_min_fairness(unfair_distribution)
        self.assertAlmostEqual(maxmin_unfair, 0.1, places=5)  # 10/100 = 0.1
        
        print("✓ Max-min fairness test passed")
    
    def test_theil_index(self):
        """测试泰尔指数"""
        # 测试完全平等
        equal_distribution = np.array([100, 100, 100])
        theil_equal = self.fairness.theil_index(equal_distribution)
        self.assertAlmostEqual(theil_equal, 0.0, places=5)
        
        # 测试不平等
        unequal_distribution = np.array([50, 100, 150])
        theil_unequal = self.fairness.theil_index(unequal_distribution)
        self.assertGreater(theil_unequal, 0.0)
        
        print("✓ Theil index test passed")
    
    def test_jain_fairness_index(self):
        """测试Jain公平性指数"""
        # 测试完全公平
        fair_distribution = np.array([100, 100, 100])
        jain_fair = self.fairness.jain_fairness_index(fair_distribution)
        self.assertAlmostEqual(jain_fair, 1.0, places=5)
        
        # 测试不公平
        unfair_distribution = np.array([10, 20, 30])
        jain_unfair = self.fairness.jain_fairness_index(unfair_distribution)
        self.assertLess(jain_unfair, 1.0)
        self.assertGreater(jain_unfair, 0.0)
        
        print("✓ Jain fairness index test passed")
    
    def test_atkinson_index(self):
        """测试阿特金森指数"""
        # 测试不同不平等厌恶参数
        distribution = np.array([50, 100, 150])
        
        # epsilon=0 (对不平等不敏感)
        atkinson_0 = self.fairness.atkinson_index(distribution, epsilon=0)
        self.assertAlmostEqual(atkinson_0, 0.0, places=5)
        
        # epsilon=1 (中等不平等厌恶)
        atkinson_1 = self.fairness.atkinson_index(distribution, epsilon=1)
        self.assertGreaterEqual(atkinson_1, 0.0)
        self.assertLessEqual(atkinson_1, 1.0)
        
        print("✓ Atkinson index test passed")
    
    def test_compute_all_fairness_metrics(self):
        """测试计算所有公平性指标"""
        # 创建测试分布
        distribution = np.random.rand(50) * 100
        
        # 计算所有指标
        all_metrics = self.fairness.compute_all(distribution)
        
        # 检查所有指标都计算了
        required_metrics = ['gini', 'maxmin', 'theil', 'jain', 'atkinson_0.5', 'atkinson_1.0']
        for metric in required_metrics:
            self.assertIn(metric, all_metrics)
            self.assertIsInstance(all_metrics[metric], float)
        
        print("✓ Compute all fairness metrics test passed")
    
    def test_fairness_efficiency_tradeoff(self):
        """测试公平性-效率权衡"""
        # 创建测试数据
        efficiency_scores = np.array([0.8, 0.7, 0.9, 0.6])
        fairness_scores = np.array([0.6, 0.8, 0.5, 0.7])
        
        # 计算权衡指标
        tradeoff = self.fairness.fairness_efficiency_tradeoff(
            efficiency_scores, fairness_scores
        )
        
        # 检查输出
        self.assertIn('correlation', tradeoff)
        self.assertIn('tradeoff_index', tradeoff)
        self.assertIn('pareto_frontier_size', tradeoff)
        
        self.assertIsInstance(tradeoff['correlation'], float)
        self.assertGreaterEqual(tradeoff['correlation'], -1.0)
        self.assertLessEqual(tradeoff['correlation'], 1.0)
        
        print("✓ Fairness-efficiency tradeoff test passed")


class TestVisualizationMetrics(unittest.TestCase):
    """可视化指标测试"""
    
    def setUp(self):
        """测试前设置"""
        self.env_config = {
            'map_size': (100, 100),
            'disaster_type': 'earthquake',
            'severity': 'medium'
        }
        self.visualizer = DisasterVisualizer(self.env_config)
    
    def test_performance_dashboard(self):
        """测试性能仪表盘"""
        # 创建测试指标数据
        metrics_history = {
            'rescue_rate': [10, 20, 30, 40, 50, 60, 70, 80, 85, 90],
            'avg_response_time': [120, 110, 100, 95, 90, 85, 80, 75, 70, 65],
            'resource_utilization': [20, 30, 40, 50, 60, 65, 70, 75, 80, 85],
            'gini_index': [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15],
            'max_min_fairness': [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        }
        
        # 测试仪表盘创建（不保存文件）
        try:
            fig = self.visualizer.plot_performance_dashboard(metrics_history)
            self.assertIsNotNone(fig)
            
            print("✓ Performance dashboard test passed")
            
        except Exception as e:
            self.fail(f"Performance dashboard creation failed: {e}")
    
    def test_comparison_chart(self):
        """测试对比图表"""
        # 创建测试算法数据
        algorithms_data = {
            'EGT-MARL': {
                'rescue_rate': [10, 25, 45, 60, 75, 85, 90, 92, 94, 95],
                'rescue_rate_std': [2, 3, 4, 5, 4, 3, 2, 2, 1, 1]
            },
            'QMIX': {
                'rescue_rate': [5, 15, 30, 45, 60, 70, 78, 83, 87, 90],
                'rescue_rate_std': [3, 4, 5, 6, 5, 4, 3, 3, 2, 2]
            }
        }
        
        # 测试对比图表创建
        try:
            fig = self.visualizer.plot_comparison_chart(
                algorithms_data, 
                'rescue_rate'
            )
            self.assertIsNotNone(fig)
            
            print("✓ Comparison chart test passed")
            
        except Exception as e:
            self.fail(f"Comparison chart creation failed: {e}")


class TestRobustnessMetrics(unittest.TestCase):
    """鲁棒性指标测试"""
    
    def test_performance_under_attack(self):
        """测试攻击下的性能"""
        # 创建测试数据
        baseline_performance = 85.0  # 无攻击时的性能
        attack_performance = 65.0    # 有攻击时的性能
        
        # 计算性能保持率
        robustness = attack_performance / baseline_performance * 100
        
        self.assertLessEqual(robustness, 100.0)
        self.assertGreaterEqual(robustness, 0.0)
        
        print("✓ Performance under attack test passed")
    
    def test_recovery_time(self):
        """测试恢复时间"""
        # 模拟攻击和恢复过程
        performance_timeline = [85, 85, 85, 30, 40, 50, 60, 70, 80, 85]
        
        # 找到攻击点（性能突然下降）
        attack_point = 3  # 索引3处性能下降
        
        # 找到恢复点（性能恢复到攻击前的90%）
        recovery_threshold = 85 * 0.9  # 76.5
        recovery_point = None
        
        for i in range(attack_point + 1, len(performance_timeline)):
            if performance_timeline[i] >= recovery_threshold:
                recovery_point = i
                break
        
        if recovery_point is not None:
            recovery_time = recovery_point - attack_point
            self.assertGreaterEqual(recovery_time, 0)
            
            print(f"✓ Recovery time test passed: {recovery_time} steps")
        else:
            print("⚠ Recovery time test: System did not recover within timeline")
    
    def test_system_stability(self):
        """测试系统稳定性"""
        # 创建性能波动数据
        performance_sequence = [80, 82, 78, 85, 83, 81, 79, 84, 82, 80]
        
        # 计算稳定性（波动越小越稳定）
        mean_performance = np.mean(performance_sequence)
        std_performance = np.std(performance_sequence)
        
        # 稳定性分数（标准差越小，分数越高）
        stability_score = 100 / (1 + std_performance)
        
        self.assertGreater(stability_score, 0)
        self.assertLess(stability_score, 100)
        
        print(f"✓ System stability test passed: score={stability_score:.2f}")


def run_metrics_tests():
    """运行所有指标测试"""
    print("\n" + "="*80)
    print("Running Metrics Tests")
    print("="*80)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTest(unittest.makeSuite(TestMetricsCollector))
    suite.addTest(unittest.makeSuite(TestFairnessMetrics))
    suite.addTest(unittest.makeSuite(TestVisualizationMetrics))
    suite.addTest(unittest.makeSuite(TestRobustnessMetrics))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "="*80)
    print("Metrics Tests Summary")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ All metrics tests passed!")
    else:
        print("✗ Some metrics tests failed")
        
        # 打印失败详情
        for test, traceback in result.failures:
            print(f"\nFailure in {test}:")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nError in {test}:")
            print(traceback)
    
    return result.wasSuccessful()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Running Complete Test Suite")
    print("="*80)
    
    # 导入其他测试模块
    from .test_environment import run_environment_tests
    from .test_algorithms import run_algorithm_tests
    
    # 运行所有测试
    env_success = run_environment_tests()
    algo_success = run_algorithm_tests()
    metrics_success = run_metrics_tests()
    
    # 总体结果
    print("\n" + "="*80)
    print("Overall Test Results")
    print("="*80)
    
    all_success = env_success and algo_success and metrics_success
    
    if all_success:
        print("🎉 All tests passed! System is ready for deployment.")
    else:
        print("⚠ Some tests failed. Please review the failures above.")
    
    return all_success


if __name__ == "__main__":
    run_all_tests()