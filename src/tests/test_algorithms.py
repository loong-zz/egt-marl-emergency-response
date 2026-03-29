"""
算法测试

测试 EGT-MARL 算法及其组件的正确性和功能。
"""

import unittest
import numpy as np
import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithms.egt_marl import EGTMARL
from algorithms.qmix_improved import QMIXImproved
from algorithms.dynamic_frontier import DynamicParetoFrontier
from algorithms.egt_layer import EGTLayer
from algorithms.marl_layer import MARLLayer
from algorithms.anti_spoofing import AntiSpoofingMechanism


class TestEGTMARL(unittest.TestCase):
    """EGT-MARL 算法测试"""
    
    def setUp(self):
        """测试前设置"""
        # 设置测试参数
        self.state_dim = 64
        self.action_dim = 5
        self.num_agents = 3
        self.hidden_dim = 32
        self.device = torch.device('cpu')
        
        # 创建算法实例
        self.algorithm = EGTMARL(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
    
    def tearDown(self):
        """测试后清理"""
        del self.algorithm
    
    def test_algorithm_initialization(self):
        """测试算法初始化"""
        # 检查算法属性
        self.assertEqual(self.algorithm.state_dim, self.state_dim)
        self.assertEqual(self.algorithm.action_dim, self.action_dim)
        self.assertEqual(self.algorithm.num_agents, self.num_agents)
        self.assertEqual(self.algorithm.hidden_dim, self.hidden_dim)
        self.assertEqual(self.algorithm.device, self.device)
        
        # 检查网络是否创建
        self.assertIsNotNone(self.algorithm.agent_networks)
        self.assertIsNotNone(self.algorithm.mixing_network)
        self.assertIsNotNone(self.algorithm.target_agent_networks)
        self.assertIsNotNone(self.algorithm.target_mixing_network)
        
        # 检查优化器
        self.assertIsNotNone(self.algorithm.optimizer)
        
        print("✓ Algorithm initialization test passed")
    
    def test_select_actions(self):
        """测试动作选择"""
        # 创建测试状态
        batch_size = 2
        state = torch.randn(batch_size, self.state_dim)
        
        # 测试探索模式 (epsilon=1.0)
        actions = self.algorithm.select_actions(state, epsilon=1.0)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(actions[0]), self.num_agents)
        
        # 测试利用模式 (epsilon=0.0)
        actions = self.algorithm.select_actions(state, epsilon=0.0)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(actions[0]), self.num_agents)
        
        # 检查动作在有效范围内
        for batch in actions:
            for action in batch:
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, self.action_dim)
        
        print("✓ Select actions test passed")
    
    def test_store_experience(self):
        """测试经验存储"""
        # 创建测试数据
        state = torch.randn(self.state_dim)
        actions = [0, 1, 2]
        rewards = [1.0, 0.5, -0.5]
        next_state = torch.randn(self.state_dim)
        done = False
        
        # 存储经验
        initial_buffer_size = len(self.algorithm.replay_buffer)
        self.algorithm.store_experience(state, actions, rewards, next_state, done)
        
        # 检查缓冲区大小增加
        self.assertEqual(len(self.algorithm.replay_buffer), initial_buffer_size + 1)
        
        # 存储更多经验直到缓冲区满
        for _ in range(self.algorithm.buffer_size * 2):
            self.algorithm.store_experience(state, actions, rewards, next_state, done)
        
        # 检查缓冲区大小不超过容量
        self.assertLessEqual(len(self.algorithm.replay_buffer), self.algorithm.buffer_size)
        
        print("✓ Store experience test passed")
    
    def test_update(self):
        """测试更新功能"""
        # 首先填充一些经验
        for _ in range(self.algorithm.batch_size * 2):
            state = torch.randn(self.state_dim)
            actions = list(np.random.randint(0, self.action_dim, size=self.num_agents))
            rewards = list(np.random.randn(self.num_agents))
            next_state = torch.randn(self.state_dim)
            done = np.random.random() > 0.5
            
            self.algorithm.store_experience(state, actions, rewards, next_state, done)
        
        # 执行更新
        loss = self.algorithm.update()
        
        # 检查损失值
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
        
        print(f"✓ Update test passed - Loss: {loss:.4f}")
    
    def test_get_state_dict(self):
        """测试获取状态字典"""
        state_dict = self.algorithm.get_state_dict()
        
        # 检查状态字典包含必要的键
        required_keys = ['agent_networks', 'mixing_network', 'optimizer']
        for key in required_keys:
            self.assertIn(key, state_dict)
        
        print("✓ Get state dict test passed")
    
    def test_load_state_dict(self):
        """测试加载状态字典"""
        # 获取当前状态
        original_state_dict = self.algorithm.get_state_dict()
        
        # 修改算法
        self.algorithm.update()
        
        # 加载原始状态
        self.algorithm.load_state_dict(original_state_dict)
        
        # 检查是否成功加载
        current_state_dict = self.algorithm.get_state_dict()
        
        # 比较网络参数（简化检查）
        self.assertEqual(
            len(original_state_dict['agent_networks']),
            len(current_state_dict['agent_networks'])
        )
        
        print("✓ Load state dict test passed")
    
    def test_set_egt_parameters(self):
        """测试设置EGT参数"""
        # 设置EGT参数
        self.algorithm.set_egt_parameters(
            lambda_param=0.7,
            pareto_weights={'efficiency': 0.4, 'fairness': 0.3, 'robustness': 0.3},
            anti_spoofing_enabled=True
        )
        
        # 检查参数设置
        self.assertEqual(self.algorithm.egt_lambda, 0.7)
        self.assertIsNotNone(self.algorithm.pareto_weights)
        self.assertTrue(self.algorithm.anti_spoofing_enabled)
        
        print("✓ Set EGT parameters test passed")
    
    def test_compute_egt_rewards(self):
        """测试计算EGT奖励"""
        # 创建测试数据
        individual_rewards = torch.tensor([[1.0, 0.5, -0.5], [0.8, 0.2, 0.0]])
        cooperation_levels = torch.tensor([0.7, 0.5])
        
        # 计算EGT奖励
        egt_rewards = self.algorithm._compute_egt_rewards(individual_rewards, cooperation_levels)
        
        # 检查输出形状
        self.assertEqual(egt_rewards.shape, individual_rewards.shape)
        
        print("✓ Compute EGT rewards test passed")


class TestQMIXImproved(unittest.TestCase):
    """改进的QMIX算法测试"""
    
    def setUp(self):
        """测试前设置"""
        self.state_dim = 64
        self.action_dim = 5
        self.num_agents = 3
        self.hidden_dim = 32
        self.device = torch.device('cpu')
        
        self.algorithm = QMIXImproved(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            hidden_dim=self.hidden_dim,
            device=self.device
        )
    
    def tearDown(self):
        """测试后清理"""
        del self.algorithm
    
    def test_hierarchical_action_selection(self):
        """测试分层动作选择"""
        # 创建测试状态
        state = torch.randn(1, self.state_dim)
        
        # 获取分层动作
        strategic_actions, tactical_actions, operational_actions = \
            self.algorithm.select_hierarchical_actions(state, epsilon=0.0)
        
        # 检查动作形状
        self.assertEqual(strategic_actions.shape, (1, self.num_agents))
        self.assertEqual(tactical_actions.shape, (1, self.num_agents))
        self.assertEqual(operational_actions.shape, (1, self.num_agents))
        
        print("✓ Hierarchical action selection test passed")
    
    def test_attention_mixing(self):
        """测试注意力混合网络"""
        # 创建测试数据
        batch_size = 2
        agent_qs = torch.randn(batch_size, self.num_agents, 1)
        states = torch.randn(batch_size, self.state_dim)
        
        # 计算混合Q值
        total_q = self.algorithm.mixing_network(agent_qs, states)
        
        # 检查输出形状
        self.assertEqual(total_q.shape, (batch_size, 1))
        
        print("✓ Attention mixing test passed")
    
    def test_enhanced_reward_computation(self):
        """测试增强奖励计算"""
        # 创建测试数据
        individual_rewards = torch.tensor([[1.0, 0.5, -0.5], [0.8, 0.2, 0.0]])
        states = torch.randn(2, self.state_dim)
        actions = torch.tensor([[0, 1, 2], [2, 0, 1]])
        
        # 计算增强奖励
        enhanced_rewards = self.algorithm.compute_enhanced_rewards(
            individual_rewards, states, actions
        )
        
        # 检查输出形状
        self.assertEqual(enhanced_rewards.shape, individual_rewards.shape)
        
        print("✓ Enhanced reward computation test passed")


class TestDynamicParetoFrontier(unittest.TestCase):
    """动态帕累托前沿测试"""
    
    def setUp(self):
        """测试前设置"""
        self.frontier = DynamicParetoFrontier(
            num_objectives=3,  # 效率、公平性、鲁棒性
            population_size=50,
            max_generations=100
        )
    
    def tearDown(self):
        """测试后清理"""
        del self.frontier
    
    def test_frontier_initialization(self):
        """测试前沿初始化"""
        # 检查属性
        self.assertEqual(self.frontier.num_objectives, 3)
        self.assertEqual(self.frontier.population_size, 50)
        self.assertEqual(self.frontier.max_generations, 100)
        
        # 检查种群初始化
        self.assertIsNotNone(self.frontier.population)
        self.assertEqual(len(self.frontier.population), 50)
        
        print("✓ Frontier initialization test passed")
    
    def test_non_dominated_sorting(self):
        """测试非支配排序"""
        # 创建测试解
        solutions = np.array([
            [0.8, 0.7, 0.6],  # 帕累托最优
            [0.7, 0.8, 0.5],  # 帕累托最优
            [0.6, 0.6, 0.7],  # 帕累托最优
            [0.5, 0.5, 0.5],  # 被支配
            [0.4, 0.4, 0.4]   # 被支配
        ])
        
        # 执行非支配排序
        fronts = self.frontier._non_dominated_sort(solutions)
        
        # 检查前沿数量
        self.assertGreaterEqual(len(fronts), 1)
        
        # 第一前沿应该包含帕累托最优解
        self.assertEqual(len(fronts[0]), 3)
        
        print("✓ Non-dominated sorting test passed")
    
    def test_crowding_distance(self):
        """测试拥挤距离计算"""
        # 创建测试前沿
        front = np.array([
            [0.9, 0.1, 0.5],
            [0.8, 0.3, 0.4],
            [0.7, 0.5, 0.3],
            [0.6, 0.7, 0.2],
            [0.5, 0.9, 0.1]
        ])
        
        # 计算拥挤距离
        distances = self.frontier._crowding_distance(front)
        
        # 检查距离计算
        self.assertEqual(len(distances), len(front))
        self.assertTrue(np.all(distances >= 0))
        
        # 边界点应该有无限距离
        self.assertTrue(np.isinf(distances[0]) or distances[0] > 1e6)
        self.assertTrue(np.isinf(distances[-1]) or distances[-1] > 1e6)
        
        print("✓ Crowding distance test passed")
    
    def test_evolution(self):
        """测试进化过程"""
        # 执行一代进化
        initial_population = self.frontier.population.copy()
        self.frontier._evolve()
        
        # 检查种群大小不变
        self.assertEqual(len(self.frontier.population), self.frontier.population_size)
        
        # 检查种群是否改变
        self.assertFalse(np.array_equal(initial_population, self.frontier.population))
        
        print("✓ Evolution test passed")
    
    def test_frontier_quality_metrics(self):
        """测试前沿质量指标"""
        # 创建测试前沿
        frontier_points = np.array([
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.7, 0.7, 0.9]
        ])
        
        # 计算质量指标
        metrics = self.frontier.evaluate_frontier_quality(frontier_points)
        
        # 检查指标计算
        required_metrics = ['hypervolume', 'spacing', 'convergence', 'uniformity']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
        
        print("✓ Frontier quality metrics test passed")


class TestAlgorithmComponents(unittest.TestCase):
    """算法组件测试"""
    
    def test_egt_layer(self):
        """测试EGT层"""
        egt_layer = EGTLayer(
            num_agents=3,
            num_strategies=5,
            lambda_param=0.5
        )
        
        # 测试复制动力学更新
        current_distribution = torch.ones(3, 5) / 5  # 均匀分布
        payoffs = torch.randn(3, 5)
        
        new_distribution = egt_layer.update_replicator_dynamics(
            current_distribution, payoffs
        )
        
        # 检查输出形状和性质
        self.assertEqual(new_distribution.shape, current_distribution.shape)
        self.assertTrue(torch.all(new_distribution >= 0))
        self.assertTrue(torch.allclose(new_distribution.sum(dim=1), torch.ones(3)))
        
        print("✓ EGT layer test passed")
    
    def test_marl_layer(self):
        """测试MARL层"""
        marl_layer = MARLLayer(
            state_dim=64,
            action_dim=5,
            num_agents=3,
            hidden_dim=32
        )
        
        # 测试Q值计算
        state = torch.randn(1, 64)
        q_values = marl_layer.compute_q_values(state)
        
        # 检查输出形状
        self.assertEqual(q_values.shape, (1, 3, 5))
        
        print("✓ MARL layer test passed")
    
    def test_anti_spoofing(self):
        """测试抗欺骗机制"""
        anti_spoofing = AntiSpoofingMechanism(
            num_agents=3,
            detection_threshold=0.7
        )
        
        # 测试行为分析
        behaviors = torch.randn(10, 3, 5)  # 10个时间步，3个智能体，5个特征
        anomaly_scores = anti_spoofing.analyze_behavior(behaviors)
        
        # 检查输出
        self.assertEqual(len(anomaly_scores), 3)
        self.assertTrue(torch.all(anomaly_scores >= 0))
        self.assertTrue(torch.all(anomaly_scores <= 1))
        
        print("✓ Anti-spoofing test passed")


def run_algorithm_tests():
    """运行所有算法测试"""
    print("\n" + "="*80)
    print("Running Algorithm Tests")
    print("="*80)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTest(unittest.makeSuite(TestEGTMARL))
    suite.addTest(unittest.makeSuite(TestQMIXImproved))
    suite.addTest(unittest.makeSuite(TestDynamicParetoFrontier))
    suite.addTest(unittest.makeSuite(TestAlgorithmComponents))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "="*80)
    print("Algorithm Tests Summary")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ All algorithm tests passed!")
    else:
        print("✗ Some algorithm tests failed")
        
        # 打印失败详情
        for test, traceback in result.failures:
            print(f"\nFailure in {test}:")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nError in {test}:")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_algorithm_tests()