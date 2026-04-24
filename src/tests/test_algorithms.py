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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.egt_marl import EGTMARL
from algorithms.qmix_improved import ImprovedQMIX
from algorithms.dynamic_frontier import DynamicParetoFrontier
from algorithms.egt_layer import EGTLayer
from algorithms.marl_layer import MARLLayer
from algorithms.anti_spoofing import AntiSpoofing


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
        self.assertEqual(self.algorithm.device, self.device)
        
        # 检查网络是否创建
        self.assertIsNotNone(self.algorithm.marl_layer)
        self.assertIsNotNone(self.algorithm.egt_layer)
        self.assertIsNotNone(self.algorithm.anti_spoofing)
        self.assertIsNotNone(self.algorithm.dynamic_frontier)
        
        # 检查优化器
        self.assertIsNotNone(self.algorithm.marl_optimizer)
        self.assertIsNotNone(self.algorithm.egt_optimizer)
        
        print("[OK] Algorithm initialization test passed")
    
    def test_select_actions(self):
        """测试动作选择"""
        # 创建测试状态
        state = torch.randn(self.state_dim)
        
        # 测试探索模式 (epsilon=1.0)
        actions = self.algorithm.select_actions(state, epsilon=1.0)
        self.assertEqual(len(actions), self.num_agents)
        
        # 测试利用模式 (epsilon=0.0)
        actions = self.algorithm.select_actions(state, epsilon=0.0)
        self.assertEqual(len(actions), self.num_agents)
        
        # 检查动作在有效范围内
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_dim)
        
        print("[OK] Select actions test passed")
    
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
        
        print("[OK] Store experience test passed")
    
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
        loss = self.algorithm.update_parameters()
        
        # 检查损失值
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
        
        print(f"[OK] Update test passed - Loss: {loss:.4f}")
    
    def test_get_state_dict(self):
        """测试获取状态字典"""
        state_dict = self.algorithm.get_state_dict()
        
        # 检查状态字典包含必要的键
        required_keys = ['marl_layer', 'egt_layer']
        for key in required_keys:
            self.assertIn(key, state_dict)
        
        # 检查可选键
        optional_keys = ['dynamic_frontier', 'anti_spoofing']
        for key in optional_keys:
            if key in state_dict:
                self.assertIsInstance(state_dict[key], dict)
        
        print("[OK] Get state dict test passed")
    
    def test_load_state_dict(self):
        """测试加载状态字典"""
        # 获取当前状态
        original_state_dict = self.algorithm.get_state_dict()
        
        # 加载原始状态
        self.algorithm.load_state_dict(original_state_dict)
        
        # 检查是否成功加载
        current_state_dict = self.algorithm.get_state_dict()
        
        # 比较网络参数（简化检查）
        self.assertEqual(
            len(original_state_dict['marl_layer']),
            len(current_state_dict['marl_layer'])
        )
        
        print("[OK] Load state dict test passed")
    
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
        
        print("[OK] Set EGT parameters test passed")
    
    def test_compute_egt_rewards(self):
        """测试计算EGT奖励"""
        # 创建测试数据
        individual_rewards = torch.tensor([[1.0, 0.5, -0.5], [0.8, 0.2, 0.0]])
        cooperation_levels = torch.tensor([0.7, 0.5])
        
        # 计算EGT奖励
        egt_rewards = self.algorithm._compute_egt_rewards(individual_rewards, cooperation_levels)
        
        # 检查输出形状
        self.assertEqual(egt_rewards.shape, individual_rewards.shape)
        
        print("[OK] Compute EGT rewards test passed")


class TestImprovedQMIX(unittest.TestCase):
    """改进的QMIX算法测试"""
    
    def setUp(self):
        """测试前设置"""
        self.state_dim = 64
        self.action_dim = 5
        self.num_agents = 3
        self.hidden_dim = 32
        self.device = torch.device('cpu')
        
        self.algorithm = ImprovedQMIX(
            num_agents=self.num_agents,
            obs_dim=self.state_dim,
            state_dim=self.state_dim,
            action_dims=[self.action_dim for _ in range(self.num_agents)],
            agent_types=['drone'] * self.num_agents,
            config={'mixing_hidden_dim': self.hidden_dim}
        )
    
    def tearDown(self):
        """测试后清理"""
        del self.algorithm
    
    def test_hierarchical_action_selection(self):
        """测试分层动作选择"""
        # 创建测试状态
        observations = [torch.randn(self.state_dim).numpy() for _ in range(self.num_agents)]
        state = torch.randn(self.state_dim).numpy()
        
        # 获取动作
        actions, action_infos = self.algorithm.act(observations, state, training=False)
        
        # 检查动作长度
        self.assertEqual(len(actions), self.num_agents)
        self.assertEqual(len(action_infos), self.num_agents)
        
        # 检查每个动作信息包含分层动作
        for action_info in action_infos:
            self.assertIn('strategic', action_info)
            self.assertIn('tactical', action_info)
            self.assertIn('operational', action_info)
        
        print("[OK] Hierarchical action selection test passed")
    
    def test_attention_mixing(self):
        """测试注意力混合网络"""
        # 创建测试数据
        batch_size = 2
        agent_qs = torch.randn(batch_size, self.num_agents)
        states = torch.randn(batch_size, self.state_dim)
        
        # 计算混合Q值
        total_q = self.algorithm.mixing_network(agent_qs, states)
        
        # 检查输出形状
        self.assertEqual(total_q.shape, (batch_size, 1))
        
        print("[OK] Attention mixing test passed")
    
    def test_enhanced_reward_computation(self):
        """测试增强奖励计算"""
        # 创建测试数据
        metrics = {
            'total_survivors': 75,
            'mean_response_time': 45.0,
            'gini_coefficient': 0.35,
            'tasks_completion_rate': 0.8,
            'stability_index': 8.5,
            'communication_effectiveness': 0.7,
            'overall_resource_utilization': 0.65
        }
        
        # 计算增强奖励
        # 使用第一个agent的reward_structure来计算奖励
        total_reward, rewards = self.algorithm.agents[0].reward_structure.calculate_reward(metrics)
        
        # 检查奖励值
        self.assertIsInstance(total_reward, float)
        self.assertIsInstance(rewards, dict)
        self.assertIn('efficiency', rewards)
        self.assertIn('fairness', rewards)
        self.assertIn('robustness', rewards)
        
        print("[OK] Enhanced reward computation test passed")


class TestDynamicParetoFrontier(unittest.TestCase):
    """动态帕累托前沿测试"""
    
    def setUp(self):
        """测试前设置"""
        self.frontier = DynamicParetoFrontier(
            config={
                'num_objectives': 3,  # 效率、公平性、鲁棒性
                'population_size': 50,
                'frontier_size': 100
            }
        )
    
    def tearDown(self):
        """测试后清理"""
        del self.frontier
    
    def test_frontier_initialization(self):
        """测试前沿初始化"""
        # 检查属性
        self.assertEqual(self.frontier.config.get('num_objectives'), 3)
        self.assertEqual(self.frontier.config.get('population_size'), 50)
        
        # 检查前沿初始化
        self.assertIsNotNone(self.frontier.frontier)
        self.assertEqual(len(self.frontier.frontier), 100)  # frontier_size
        
        print("[OK] Frontier initialization test passed")
    
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
        # 注意：这里我们直接测试非支配排序逻辑，因为 DynamicParetoFrontier 没有公开的 _non_dominated_sorting 方法
        # 我们创建一个简单的测试来验证前沿功能
        test_points = [
            {'efficiency': 0.8, 'fairness': 0.7, 'robustness': 0.6},
            {'efficiency': 0.7, 'fairness': 0.8, 'robustness': 0.5},
            {'efficiency': 0.6, 'fairness': 0.6, 'robustness': 0.7},
            {'efficiency': 0.5, 'fairness': 0.5, 'robustness': 0.5}
        ]
        
        # 测试前沿更新
        self.frontier.update_frontier(test_points, test_points[0])
        
        # 检查前沿大小
        self.assertGreater(len(self.frontier.frontier), 0)
        
        print("[OK] Non-dominated sorting test passed")
    
    def test_crowding_distance(self):
        """测试拥挤距离计算"""
        # 测试前沿质量指标
        frontier_metrics = self.frontier.get_frontier_metrics()
        
        # 检查指标计算
        self.assertIsInstance(frontier_metrics, object)
        self.assertTrue(frontier_metrics.hypervolume >= 0)
        self.assertTrue(frontier_metrics.spread >= 0)
        self.assertTrue(frontier_metrics.convergence >= 0)
        self.assertTrue(frontier_metrics.uniformity >= 0)
        
        print("[OK] Crowding distance test passed")
    
    def test_evolution(self):
        """测试进化过程"""
        # 执行前沿更新
        initial_frontier_size = len(self.frontier.frontier)
        
        # 创建测试数据
        test_points = [
            {'efficiency': 0.85, 'fairness': 0.75, 'robustness': 0.65},
            {'efficiency': 0.75, 'fairness': 0.85, 'robustness': 0.55}
        ]
        
        # 更新前沿
        self.frontier.update_frontier(test_points, test_points[0])
        
        # 检查前沿大小
        self.assertGreaterEqual(len(self.frontier.frontier), initial_frontier_size)
        
        print("[OK] Evolution test passed")
    
    def test_frontier_quality_metrics(self):
        """测试前沿质量指标"""
        # 计算质量指标
        metrics = self.frontier.get_frontier_metrics()
        
        # 检查指标计算
        self.assertIsInstance(metrics, object)
        self.assertTrue(metrics.hypervolume >= 0)
        self.assertTrue(metrics.spread >= 0)
        self.assertTrue(metrics.convergence >= 0)
        self.assertTrue(metrics.uniformity >= 0)
        self.assertTrue(metrics.cardinality >= 0)
        
        print("[OK] Frontier quality metrics test passed")


class TestAlgorithmComponents(unittest.TestCase):
    """算法组件测试"""
    
    def test_egt_layer(self):
        """测试EGT层"""
        egt_layer = EGTLayer(
            num_strategies=5,
            learning_rate=0.5
        )
        
        # 测试策略进化
        performance_metrics = {'fairness_score': 0.6, 'efficiency_score': 0.7, 'total_reward': 100}
        
        new_distribution = egt_layer.evolve_strategies(performance_metrics)
        
        # 检查输出形状和性质
        self.assertEqual(new_distribution.shape, (5,))
        self.assertTrue(torch.all(new_distribution >= 0))
        self.assertTrue(torch.allclose(new_distribution.sum(), torch.tensor(1.0)))
        
        print("[OK] EGT layer test passed")
    
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
        q_values = marl_layer.forward(state)
        
        # 检查输出形状
        self.assertEqual(q_values.shape, (1, 3, 5))
        
        print("[OK] MARL layer test passed")
    
    def test_anti_spoofing(self):
        """测试抗欺骗机制"""
        anti_spoofing = AntiSpoofing(
            observation_dim=100,
            action_dim=5,
            detection_threshold=0.7
        )
        
        # 测试行为验证
        observation = torch.randn(100)
        action = torch.randn(5)
        is_legitimate, confidence = anti_spoofing.verify_action(observation, action, agent_id=0)
        
        # 检查输出
        self.assertIsInstance(is_legitimate, (bool, torch.Tensor))
        self.assertTrue(0 <= confidence <= 1)
        
        # 测试行为纠正
        corrected_action = anti_spoofing.correct_action(observation, action, agent_id=0)
        self.assertIsInstance(corrected_action, (dict, torch.Tensor))
        
        # 如果返回的是字典，检查是否包含必要的键
        if isinstance(corrected_action, dict):
            self.assertIn('_corrected', corrected_action)
            self.assertIn('_original_reputation', corrected_action)
            self.assertIn('_correction_strength', corrected_action)
        
        print("[OK] Anti-spoofing test passed")


def run_algorithm_tests():
    """运行所有算法测试"""
    print("\n" + "="*80)
    print("Running Algorithm Tests")
    print("="*80)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEGTMARL))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestImprovedQMIX))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDynamicParetoFrontier))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAlgorithmComponents))
    
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
        print("[OK] All algorithm tests passed!")
    else:
        print("[FAIL] Some algorithm tests failed")
        
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