"""
环境测试

测试 DisasterSim-2026 环境的正确性和功能。
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.disaster_sim import DisasterSim
from environments.disaster_scenarios import DisasterScenarioFactory


class TestDisasterSim(unittest.TestCase):
    """DisasterSim 环境测试"""
    
    def setUp(self):
        """测试前设置"""
        self.env_config = {
            'map_size': (100, 100),
            'num_agents': 3,
            'num_victims': 10,
            'num_resources': 5,
            'num_hospitals': 2,
            'disaster_type': 'earthquake',
            'severity': 'medium'
        }
        
        self.env = DisasterSim(**self.env_config)
    
    def tearDown(self):
        """测试后清理"""
        del self.env
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        # 检查环境属性
        self.assertEqual(self.env.map_size, (100, 100))
        self.assertEqual(self.env.num_agents, 3)
        self.assertEqual(self.env.num_victims, 10)
        self.assertEqual(self.env.num_resources, 5)
        self.assertEqual(self.env.num_hospitals, 2)
        self.assertEqual(self.env.disaster_type, 'earthquake')
        self.assertEqual(self.env.severity, 'medium')
        
        # 检查组件初始化
        self.assertEqual(len(self.env.agents), 3)
        self.assertEqual(len(self.env.victims), 10)
        self.assertEqual(len(self.env.resources), 5)
        self.assertEqual(len(self.env.hospitals), 2)
        
        print("✓ Environment initialization test passed")
    
    def test_reset_function(self):
        """测试重置功能"""
        # 执行一些步骤
        state = self.env.reset()
        actions = [0, 1, 2]  # 示例动作
        next_state, rewards, done, info = self.env.step(actions)
        
        # 重置环境
        reset_state = self.env.reset()
        
        # 检查状态是否重置
        self.assertIsNotNone(reset_state)
        self.assertEqual(len(reset_state), self.env.get_state_dimension())
        
        # 检查组件是否重置
        self.assertEqual(len(self.env.agents), 3)
        self.assertEqual(len(self.env.victims), 10)
        
        print("✓ Reset function test passed")
    
    def test_step_function(self):
        """测试步进功能"""
        state = self.env.reset()
        
        # 测试有效动作
        actions = [0, 1, 2]  # 有效动作
        next_state, rewards, done, info = self.env.step(actions)
        
        # 检查返回值的类型和形状
        self.assertIsInstance(next_state, (np.ndarray, list))
        self.assertIsInstance(rewards, list)
        self.assertEqual(len(rewards), self.env.num_agents)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # 检查状态维度
        self.assertEqual(len(next_state), self.env.get_state_dimension())
        
        print("✓ Step function test passed")
    
    def test_state_dimension(self):
        """测试状态维度"""
        state_dim = self.env.get_state_dimension()
        
        # 状态维度应该大于0
        self.assertGreater(state_dim, 0)
        
        # 状态维度应该合理
        expected_min_dim = self.env.num_agents * 2 + self.env.num_victims * 3
        self.assertGreaterEqual(state_dim, expected_min_dim)
        
        print(f"✓ State dimension test passed: {state_dim} dimensions")
    
    def test_action_dimension(self):
        """测试动作维度"""
        action_dim = self.env.get_action_dimension()
        
        # 动作维度应该大于0
        self.assertGreater(action_dim, 0)
        
        # 动作维度应该合理（假设每个智能体有5个动作）
        self.assertEqual(action_dim, 5)
        
        print(f"✓ Action dimension test passed: {action_dim} actions")
    
    def test_reward_range(self):
        """测试奖励范围"""
        state = self.env.reset()
        
        # 测试多个随机动作的奖励
        for _ in range(10):
            actions = list(np.random.randint(0, 5, size=self.env.num_agents))
            _, rewards, _, _ = self.env.step(actions)
            
            # 检查奖励值在合理范围内
            for reward in rewards:
                self.assertIsInstance(reward, (int, float))
                # 奖励通常在[-100, 100]范围内
                self.assertGreaterEqual(reward, -100)
                self.assertLessEqual(reward, 100)
        
        print("✓ Reward range test passed")
    
    def test_done_condition(self):
        """测试完成条件"""
        state = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        # 运行直到完成或达到最大步数
        while not done and steps < max_steps:
            actions = list(np.random.randint(0, 5, size=self.env.num_agents))
            _, _, done, _ = self.env.step(actions)
            steps += 1
        
        # 应该在一定步数内完成
        self.assertLess(steps, max_steps)
        self.assertTrue(done)
        
        print(f"✓ Done condition test passed: completed in {steps} steps")
    
    def test_info_dict(self):
        """测试信息字典"""
        state = self.env.reset()
        actions = [0, 1, 2]
        _, _, _, info = self.env.step(actions)
        
        # 检查必需的信息字段
        required_fields = ['rescued', 'deaths', 'resources_used']
        for field in required_fields:
            self.assertIn(field, info)
            self.assertIsInstance(info[field], (int, float))
        
        # 检查可选字段
        optional_fields = ['response_time', 'victim_severity', 'task_conflicts']
        for field in optional_fields:
            if field in info:
                self.assertIsInstance(info[field], (int, float, list))
        
        print("✓ Info dictionary test passed")
    
    def test_agent_positions(self):
        """测试智能体位置"""
        state = self.env.reset()
        
        # 检查所有智能体位置在地图范围内
        for agent in self.env.agents:
            x, y = agent['position']
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, self.env.map_size[0])
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, self.env.map_size[1])
        
        print("✓ Agent positions test passed")
    
    def test_victim_severities(self):
        """测试受害者严重程度"""
        # 检查所有受害者都有严重程度
        for victim in self.env.victims:
            self.assertIn('severity', victim)
            severity = victim['severity']
            self.assertIn(severity, ['low', 'medium', 'high', 'critical'])
        
        print("✓ Victim severities test passed")
    
    def test_resource_capacities(self):
        """测试资源容量"""
        # 检查所有资源都有容量
        for resource in self.env.resources:
            self.assertIn('capacity', resource)
            self.assertIn('remaining', resource)
            
            capacity = resource['capacity']
            remaining = resource['remaining']
            
            self.assertGreater(capacity, 0)
            self.assertGreaterEqual(remaining, 0)
            self.assertLessEqual(remaining, capacity)
        
        print("✓ Resource capacities test passed")


class TestDisasterScenarios(unittest.TestCase):
    """灾害场景测试"""
    
    def setUp(self):
        """测试前设置"""
        self.factory = DisasterScenarioFactory()
    
    def test_scenario_creation(self):
        """测试场景创建"""
        # 测试所有灾害类型
        disaster_types = ['earthquake', 'flood', 'hurricane', 'industrial', 'pandemic', 'compound']
        
        for disaster_type in disaster_types:
            scenario = self.factory.create_scenario(
                disaster_type=disaster_type,
                severity='medium',
                map_size=(100, 100)
            )
            
            # 检查场景属性
            self.assertEqual(scenario.disaster_type, disaster_type)
            self.assertEqual(scenario.severity, 'medium')
            self.assertEqual(scenario.map_size, (100, 100))
            
            # 检查场景参数
            self.assertIn('epicenter', scenario.params)
            self.assertIn('radius', scenario.params)
            self.assertIn('intensity', scenario.params)
        
        print("✓ Scenario creation test passed")
    
    def test_scenario_parameters(self):
        """测试场景参数"""
        # 测试不同严重程度
        severities = ['low', 'medium', 'high', 'critical']
        
        for severity in severities:
            scenario = self.factory.create_scenario(
                disaster_type='earthquake',
                severity=severity,
                map_size=(100, 100)
            )
            
            # 检查参数随严重程度变化
            intensity = scenario.params['intensity']
            
            if severity == 'low':
                self.assertLess(intensity, 0.4)
            elif severity == 'medium':
                self.assertGreaterEqual(intensity, 0.4)
                self.assertLessEqual(intensity, 0.7)
            elif severity == 'high':
                self.assertGreaterEqual(intensity, 0.7)
                self.assertLessEqual(intensity, 0.9)
            elif severity == 'critical':
                self.assertGreater(intensity, 0.9)
        
        print("✓ Scenario parameters test passed")
    
    def test_predefined_scenarios(self):
        """测试预定义场景"""
        # 获取预定义场景
        scenarios = self.factory.get_predefined_scenarios()
        
        # 检查场景数量
        self.assertGreater(len(scenarios), 0)
        
        # 检查每个场景
        for scenario_name, scenario_config in scenarios.items():
            self.assertIsInstance(scenario_name, str)
            self.assertIsInstance(scenario_config, dict)
            
            # 检查必需配置字段
            required_fields = ['disaster_type', 'severity', 'map_size']
            for field in required_fields:
                self.assertIn(field, scenario_config)
        
        print("✓ Predefined scenarios test passed")


def run_environment_tests():
    """运行所有环境测试"""
    print("\n" + "="*80)
    print("Running Environment Tests")
    print("="*80)
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTest(unittest.makeSuite(TestDisasterSim))
    suite.addTest(unittest.makeSuite(TestDisasterScenarios))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "="*80)
    print("Environment Tests Summary")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ All environment tests passed!")
    else:
        print("✗ Some environment tests failed")
        
        # 打印失败详情
        for test, traceback in result.failures:
            print(f"\nFailure in {test}:")
            print(traceback)
        
        for test, traceback in result.errors:
            print(f"\nError in {test}:")
            print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_environment_tests()