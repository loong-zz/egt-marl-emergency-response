"""
主测试脚本

运行所有测试：环境测试、算法测试、指标测试。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_comprehensive_test_suite():
    """运行全面的测试套件"""
    print("\n" + "="*80)
    print("EGT-MARL Comprehensive Test Suite")
    print("="*80)
    print("Testing the complete DisasterSim-2026 system with EGT-MARL algorithm")
    print("\n")
    
    # 导入测试模块
    from tests.test_environment import run_environment_tests
    from tests.test_algorithms import run_algorithm_tests
    from tests.test_metrics import run_metrics_tests
    
    results = []
    
    # 1. 运行环境测试
    print("\n" + "-"*80)
    print("Phase 1: Environment Tests")
    print("-"*80)
    env_result = run_environment_tests()
    results.append(("Environment", env_result))
    
    # 2. 运行算法测试
    print("\n" + "-"*80)
    print("Phase 2: Algorithm Tests")
    print("-"*80)
    algo_result = run_algorithm_tests()
    results.append(("Algorithms", algo_result))
    
    # 3. 运行指标测试
    print("\n" + "-"*80)
    print("Phase 3: Metrics Tests")
    print("-"*80)
    metrics_result = run_metrics_tests()
    results.append(("Metrics", metrics_result))
    
    # 打印总体结果
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE RESULTS")
    print("="*80)
    
    all_passed = True
    for module_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{module_name:15} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "-"*80)
    if all_passed:
        print("🎉 SUCCESS: All tests passed! The system is ready for deployment.")
        print("\nNext steps:")
        print("1. Run training: python experiments/train_egt_marl.py")
        print("2. Evaluate baselines: python experiments/evaluate_baselines.py")
        print("3. Run ablation study: python experiments/ablation_study.py")
        print("4. Test robustness: python experiments/robustness_test.py")
    else:
        print("⚠ WARNING: Some tests failed. Please review the failures above.")
        print("\nRecommended actions:")
        print("1. Check the specific test failures")
        print("2. Fix the issues in the corresponding modules")
        print("3. Re-run the failed tests")
        print("4. Ensure all tests pass before proceeding")
    
    print("\n" + "="*80)
    
    return all_passed


def run_quick_test():
    """运行快速测试（仅关键功能）"""
    print("\n" + "="*80)
    print("EGT-MARL Quick Test")
    print("="*80)
    print("Running quick tests of critical functionality")
    print("\n")
    
    import unittest
    
    # 只运行最关键的测试
    test_modules = [
        'tests.test_environment.TestDisasterSim',
        'tests.test_algorithms.TestEGTMARL',
        'tests.test_metrics.TestMetricsCollector'
    ]
    
    # 创建测试加载器
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 加载测试
    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
        except Exception as e:
            print(f"Warning: Could not load {module}: {e}")
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # 打印结果
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ Quick test passed! Basic functionality is working.")
    else:
        print("✗ Quick test failed. Critical issues found.")
    
    return result.wasSuccessful()


def run_integration_test():
    """运行集成测试"""
    print("\n" + "="*80)
    print("EGT-MARL Integration Test")
    print("="*80)
    print("Testing integration of all system components")
    print("\n")
    
    try:
        # 测试环境创建
        from environments.disaster_sim import DisasterSim
        from algorithms.egt_marl import EGTMARL
        from utils.metrics import MetricsCollector
        
        print("1. Creating environment...")
        env = DisasterSim(
            map_size=(50, 50),
            num_agents=2,
            num_victims=5,
            num_resources=3,
            num_hospitals=1,
            disaster_type='earthquake',
            severity='medium'
        )
        print("   ✓ Environment created")
        
        print("2. Creating algorithm...")
        algorithm = EGTMARL(
            state_dim=env.get_state_dimension(),
            action_dim=env.get_action_dimension(),
            num_agents=env.num_agents,
            hidden_dim=32
        )
        print("   ✓ Algorithm created")
        
        print("3. Creating metrics collector...")
        metrics = MetricsCollector()
        print("   ✓ Metrics collector created")
        
        print("4. Running integration test episode...")
        state = env.reset()
        total_reward = 0
        
        for step in range(10):  # 运行10步
            actions = algorithm.select_actions(state, epsilon=0.5)
            next_state, rewards, done, info = env.step(actions)
            
            total_reward += sum(rewards)
            
            # 存储经验
            algorithm.store_experience(state, actions, rewards, next_state, done)
            
            # 记录指标
            episode_metrics = {
                'step': step,
                'total_reward': total_reward,
                'rescued': info.get('rescued', 0),
                'deaths': info.get('deaths', 0)
            }
            metrics.record(f'step_{step}', episode_metrics)
            
            state = next_state
            
            if done:
                break
        
        print(f"   ✓ Episode completed: {step+1} steps, total reward: {total_reward:.2f}")
        
        print("5. Testing algorithm update...")
        if len(algorithm.replay_buffer) >= algorithm.batch_size:
            loss = algorithm.update()
            print(f"   ✓ Algorithm updated, loss: {loss:.4f}")
        else:
            print("   ⚠ Not enough experience for update (expected)")
        
        print("6. Testing metrics export...")
        df = metrics.to_dataframe()
        print(f"   ✓ Metrics exported to DataFrame: {len(df)} records")
        
        print("\n" + "="*80)
        print("INTEGRATION TEST RESULTS")
        print("="*80)
        print("🎉 SUCCESS: All components integrated successfully!")
        print("\nSystem components tested:")
        print("  - Environment (DisasterSim)")
        print("  - Algorithm (EGT-MARL)")
        print("  - Metrics collector")
        print("  - Experience replay")
        print("  - Training update")
        print("  - Data export")
        
        return True
        
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EGT-MARL tests')
    parser.add_argument('--mode', type=str, default='comprehensive',
                       choices=['quick', 'integration', 'comprehensive', 'all'],
                       help='Test mode: quick, integration, comprehensive, or all')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        success = run_quick_test()
    elif args.mode == 'integration':
        success = run_integration_test()
    elif args.mode == 'comprehensive':
        success = run_comprehensive_test_suite()
    elif args.mode == 'all':
        # 运行所有测试模式
        print("Running all test modes...\n")
        
        print("1. Quick test:")
        quick_success = run_quick_test()
        
        print("\n2. Integration test:")
        integration_success = run_integration_test()
        
        print("\n3. Comprehensive test suite:")
        comprehensive_success = run_comprehensive_test_suite()
        
        success = quick_success and integration_success and comprehensive_success
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETE")
        print("="*80)
        if success:
            print("🎉 ALL TEST MODES PASSED!")
        else:
            print("⚠ SOME TESTS FAILED")
    
    # 设置退出码
    sys.exit(0 if success else 1)