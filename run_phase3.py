"""
第三阶段：系统集成与测试

运行完整的第三阶段工作，包括：
1. 运行所有测试
2. 训练EGT-MARL
3. 评估基线算法
4. 运行消融研究
5. 测试鲁棒性
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def run_command(cmd, description):
    """运行命令并处理输出"""
    print(f"\n{description}...")
    print(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {description} 完成")
            if result.stdout:
                print("输出:", result.stdout[:500])  # 只显示前500字符
            return True
        else:
            print(f"✗ {description} 失败")
            print("错误:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ {description} 异常: {e}")
        return False

def run_phase3(args):
    """运行第三阶段"""
    print_header("第三阶段：系统集成与测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    
    results = {}
    
    # 1. 运行测试
    if args.run_tests:
        print_header("步骤1: 运行系统测试")
        
        test_modes = ['quick', 'integration', 'comprehensive']
        for mode in test_modes:
            if mode == 'quick' or mode == args.test_mode:
                cmd = f'python src/tests/run_all_tests.py --mode {mode}'
                success = run_command(cmd, f"运行{mode}测试")
                results[f'test_{mode}'] = success
                
                if not success and args.stop_on_failure:
                    print("测试失败，停止执行")
                    return False
    
    # 2. 训练EGT-MARL
    if args.train_egt_marl:
        print_header("步骤2: 训练EGT-MARL")
        
        output_dir = f"experiment_results/phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = f'python src/experiments/train_egt_marl.py '
        cmd += f'--config configs/training.yaml '
        cmd += f'--output_dir {output_dir} '
        
        if args.num_episodes:
            cmd += f'--num_episodes {args.num_episodes} '
        
        success = run_command(cmd, "训练EGT-MARL算法")
        results['train_egt_marl'] = success
        
        if success:
            # 保存训练输出目录供后续使用
            args.trained_model_dir = output_dir
            print(f"训练完成，模型保存在: {output_dir}")
        
        if not success and args.stop_on_failure:
            print("训练失败，停止执行")
            return False
    
    # 3. 评估基线算法
    if args.evaluate_baselines:
        print_header("步骤3: 评估基线算法")
        
        output_dir = f"evaluation_results/phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = f'python src/experiments/evaluate_baselines.py '
        cmd += f'--config configs/evaluation.yaml '
        cmd += f'--output_dir {output_dir} '
        
        if args.num_episodes:
            cmd += f'--num_episodes {args.num_episodes} '
        
        success = run_command(cmd, "评估基线算法")
        results['evaluate_baselines'] = success
        
        if not success and args.stop_on_failure:
            print("评估失败，停止执行")
            return False
    
    # 4. 运行消融研究
    if args.run_ablation:
        print_header("步骤4: 运行消融研究")
        
        output_dir = f"ablation_results/phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = f'python src/experiments/ablation_study.py '
        cmd += f'--config configs/ablation.yaml '
        cmd += f'--output_dir {output_dir} '
        
        success = run_command(cmd, "运行消融研究")
        results['ablation_study'] = success
        
        if not success and args.stop_on_failure:
            print("消融研究失败，停止执行")
            return False
    
    # 5. 测试鲁棒性
    if args.test_robustness:
        print_header("步骤5: 测试鲁棒性")
        
        output_dir = f"robustness_results/phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = f'python src/experiments/robustness_test.py '
        cmd += f'--config configs/robustness.yaml '
        cmd += f'--output_dir {output_dir} '
        
        # 如果训练了模型，使用训练好的模型
        if hasattr(args, 'trained_model_dir') and args.trained_model_dir:
            model_path = Path(args.trained_model_dir) / 'models' / 'best_model.pt'
            if model_path.exists():
                cmd += f'--model_path "{model_path}" '
        
        success = run_command(cmd, "测试鲁棒性")
        results['robustness_test'] = success
        
        if not success and args.stop_on_failure:
            print("鲁棒性测试失败，停止执行")
            return False
    
    # 打印总结
    print_header("第三阶段完成总结")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n执行结果:")
    all_success = True
    for task, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {task:20} {status}")
        if not success:
            all_success = False
    
    print("\n生成的文件:")
    base_dirs = ['experiment_results', 'evaluation_results', 'ablation_results', 'robustness_results']
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            # 找到最新的phase3目录
            dirs = [d for d in os.listdir(base_dir) if d.startswith('phase3_')]
            if dirs:
                latest = max(dirs)
                print(f"  {base_dir}/{latest}/")
    
    if all_success:
        print("\n🎉 第三阶段所有任务完成成功！")
        print("\n下一步建议:")
        print("1. 查看生成的结果文件和可视化")
        print("2. 分析实验数据")
        print("3. 根据结果优化算法参数")
        print("4. 准备论文写作")
    else:
        print("\n⚠ 部分任务失败，请检查错误信息")
    
    return all_success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行第三阶段：系统集成与测试')
    
    # 任务选择
    parser.add_argument('--run-tests', action='store_true', help='运行系统测试')
    parser.add_argument('--train-egt-marl', action='store_true', help='训练EGT-MARL算法')
    parser.add_argument('--evaluate-baselines', action='store_true', help='评估基线算法')
    parser.add_argument('--run-ablation', action='store_true', help='运行消融研究')
    parser.add_argument('--test-robustness', action='store_true', help='测试鲁棒性')
    
    # 运行所有任务
    parser.add_argument('--run-all', action='store_true', help='运行所有任务')
    
    # 参数
    parser.add_argument('--test-mode', default='comprehensive', 
                       choices=['quick', 'integration', 'comprehensive'],
                       help='测试模式')
    parser.add_argument('--num-episodes', type=int, help='每个任务的episode数量')
    parser.add_argument('--stop-on-failure', action='store_true', 
                       help='任务失败时停止执行')
    
    args = parser.parse_args()
    
    # 如果指定了--run-all，启用所有任务
    if args.run_all:
        args.run_tests = True
        args.train_egt_marl = True
        args.evaluate_baselines = True
        args.run_ablation = True
        args.test_robustness = True
    
    # 如果没有指定任何任务，运行所有任务
    if not any([args.run_tests, args.train_egt_marl, args.evaluate_baselines, 
                args.run_ablation, args.test_robustness]):
        print("未指定任务，将运行所有任务")
        args.run_all = True
        args.run_tests = True
        args.train_egt_marl = True
        args.evaluate_baselines = True
        args.run_ablation = True
        args.test_robustness = True
    
    # 运行第三阶段
    success = run_phase3(args)
    
    # 设置退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()