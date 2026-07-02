"""
决策时间评估脚本

评估 EGT-MARL 及其组件的决策时间性能。
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.disaster_sim import DisasterSim
from algorithms.egt_marl import EGTMARL
from utils.metrics import MetricsCollector
import logging

# 初始化logger
logger = logging.getLogger(__name__)


class DecisionTimeEvaluator:
    """决策时间评估器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化决策时间评估器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.setup_device()
        
        # 决策时间记录
        self.decision_times = []
        self.step_times = []
        self.marl_times = []
        self.egt_times = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config = {
            'evaluation': {
                'num_episodes': 30,
                'max_steps_per_episode': 200,
                'warmup_steps': 5,  # 预热步数，不计入统计
            },
            'environment': {
                'map_size': (1000, 1000),
                'num_agents': 20,
                'num_victims': 200,
                'num_resources': 10,
                'num_areas': 3,
                'disaster_type': 'earthquake',
                'severity': 'medium'
            },
            'algorithm': {
                'hidden_dim': 64,
                'mixing_hidden_dim': 64,
                'attention_heads': 4
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                for section in file_config:
                    if section not in config:
                        config[section] = file_config[section]
                    else:
                        config[section].update(file_config.get(section, {}))
            except Exception as e:
                logger.warning(f"Config load error: {e}")
        
        return config
    
    def setup_device(self):
        """设置计算设备"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
    
    def setup_environment(self):
        """设置环境"""
        env_config = self.config['environment']
        self.env = DisasterSim(
            map_size=tuple(env_config['map_size']),
            num_agents=env_config['num_agents'],
            num_victims=env_config['num_victims'],
            num_resources=env_config['num_resources'],
            num_areas=env_config['num_areas'],
            disaster_type=env_config['disaster_type'],
            severity=env_config['severity']
        )
    
    def setup_algorithm(self):
        """设置算法"""
        algo_config = self.config['algorithm']
        state_dim = self.env.get_state_dimension()
        action_dim = 32
        
        self.algorithm = EGTMARL(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=self.env.get_num_agents(),
            device=self.device
        )
        
        self.algorithm.set_egt_parameters(
            lambda_param=0.5,
            anti_spoofing_enabled=True
        )
    
    def evaluate_single_episode(self, warmup: int = 5) -> Dict[str, Any]:
        """评估单个episode的决策时间"""
        episode_times = {
            'step_times': [],
            'decision_times': [],
            'marl_times': [],
            'egt_times': []
        }
        
        state = self.env.reset()
        done = False
        step = 0
        max_steps = self.config['evaluation']['max_steps_per_episode']
        
        while not done and step < max_steps:
            step_start = time.perf_counter()
            
            # 决策时间测量
            decision_start = time.perf_counter()
            
            # MARL层决策时间
            marl_start = time.perf_counter()
            actions_list = self.algorithm.select_actions(state, epsilon=0.0)
            actions = {i: {'tactical': actions_list[i]} for i in range(len(actions_list))}
            marl_end = time.perf_counter()
            
            # EGT层决策时间
            egt_start = time.perf_counter()
            # EGT决策已集成到select_actions中，这里只记录EGT更新时间
            self.algorithm.egt_layer.evolve_strategies(performance_metrics={})
            egt_end = time.perf_counter()
            
            decision_end = time.perf_counter()
            
            # 执行动作
            next_state, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            step_end = time.perf_counter()
            
            # 记录时间（跳过预热阶段）
            if step >= warmup:
                step_time = (step_end - step_start) * 1000  # 转换为毫秒
                decision_time = (decision_end - decision_start) * 1000
                marl_time = (marl_end - marl_start) * 1000
                egt_time = (egt_end - egt_start) * 1000
                
                episode_times['step_times'].append(step_time)
                episode_times['decision_times'].append(decision_time)
                episode_times['marl_times'].append(marl_time)
                episode_times['egt_times'].append(egt_time)
            
            state = next_state
            step += 1
        
        return episode_times
    
    def evaluate_component_times(self) -> Dict[str, Any]:
        """评估各组件决策时间"""
        logger.info("Starting decision time evaluation...")
        
        all_episode_times = []
        
        for ep in range(self.config['evaluation']['num_episodes']):
            episode_times = self.evaluate_single_episode(
                warmup=self.config['evaluation'].get('warmup_steps', 5)
            )
            all_episode_times.append(episode_times)
            
            if (ep + 1) % 5 == 0:
                avg_decision = np.mean([np.mean(ep['decision_times']) for ep in all_episode_times])
                logger.info(f"Episode {ep+1}/{self.config['evaluation']['num_episodes']} - "
                           f"Avg Decision Time: {avg_decision:.2f}ms")
        
        # 汇总统计
        results = {
            'step_time': {
                'mean': np.mean([np.mean(ep['step_times']) for ep in all_episode_times]),
                'std': np.mean([np.std(ep['step_times']) for ep in all_episode_times]),
                'min': np.min([np.min(ep['step_times']) for ep in all_episode_times]),
                'max': np.max([np.max(ep['step_times']) for ep in all_episode_times])
            },
            'decision_time': {
                'mean': np.mean([np.mean(ep['decision_times']) for ep in all_episode_times]),
                'std': np.mean([np.std(ep['decision_times']) for ep in all_episode_times]),
                'min': np.min([np.min(ep['decision_times']) for ep in all_episode_times]),
                'max': np.max([np.max(ep['decision_times']) for ep in all_episode_times])
            },
            'marl_time': {
                'mean': np.mean([np.mean(ep['marl_times']) for ep in all_episode_times]),
                'std': np.mean([np.std(ep['marl_times']) for ep in all_episode_times]),
                'min': np.min([np.min(ep['marl_times']) for ep in all_episode_times]),
                'max': np.max([np.max(ep['marl_times']) for ep in all_episode_times])
            },
            'egt_time': {
                'mean': np.mean([np.mean(ep['egt_times']) for ep in all_episode_times]),
                'std': np.mean([np.std(ep['egt_times']) for ep in all_episode_times]),
                'min': np.min([np.min(ep['egt_times']) for ep in all_episode_times]),
                'max': np.max([np.max(ep['egt_times']) for ep in all_episode_times])
            }
        }
        
        return results
    
    def evaluate_scaling(self) -> Dict[str, Any]:
        """评估不同规模下的决策时间"""
        scales = [
            {'num_agents': 10, 'num_victims': 100},
            {'num_agents': 20, 'num_victims': 200},
            {'num_agents': 30, 'num_victims': 300},
        ]
        
        scaling_results = []
        
        for scale in scales:
            logger.info(f"Testing scale: {scale['num_agents']} agents, {scale['num_victims']} victims")
            
            # 临时修改环境配置
            original_env_config = self.config['environment'].copy()
            self.config['environment']['num_agents'] = scale['num_agents']
            self.config['environment']['num_victims'] = scale['num_victims']
            
            self.setup_environment()
            self.setup_algorithm()
            
            episode_times = self.evaluate_single_episode(warmup=5)
            
            scaling_results.append({
                'num_agents': scale['num_agents'],
                'num_victims': scale['num_victims'],
                'decision_time_mean': np.mean(episode_times['decision_times']),
                'decision_time_std': np.std(episode_times['decision_times'])
            })
            
            # 恢复配置
            self.config['environment'] = original_env_config
        
        return scaling_results
    
    def run_evaluation(self, output_dir: str = None) -> Dict[str, Any]:
        """运行完整评估"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir or f'decision_time_results/decision_time_{timestamp}'
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置环境和算法
        self.setup_environment()
        self.setup_algorithm()
        
        # 组件时间评估
        component_results = self.evaluate_component_times()
        
        # 规模扩展评估
        scaling_results = self.evaluate_scaling()
        
        # 保存结果
        results = {
            'timestamp': timestamp,
            'config': self.config,
            'component_times': component_results,
            'scaling_results': scaling_results
        }
        
        import json
        with open(output_path / 'decision_time_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 生成报告
        self.generate_report(results, output_path)
        
        logger.info(f"Decision time evaluation completed!")
        logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: Path):
        """生成决策时间评估报告"""
        report = []
        report.append("=" * 80)
        report.append("EGT-MARL Decision Time Evaluation Report")
        report.append("=" * 80)
        report.append("")
        report.append(f"Timestamp: {results['timestamp']}")
        report.append("")
        
        report.append("1. Component Time Analysis")
        report.append("-" * 40)
        comp = results['component_times']
        
        report.append("")
        report.append("Step Time (total per decision):")
        report.append(f"  Mean: {comp['step_time']['mean']:.2f}ms")
        report.append(f"  Std:  {comp['step_time']['std']:.2f}ms")
        report.append(f"  Range: {comp['step_time']['min']:.2f}ms - {comp['step_time']['max']:.2f}ms")
        
        report.append("")
        report.append("Decision Time (algorithm only):")
        report.append(f"  Mean: {comp['decision_time']['mean']:.2f}ms")
        report.append(f"  Std:  {comp['decision_time']['std']:.2f}ms")
        report.append(f"  Range: {comp['decision_time']['min']:.2f}ms - {comp['decision_time']['max']:.2f}ms")
        
        report.append("")
        report.append("MARL Layer Time:")
        report.append(f"  Mean: {comp['marl_time']['mean']:.2f}ms")
        report.append(f"  Std:  {comp['marl_time']['std']:.2f}ms")
        report.append(f"  Range: {comp['marl_time']['min']:.2f}ms - {comp['marl_time']['max']:.2f}ms")
        
        report.append("")
        report.append("EGT Layer Time:")
        report.append(f"  Mean: {comp['egt_time']['mean']:.2f}ms")
        report.append(f"  Std:  {comp['egt_time']['std']:.2f}ms")
        report.append(f"  Range: {comp['egt_time']['min']:.2f}ms - {comp['egt_time']['max']:.2f}ms")
        
        report.append("")
        report.append("2. Scaling Analysis")
        report.append("-" * 40)
        report.append("")
        report.append("| Agents | Victims | Decision Time (ms) |")
        report.append("|--------|---------|-------------------|")
        
        for scale in results['scaling_results']:
            report.append(f"| {scale['num_agents']:6} | {scale['num_victims']:7} | {scale['decision_time_mean']:>16.2f} ± {scale['decision_time_std']:.2f} |")
        
        report.append("")
        report.append("3. Performance Summary")
        report.append("-" * 40)
        
        # 计算时间占比
        marl_pct = comp['marl_time']['mean'] / comp['decision_time']['mean'] * 100
        egt_pct = comp['egt_time']['mean'] / comp['decision_time']['mean'] * 100
        other_pct = 100 - marl_pct - egt_pct
        
        report.append(f"MARL Layer: {marl_pct:.1f}% of decision time")
        report.append(f"EGT Layer:  {egt_pct:.1f}% of decision time")
        report.append(f"Other:      {other_pct:.1f}% of decision time")
        report.append("")
        
        # 实时性评估
        avg_decision = comp['decision_time']['mean']
        if avg_decision < 10:
            classification = "EXCELLENT (suitable for real-time control)"
        elif avg_decision < 50:
            classification = "GOOD (suitable for most applications)"
        elif avg_decision < 100:
            classification = "ACCEPTABLE (may cause delays in high-frequency scenarios)"
        else:
            classification = "SLOW (optimization recommended)"
        
        report.append(f"Real-time Performance: {classification}")
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(output_path / 'decision_time_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        with open(output_path / 'decision_time_report.md', 'w', encoding='utf-8') as f:
            f.write("# EGT-MARL Decision Time Evaluation Report\n\n")
            f.write(f"**Timestamp**: {results['timestamp']}\n\n")
            f.write("## Component Time Analysis\n\n")
            f.write("| Metric | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |\n")
            f.write("|--------|-----------|----------|----------|----------|\n")
            f.write(f"| Step Time | {comp['step_time']['mean']:.2f} | {comp['step_time']['std']:.2f} | {comp['step_time']['min']:.2f} | {comp['step_time']['max']:.2f} |\n")
            f.write(f"| Decision Time | {comp['decision_time']['mean']:.2f} | {comp['decision_time']['std']:.2f} | {comp['decision_time']['min']:.2f} | {comp['decision_time']['max']:.2f} |\n")
            f.write(f"| MARL Time | {comp['marl_time']['mean']:.2f} | {comp['marl_time']['std']:.2f} | {comp['marl_time']['min']:.2f} | {comp['marl_time']['max']:.2f} |\n")
            f.write(f"| EGT Time | {comp['egt_time']['mean']:.2f} | {comp['egt_time']['std']:.2f} | {comp['egt_time']['min']:.2f} | {comp['egt_time']['max']:.2f} |\n\n")
            f.write("## Scaling Analysis\n\n")
            f.write("| Agents | Victims | Decision Time (ms) |\n")
            f.write("|--------|---------|-------------------|\n")
            for scale in results['scaling_results']:
                f.write(f"| {scale['num_agents']} | {scale['num_victims']} | {scale['decision_time_mean']:.2f} ± {scale['decision_time_std']:.2f} |\n")
            f.write("\n## Performance Classification\n\n")
            f.write(f"**Real-time Performance**: {classification}\n\n")
            f.write(f"- MARL Layer: {marl_pct:.1f}% of decision time\n")
            f.write(f"- EGT Layer: {egt_pct:.1f}% of decision time\n")
            f.write(f"- Other overhead: {other_pct:.1f}%\n")
        
        print(report_text)


def main():
    parser = argparse.ArgumentParser(description='EGT-MARL Decision Time Evaluation')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=30,
                       help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    evaluator = DecisionTimeEvaluator(args.config)
    
    # Override config with command line args
    if args.num_episodes:
        evaluator.config['evaluation']['num_episodes'] = args.num_episodes
    if args.max_steps:
        evaluator.config['evaluation']['max_steps_per_episode'] = args.max_steps
    
    results = evaluator.run_evaluation(args.output_dir)


if __name__ == "__main__":
    main()
