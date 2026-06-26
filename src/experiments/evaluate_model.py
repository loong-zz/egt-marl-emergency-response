"""
模型评估脚本
评估 EGT-MARL 训练模型的性能
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.disaster_sim import DisasterSim
from environments.config.constants import SimulationConfig, NUM_STRATEGIES
from algorithms.egt_marl import EGTMARL
from utils.metrics import MetricsCollector
import logging

# 初始化logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model(config: Dict, checkpoint_path: str):
    """加载训练好的模型"""
    # 从配置中提取environment参数
    env_config = config.get('environment', {})
    algo_config = config.get('algorithm', {})
    training_config = config.get('training', {})
    
    # 创建SimulationConfig对象
    sim_config = SimulationConfig()
    sim_config.num_agents = env_config.get('num_agents', 20)
    sim_config.num_victims = env_config.get('num_victims', 200)
    sim_config.num_resources = env_config.get('num_resources', 10)
    sim_config.num_hospitals = env_config.get('num_hospitals', 3)
    sim_config.map_size = env_config.get('map_size', (1000, 1000))
    sim_config.disaster_type = env_config.get('disaster_type', 'earthquake')
    sim_config.severity = env_config.get('severity', 'medium')
    
    # 初始化环境
    env = DisasterSim(
        scenario=env_config.get('scenario', 'earthquake_standard'),
        map_size=sim_config.map_size,
        num_agents=sim_config.num_agents,
        num_victims=sim_config.num_victims,
        num_resources=sim_config.num_resources,
        num_hospitals=sim_config.num_hospitals,
        disaster_type=sim_config.disaster_type,
        severity=sim_config.severity,
        config=sim_config
    )
    
    # 创建算法配置（与训练代码一致）
    algo_config_dict = {
        'marl': {
            'state_dim': env.get_state_dimension(),
            'action_dim': 32,  # 8 tactical * 4 communication
            'num_agents': len(env.rescue_agents),
            'hidden_dim': algo_config.get('hidden_dim', 64),
            'mixing_hidden_dim': algo_config.get('mixing_hidden_dim', 64),
            'attention_heads': algo_config.get('attention_heads', 4),
            'learning_rate': training_config.get('learning_rate', 0.0001),
            'batch_size': training_config.get('batch_size', 32),
            'buffer_size': training_config.get('buffer_size', 10000)
        },
        'egt': {
            'num_strategies': NUM_STRATEGIES,
            'learning_rate': 0.01
        },
        'anti_spoofing': {
            'observation_dim': env.get_state_dimension(),
            'action_dim': 32
        },
        'dynamic_frontier': {
            'alpha': algo_config.get('pareto_weight_alpha', 0.3),
            'beta': algo_config.get('pareto_weight_beta', 0.4),
            'gamma': algo_config.get('pareto_weight_gamma', 0.3)
        }
    }
    
    # 初始化算法
    algorithm = EGTMARL(env=env, config=algo_config_dict, hidden_dim=algo_config.get('hidden_dim', 64))
    
    # 加载检查点（设置weights_only=False以支持numpy对象）
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 加载模型参数（使用strict=False处理可能的参数不匹配）
    if 'marl_layer_state' in checkpoint:
        try:
            algorithm.marl_layer.load_state_dict(checkpoint['marl_layer_state'])
        except RuntimeError:
            # 如果参数不匹配，尝试使用strict=False
            algorithm.marl_layer.load_state_dict(checkpoint['marl_layer_state'], strict=False)
    
    if 'egt_layer_state' in checkpoint:
        try:
            algorithm.egt_layer.load_state_dict(checkpoint['egt_layer_state'])
        except RuntimeError:
            algorithm.egt_layer.load_state_dict(checkpoint['egt_layer_state'], strict=False)
    
    if 'anti_spoofing_state' in checkpoint:
        anti_spoofing_state = checkpoint['anti_spoofing_state']
        try:
            # AntiSpoofing使用自定义的load方法
            if hasattr(algorithm.anti_spoofing, 'load_state_dict'):
                algorithm.anti_spoofing.load_state_dict(anti_spoofing_state)
            else:
                # 手动加载各个子组件
                if 'verifier' in anti_spoofing_state:
                    algorithm.anti_spoofing.verifier.load_state_dict(anti_spoofing_state['verifier'], strict=False)
                if 'spoofing_detector' in anti_spoofing_state:
                    algorithm.anti_spoofing.spoofing_detector.load_state_dict(anti_spoofing_state['spoofing_detector'], strict=False)
                if 'corrector' in anti_spoofing_state:
                    # 兼容旧版本的corrector属性（现已重命名为correction_network）
                    if hasattr(algorithm.anti_spoofing, 'corrector'):
                        algorithm.anti_spoofing.corrector.load_state_dict(anti_spoofing_state['corrector'], strict=False)
                    elif hasattr(algorithm.anti_spoofing, 'correction_network'):
                        algorithm.anti_spoofing.correction_network.load_state_dict(anti_spoofing_state['corrector'], strict=False)
                if 'correction_network' in anti_spoofing_state:
                    if hasattr(algorithm.anti_spoofing, 'correction_network'):
                        algorithm.anti_spoofing.correction_network.load_state_dict(anti_spoofing_state['correction_network'], strict=False)
                if 'reputation_system' in anti_spoofing_state:
                    algorithm.anti_spoofing.reputation_system = anti_spoofing_state['reputation_system']
        except Exception as e:
            logger.warning(f"Failed to load anti_spoofing state: {e}")
    
    return algorithm, env


def evaluate_model(algorithm, env, num_episodes: int = 50):
    """评估模型性能"""
    metrics = {
        'rescue_rate': [],
        'avg_response_time': [],
        'total_reward': [],
        'survivors': []
    }
    
    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        
        done = False
        step = 0
        max_steps = 1200
        
        while not done and step < max_steps:
            actions = algorithm.select_action(state, training=False, epsilon=0.0)
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            step_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
            episode_reward += step_reward
            
            state = next_state
            step += 1
        
        # 获取统计数据
        statistics = info.get('statistics', {})
        episode_rescued = statistics.get('total_rescued', 0)
        total_victims = len(env.casualties)
        
        rescue_rate = (episode_rescued / total_victims * 100) if total_victims > 0 else 0.0
        response_times = statistics.get('response_times', [])
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        metrics['rescue_rate'].append(rescue_rate)
        metrics['avg_response_time'].append(avg_response_time)
        metrics['total_reward'].append(episode_reward)
        metrics['survivors'].append(episode_rescued)
        
        if (ep + 1) % 10 == 0:
            logger.info(f"Episode {ep+1}/{num_episodes} - Rescue Rate: {rescue_rate:.1f}% - Reward: {episode_reward:.2f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估 EGT-MARL 模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num_episodes', type=int, default=50, help='评估episode数量')
    
    args = parser.parse_args()
    
    # 加载配置（使用FullLoader以支持Python对象）
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 加载模型
    logger.info(f"加载模型: {args.checkpoint}")
    algorithm, env = load_model(config, args.checkpoint)
    
    # 评估模型
    logger.info(f"开始评估，共 {args.num_episodes} 个 episodes...")
    metrics = evaluate_model(algorithm, env, args.num_episodes)
    
    # 输出评估结果
    logger.info("\n" + "="*60)
    logger.info("评估结果汇总")
    logger.info("="*60)
    logger.info(f"评估Episodes: {args.num_episodes}")
    logger.info(f"平均救援率: {np.mean(metrics['rescue_rate']):.2f}% ± {np.std(metrics['rescue_rate']):.2f}")
    logger.info(f"最高救援率: {np.max(metrics['rescue_rate']):.2f}%")
    logger.info(f"最低救援率: {np.min(metrics['rescue_rate']):.2f}%")
    logger.info(f"平均响应时间: {np.mean(metrics['avg_response_time']):.1f}s")
    logger.info(f"平均奖励: {np.mean(metrics['total_reward']):.2f}")
    logger.info(f"平均获救人数: {np.mean(metrics['survivors']):.1f}")
    logger.info("="*60)
    
    # 关闭环境
    if hasattr(env, 'close'):
        env.close()


if __name__ == '__main__':
    main()