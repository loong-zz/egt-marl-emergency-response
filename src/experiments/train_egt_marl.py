"""
EGT-MARL 训练脚本

训练 EGT-MARL 算法并保存模型。
支持分布式训练、超参数调优和模型检查点。
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.multiprocessing as mp
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
from algorithms.qmix_improved import ImprovedQMIX
from algorithms.dynamic_frontier import DynamicParetoFrontier
from utils.metrics import MetricsCollector
from environments.visualization import DisasterVisualizer
import logging

# 初始化logger
logger = logging.getLogger(__name__)


class EGTMARLTrainer:
    """EGT-MARL 训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.setup_device()
        
        # 初始化组件
        self.env = None
        self.algorithm = None
        self.metrics_collector = None
        self.visualizer = None
        
        logger.info(f"EGT-MARL Trainer initialized with config: {config_path}")
    
    def setup_directories(self):
        """设置目录结构"""
        base_dir = Path(self.config.get('output_dir', 'experiment_results'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = base_dir / f'egt_marl_{timestamp}'
        
        # 创建目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'models').mkdir(exist_ok=True)
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        (self.experiment_dir / 'visualizations').mkdir(exist_ok=True)
        
        # 保存配置
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 设置默认值
        defaults = {
            'training': {
                'num_episodes': 200,
                'max_steps_per_episode': 600,
                'batch_size': 32,
                'buffer_size': 5000,
                'gamma': 0.99,
                'tau': 0.01,
                'learning_rate': 0.001,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.99,
                'target_update_interval': 100,
                'checkpoint_interval': 100,
                'eval_interval': 50,
                'num_eval_episodes': 10,
                'save_best_model': True,
                'update_frequency': 10
            },
            'environment': {
                'map_size': (1000, 1000),
                'num_agents': 20,
                'num_victims': 200,
                'num_resources': 10,
                'num_hospitals': 3,
                'disaster_type': 'earthquake',
                'severity': 'medium'
            },
            'algorithm': {
                'hidden_dim': 128,
                'mixing_hidden_dim': 64,
                'attention_heads': 4,
                'egt_lambda': 0.5,
                'pareto_weight_alpha': 0.3,
                'pareto_weight_beta': 0.4,
                'pareto_weight_gamma': 0.3,
                'anti_spoofing_enabled': True
            }
        }
        
        # 合并配置
        for section in defaults:
            if section not in config:
                config[section] = defaults[section]
            else:
                for key, value in defaults[section].items():
                    if key not in config[section]:
                        config[section][key] = value
        
        return config
    
    def setup_directories(self):
        """设置目录结构"""
        base_dir = Path(self.config.get('output_dir', 'experiment_results'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = base_dir / f'egt_marl_{timestamp}'
        
        # 创建目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'models').mkdir(exist_ok=True)
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        (self.experiment_dir / 'visualizations').mkdir(exist_ok=True)
        
        # 配置日志
        log_file = self.experiment_dir / 'logs' / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # 保存配置
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
        logger.info(f"Log file: {log_file}")
    
    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
    
    def setup_environment(self):
        """设置环境"""
        # 使用配置文件中的参数
        env_config = self.config['environment']
        self.env = DisasterSim(
            map_size=env_config['map_size'],
            num_agents=env_config['num_agents'],
            num_victims=env_config['num_victims'],
            num_resources=env_config['num_resources'],
            num_hospitals=env_config['num_hospitals'],
            disaster_type=env_config['disaster_type'],
            severity=env_config['severity']
        )
        
        # 为了兼容训练脚本，添加必要的属性
        self.env.num_agents = len(self.env.rescue_agents)
        self.env.num_victims = len(self.env.casualties)
        self.env.num_resources = len(self.env.resource_depots)
        
        logger.info(f"Environment initialized: {self.env}")
    
    def setup_algorithm(self):
        """设置算法"""
        # 初始化算法，只传递 env 参数
        self.algorithm = EGTMARL(env=self.env)
        
        # 为了兼容训练脚本，设置必要的属性
        self.algorithm.device = self.device
        
        logger.info(f"Algorithm initialized: {self.algorithm}")
    
    def setup_metrics(self):
        """设置指标收集器"""
        self.metrics_collector = MetricsCollector()
        self.visualizer = DisasterVisualizer(self.config['environment'])
        
        logger.info("Metrics collector and visualizer initialized")
    
    def train_episode(self, episode_idx: int, epsilon: float) -> Dict[str, float]:
        """
        训练一个episode
        
        Args:
            episode_idx: episode索引
            epsilon: 探索率
            
        Returns:
            指标字典
        """
        # 重置环境
        state, info = self.env.reset()
        logger.info(f"Episode {episode_idx} reset - Num casualties: {info.get('num_casualties', 0)}, Num agents: {info.get('num_rescue_agents', 0)}")
        
        episode_metrics = {
            'total_reward': 0.0,
            'steps': 0,
            'rescued': 0,
            'deaths': 0,
            'resources_used': 0,
            'response_times': []
        }
        
        done = False
        step = 0
        max_steps = self.config['training']['max_steps_per_episode']
        
        while not done and step < max_steps:
            # 获取动作 - 传递训练参数以启用探索
            actions = self.algorithm.select_action(state, training=True)
            
            # 执行动作
            next_state, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # 存储经验并更新算法
            self.algorithm.store_experience(state, actions, rewards, next_state, done)
            if step % self.config['training']['update_frequency'] == 0:
                self.algorithm.update()
            
            # 记录奖励和状态信息
            if step % 50 == 0:
                logger.debug(f"Step {step}: Reward={rewards:.4f}, Rescued={info.get('rescued', 0)}, Deaths={info.get('deaths', 0)}")
            
            # 更新状态
            state = next_state
            
            # 收集指标
            episode_metrics['total_reward'] += rewards
            episode_metrics['steps'] += 1
            # 不累加rescued和deaths，因为info中已经是累计值
            # episode_metrics['rescued'] += info.get('rescued', 0)
            # episode_metrics['deaths'] += info.get('deaths', 0)
            episode_metrics['resources_used'] += info.get('resources_used', 0)
            
            if 'response_time' in info:
                episode_metrics['response_times'].append(info['response_time'])
            
            step += 1
        
        # 计算平均响应时间
        if episode_metrics['response_times']:
            episode_metrics['avg_response_time'] = np.mean(episode_metrics['response_times'])
        else:
            episode_metrics['avg_response_time'] = 0.0
        
        # 在episode结束时获取最终的rescued和deaths值
        episode_metrics['rescued'] = info.get('rescued', 0)
        episode_metrics['deaths'] = info.get('deaths', 0)
        
        # 计算救援成功率
        total_victims = self.env.num_victims
        if total_victims > 0:
            episode_metrics['rescue_rate'] = (episode_metrics['rescued'] / total_victims) * 100
        else:
            episode_metrics['rescue_rate'] = 0.0
        
        # 计算资源利用率
        total_resources = self.env.num_resources * 100  # 假设每个资源点容量为100
        if total_resources > 0:
            episode_metrics['resource_utilization'] = (episode_metrics['resources_used'] / total_resources) * 100
        else:
            episode_metrics['resource_utilization'] = 0.0
        
        return episode_metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估当前策略
        
        Args:
            num_episodes: 评估episode数量
            
        Returns:
            评估指标字典
        """
        eval_metrics = {
            'rescue_rate': [],
            'avg_response_time': [],
            'resource_utilization': [],
            'total_reward': []
        }
        
        for ep in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0.0
            episode_rescued = 0
            episode_resources_used = 0
            response_times = []
            
            done = False
            step = 0
            max_steps = self.config['training']['max_steps_per_episode']
            
            while not done and step < max_steps:
                # 使用确定性策略
                actions = self.algorithm.select_action(state)
                next_state, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                episode_reward += rewards
                # 不累加rescued，因为info中已经是累计值
                # episode_rescued += info.get('rescued', 0)
                episode_resources_used += info.get('resources_used', 0)
                
                if 'response_time' in info:
                    response_times.append(info['response_time'])
                
                state = next_state
                step += 1
            
            # 在episode结束时获取最终的rescued值
            episode_rescued = info.get('rescued', 0)
            
            # 计算指标
            total_victims = self.env.num_victims
            rescue_rate = (episode_rescued / total_victims * 100) if total_victims > 0 else 0.0
            
            avg_response_time = np.mean(response_times) if response_times else 0.0
            
            total_resources = self.env.num_resources * 100
            resource_utilization = (episode_resources_used / total_resources * 100) if total_resources > 0 else 0.0
            
            eval_metrics['rescue_rate'].append(rescue_rate)
            eval_metrics['avg_response_time'].append(avg_response_time)
            eval_metrics['resource_utilization'].append(resource_utilization)
            eval_metrics['total_reward'].append(episode_reward)
        
        # 计算平均指标
        avg_metrics = {}
        for key, values in eval_metrics.items():
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def save_checkpoint(self, episode_idx: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_path = self.experiment_dir / 'checkpoints' / f'checkpoint_ep{episode_idx}.pt'
        
        checkpoint = {
            'episode': episode_idx,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self, episode_idx: int, metrics: Dict[str, float]):
        """保存最佳模型"""
        best_model_path = self.experiment_dir / 'models' / 'best_model.pt'
        
        model_state = {
            'episode': episode_idx,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(model_state, best_model_path)
        logger.info(f"Best model saved: {best_model_path}")
    
    def train(self):
        """主训练循环"""
        logger.info("Starting training...")
        
        # 设置目录（在参数覆盖后）
        self.setup_directories()
        
        # 设置组件
        self.setup_environment()
        self.setup_algorithm()
        self.setup_metrics()
        
        # 训练参数
        training_config = self.config['training']
        num_episodes = training_config['num_episodes']
        epsilon = training_config['epsilon_start']
        epsilon_decay = training_config['epsilon_decay']
        epsilon_end = training_config['epsilon_end']
        
        checkpoint_interval = training_config['checkpoint_interval']
        eval_interval = training_config['eval_interval']
        num_eval_episodes = training_config['num_eval_episodes']
        save_best_model = training_config['save_best_model']
        
        # 训练统计
        best_rescue_rate = 0.0
        training_history = {
            'episodes': [],
            'rescue_rate': [],
            'avg_response_time': [],
            'resource_utilization': [],
            'total_reward': [],
            'loss': []
        }
        
        # 训练循环
        for episode in range(1, num_episodes + 1):
            # 训练一个episode
            episode_metrics = self.train_episode(episode, epsilon)
            
            # 更新探索率
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # 记录训练历史
            training_history['episodes'].append(episode)
            training_history['rescue_rate'].append(episode_metrics.get('rescue_rate', 0.0))
            training_history['avg_response_time'].append(episode_metrics.get('avg_response_time', 0.0))
            training_history['resource_utilization'].append(episode_metrics.get('resource_utilization', 0.0))
            training_history['total_reward'].append(episode_metrics.get('total_reward', 0.0))
            training_history['loss'].append(episode_metrics.get('loss', 0.0))
            
            # 定期评估
            if episode % eval_interval == 0:
                eval_metrics = self.evaluate(num_eval_episodes)
                
                logger.info(f"Episode {episode}/{num_episodes} - "
                           f"Epsilon: {epsilon:.3f} - "
                           f"Train Rescue Rate: {episode_metrics.get('rescue_rate', 0.0):.1f}% - "
                           f"Eval Rescue Rate: {eval_metrics.get('rescue_rate', 0.0):.1f}% ± {eval_metrics.get('rescue_rate_std', 0.0):.1f}")
                
                # 保存最佳模型
                if save_best_model and eval_metrics.get('rescue_rate', 0.0) > best_rescue_rate:
                    best_rescue_rate = eval_metrics.get('rescue_rate', 0.0)
                    self.save_best_model(episode, eval_metrics)
            
            # 定期保存检查点
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode, episode_metrics)
            
            # 定期打印进度
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes} - "
                           f"Rescue Rate: {episode_metrics.get('rescue_rate', 0.0):.1f}% - "
                           f"Reward: {episode_metrics.get('total_reward', 0.0):.1f} - "
                           f"Steps: {episode_metrics.get('steps', 0)} - "
                           f"Remaining Casualties: {len(self.env.casualties)}")
        
        # 训练完成
        logger.info("Training completed!")
        
        # 最终评估
        final_metrics = self.evaluate(num_eval_episodes * 2)
        logger.info(f"Final Evaluation - "
                   f"Rescue Rate: {final_metrics.get('rescue_rate', 0.0):.1f}% ± {final_metrics.get('rescue_rate_std', 0.0):.1f} - "
                   f"Response Time: {final_metrics.get('avg_response_time', 0.0):.1f}s - "
                   f"Resource Utilization: {final_metrics.get('resource_utilization', 0.0):.1f}%")
        
        # 保存最终模型
        final_model_path = self.experiment_dir / 'models' / 'final_model.pt'
        torch.save({
            'final_metrics': final_metrics,
            'training_history': training_history,
            'config': self.config
        }, final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
        
        # 生成训练报告
        self.generate_training_report(training_history, final_metrics)
        
        return training_history, final_metrics
    
    def generate_training_report(self, 
                                training_history: Dict[str, List[float]],
                                final_metrics: Dict[str, float]):
        """生成训练报告"""
        report_path = self.experiment_dir / 'training_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EGT-MARL Training Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. Experiment Information\n")
            f.write("-" * 40 + "\n")
            f.write(f"Experiment Directory: {self.experiment_dir}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("2. Configuration Summary\n")
            f.write("-" * 40 + "\n")
            f.write(f"Environment: {self.config['environment']['disaster_type']} "
                   f"({self.config['environment']['severity']})\n")
            f.write(f"Agents: {self.config['environment']['num_agents']}\n")
            f.write(f"Victims: {self.config['environment']['num_victims']}\n")
            f.write(f"Episodes: {self.config['training']['num_episodes']}\n")
            f.write(f"Learning Rate: {self.config['training']['learning_rate']}\n")
            f.write(f"Gamma: {self.config['training']['gamma']}\n\n")
            
            f.write("3. Final Performance Metrics\n")
            f.write("-" * 40 + "\n")
            f.write(f"Rescue Rate: {final_metrics.get('rescue_rate', 0.0):.1f}% "
                   f"(±{final_metrics.get('rescue_rate_std', 0.0):.1f})\n")
            f.write(f"Average Response Time: {final_metrics.get('avg_response_time', 0.0):.1f}s\n")
            f.write(f"Resource Utilization: {final_metrics.get('resource_utilization', 0.0):.1f}%\n")
            f.write(f"Total Reward: {final_metrics.get('total_reward', 0.0):.1f}\n\n")
            
            f.write("4. Training Statistics\n")
            f.write("-" * 40 + "\n")
            if training_history['rescue_rate']:
                f.write(f"Best Rescue Rate: {max(training_history['rescue_rate']):.1f}%\n")
                f.write(f"Final Rescue Rate: {training_history['rescue_rate'][-1]:.1f}%\n")
                f.write(f"Average Rescue Rate: {np.mean(training_history['rescue_rate']):.1f}%\n")
            
            if training_history['loss']:
                valid_losses = [l for l in training_history['loss'] if l is not None and l > 0]
                if valid_losses:
                    f.write(f"Final Loss: {valid_losses[-1]:.4f}\n")
                    f.write(f"Average Loss: {np.mean(valid_losses):.4f}\n")
            
            f.write("\n5. Files Generated\n")
            f.write("-" * 40 + "\n")
            f.write(f"Config: {self.experiment_dir}/config.yaml\n")
            f.write(f"Best Model: {self.experiment_dir}/models/best_model.pt\n")
            f.write(f"Final Model: {self.experiment_dir}/models/final_model.pt\n")
            f.write(f"Checkpoints: {self.experiment_dir}/checkpoints/\n")
            f.write(f"Logs: {self.experiment_dir}/logs/\n")
            f.write(f"Visualizations: {self.experiment_dir}/visualizations/\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Training Completed Successfully!\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Training report saved: {report_path}")
        
        # 生成可视化
        self.generate_training_visualizations(training_history)
    
    def generate_training_visualizations(self, training_history: Dict[str, List[float]]):
        """生成训练可视化"""
        try:
            # 创建可视化器
            visualizer = DisasterVisualizer(self.config['environment'])
            
            # 准备指标数据
            metrics_data = {
                'rescue_rate': training_history['rescue_rate'],
                'avg_response_time': training_history['avg_response_time'],
                'resource_utilization': training_history['resource_utilization'],
                'total_reward': training_history['total_reward']
            }
            
            # 绘制性能仪表盘
            dashboard_path = self.experiment_dir / 'visualizations' / 'training_dashboard.png'
            visualizer.plot_performance_dashboard(metrics_data, str(dashboard_path))
            
            logger.info(f"Training visualizations saved to {self.experiment_dir}/visualizations/")
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train EGT-MARL algorithm')
    parser.add_argument('--config', type=str, default='configs/training.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                       help='Output directory for results')
    # 支持多种参数格式
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of training episodes (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (alias for --num_episodes)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (alias for --learning_rate)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (alias for --batch_size)')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = EGTMARLTrainer(args.config)
    
    # 覆盖配置参数
    if args.output_dir:
        trainer.config['output_dir'] = args.output_dir
    
    # 处理训练轮数参数
    if args.num_episodes:
        trainer.config['training']['num_episodes'] = args.num_episodes
    elif args.epochs:
        trainer.config['training']['num_episodes'] = args.epochs
    
    # 处理学习率参数
    if args.learning_rate:
        trainer.config['training']['learning_rate'] = args.learning_rate
    
    # 处理批大小参数
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    
    # 开始训练
    try:
        training_history, final_metrics = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {trainer.experiment_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()