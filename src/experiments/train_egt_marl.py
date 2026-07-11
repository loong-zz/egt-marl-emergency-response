"""
EGT-MARL 训练脚本

训练 EGT-MARL 算法并保存模型。
支持分布式训练、超参数调优和模型检查点。
"""

import os
import sys
import argparse
import math
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

from environments.disaster_sim import DisasterSim, ResourceType
from environments.config.constants import (
    RESOURCE_ABBR, EGT_CONFIG, REPUTATION_CONFIG, PARETO_CONFIG,
    COMMUNICATION_CONFIG, INTERFERENCE_CONFIG, NUM_REGIONS,
    NUM_STRATEGIES,
)
from algorithms.egt_marl import EGTMARL
from algorithms.qmix_improved import ImprovedQMIX
from algorithms.dynamic_frontier import DynamicParetoFrontier
from utils.metrics import MetricsCollector
from environments.visualization import DisasterVisualizer
from utils.visualization import plot_egt_strategy_evolution, plot_egt_strategy_recommendation
from environments.managers.manager_integration import ManagerIntegration
from torch.utils.tensorboard import SummaryWriter
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
        
        # 初始化 TensorBoard
        self.writer = None
        if self.config.get('logging', {}).get('tensorboard', {}).get('enabled', False):
            log_dir = self.config['logging']['tensorboard'].get('log_dir', 'runs/{experiment_name}')
            experiment_name = self.config.get('experiment', {}).get('name', 'egt_marl')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = log_dir.format(experiment_name=experiment_name, timestamp=timestamp)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging enabled, log directory: {log_dir}")
        
        # 初始化组件
        self.env = None
        self.algorithm = None
        self.metrics_collector = None
        self.visualizer = None
        self.manager_integration = None  # Manager 集成
        
        logger.info(f"EGT-MARL Trainer initialized with config: {config_path}")
    

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 设置默认值
        defaults = {
            'training': {
                'num_episodes': 200,
                'max_steps_per_episode': 1200,
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
            'schedule': {
                'exploration_schedule': {
                    'type': 'exponential',
                    'start': 0.8,
                    'end': 0.01,
                    'decay': 0.997
                },
                'lr_schedule': {
                    'type': 'cosine',
                    'warmup_episodes': 100,
                    'min_lr': 1e-6,
                    'max_lr': 0.0001
                },
                'phases': []
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
                'hidden_dim': 128,
                'mixing_hidden_dim': 64,
                'attention_heads': 4,
                'egt_lambda': 0.5,
                'pareto_weight_alpha': 0.3,
                'pareto_weight_beta': 0.4,
                'pareto_weight_gamma': 0.3,
                'anti_spoofing_enabled': True
            },
            # Manager配置默认值（从constants.py导入）
            'egt': EGT_CONFIG,
            'reputation': REPUTATION_CONFIG,
            'pareto': PARETO_CONFIG,
            'communication': COMMUNICATION_CONFIG,
            'interference': INTERFERENCE_CONFIG
        }
        
        # 合并配置
        for section in defaults:
            if section not in config:
                config[section] = defaults[section]
            else:
                for key, value in defaults[section].items():
                    if key not in config[section]:
                        config[section][key] = value
        
        # 从schedule.exploration_schedule读取并覆盖training中的探索率配置
        if 'schedule' in config and 'exploration_schedule' in config['schedule']:
            exp_sched = config['schedule']['exploration_schedule']
            if 'start' in exp_sched:
                config['training']['epsilon_start'] = exp_sched['start']
            if 'end' in exp_sched:
                config['training']['epsilon_end'] = exp_sched['end']
            if 'decay' in exp_sched:
                config['training']['epsilon_decay'] = exp_sched['decay']
        
        return config
    
    def setup_directories(self, experiment_dir: str = None):
        """设置目录结构

        Args:
            experiment_dir: 可选，指定现有的实验目录（用于恢复训练）。
                          如果为 None，则根据时间戳创建新目录。
        """
        # 使用项目根目录作为基准，确保结果目录位置一致
        project_root = Path(__file__).parent.parent

        if experiment_dir:
            # 恢复训练：复用已有实验目录
            self.experiment_dir = Path(experiment_dir)
            if not self.experiment_dir.exists():
                logger.warning(f"Specified experiment directory does not exist: {self.experiment_dir}")
                self.experiment_dir.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"Reusing existing experiment directory: {self.experiment_dir}")
        else:
            # 新训练：创建带时间戳的新目录
            base_dir = project_root / self.config.get('output_dir', 'experiment_results')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_dir = base_dir / f'egt_marl_{timestamp}'
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.experiment_dir / 'models').mkdir(exist_ok=True)
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        (self.experiment_dir / 'visualizations').mkdir(exist_ok=True)

        # 配置日志（恢复训练时追加写入，避免覆盖旧日志）
        log_file = self.experiment_dir / 'logs' / 'training.log'
        log_level_str = self.config.get('logging', {}).get('level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)

        # 避免重复添加 handler（当多次调用时）
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler)
                  and getattr(h, 'baseFilename', None) == str(log_file)
                  for h in root_logger.handlers):
            file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s'))
            root_logger.addHandler(file_handler)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s'))
            root_logger.addHandler(stream_handler)

        root_logger.setLevel(log_level)

        # 保存配置（新训练时覆盖；恢复训练时重新写入，以便保存最新配置）
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
    
    def setup_manager_integration(self):
        """设置Manager集成"""
        manager_config = {
            'egt': self.config['egt'],
            'reputation': self.config['reputation'],
            'pareto': self.config['pareto'],
            'communication': self.config['communication'],
            'interference': self.config['interference']
        }
        
        self.manager_integration = ManagerIntegration(manager_config)
        logger.info("Manager Integration initialized")
    
    def setup_environment(self):
        """设置环境"""
        # 使用配置文件中的参数
        env_config = self.config['environment']
        self.env = DisasterSim(
            map_size=env_config['map_size'],
            num_agents=env_config['num_agents'],
            num_victims=env_config['num_victims'],
            num_resources=env_config['num_resources'],
            num_areas=env_config['num_areas'],
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
        # 构建算法配置
        algo_config = {
            'marl': {
                'state_dim': self.env.get_state_dimension(),
                'action_dim': 32,  # 8 tactical * 4 communication
                'num_agents': self.env.num_agents,
                'hidden_dim': self.config['algorithm'].get('hidden_dim', 64),
                'mixing_hidden_dim': self.config['algorithm'].get('mixing_hidden_dim', 64),
                'attention_heads': self.config['algorithm'].get('attention_heads', 4),
                'learning_rate': self.config['training']['learning_rate'],
                'batch_size': self.config['training'].get('batch_size', 32),
                'buffer_size': self.config['training'].get('buffer_size', 10000)
            },

            'egt': {
                'num_strategies': self.config.get('egt', {}).get('num_strategies', NUM_STRATEGIES),
                'learning_rate': self.config.get('egt', {}).get('learning_rate', 0.01),
                'mutation_rate': self.config.get('egt', {}).get('mutation_rate', 0.05),
                'selection_pressure': self.config.get('egt', {}).get('selection_pressure', 2.0),
                'population_size': self.config.get('egt', {}).get('population_size', 100),
                'tradeoff_adaptation_rate': self.config.get('egt', {}).get('tradeoff_adaptation_rate', 0.1),
                'min_fairness_weight': self.config.get('egt', {}).get('min_fairness_weight', 0.2),
                'max_fairness_weight': self.config.get('egt', {}).get('max_fairness_weight', 0.8),
            },
            'anti_spoofing': {
                'observation_dim': self.env.get_state_dimension(),
                'action_dim': 32
            },
            'dynamic_frontier': {
                'alpha': self.config['algorithm'].get('pareto_weight_alpha', 0.3),
                'beta': self.config['algorithm'].get('pareto_weight_beta', 0.4),
                'gamma': self.config['algorithm'].get('pareto_weight_gamma', 0.3)
            }
        }
        
        # 初始化算法，传递环境和配置
        self.algorithm = EGTMARL(
            env=self.env,
            config=algo_config,
            hidden_dim=self.config['algorithm'].get('hidden_dim', 64)
        )
        
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
        
        # Manager集成：Episode开始回调
        if self.manager_integration is not None:
            self.manager_integration.on_episode_start(
                num_agents=self.env.num_agents,
                num_regions=NUM_REGIONS,
                map_size=self.env.map_size
            )
            
            # 注册伤员和agent区域信息
            casualties = {cid: {'position': casualty.position} for cid, casualty in self.env.casualties.items()}
            self.manager_integration.register_casualties(casualties)
            
            agents = {aid: {'position': agent.position} for aid, agent in self.env.rescue_agents.items()}
            self.manager_integration.register_agents(agents)
        
        episode_metrics = {
            'total_reward': 0.0,
            'steps': 0,
            'rescued': 0,
            'deaths': 0,
            'resources_used': 0,
            'response_times': [],
            'final_lambda': 0.0,
            'total_communications': 0,
            'shared_casualties': 0
        }
        
        # Track rescued casualties for region manager
        previously_rescued = set()
        
        done = False
        step = 0
        max_steps = self.config['training']['max_steps_per_episode']
        
        while not done and step < max_steps:
            # Manager集成：Step开始回调
            if self.manager_integration is not None:
                # P1 fix: was `step * 0.1 / 3600.0` (每步0.1秒), which made
                # max hours = 1200*0.1/3600 ≈ 0.033h,永远 < phase1_threshold=24h,
                # 导致 pareto_manager 永远停在 'early' 阶段. 现每步=0.1小时(6分钟),
                # 1200步=120小时,能覆盖 early(0-24h)/mid(24-72h)/recovery(72h+) 三阶段.
                hours_elapsed = step * 0.1
                aftershock = info.get('aftershock_happening', False)
                self.manager_integration.on_step_start(hours_elapsed, aftershock)
            
            # 获取动作 - 传递训练参数和epsilon以启用探索
            actions = self.algorithm.select_action(state, training=True, epsilon=epsilon)
            
            # 执行动作
            next_state, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated
            
            # Manager集成：Agent通信（每5步）
            if self.manager_integration is not None and step % 5 == 0:
                self._process_agent_communication()
            
            # Manager集成：奖励塑形
            if self.manager_integration is not None:
                # rewards现在是字典格式（个体奖励）
                if not isinstance(rewards, dict):
                    # 如果是旧格式（全局奖励），转换为字典格式
                    num_agents = len(self.env.rescue_agents)
                    per_agent_reward = rewards / num_agents if num_agents > 0 else 0.0
                    rewards = {aid: per_agent_reward for aid in self.env.rescue_agents}
                
                shaped_rewards = {}
                for agent_id in self.env.rescue_agents:
                    base_reward = rewards.get(agent_id, 0.0)
                    shaped_reward = self.manager_integration.get_shaped_reward(
                        base_reward=base_reward,
                        agent_id=agent_id,
                        action_type='step',
                        context=info
                    )
                    shaped_rewards[agent_id] = shaped_reward
                rewards = shaped_rewards
            else:
                # 如果没有manager_integration，确保rewards是字典格式
                if not isinstance(rewards, dict):
                    num_agents = len(self.env.rescue_agents)
                    per_agent_reward = rewards / num_agents if num_agents > 0 else 0.0
                    rewards = {aid: per_agent_reward for aid in self.env.rescue_agents}
            
            # M1 audit fix: forward resource claim / allocation callbacks
            # to the ManagerIntegration so the reputation system and EGT layer
            # can react to anti-spoofing events.
            if self.manager_integration is not None and isinstance(info, dict):
                # Pull per-agent claim/usage from info when the env exposes it
                for agent_id, agent in self.env.rescue_agents.items():
                    claimed = info.get('claimed_demand', {}).get(agent_id)
                    actual = info.get('actual_demand', {}).get(agent_id)
                    if claimed is not None and actual is not None:
                        try:
                            self.manager_integration.on_resource_claim(
                                agent_id=agent_id,
                                claimed_demand=float(claimed),
                                actual_demand=float(actual),
                                context={'severity': info.get('severity'),
                                         'time': step},
                            )
                        except Exception:
                            pass
                    allocated = info.get('allocated', {}).get(agent_id)
                    if allocated is not None:
                        try:
                            self.manager_integration.on_resource_allocation(
                                agent_id=agent_id,
                                allocated_amount=float(allocated),
                            )
                        except Exception:
                            pass

            # 存储经验并更新算法
            # P0 fix: 传入 info，让 EGT 层拿到真实的 fairness_score / efficiency_score
            self.algorithm.store_experience(state, actions, rewards, next_state, done, info=info)
            if step % self.config['training']['update_frequency'] == 0:
                update_out = self.algorithm.update()
                if not isinstance(update_out, dict):
                    update_out = {}
                # Track the most informative training signal for the
                # episode-level loss field.  We prefer mixing_loss (the
                # end-to-end QMIX signal), then egt_loss, then marl_loss.
                ep_loss = update_out.get(
                    'mixing_loss',
                    update_out.get(
                        'marl_loss',
                        update_out.get('egt_loss', 0.0),
                    ),
                )
                if isinstance(ep_loss, (int, float)) and ep_loss != 0.0:
                    running = episode_metrics.get('__loss_buf', [])
                    if not isinstance(running, list):
                        running = []
                    running.append(float(ep_loss))
                    episode_metrics['__loss_buf'] = running
                # Stash the last raw update dict for episode-end logging.
                episode_metrics['__last_update'] = update_out
            
            # Manager集成：记录新救援的伤员
            if self.manager_integration is not None:
                current_rescued = {cid for cid, casualty in self.env.casualties.items() if casualty.treated}
                newly_rescued = current_rescued - previously_rescued
                for cid in newly_rescued:
                    # 找到救援该伤员的agent（通过检查rescued_count变化）
                    rescuer_id = None
                    for aid, agent in self.env.rescue_agents.items():
                        if hasattr(agent, 'current_mission') and agent.current_mission:
                            mission_target = getattr(agent.current_mission, 'target_id', None)
                            if mission_target == cid:
                                rescuer_id = aid
                                break
                    # 如果找不到具体救援者，使用-1表示未知
                    self.manager_integration.record_rescue(rescuer_id if rescuer_id else -1, cid, success=True)
                previously_rescued.update(newly_rescued)
            
            # Manager集成：更新区域公平性指标
            if self.manager_integration is not None:
                self.manager_integration.update_region_fairness_metrics()
                self.manager_integration.record_fairness_step()
            
            # Manager集成：Step结束回调
            if self.manager_integration is not None:
                agent_states = self._get_agent_states()
                agent_rewards = {aid: rewards.get(aid, 0.0) for aid in self.env.rescue_agents}
                self.manager_integration.on_step_end(agent_states, agent_rewards, {})
            
            # 记录奖励和状态信息
            if step % 50 == 0:
                stats = info.get('statistics', {})
                rescued = stats.get('total_rescued', 0)
                deaths = stats.get('total_deaths', 0)
                total_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
                logger.debug(f"Step {step}: Reward={total_reward:.4f}, Rescued={rescued}, Deaths={deaths}")
                self._log_entity_info(step)
            
            # 更新状态
            state = next_state
            
            # 收集指标
            total_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
            episode_metrics['total_reward'] += total_reward
            episode_metrics['steps'] += 1
            
            step += 1
        
        # 计算平均响应时间（从statistics中获取）
        response_times = info.get('statistics', {}).get('response_times', [])
        if response_times:
            episode_metrics['response_times'] = response_times
            episode_metrics['avg_response_time'] = np.mean(response_times)
        else:
            episode_metrics['avg_response_time'] = 0.0
        
        # 在episode结束时获取最终的rescued和deaths值
        episode_metrics['rescued'] = info.get('statistics', {}).get('total_rescued', 0)
        episode_metrics['deaths'] = info.get('statistics', {}).get('total_deaths', 0)
        episode_metrics['resources_used'] = info.get('statistics', {}).get('resources_used', 0)
        # T-log: 也记录总casualty数，供 Survivors X/Y 格式使用
        episode_metrics['total_casualties'] = getattr(self.env, 'num_victims', 0) or len(getattr(self.env, 'casualties', {}))
        
        # 计算救援成功率
        total_victims = self.env.num_victims
        if total_victims > 0:
            episode_metrics['rescue_rate'] = (episode_metrics['rescued'] / total_victims) * 100
        else:
            episode_metrics['rescue_rate'] = 0.0
        
        # 计算资源利用率
        # 使用环境中已计算好的初始资源总量（depot初始资源 + agent初始资源），确保与评估阶段计算口径一致
        total_initial = sum(sum(r.values()) for r in self.env.initial_resources.values()) + self.env.initial_agent_resources
        
        if total_initial > 0:
            episode_metrics['resource_utilization'] = (episode_metrics['resources_used'] / total_initial) * 100
        else:
            episode_metrics['resource_utilization'] = 0.0
        
        # Manager集成：Episode结束回调
        if self.manager_integration is not None:
            episode_summary = self.manager_integration.on_episode_end()
            episode_metrics['final_lambda'] = episode_summary.get('final_lambda', 0.0)
            episode_metrics['total_communications'] = episode_summary.get('total_communications', 0)
            episode_metrics['shared_casualties'] = episode_summary.get('shared_casualties', 0)
        
        # 记录Manager指标
        self._log_manager_metrics(episode_idx)

        # Aggregate the per-step loss buffer into a single episode-level
        # 'loss' field (mean of non-zero updates; 0.0 if none fired).
        loss_buf = episode_metrics.pop('__loss_buf', None)
        last_update = episode_metrics.pop('__last_update', None)
        if isinstance(loss_buf, list) and loss_buf:
            episode_metrics['loss'] = float(np.mean(loss_buf))
        else:
            episode_metrics['loss'] = 0.0
        # Stash the most-recent raw update dict for the trainer's history
        if isinstance(last_update, dict):
            for k, v in last_update.items():
                if isinstance(v, (int, float)):
                    episode_metrics[f"last_{k}"] = float(v)

        return episode_metrics
    
    def _log_entity_info(self, step: int) -> None:
        """Log detailed entity information every 50 steps for debugging."""
        # Log agent information (single line per agent)
        logger.debug(f"=== Step {step} - Agents ===")
        for agent_id, agent in sorted(self.env.rescue_agents.items()):
            logger.debug(f"  {agent.format_log_line()}")
        
        # Log casualty information (single line per casualty) - skip treated casualties
        logger.debug(f"=== Step {step} - Casualties ===")
        for casualty_id, casualty in sorted(self.env.casualties.items()):
            # Skip already treated casualties
            if casualty.treated:
                continue
            
            # Find nearest agent
            nearest_agent = None
            min_dist = float('inf')
            for agent in self.env.rescue_agents.values():
                dist = np.linalg.norm(agent.position - casualty.position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_agent = agent.id
            
            nearest_info = {'agent_id': nearest_agent, 'distance': min_dist}
            logger.debug(f"  {casualty.format_log_line(nearest_info)}")
    
    def _get_agent_states(self) -> Dict:
        """获取所有agent的状态（用于EGT fitness计算）"""
        agent_states = {}
        for agent_id, agent in self.env.rescue_agents.items():
            # 使用agent自己救援的人数作为fitness指标
            rescued_count = getattr(agent, 'rescued_count', 0)
            
            # 资源效率：当前资源量与最大容量的比例（反映资源利用情况）
            current_resources = sum(agent.capacity.values())
            max_resources = sum(agent.max_capacity.values())
            resource_efficiency = current_resources / max(max_resources, 1)
            
            agent_states[agent_id] = {
                'survival_rate': float(rescued_count),  # 使用绝对救援数
                'resource_efficiency': resource_efficiency
            }
        return agent_states
    
    def _process_agent_communication(self):
        """处理Agent间的信息共享"""
        if self.manager_integration is None:
            return
        
        # 更新每个agent的已知伤员
        for agent_id, agent in self.env.rescue_agents.items():
            # 准备邻近agent列表
            nearby_agents = []
            for other_id, other_agent in self.env.rescue_agents.items():
                if other_id == agent_id:
                    continue
                
                # 检查通信是否成功
                can_comm, _ = self.manager_integration.check_communication(
                    agent.position,
                    other_agent.position
                )
                
                if can_comm:
                    nearby_agents.append((other_id, other_agent.position))
            
            # 广播自己的已知伤员
            known_casualties = {}
            for cid in agent.known_casualties:
                if cid in self.env.casualties:
                    casualty = self.env.casualties[cid]
                    known_casualties[cid] = {
                        'position': casualty.position,
                        'severity': casualty.severity.name
                    }
            
            self.manager_integration.broadcast_casualties(agent_id, agent.position, known_casualties)
            
            # 接收邻近agent的广播
            new_casualties = self.manager_integration.receive_broadcasts(
                agent_id, agent.position, nearby_agents
            )
            
            # 更新agent的known_casualties（字典类型）
            for casualty_id, casualty_info in new_casualties.items():
                if casualty_id not in agent.known_casualties:
                    agent.known_casualties[casualty_id] = casualty_info
    
    def _log_manager_metrics(self, episode: int):
        """记录Manager指标到日志"""
        if self.manager_integration is None:
            return
        
        metrics = self.manager_integration.get_metrics()
        
        # EGT指标
        egt = metrics['egt']
        logger.info(f"[EGT] Episode {episode} - λ={egt['lambda_t']:.4f}, "
                    f"HistoryLen={egt['history_length']}")
        
        # Pareto指标
        pareto = metrics['pareto']
        logger.info(f"[PARETO] Episode {episode} - "
                    f"Efficiency={pareto['current_efficiency_weight']:.2f}, "
                    f"Fairness={pareto['current_fairness_weight']:.2f}")
        
        # Reputation指标
        rep = metrics['reputation']
        logger.info(f"[REPUTATION] Episode {episode} - "
                    f"AvgReputation={rep.get('avg_reputation', 0.0):.2f}, "
                    f"AgentCount={rep.get('agent_count', 0)}")
        
        # Communication指标
        comm = metrics['communication']
        logger.info(f"[COMM] Episode {episode} - "
                    f"SharedCasualties={comm['shared_casualties_count']}, "
                    f"Events={comm['communication_events']}")
        
        # Interference指标
        interf = metrics['interference']
        logger.info(f"[INTERFERENCE] Episode {episode} - "
                    f"AvgDelay={interf.get('avg_delay', 0.0):.3f}s, "
                    f"LossRate={interf.get('loss_rate', 0.0):.2%}, "
                    f"Interrupted={interf['is_interrupted']}")
    
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
            response_times = []

            done = False
            step = 0
            max_steps = self.config['training']['max_steps_per_episode']

            while not done and step < max_steps:
                actions = self.algorithm.select_action(state)
                next_state, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated

                # Handle individual rewards (dict) or global reward (float)
                step_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
                episode_reward += step_reward

                state = next_state
                step += 1
            
            # 在episode结束时获取最终的统计数据
            statistics = info.get('statistics', {})
            episode_rescued = statistics.get('total_rescued', 0)
            
            # 计算指标
            total_victims = self.env.num_victims
            rescue_rate = (episode_rescued / total_victims * 100) if total_victims > 0 else 0.0

            # 计算平均响应时间
            response_times = statistics.get('response_times', [])
            avg_response_time = np.mean(response_times) if response_times else 0.0

            # 计算资源利用率
            resources_used = statistics.get('resources_used', 0.0)
            total_initial = sum(sum(r.values()) for r in self.env.initial_resources.values()) + self.env.initial_agent_resources
            resource_utilization = (resources_used / total_initial * 100) if total_initial > 0 else 0.0

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
        
        # 保存完整的算法状态（包括EGT层、MARL层等）
        algorithm_checkpoint_path = self.experiment_dir / 'checkpoints' / f'checkpoint_ep{episode_idx}_algorithm.pt'
        if self.algorithm is not None:
            self.algorithm.save_checkpoint(algorithm_checkpoint_path)
            logger.info(f"Algorithm checkpoint saved: {algorithm_checkpoint_path}")
        
        checkpoint = {
            'episode': episode_idx,
            'metrics': metrics,
            'config': self.config,
            'algorithm_checkpoint_path': str(algorithm_checkpoint_path),
            'best_rescue_rate': getattr(self, 'best_rescue_rate', 0.0),
            'best_model_episode': getattr(self, '_best_model_episode', None),
            # 保存训练时的环境维度，evaluate_model.py 据此检测
            # "训练/评估环境不一致"导致的 RR 下降。
            'env_dims': {
                'map_size': self.config.get('environment', {}).get('map_size'),
                'num_agents': self.config.get('environment', {}).get('num_agents'),
                'num_victims': self.config.get('environment', {}).get('num_victims'),
                'num_resources': self.config.get('environment', {}).get('num_resources'),
                'num_areas': self.config.get('environment', {}).get('num_areas'),
            },
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            恢复时的episode编号
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        episode = checkpoint.get('episode', 0)
        self.config = checkpoint.get('config', self.config)

        # Restore best_rescue_rate so that resume does not overwrite
        # the previously best model with a worse one (Issue 5 fix).
        if 'best_rescue_rate' in checkpoint:
            self.best_rescue_rate = checkpoint['best_rescue_rate']
            logger.info(
                f"Restored best_rescue_rate={self.best_rescue_rate:.4f} "
                f"from checkpoint (best_model_episode={checkpoint.get('best_model_episode')})"
            )
        
        # 加载算法权重（包括EGT层、MARL层等）
        if self.algorithm is not None:
            # 首先尝试从checkpoint中指定的路径加载
            algo_ckpt_path = checkpoint.get('algorithm_checkpoint_path')
            if algo_ckpt_path and os.path.exists(algo_ckpt_path):
                self.algorithm.load_checkpoint(algo_ckpt_path)
                logger.info(f"Algorithm checkpoint loaded from: {algo_ckpt_path}")
            else:
                # 旧格式兼容：尝试从同一目录加载算法checkpoint
                algo_ckpt_path = Path(checkpoint_path).parent / f'checkpoint_ep{episode}_algorithm.pt'
                if algo_ckpt_path.exists():
                    self.algorithm.load_checkpoint(algo_ckpt_path)
                    logger.info(f"Algorithm checkpoint loaded from: {algo_ckpt_path}")
                else:
                    logger.warning(f"Algorithm checkpoint not found, EGT/MARL state will be reset")
        
        logger.info(f"Checkpoint loaded: {checkpoint_path} (resuming from episode {episode})")
        return episode
    
    def save_best_model(self, episode_idx: int, metrics: Dict[str, float]):
        """保存最佳模型"""
        best_model_path = self.experiment_dir / 'models' / 'best_model.pt'
        
        # 调用EGT-MARL的save_checkpoint方法保存模型权重
        self.algorithm.save_checkpoint(best_model_path)
        logger.info(f"Best model saved: {best_model_path}")
    
    def save_final_model(self, metrics: Dict[str, float]):
        """保存最终模型"""
        final_model_path = self.experiment_dir / 'models' / 'final_model.pt'
        
        # 调用EGT-MARL的save_checkpoint方法保存模型权重
        self.algorithm.save_checkpoint(final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
    
    def train(self, resume_from: str = None):
        """主训练循环"""
        logger.info("Starting training...")

        # 设置目录（恢复训练时复用checkpoint所在的实验目录
        if resume_from:
            ckpt_path = Path(resume_from)
            # checkpoints 位于 experiment_dir/checkpoints/checkpoint_epXXX.pt
            experiment_dir = ckpt_path.parent.parent
            self.setup_directories(str(experiment_dir))
        else:
            self.setup_directories()
        
        # 设置组件（包括Manager集成）
        self.setup_manager_integration()
        self.setup_environment()
        self.setup_algorithm()
        self.setup_metrics()
        
        # 训练参数
        training_config = self.config['training']
        
        # 检查是否有阶段配置，如果有则计算总 episode 数
        if ('schedule' in self.config
                and self.config['schedule'].get('phases')):
            phases = self.config['schedule']['phases']
            num_episodes = sum(phase['episodes'] for phase in phases)
            logger.info(f"Multi-phase training enabled: {len(phases)} phases, total episodes: {num_episodes}")
            for phase in phases:
                logger.info(f"  - {phase['name']}: {phase['episodes']} episodes")
        else:
            # Audit fix T1: default 4-stage curriculum
            # (Warmup / Transition / Main / FineTune) when no explicit phases
            # are configured.  Stages progress from random exploration to
            # fairness-aware fine-tuning.
            #
            # base_lr default = 0.0001 matches transition stage. With the
            # 2.0/1.0/0.5/0.1 multiplier schedule this yields:
            #   Warmup    = 0.0002
            #   Transition= 0.0001
            #   Main      = 0.00005
            #   FineTune  = 0.00001
            # which is identical to the explicit phases in training.yaml.
            num_episodes = training_config['num_episodes']
            warmup = int(num_episodes * 0.10)
            transition = int(num_episodes * 0.20)
            main = int(num_episodes * 0.55)
            finetune = num_episodes - warmup - transition - main
            base_lr = training_config.get('learning_rate', 0.0001)
            phases = [
                {'name': 'Warmup',     'episodes': warmup,
                 'learning_rate': base_lr * 2.0,   'epsilon_scale': 1.0},
                {'name': 'Transition', 'episodes': transition,
                 'learning_rate': base_lr,          'epsilon_scale': 0.7},
                {'name': 'Main',       'episodes': main,
                 'learning_rate': base_lr * 0.5,    'epsilon_scale': 0.3},
                {'name': 'FineTune',   'episodes': finetune,
                 'learning_rate': base_lr * 0.1,    'epsilon_scale': 0.05},
            ]
            self.config.setdefault('schedule', {})['phases'] = phases
            logger.info(
                f"Default 4-stage curriculum: Warmup={warmup}, "
                f"Transition={transition}, Main={main}, FineTune={finetune} "
                f"(total={num_episodes})"
            )
        
        epsilon = training_config['epsilon_start']
        epsilon_decay = training_config['epsilon_decay']
        epsilon_end = training_config['epsilon_end']
        
        checkpoint_interval = training_config['checkpoint_interval']
        eval_interval = training_config['eval_interval']
        num_eval_episodes = training_config['num_eval_episodes']
        save_best_model = training_config['save_best_model']
        
        # 训练统计（best_rescue_rate 改为实例属性以便 resume 时恢复）
        if not hasattr(self, 'best_rescue_rate'):
            self.best_rescue_rate = 0.0
        if not hasattr(self, '_best_model_episode'):
            self._best_model_episode = None
        training_history = {
            'episodes': [],
            'rescue_rate': [],
            'avg_response_time': [],
            'resource_utilization': [],
            'total_reward': [],
            'loss': []
        }
        
        # 计算phase边界
        phases = self.config.get('schedule', {}).get('phases', [])
        phase_starts = []
        phase_ends = []
        current_start = 1
        for phase in phases:
            phase_starts.append(current_start)
            current_end = current_start + phase.get('episodes', 100) - 1
            phase_ends.append(current_end)
            current_start = current_end + 1
        
        # 读取学习率调度配置
        lr_schedule_config = self.config.get('schedule', {}).get('lr_schedule', {})
        lr_warmup = int(lr_schedule_config.get('warmup_episodes', 100))
        lr_min = float(lr_schedule_config.get('min_lr', 1e-6))
        lr_max = float(lr_schedule_config.get('max_lr', 0.001))
        
        # 检查是否需要恢复训练
        start_episode = 1
        if resume_from:
            start_episode = self.load_checkpoint(resume_from) + 1
            # 计算恢复后的epsilon值
            epsilon = max(epsilon_end, epsilon * (epsilon_decay ** (start_episode - 1)))
            logger.info(f"Resuming training from episode {start_episode} with epsilon={epsilon:.4f}")
        
        # 训练循环
        for episode in range(start_episode, num_episodes + 1):
            # Issue 6 fix: sync algorithm's internal episode counter so the
            # checkpoint records the real episode, not 0.
            if self.algorithm is not None and hasattr(self.algorithm, 'set_external_episode'):
                self.algorithm.set_external_episode(episode)

            # 检查是否需要切换phase
            current_phase_idx = -1
            for i, (start, end) in enumerate(zip(phase_starts, phase_ends)):
                if start <= episode <= end:
                    current_phase_idx = i
                    break
            
            # 根据当前phase设置learning_rate（余弦退火）
            if phases and current_phase_idx >= 0:
                phase = phases[current_phase_idx]
                if 'learning_rate' in phase:
                    phase_lr = float(phase['learning_rate'])
                    # 在phase内部应用余弦退火
                    phase_start = phase_starts[current_phase_idx]
                    phase_end = phase_ends[current_phase_idx]
                    phase_progress = (episode - phase_start) / max(1, phase_end - phase_start)
                    # 余弦退火：从phase_lr逐渐降到lr_min
                    current_lr = lr_min + (phase_lr - lr_min) * (1 + math.cos(math.pi * phase_progress)) / 2
                    # 更新optimizer的学习率
                    if hasattr(self, 'algorithm') and hasattr(self.algorithm, 'marl_layer') and hasattr(self.algorithm.marl_layer, 'optimizer'):
                        for param_group in self.algorithm.marl_layer.optimizer.param_groups:
                            param_group['lr'] = current_lr
                        logger.debug(f"Phase {phase.get('name', current_phase_idx)} - Episode {episode}: Learning rate set to {current_lr:.6f}")

                # T1 fix: when a new phase begins, also apply phase['exploration_rate']
                # (absolute) and phase['lambda'] (injected into EGT layer), and
                # phase['difficulty'] (drives env casualty severity mix).
                if current_phase_idx != getattr(self, '_last_phase_idx', -2):
                    # ----- exploration_rate (absolute, takes priority over epsilon_scale) -----
                    if 'exploration_rate' in phase:
                        target_eps = float(phase['exploration_rate'])
                        epsilon = max(epsilon_end, target_eps)
                        logger.info(
                            f"=== Entering phase '{phase.get('name', current_phase_idx)}' "
                            f"(eps → {epsilon:.3f}, lr={current_lr:.6f}) ==="
                        )
                    # ----- lambda_param (anchor for EGT evolution) -----
                    # Fix audit Issue 2: previously we OVERWROTE
                    # egt_layer.lambda_param directly, which meant phase changes
                    # reset whatever the replicator dynamics had evolved. That
                    # made the EGT layer's micro-→macro feedback loop useless.
                    #
                    # New behaviour: store the phase's target as
                    # `lambda_anchor` and let `egt_layer._update_lambda()`
                    # BLEND it with the evolved strategy distribution.  See
                    # `egt_layer.py:phase_anchor_blend` for the blending rule.
                    if 'lambda' in phase and hasattr(self, 'algorithm') and hasattr(self.algorithm, 'egt_layer'):
                        lam = float(phase['lambda'])
                        clamped_lam = max(0.0, min(1.0, lam))
                        self.algorithm.egt_layer.lambda_anchor = clamped_lam
                        logger.info(
                            f"  phase lambda -> EGT layer lambda_anchor = {clamped_lam:.3f} "
                            f"(current lambda_param = "
                            f"{self.algorithm.egt_layer.lambda_param:.3f})"
                        )
                    # ----- difficulty (drive env casualty severity distribution) -----
                    if 'difficulty' in phase and hasattr(self, 'env') and hasattr(self.env, 'set_difficulty'):
                        diff = float(phase['difficulty'])
                        self.env.set_difficulty(diff)
                        logger.info(
                            f"  phase difficulty -> env.set_difficulty({diff:.2f})"
                        )
                    self._last_phase_idx = current_phase_idx

                # Backward-compat: epsilon_scale (relative) still works when
                # phase['exploration_rate'] is not provided.
                if 'epsilon_scale' in phase and 'exploration_rate' not in phase:
                    # Recompute target epsilon for this phase (relative to epsilon_end)
                    phase_epsilon = max(
                        epsilon_end,
                        epsilon * float(phase['epsilon_scale'])
                    )
                    if current_phase_idx != getattr(self, '_last_phase_idx', -2):
                        # Phase transition — log and apply
                        self._last_phase_idx = current_phase_idx
                        logger.info(
                            f"=== Entering phase '{phase.get('name', current_phase_idx)}' "
                            f"(eps {epsilon:.3f} → target {phase_epsilon:.3f}, "
                            f"lr={current_lr:.6f}) ==="
                        )
                        # Snap epsilon down at the start of each new phase
                        # (subsequent episodes continue the exponential decay
                        # from this snapped value)
                        epsilon = phase_epsilon
            
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
            
            # TensorBoard 日志记录
            if self.writer is not None:
                tb_step = episode  # 使用 episode 作为 step
                self.writer.add_scalar('Metrics/Rescue_Rate', episode_metrics.get('rescue_rate', 0.0), tb_step)
                self.writer.add_scalar('Metrics/Avg_Response_Time', episode_metrics.get('avg_response_time', 0.0), tb_step)
                self.writer.add_scalar('Metrics/Resource_Utilization', episode_metrics.get('resource_utilization', 0.0), tb_step)
                self.writer.add_scalar('Metrics/Total_Reward', episode_metrics.get('total_reward', 0.0), tb_step)
                self.writer.add_scalar('Metrics/Rescued_Count', episode_metrics.get('rescued', 0), tb_step)
                self.writer.add_scalar('Metrics/Deaths', episode_metrics.get('deaths', 0), tb_step)
                self.writer.add_scalar('Metrics/Epsilon', epsilon, tb_step)
                
                # 记录 Manager 指标
                if self.manager_integration is not None:
                    self.writer.add_scalar('EGT/Lambda', episode_metrics.get('final_lambda', 0.0), tb_step)
                    self.writer.add_scalar('Communication/Total', episode_metrics.get('total_communications', 0), tb_step)
                    self.writer.add_scalar('Communication/Shared', episode_metrics.get('shared_casualties', 0), tb_step)
            
            # 定期评估
            if episode % eval_interval == 0:
                eval_metrics = self.evaluate(num_eval_episodes)
                
                logger.info(f"Episode {episode}/{num_episodes} - "
                           f"Epsilon: {epsilon:.3f} - "
                           f"Train Rescue Rate: {episode_metrics.get('rescue_rate', 0.0):.1f}% - "
                           f"Eval Rescue Rate: {eval_metrics.get('rescue_rate', 0.0):.1f}% ± {eval_metrics.get('rescue_rate_std', 0.0):.1f}")
                
                # 保存最佳模型（Issue 5: 使用 self.best_rescue_rate 实例属性）
                if save_best_model and eval_metrics.get('rescue_rate', 0.0) > self.best_rescue_rate:
                    self.best_rescue_rate = eval_metrics.get('rescue_rate', 0.0)
                    self._best_model_episode = episode
                    self.save_best_model(episode, eval_metrics)
                    logger.info(
                        f"New best model: RR={self.best_rescue_rate:.4f} at episode {episode}"
                    )
            
            # 定期保存检查点
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode, episode_metrics)
            
            # 每5个episode打印详细进度（增加频率）
            if episode % 5 == 0:
                logger.info(f"Episode {episode}/{num_episodes} - "
                           f"Epsilon: {epsilon:.3f} - "
                           f"Rescue Rate: {episode_metrics.get('rescue_rate', 0.0):.1f}% - "
                           f"Survivors: {episode_metrics.get('rescued', 0)}/{episode_metrics.get('total_casualties', 0)} - "
                           f"Reward: {episode_metrics.get('total_reward', 0.0):.2f} - "
                           f"Steps: {episode_metrics.get('steps', 0)} - "
                           f"Response Time: {episode_metrics.get('avg_response_time', 0.0):.1f}s")
            
            # 每个episode都打印简要的救援率信息（确保能看到每个episode的救援率）
            if episode % 1 == 0:
                logger.info(f"Episode {episode} | Rescue Rate: {episode_metrics.get('rescue_rate', 0.0):.1f}% | "
                           f"Rescued: {episode_metrics.get('rescued', 0)} | Deaths: {episode_metrics.get('deaths', 0)} | "
                           f"Response Time: {episode_metrics.get('avg_response_time', 0.0):.1f}s | "
                           f"Resource Utilization: {episode_metrics.get('resource_utilization', 0.0):.1f}%")
                
                # 打印每个agent的详细信息
                resource_abbr = {'BROAD_SPECTRUM_ANTIBIOTICS': 'ANT', 'BLOOD_PACKS': 'BLD', 'OXYGEN': 'OXY', 'PAIN_MEDICATION': 'PAIN'}
                for agent_id, agent in sorted(self.env.rescue_agents.items()):
                    known_count = len(agent.known_casualties)
                    rescued_count = getattr(agent, 'rescued_count', 0)
                    mission = getattr(agent, 'current_mission', 'None')
                    resources = {resource_abbr.get(rt.name, rt.name[:4]): round(agent.capacity[rt], 1) for rt in ResourceType}
                    logger.info(f"  [AGENT {agent_id}/{agent.agent_type}] Status={mission} | Position=[{agent.position[0]:.1f}, {agent.position[1]:.1f}] | KNOWN CASUALTIES={known_count} | Rescued={rescued_count} | Resources={resources}")

            # EGT策略建议：每100个episode输出一次
            if episode % 100 == 0 and hasattr(self.algorithm, 'egt_layer'):
                rec = self.algorithm.egt_layer.get_strategy_recommendation()
                logger.info(
                    f"[EGT-Strategy] Episode {episode} | "
                    f"Dominant={rec['dominant_strategy']} | "
                    f"Dist=[{', '.join(f'{x:.2f}' for x in rec['strategy_distribution'])}] | "
                    f"F={rec['fairness_weight']:.2f}, E={rec['efficiency_weight']:.2f} | "
                    f"Conv={rec['convergence_status']} | "
                    f"Rec={rec['recommendation'][:50]}..."
                )
                # 记录到训练历史
                training_history.setdefault('strategy_recommendations', []).append({
                    'episode': episode,
                    'dominant_strategy': rec['dominant_strategy'],
                    'strategy_distribution': rec['strategy_distribution'],
                    'fairness_weight': rec['fairness_weight'],
                    'efficiency_weight': rec['efficiency_weight'],
                    'convergence_status': rec['convergence_status'],
                    'avg_fitness': rec['avg_fitness'],
                    'diversity': rec['diversity']
                })

        # 训练完成
        logger.info("Training completed!")

        # 最终EGT策略报告
        if hasattr(self.algorithm, 'egt_layer'):
            final_rec = self.algorithm.egt_layer.get_strategy_recommendation()
            logger.info("=" * 60)
            logger.info("Final EGT Strategy Report:")
            logger.info(f"  Dominant Strategy: {final_rec['dominant_strategy']}")
            logger.info(f"  Strategy Distribution: {final_rec['strategy_distribution']}")
            logger.info(f"  Fairness Weight: {final_rec['fairness_weight']:.3f}")
            logger.info(f"  Efficiency Weight: {final_rec['efficiency_weight']:.3f}")
            logger.info(f"  Convergence Status: {final_rec['convergence_status']}")
            logger.info(f"  Avg Fitness: {final_rec['avg_fitness']:.3f}")
            logger.info(f"  Diversity: {final_rec['diversity']:.3f}")
            logger.info(f"  Recommendation: {final_rec['recommendation']}")
            logger.info("=" * 60)

        # 最终评估
        final_metrics = self.evaluate(num_eval_episodes * 2)
        logger.info(f"Final Evaluation - "
                   f"Rescue Rate: {final_metrics.get('rescue_rate', 0.0):.1f}% ± {final_metrics.get('rescue_rate_std', 0.0):.1f} - "
                   f"Response Time: {final_metrics.get('avg_response_time', 0.0):.1f}s - "
                   f"Resource Utilization: {final_metrics.get('resource_utilization', 0.0):.1f}%")
        
        # 保存最终模型
        self.save_final_model(final_metrics)
        
        # 生成训练报告
        self.generate_training_report(training_history, final_metrics)
        
        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        
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
            
            f.write("3. Evaluation Performance Metrics\n")
            f.write("-" * 40 + "\n")
            num_eval_episodes = self.config['training']['num_eval_episodes']
            f.write(f"(Evaluated on {num_eval_episodes * 2} episodes after training)\n")
            f.write(f"Rescue Rate: {final_metrics.get('rescue_rate', 0.0):.1f}% "
                   f"(±{final_metrics.get('rescue_rate_std', 0.0):.1f})\n")
            f.write(f"Average Response Time: {final_metrics.get('avg_response_time', 0.0):.1f}s\n")
            f.write(f"Resource Utilization: {final_metrics.get('resource_utilization', 0.0):.1f}%\n")
            f.write(f"Total Reward: {final_metrics.get('total_reward', 0.0):.1f}\n\n")
            
            f.write("4. Training Statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"(Recorded during {self.config['training']['num_episodes']} training episodes)\n")
            if training_history['rescue_rate']:
                f.write(f"Best Training Rescue Rate: {max(training_history['rescue_rate']):.1f}%\n")
                f.write(f"Final Training Rescue Rate: {training_history['rescue_rate'][-1]:.1f}%\n")
                f.write(f"Average Training Rescue Rate: {np.mean(training_history['rescue_rate']):.1f}%\n")
            
            if training_history['loss']:
                valid_losses = [l for l in training_history['loss'] if l is not None and l > 0]
                if valid_losses:
                    f.write(f"Final Loss: {valid_losses[-1]:.4f}\n")
                    f.write(f"Average Loss: {np.mean(valid_losses):.4f}\n")

            # 新增：EGT策略推荐报告
            if hasattr(self.algorithm, 'egt_layer'):
                f.write("\n5. EGT Strategy Recommendation\n")
                f.write("-" * 40 + "\n")
                rec = self.algorithm.egt_layer.get_strategy_recommendation()
                f.write(f"Dominant Strategy: {rec['dominant_strategy']}\n")
                f.write(f"Strategy Distribution: "
                       f"[{', '.join(f'{x:.4f}' for x in rec['strategy_distribution'])}]\n")
                f.write(f"Fairness Weight: {rec['fairness_weight']:.4f}\n")
                f.write(f"Efficiency Weight: {rec['efficiency_weight']:.4f}\n")
                f.write(f"Convergence Status: {rec['convergence_status']}\n")
                f.write(f"Average Fitness: {rec['avg_fitness']:.4f}\n")
                f.write(f"Diversity (Entropy): {rec['diversity']:.4f}\n")
                f.write(f"Recommendation: {rec['recommendation']}\n")
                # 记录演化历史摘要
                if 'strategy_recommendations' in training_history:
                    f.write(f"\nStrategy Evolution Milestones:\n")
                    for milestone in training_history['strategy_recommendations']:
                        f.write(f"  Episode {milestone['episode']}: "
                               f"{milestone['dominant_strategy']} "
                               f"(F={milestone['fairness_weight']:.2f}, "
                               f"E={milestone['efficiency_weight']:.2f})\n")

            f.write("\n6. Files Generated\n")
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

            # EGT策略演化图
            if 'strategy_recommendations' in training_history and training_history['strategy_recommendations']:
                strategy_names = getattr(self.algorithm.egt_layer, 'strategy_names',
                                         ['Strategy-' + str(i) for i in range(self.algorithm.egt_layer.num_strategies)])
                evolution_path = self.experiment_dir / 'visualizations' / 'egt_strategy_evolution.png'
                plot_egt_strategy_evolution(
                    training_history['strategy_recommendations'],
                    strategy_names,
                    str(evolution_path),
                    show=False
                )
                # 最终策略推荐饼图
                if hasattr(self.algorithm, 'egt_layer'):
                    final_rec = self.algorithm.egt_layer.get_strategy_recommendation()
                    pie_path = self.experiment_dir / 'visualizations' / 'egt_final_recommendation.png'
                    plot_egt_strategy_recommendation(
                        final_rec,
                        strategy_names,
                        str(pie_path),
                        show=False
                    )

            logger.info(f"Training visualizations saved to {self.experiment_dir}/visualizations/")

        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")


def train_egt_marl(config_path: str = 'configs/training.yaml', **kwargs):
    """
    训练EGT-MARL算法的入口函数
    
    Args:
        config_path: 配置文件路径
        **kwargs: 额外参数（用于覆盖配置）
    
    Returns:
        training_history: 训练历史数据
        final_metrics: 最终评估指标
    """
    # 创建训练器
    trainer = EGTMARLTrainer(config_path)
    
    # 应用额外参数覆盖
    for key, value in kwargs.items():
        if key == 'output_dir':
            trainer.config['output_dir'] = value
        elif key == 'num_episodes':
            trainer.config['training']['num_episodes'] = value
        elif key == 'learning_rate':
            trainer.config['training']['learning_rate'] = value
        elif key == 'batch_size':
            trainer.config['training']['batch_size'] = value
    
    # 开始训练
    training_history, final_metrics = trainer.train()
    
    return training_history, final_metrics


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
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
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
    
    # 设置恢复检查点路径
    resume_checkpoint = args.resume
    
    # 开始训练
    try:
        if resume_checkpoint:
            training_history, final_metrics = trainer.train(resume_from=resume_checkpoint)
        else:
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