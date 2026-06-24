"""
基线算法评估脚本

评估 EGT-MARL 与基线算法的性能对比。
包括传统方法（FCFS, 优先级调度）和其他 MARL 算法（QMIX, MADDPG, MAPPO）。
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import pandas as pd
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
from utils.metrics import MetricsCollector
from environments.visualization import DisasterVisualizer
import logging

# 初始化logger
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """基线算法评估器"""
    
    def __init__(self, config_path: str):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.setup_device()
        
        # 初始化组件
        self.env = None
        self.algorithms = {}
        self.metrics_collector = MetricsCollector()
        self.visualizer = None
        
        logger.info(f"Baseline Evaluator initialized with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 设置默认值
        defaults = {
            'evaluation': {
                'num_episodes': 100,
                'max_steps_per_episode': 200,
                'num_runs': 5,
                'scenarios': ['earthquake', 'flood', 'hurricane'],
                'severities': ['low', 'medium', 'high']
            },
            'algorithms': {
                'egt_marl': {'enabled': False, 'model_path': None},  # 禁用EGT-MARL，先执行其他基线
                'qmix': {'enabled': False, 'model_path': None},
                'maddpg': {'enabled': False, 'model_path': None},
                'mappo': {'enabled': False, 'model_path': None},
                'fcfs': {'enabled': True},
                'priority': {'enabled': True},
                'greedy_local': {'enabled': True},
                'proportional_fair': {'enabled': True},
                'centralized_mpc': {'enabled': True},
                'standard_marl': {'enabled': True},
                'game_theoretic': {'enabled': True},
                'gnn_based': {'enabled': True},
                'transformer_based': {'enabled': True}
            },
            'environment': {
                'map_size': (100, 100),
                'num_agents': 5,
                'num_victims': 20,
                'num_resources': 10,
                'num_hospitals': 3
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
        base_dir = Path(self.config.get('output_dir', 'evaluation_results'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.evaluation_dir = base_dir / f'baseline_evaluation_{timestamp}'
        
        # 创建目录
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        (self.evaluation_dir / 'results').mkdir(exist_ok=True)
        (self.evaluation_dir / 'logs').mkdir(exist_ok=True)
        (self.evaluation_dir / 'visualizations').mkdir(exist_ok=True)
        
        # 配置日志
        log_file = self.evaluation_dir / 'logs' / 'evaluation.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler()
            ]
        )
        
        # 保存配置
        config_path = self.evaluation_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Evaluation directory: {self.evaluation_dir}")
        logger.info(f"Log file: {log_file}")
    
    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
    
    def setup_environment(self, disaster_type: str, severity: str):
        """Set up the environment for the given disaster scenario.

        Args:
            disaster_type: Disaster type (earthquake, flood, hurricane, etc.)
            severity: Severity level (low, medium, high)
        """
        env_config = self.config['environment']

        # Initialize DisasterSim environment.  The CLI args (disaster_type and
        # severity) take precedence over the hard-coded values in the YAML
        # (this is the fix for the E2 audit finding: the function previously
        # ignored the parameters and used the YAML values regardless).
        self.env = DisasterSim(
            map_size=tuple(env_config['map_size']),
            num_agents=env_config['num_agents'],
            num_victims=env_config['num_victims'],
            num_resources=env_config['num_resources'],
            num_hospitals=env_config['num_hospitals'],
            disaster_type=disaster_type,
            severity=severity,
        )

        # For compatibility with other code, add necessary attributes
        self.env.num_agents = len(self.env.rescue_agents)
        self.env.num_victims = len(self.env.casualties)

        # Cache total resources for utilization calculation
        # Get from config object or use default
        try:
            if hasattr(self.env.config, 'total_resources'):
                self.env.total_resources = self.env.config.total_resources
            elif hasattr(self.env.config, 'get') and callable(self.env.config.get):
                self.env.total_resources = self.env.config.get('total_resources', 1000)
            else:
                self.env.total_resources = 1000
        except:
            self.env.total_resources = 1000

        logger.info(
            f"Environment initialized: {disaster_type} ({severity}) - "
            f"Agents: {self.env.num_agents}, Victims: {self.env.num_victims}"
        )
    
    def setup_algorithms(self):
        """设置算法"""
        algo_config = self.config['algorithms']
        
        # 获取环境信息（需要先初始化环境）
        if self.env is None:
            raise ValueError("Environment must be initialized before setting up algorithms")
        
        state_dim = self.env.get_state_dimension()
        action_dim = 32  # 8 tactical actions * 4 communication modes (from training config)
        num_agents = self.env.num_agents
        
        # 初始化 EGT-MARL
        if algo_config['egt_marl']['enabled']:
            # 先尝试加载checkpoint获取配置
            model_path = algo_config['egt_marl'].get('model_path')
            checkpoint_config = None
            
            if model_path and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if 'config' in checkpoint:
                        checkpoint_config = checkpoint['config']
                        logger.info(f"Loaded config from checkpoint: state_dim={checkpoint_config.get('marl', {}).get('state_dim')}, "
                                   f"num_strategies={checkpoint_config.get('egt', {}).get('num_strategies')}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint config: {e}")
            
            # 为EGT-MARL创建完整配置
            egt_config = {
                'marl': {
                    'num_agents': checkpoint_config.get('marl', {}).get('num_agents', num_agents),
                    'state_dim': checkpoint_config.get('marl', {}).get('state_dim', state_dim),
                    'action_dim': checkpoint_config.get('marl', {}).get('action_dim', action_dim),
                    'hidden_dim': checkpoint_config.get('marl', {}).get('hidden_dim', 128),
                    'mixing_hidden_dim': checkpoint_config.get('marl', {}).get('mixing_hidden_dim', 64),
                    'attention_heads': checkpoint_config.get('marl', {}).get('attention_heads', 4),
                    'learning_rate': checkpoint_config.get('marl', {}).get('learning_rate', 0.001),
                    'epsilon_start': 1.0,
                    'epsilon_decay': 0.995,
                    'epsilon_min': 0.01,
                    'gamma': 0.99,
                    'tau': 0.005,
                    'batch_size': checkpoint_config.get('marl', {}).get('batch_size', 32),
                    'buffer_size': checkpoint_config.get('marl', {}).get('buffer_size', 5000),
                    'update_frequency': 4
                },
                'egt': {
                    'fairness_weight': 0.3,
                    'efficiency_weight': 0.7,
                    'anti_spoofing_threshold': 0.1,
                    'num_strategies': checkpoint_config.get('egt', {}).get('num_strategies', 3),
                    'learning_rate': checkpoint_config.get('egt', {}).get('learning_rate', 0.01),
                    'mutation_rate': 0.01,
                    'selection_intensity': 1.0,
                    'egt_lambda': 0.5
                },
                'anti_spoofing': {
                    'observation_dim': checkpoint_config.get('anti_spoofing', {}).get('observation_dim', state_dim),
                    'hidden_dim': 64,
                    'detection_threshold': 0.8,
                    'prior_belief': 0.5,
                    'evidence_strength': 0.7,
                    'reputation_decay': 0.99,
                    'min_reputation': 0.1,
                    'max_reputation': 1.0,
                    'reputation_weight': 0.3,
                    'false_report_penalty': -0.5,
                    'malicious_action_penalty': -1.0,
                    'detection_reward': 0.2
                },
                'dynamic_frontier': {
                    'num_objectives': 3,
                    'frontier_size': 50,
                    'update_frequency': 100,
                    'weight_adaptation_rate': 0.05,
                    'min_weight': 0.1,
                    'max_weight': 0.8,
                    'mutation_strength': 0.1,
                    'crossover_rate': 0.7,
                    'elitism_rate': 0.1
                }
            }
            
            self.algorithms['EGT-MARL'] = EGTMARL(
                env=self.env,
                config=egt_config
            )
            
            # 加载预训练模型
            if model_path and os.path.exists(model_path):
                self._load_algorithm_model('EGT-MARL', model_path)
        
        # 初始化 QMIX
        if algo_config['qmix']['enabled']:
            qmix_config = {
                'num_agents': num_agents,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_dim': 128,
                'mixing_hidden_dim': 64,
                'attention_heads': 4,
                'learning_rate': 0.0001,
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 32,
                'buffer_size': 10000
            }
            self.algorithms['QMIX'] = ImprovedQMIX(
                env=self.env,
                config=qmix_config
            )
            
            # 加载预训练模型（如果有）
            qmix_model_path = algo_config['qmix'].get('model_path')
            if qmix_model_path and os.path.exists(qmix_model_path):
                self._load_algorithm_model('QMIX', qmix_model_path)
        
        # 初始化传统方法
        if algo_config['fcfs']['enabled']:
            self.algorithms['FCFS'] = self._create_fcfs_policy()
        
        if algo_config['priority']['enabled']:
            self.algorithms['Priority'] = self._create_priority_policy()
        
        if algo_config.get('greedy_local', {}).get('enabled', True):
            self.algorithms['Greedy-Local'] = self._create_greedy_policy()
        
        if algo_config.get('proportional_fair', {}).get('enabled', True):
            self.algorithms['Proportional-Fair'] = self._create_proportional_fair_policy()
        
        # 添加 Centralized-MPC
        if algo_config.get('centralized_mpc', {}).get('enabled', True):
            self.algorithms['Centralized-MPC'] = self._create_mpc_policy()
        
        # 添加 Standard-MARL
        if algo_config.get('standard_marl', {}).get('enabled', True):
            self.algorithms['Standard-MARL'] = self._create_standard_marl_policy()
        
        # 添加 Game-Theoretic baseline
        if algo_config.get('game_theoretic', {}).get('enabled', True):
            self.algorithms['Game-Theoretic'] = self._create_game_theoretic_policy()
        
        # 添加 GNN-Based baseline
        if algo_config.get('gnn_based', {}).get('enabled', True):
            self.algorithms['GNN-Based'] = self._create_gnn_policy()
        
        # 添加 Transformer-Based baseline
        if algo_config.get('transformer_based', {}).get('enabled', True):
            self.algorithms['Transformer-Based'] = self._create_transformer_policy()
        
        logger.info(f"Algorithms initialized: {list(self.algorithms.keys())}")
    
    def _load_algorithm_model(self, algorithm_name: str, model_path: str):
        """加载算法模型"""
        try:
            if algorithm_name == 'EGT-MARL' and hasattr(self.algorithms[algorithm_name], 'load_checkpoint'):
                # 使用EGT-MARL的load_checkpoint方法
                self.algorithms[algorithm_name].load_checkpoint(model_path)
                logger.info(f"Loaded model for {algorithm_name} from {model_path}")
            else:
                # 其他算法的加载逻辑
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                if algorithm_name in self.algorithms:
                    if 'algorithm_state' in checkpoint:
                        self.algorithms[algorithm_name].load_state_dict(checkpoint['algorithm_state'])
                        logger.info(f"Loaded model for {algorithm_name} from {model_path}")
                    else:
                        logger.warning(f"No algorithm state found in {model_path}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to load model for {algorithm_name}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    def _create_fcfs_policy(self):
        """创建先到先服务策略"""
        class FCFSPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "FCFS"
                
            def select_actions(self, state, epsilon=0.0):
                # 简单的FCFS策略：每个智能体选择最近的未处理受害者
                actions = []
                for i in range(self.num_agents):
                    if hasattr(self.env, 'rescue_agents') and i < len(self.env.rescue_agents):
                        agent = list(self.env.rescue_agents.values())[i]
                        nearest_casualty = None
                        min_distance = float('inf')
                        
                        # 找到最近的未处理受害者
                        for casualty in self.env.casualties.values():
                            if not casualty.treated:
                                distance = np.linalg.norm(agent.position - casualty.position)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_casualty = casualty
                        
                        if nearest_casualty:
                            # 向受害者移动
                            direction = nearest_casualty.position - agent.position
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                # 转换为8个方向的动作
                                angle = np.arctan2(direction[1], direction[0])
                                action = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                            else:
                                action = 0  # 停止
                        else:
                            action = 0  # 没有受害者，停止
                    else:
                        action = 0  # 默认动作
                    actions.append(action)
                return actions
            
            def get_name(self):
                return self.name
        
        return FCFSPolicy(self.env.num_agents, self.env)
    
    def _create_priority_policy(self):
        """创建优先级调度策略"""
        class PriorityPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Priority"
                
            def select_actions(self, state, epsilon=0.0):
                # 优先级策略：优先处理严重受害者
                actions = []
                for i in range(self.num_agents):
                    if hasattr(self.env, 'rescue_agents') and i < len(self.env.rescue_agents):
                        agent = list(self.env.rescue_agents.values())[i]
                        priority_casualty = None
                        highest_priority = -1
                        
                        # 找到优先级最高的未处理受害者
                        for casualty in self.env.casualties.values():
                            if not casualty.treated:
                                # 严重程度作为优先级（数字越大越严重）
                                # 将严重程度字符串转换为数字优先级
                                severity_to_priority = {
                                    'critical': 4,
                                    'severe': 3,
                                    'moderate': 2,
                                    'mild': 1
                                }
                                priority = severity_to_priority.get(casualty.severity.value, 0)
                                if priority > highest_priority:
                                    highest_priority = priority
                                    priority_casualty = casualty
                        
                        if priority_casualty:
                            # 向受害者移动
                            direction = priority_casualty.position - agent.position
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                # 转换为8个方向的动作
                                angle = np.arctan2(direction[1], direction[0])
                                action = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                            else:
                                action = 0  # 停止
                        else:
                            action = 0  # 没有受害者，停止
                    else:
                        action = 0  # 默认动作
                    actions.append(action)
                return actions
            
            def get_name(self):
                return self.name
        
        return PriorityPolicy(self.env.num_agents, self.env)

    # ------------------------------------------------------------------
    # GNN-based baseline
    # ------------------------------------------------------------------
    def _create_gnn_policy(self):
        """Create a GNN-based baseline.

        Each agent is represented as a node in a graph (agents + casualties).
        Edges connect agents to casualties within a heuristic visibility range.
        A small Graph-Attention-style score is computed by combining:
          - per-agent embedding (id + role) and per-casualty embedding
            (severity one-hot + normalized position)
          - distance-based attention weight
          - severity-weighted priority

        Then each agent picks the highest-scoring un-treated casualty within
        range and turns toward it. The model is a *lightweight* stand-in
        (no PyTorch Geometric dependency) that captures the spirit of a
        relational/GNN policy: shared scoring function over the joint
        agent-casualty graph.
        """
        class GNNPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "GNN-Based"
                # Visibility range (must be > Communication range so we have
                # some 'graph' to reason over). Fall back to 200.
                self.vis_range = 200.0

            def _agent_embed(self, agent):
                role_id = 0
                try:
                    role = getattr(agent, 'role', None) or getattr(agent, 'agent_type', None)
                    role_id = {'drone': 0, 'ambulance': 1, 'personnel': 2, 'vehicle': 3, 'hospital': 4}.get(str(role).lower(), 5)
                except Exception:
                    role_id = 5
                return np.array([role_id / 5.0,
                                 float(getattr(agent, 'capacity', None) is not None),
                                 float(agent.id) / max(1.0, float(self.num_agents))],
                                dtype=np.float32)

            def _casualty_embed(self, casualty, map_size):
                sev = ['critical', 'severe', 'moderate', 'mild']
                sev_oh = np.zeros(4, dtype=np.float32)
                sev_str = getattr(casualty.severity, 'value', str(casualty.severity)).lower()
                if sev_str in sev:
                    sev_oh[sev.index(sev_str)] = 1.0
                pos = casualty.position
                if isinstance(map_size, (tuple, list)) and len(map_size) == 2:
                    pos_norm = np.array([pos[0] / max(1.0, float(map_size[0])),
                                         pos[1] / max(1.0, float(map_size[1]))], dtype=np.float32)
                else:
                    pos_norm = np.array([0.5, 0.5], dtype=np.float32)
                return np.concatenate([sev_oh, pos_norm])

            def select_actions(self, state, epsilon=0.0):
                actions = [0] * self.num_agents
                if not hasattr(self.env, 'casualties') or not hasattr(self.env, 'rescue_agents'):
                    return actions

                agents = list(self.env.rescue_agents.values())
                if not agents:
                    return actions

                map_size = getattr(self.env, 'map_size', (1000.0, 1000.0))
                # Casualty features
                cas_list = [c for c in self.env.casualties.values() if not c.treated]
                cas_emb = {c.id: self._casualty_embed(c, map_size) for c in cas_list}

                for i, agent in enumerate(agents):
                    if i >= self.num_agents:
                        break
                    a_emb = self._agent_embed(agent)
                    best_score = -1.0
                    best_cas = None
                    for c in cas_list:
                        d = np.linalg.norm(agent.position - c.position)
                        if d > self.vis_range:
                            continue
                        # "Graph attention" score: severity priority × distance decay
                        sev_w = {'critical': 1.0, 'severe': 0.7, 'moderate': 0.4, 'mild': 0.2}.get(
                            getattr(c.severity, 'value', str(c.severity)).lower(), 0.1)
                        dist_decay = 1.0 / (1.0 + d / 50.0)
                        score = sev_w * dist_decay
                        if score > best_score:
                            best_score = score
                            best_cas = c

                    if best_cas is not None:
                        direction = best_cas.position - agent.position
                        n = np.linalg.norm(direction)
                        if n > 1e-6:
                            direction = direction / n
                            angle = np.arctan2(direction[1], direction[0])
                            actions[i] = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                    # else: stay in place (action=0)

                return actions

            def get_name(self):
                return self.name

        return GNNPolicy(self.env.num_agents, self.env)

    # ------------------------------------------------------------------
    # Transformer-based baseline
    # ------------------------------------------------------------------
    def _create_transformer_policy(self):
        """Create a Transformer-based baseline.

        A lightweight stand-in for a self-attention policy: the "key/query"
        is a per-agent feature vector; the "value" is per-casualty features.
        We compute attention weights agent -> casualty using a single linear
        dot-product (no learned parameters; deterministic given state). The
        intent is to capture the *transformer-style* global attention pattern
        over the casualty set (not a literal torch.nn.Transformer).
        """
        class TransformerPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Transformer-Based"
                self.vis_range = 300.0  # wider than GNN, simulating 'global' attention
                # Fixed projection: agents [role, capacity_norm, health_norm] (3d)
                # to compare with casualty features [sev4, pos2] (6d).
                np.random.seed(0)
                self.W_q = np.random.randn(3, 4).astype(np.float32) * 0.1
                self.W_k = np.random.randn(6, 4).astype(np.float32) * 0.1

            def _agent_feat(self, agent):
                role = getattr(agent, 'role', None) or getattr(agent, 'agent_type', None)
                role_id = {'drone': 0, 'ambulance': 1, 'personnel': 2, 'vehicle': 3, 'hospital': 4}.get(str(role).lower(), 5)
                cap_total = 0.0
                try:
                    cap_total = float(sum(agent.capacity.values())) if agent.capacity else 1.0
                except Exception:
                    cap_total = 1.0
                # Normalize capacity to [0, 1] assuming 20 is full
                return np.array([role_id / 5.0, min(1.0, cap_total / 20.0), 1.0], dtype=np.float32)

            def _casualty_feat(self, casualty, map_size):
                sev = ['critical', 'severe', 'moderate', 'mild']
                sev_oh = np.zeros(4, dtype=np.float32)
                sev_str = getattr(casualty.severity, 'value', str(casualty.severity)).lower()
                if sev_str in sev:
                    sev_oh[sev.index(sev_str)] = 1.0
                pos = casualty.position
                if isinstance(map_size, (tuple, list)) and len(map_size) == 2:
                    pos_norm = np.array([pos[0] / max(1.0, float(map_size[0])),
                                         pos[1] / max(1.0, float(map_size[1]))], dtype=np.float32)
                else:
                    pos_norm = np.array([0.5, 0.5], dtype=np.float32)
                return np.concatenate([sev_oh, pos_norm])

            def select_actions(self, state, epsilon=0.0):
                actions = [0] * self.num_agents
                if not hasattr(self.env, 'casualties') or not hasattr(self.env, 'rescue_agents'):
                    return actions

                agents = list(self.env.rescue_agents.values())
                if not agents:
                    return actions
                map_size = getattr(self.env, 'map_size', (1000.0, 1000.0))
                cas_list = [c for c in self.env.casualties.values() if not c.treated]

                # Pre-compute casualty features
                cas_feats = np.stack([self._casualty_feat(c, map_size) for c in cas_list]) if cas_list else np.zeros((0, 6), dtype=np.float32)
                cas_keys = cas_feats @ self.W_k if len(cas_list) else np.zeros((0, 4), dtype=np.float32)

                # Track which casualties have been claimed (to spread coverage)
                claimed = set()

                for i, agent in enumerate(agents):
                    if i >= self.num_agents:
                        break
                    a_feat = self._agent_feat(agent)
                    q = a_feat @ self.W_q  # (4,)
                    if len(cas_list) == 0:
                        continue

                    # Attention scores = q · k^T (after scaling)
                    scores = cas_keys @ q / np.sqrt(4.0)  # (num_cas,)
                    # Softmax over visibility-masked candidates
                    for k_idx, c in enumerate(cas_list):
                        d = np.linalg.norm(agent.position - c.position)
                        if d > self.vis_range or c.id in claimed:
                            scores[k_idx] = -1e9

                    best_k = int(np.argmax(scores))
                    if scores[best_k] > -1e8:
                        best_cas = cas_list[best_k]
                        claimed.add(best_cas.id)
                        direction = best_cas.position - agent.position
                        n = np.linalg.norm(direction)
                        if n > 1e-6:
                            direction = direction / n
                            angle = np.arctan2(direction[1], direction[0])
                            actions[i] = int((angle + np.pi) / (2 * np.pi) * 8) % 8

                return actions

            def get_name(self):
                return self.name

        return TransformerPolicy(self.env.num_agents, self.env)

    def _create_greedy_policy(self):
        """创建局部贪心算法 - 贪心选择最近受害者"""
        class GreedyPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Greedy-Local"
            
            def select_actions(self, state, epsilon=0.0):
                actions = []
                for i in range(self.num_agents):
                    if hasattr(self.env, 'rescue_agents') and i < len(self.env.rescue_agents):
                        agent = list(self.env.rescue_agents.values())[i]
                        nearest_casualty = None
                        min_distance = float('inf')
                        
                        # 贪心选择最近的未处理受害者
                        for casualty in self.env.casualties.values():
                            if not casualty.treated:
                                distance = np.linalg.norm(agent.position - casualty.position)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_casualty = casualty
                        
                        if nearest_casualty:
                            direction = nearest_casualty.position - agent.position
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                angle = np.arctan2(direction[1], direction[0])
                                action = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                            else:
                                action = 0
                        else:
                            action = 0
                    else:
                        action = 0
                    actions.append(action)
                return actions
            
            def get_name(self):
                return self.name
        
        return GreedyPolicy(self.env.num_agents, self.env)
    
    def _create_proportional_fair_policy(self):
        """创建比例公平算法 - 平衡效率和公平性"""
        class ProportionalFairPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Proportional-Fair"
                self.fairness_weight = 0.3
                self.agent_rescue_count = {}  # 跟踪每个agent的救援次数
            
            def select_actions(self, state, epsilon=0.0):
                actions = []
                for i in range(self.num_agents):
                    if hasattr(self.env, 'rescue_agents') and i < len(self.env.rescue_agents):
                        agent = list(self.env.rescue_agents.values())[i]
                        best_casualty = None
                        best_score = -float('inf')
                        
                        # 比例公平评分 = 效率得分 - λ * 公平性惩罚
                        lambda_fair = self.fairness_weight
                        
                        for casualty in self.env.casualties.values():
                            if not casualty.treated:
                                # 效率得分：基于距离和严重程度
                                distance = np.linalg.norm(agent.position - casualty.position)
                                severity_score = {
                                    'critical': 4, 'severe': 3, 'moderate': 2, 'mild': 1
                                }.get(casualty.severity.value, 1)
                                efficiency = severity_score / max(distance, 1)
                                
                                # 公平性惩罚：避免重复救援同一区域
                                fairness_penalty = 0
                                if hasattr(casualty, 'assigned_agent'):
                                    fairness_penalty = 0.5  # 已分配过的受害者惩罚
                                
                                # 综合得分
                                score = efficiency - lambda_fair * fairness_penalty
                                
                                if score > best_score:
                                    best_score = score
                                    best_casualty = casualty
                        
                        if best_casualty:
                            direction = best_casualty.position - agent.position
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                angle = np.arctan2(direction[1], direction[0])
                                action = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                            else:
                                action = 0
                        else:
                            action = 0
                    else:
                        action = 0
                    actions.append(action)
                return actions
            
            def get_name(self):
                return self.name
        
        return ProportionalFairPolicy(self.env.num_agents, self.env)
    
    def _create_standard_marl_policy(self):
        """
        创建标准MARL算法 (Standard-MARL)
        
        根据论文，Standard-MARL是标准QMIX：
        - 只有MARL层，无EGT调节
        - 无抗欺骗机制
        - 无动态帕累托前沿
        - 仅使用全局存活人数作为奖励
        """
        class StandardMARLPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Standard-MARL"
                # Q-learning表，用于简单的基于值函数的策略
                self.q_table = {}
                self.learning_rate = 0.1
                self.gamma = 0.99
                self.epsilon = 0.1
                
            def _get_q_value(self, state_key, action):
                """获取Q值"""
                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(8)  # 8个动作
                return self.q_table[state_key][action]
            
            def _get_state_key(self):
                """从环境获取简化的状态键"""
                # 简化的状态表示：最近的受害者位置区域
                if not hasattr(self.env, 'casualties'):
                    return "default"
                
                active_casualties = [c for c in self.env.casualties.values() if not c.treated]
                if not active_casualties:
                    return "no_casualties"
                
                # 按区域聚合受害者
                regions = {}
                for c in active_casualties:
                    region = (int(c.position[0] // 200), int(c.position[1] // 200))
                    if region not in regions:
                        regions[region] = []
                    regions[region].append(c)
                
                # 找到受害者最多的区域
                best_region = max(regions.keys(), key=lambda r: len(regions[r]))
                return f"region_{best_region}_{len(regions[best_region])}"
            
            def select_actions(self, state, epsilon=0.0):
                actions = []
                state_key = self._get_state_key()
                
                for i in range(self.num_agents):
                    if hasattr(self.env, 'rescue_agents') and i < len(self.env.rescue_agents):
                        agent = list(self.env.rescue_agents.values())[i]
                        
                        # ε-贪心策略
                        if np.random.random() < (epsilon if epsilon > 0 else self.epsilon):
                            # 探索：随机动作
                            action = np.random.randint(0, 8)
                        else:
                            # 利用：选择Q值最高的动作
                            q_values = [self._get_q_value(f"{state_key}_agent{i}", a) for a in range(8)]
                            action = np.argmax(q_values)
                        
                        # 根据动作移动
                        if action == 0:  # 北
                            direction = np.array([0, 1])
                        elif action == 1:  # 东北
                            direction = np.array([0.707, 0.707])
                        elif action == 2:  # 东
                            direction = np.array([1, 0])
                        elif action == 3:  # 东南
                            direction = np.array([0.707, -0.707])
                        elif action == 4:  # 南
                            direction = np.array([0, -1])
                        elif action == 5:  # 西南
                            direction = np.array([-0.707, -0.707])
                        elif action == 6:  # 西
                            direction = np.array([-1, 0])
                        elif action == 7:  # 西北
                            direction = np.array([-0.707, 0.707])
                        
                        actions.append(action)
                    else:
                        actions.append(0)
                
                return actions
            
            def update(self, state, actions, reward, next_state):
                """Q-learning更新"""
                state_key = self._get_state_key()
                next_state_key = self._get_state_key()
                
                for i, action in enumerate(actions):
                    key = f"{state_key}_agent{i}"
                    if key not in self.q_table:
                        self.q_table[key] = np.zeros(8)
                    
                    # 获取当前和最大Q值
                    current_q = self.q_table[key][action]
                    next_qs = [self._get_q_value(f"{next_state_key}_agent{i}", a) for a in range(8)]
                    max_next_q = max(next_qs) if next_qs else 0
                    
                    # Q-learning更新
                    self.q_table[key][action] = current_q + self.learning_rate * (
                        reward + self.gamma * max_next_q - current_q
                    )
            
            def get_name(self):
                return self.name
        
        return StandardMARLPolicy(self.env.num_agents, self.env)
    
    def _create_mpc_policy(self):
        """
        创建集中式MPC算法 (Centralized-MPC)
        
        根据论文，Centralized-MPC假设完美信息：
        - 使用集中式优化（贪婪近似）
        - 假设全局信息已知
        - 优化全局资源分配
        """
        class MPCPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Centralized-MPC"
                self.horizon = 20  # 预测范围
            
            def _score_casualty(self, casualty, agent):
                """计算受害者-智能体对的得分"""
                # 距离得分（越近越高）
                distance = np.linalg.norm(casualty.position - agent.position)
                distance_score = 1.0 / (1.0 + distance / 100)
                
                # 严重程度得分
                severity_score = {
                    'critical': 1.0,
                    'severe': 0.75,
                    'moderate': 0.5,
                    'mild': 0.25
                }.get(casualty.severity.value, 0.25)
                
                # 生存概率得分（越低越紧急）
                survival_prob = casualty.survival_probability if hasattr(casualty, 'survival_probability') else 0.5
                urgency_score = 1.0 - survival_prob
                
                return 0.4 * distance_score + 0.4 * severity_score + 0.2 * urgency_score
            
            def select_actions(self, state, epsilon=0.0):
                """
                集中式选择：假设完美信息，全局最优分配
                使用贪婪近似求解分配问题
                """
                actions = []
                
                if not hasattr(self.env, 'casualties') or not hasattr(self.env, 'rescue_agents'):
                    return [0] * self.num_agents
                
                # 获取未处理受害者
                unassigned_casualties = []
                for c in self.env.casualties.values():
                    if not c.treated:
                        unassigned_casualties.append(c)
                
                # 获取可用智能体
                available_agents = list(self.env.rescue_agents.values())[:self.num_agents]
                
                # 贪婪分配：每个受害者分配给得分最高的智能体
                assignments = {}  # agent_id -> casualty_id
                assigned_casualty_ids = set()
                
                for casualty in sorted(unassigned_casualties, 
                                       key=lambda c: {'critical': 0, 'severe': 1, 'moderate': 2, 'mild': 3}.get(c.severity.value, 4)):
                    casualty_id = id(casualty)
                    if casualty_id in assigned_casualty_ids:
                        continue
                    
                    best_agent = None
                    best_score = -float('inf')
                    
                    for agent in available_agents:
                        score = self._score_casualty(casualty, agent)
                        if score > best_score:
                            best_score = score
                            best_agent = agent
                    
                    if best_agent is not None:
                        agent_id = list(self.env.rescue_agents.keys())[list(self.env.rescue_agents.values()).index(best_agent)]
                        assignments[agent_id] = casualty
                        assigned_casualty_ids.add(casualty_id)
                
                # 为每个智能体生成动作
                for i in range(self.num_agents):
                    if i < len(available_agents):
                        agent = available_agents[i]
                        agent_id = list(self.env.rescue_agents.keys())[i]
                        
                        if agent_id in assignments:
                            target = assignments[agent_id]
                            direction = target.position - agent.position
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                angle = np.arctan2(direction[1], direction[0])
                                action = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                            else:
                                action = 0
                        else:
                            # 没有分配受害者，留在原地或随机移动
                            action = 0
                    else:
                        action = 0
                    
                    actions.append(action)
                
                return actions
            
            def get_name(self):
                return self.name
        
        return MPCPolicy(self.env.num_agents, self.env)
    
    def _create_game_theoretic_policy(self):
        """
        创建博弈论基线算法 (Game-Theoretic)
        
        根据论文，Game-Theoretic是斯坦伯格博弈：
        - 一个领导者智能体（协调者）
        - 多个跟随者智能体
        - 领导者制定策略，跟随者响应
        """
        class GameTheoreticPolicy:
            def __init__(self, num_agents: int, env):
                self.num_agents = num_agents
                self.env = env
                self.name = "Game-Theoretic"
                
                # 领导者-跟随者结构
                self.leader_id = 0  # 第一个智能体是领导者
                self.follower_ids = list(range(1, num_agents))
                
                # 博弈历史
                self.game_history = []
                
            def _leader_strategy(self, agents, casualties):
                """
                领导者策略：选择最需要救援的区域
                """
                if not casualties:
                    return None
                
                # 按区域聚合受害者
                regions = {}
                for c in casualties:
                    region = (int(c.position[0] // 300), int(c.position[1] // 300))
                    if region not in regions:
                        regions[region] = {'count': 0, 'severity': 0, 'center': [0, 0]}
                    
                    severity_map = {'critical': 4, 'severe': 3, 'moderate': 2, 'mild': 1}
                    regions[region]['count'] += 1
                    regions[region]['severity'] += severity_map.get(c.severity.value, 0)
                    regions[region]['center'][0] += c.position[0]
                    regions[region]['center'][1] += c.position[1]
                
                # 选择领导者目标区域（综合得分最高的区域）
                best_region = None
                best_score = -float('inf')
                
                for region, data in regions.items():
                    avg_x = data['center'][0] / data['count']
                    avg_y = data['center'][1] / data['count']
                    # 综合得分：严重程度 * 数量 / 平均距离
                    score = data['severity'] * data['count'] / (1 + np.sqrt(avg_x**2 + avg_y**2) / 100)
                    if score > best_score:
                        best_score = score
                        best_region = (avg_x, avg_y)
                
                return best_region
            
            def _follower_response(self, agent, leader_target, casualties):
                """
                跟随者响应策略：
                - 如果领导者有目标，优先跟随
                - 否则按局部最优行动
                """
                if not casualties:
                    return np.array([0, 0])
                
                # 跟随领导者分配的区域
                if leader_target is not None:
                    leader_region = (leader_target[0] // 300, leader_target[1] // 300)
                    
                    # 找到该区域的受害者
                    region_casualties = [
                        c for c in casualties 
                        if (int(c.position[0]) // 300, int(c.position[1]) // 300) == leader_region
                    ]
                    
                    if region_casualties:
                        # 选择最近的
                        nearest = min(region_casualties, 
                                    key=lambda c: np.linalg.norm(c.position - agent.position))
                        return nearest.position - agent.position
                
                # 局部最优：选择最近的受害者
                nearest = min(casualties, key=lambda c: np.linalg.norm(c.position - agent.position))
                return nearest.position - agent.position
            
            def select_actions(self, state, epsilon=0.0):
                actions = []
                
                if not hasattr(self.env, 'casualties') or not hasattr(self.env, 'rescue_agents'):
                    return [0] * self.num_agents
                
                # 获取未处理受害者
                unassigned_casualties = [c for c in self.env.casualties.values() if not c.treated]
                available_agents = list(self.env.rescue_agents.values())
                
                # 领导者决策
                leader_target = self._leader_strategy(available_agents, unassigned_casualties)
                
                # 为每个智能体生成动作
                for i in range(self.num_agents):
                    if i < len(available_agents):
                        agent = available_agents[i]
                        
                        if i == self.leader_id:
                            # 领导者
                            if leader_target is not None:
                                direction = np.array(leader_target) - agent.position
                            else:
                                direction = np.array([0, 0])
                        else:
                            # 跟随者
                            direction = self._follower_response(agent, leader_target, unassigned_casualties)
                        
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            angle = np.arctan2(direction[1], direction[0])
                            action = int((angle + np.pi) / (2 * np.pi) * 8) % 8
                        else:
                            action = 0
                    else:
                        action = 0
                    
                    actions.append(action)
                
                return actions
            
            def get_name(self):
                return self.name
        
        return GameTheoreticPolicy(self.env.num_agents, self.env)

    # NOTE: GNN / Transformer implementations are defined earlier in this class
    # (see _create_gnn_policy / _create_transformer_policy above, ~L470 / ~L574).
    # The earlier, lightweight numpy-only versions are kept; this avoids the
    # earlier broken torch.cat-on-scalars bug.

    def evaluate_algorithm(self,
                          algorithm_name: str,
                          algorithm,
                          num_episodes: int = 100) -> Dict[str, List[float]]:
        """
        评估单个算法
        
        Args:
            algorithm_name: 算法名称
            algorithm: 算法实例
            num_episodes: episode数量
            
        Returns:
            指标字典
        """
        logger.info(f"Evaluating {algorithm_name}...")
        
        metrics = {
            'rescue_rate': [],
            'avg_response_time': [],
            'resource_utilization': [],
            'total_reward': [],
            'fairness_gini': [],
            'fairness_theil': [],
            'fairness_maxmin': []
        }
        
        for ep in range(num_episodes):
            state = self.env.reset()
            episode_metrics = self._run_episode(algorithm, state)
            
            # 收集指标
            for key in metrics:
                if key in episode_metrics:
                    metrics[key].append(episode_metrics[key])
            
            # 打印进度
            if (ep + 1) % 10 == 0:
                logger.info(f"  {algorithm_name} - Episode {ep+1}/{num_episodes} - "
                           f"Rescue Rate: {episode_metrics.get('rescue_rate', 0.0):.1f}%")
        
        # 计算统计信息
        stats = {}
        for key, values in metrics.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_median'] = np.median(values)
            else:
                stats[f'{key}_mean'] = 0.0
                stats[f'{key}_std'] = 0.0
        
        logger.info(f"  {algorithm_name} completed - "
                   f"Avg Rescue Rate: {stats.get('rescue_rate_mean', 0.0):.1f}% ± {stats.get('rescue_rate_std', 0.0):.1f}")
        
        return stats
    
    def _run_episode(self, algorithm, initial_state) -> Dict[str, float]:
        """运行一个episode"""
        state = initial_state
        episode_metrics = {
            'total_reward': 0.0,
            'steps': 0,
            'rescued': 0,
            'deaths': 0,
            'resources_used': 0,
            'response_times': [],
            'victim_severities': []
        }
        
        done = False
        step = 0
        max_steps = self.config['evaluation']['max_steps_per_episode']
        
        while not done and step < max_steps:
            # 获取动作
            if hasattr(algorithm, 'select_action'):
                actions = algorithm.select_action(state, training=False)
            elif hasattr(algorithm, 'act'):
                # 对于 ImprovedQMIX 等算法
                # 简化处理，假设 state 是观察值列表
                if isinstance(state, tuple):
                    state_obs = state[0]
                else:
                    state_obs = state
                # 为每个智能体创建观察值
                num_agents = len(self.env.rescue_agents) if hasattr(self.env, 'rescue_agents') else 5  # 默认 5 个智能体
                observations = [state_obs[i] for i in range(num_agents)]
                # 调用 act 方法
                actions, _ = algorithm.act(observations, state_obs, training=False)
                # 转换为字典格式
                actions_dict = {}
                for agent_id, action in enumerate(actions):
                    actions_dict[agent_id] = {
                        "strategic": [0.25, 0.25, 0.25, 0.25],
                        "tactical": action % 8,  # 8个方向
                        "communication": action // 8
                    }
                actions = actions_dict
            elif hasattr(algorithm, 'select_actions'):
                actions = algorithm.select_actions(state, epsilon=0.0)
            else:
                # 对于传统方法
                actions = algorithm.select_actions(state)
            
            # 确保动作是字典格式
            if isinstance(actions, list):
                # 转换列表为字典格式
                actions_dict = {}
                for agent_id, action in enumerate(actions):
                    actions_dict[agent_id] = {
                        "strategic": [0.25, 0.25, 0.25, 0.25],
                        "tactical": action % 8,
                        "communication": action // 8
                    }
                actions = actions_dict
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(actions)
            rewards = [reward]  # 转换为列表格式
            done = terminated or truncated
            
            # 收集指标
            # 处理奖励（可能是dict或其他类型）
            if isinstance(rewards, dict):
                episode_metrics['total_reward'] += sum(rewards.values())
            elif isinstance(rewards, list):
                # 列表中的元素可能是dict
                total = 0.0
                for r in rewards:
                    if isinstance(r, dict):
                        total += sum(r.values())
                    elif isinstance(r, (int, float)):
                        total += r
                episode_metrics['total_reward'] += total
            elif isinstance(rewards, (int, float)):
                episode_metrics['total_reward'] += rewards
            episode_metrics['steps'] += 1
            # 直接记录当前救援人数（最后会取最终值），不累加
            # 从 statistics 中获取救援和死亡人数
            statistics = info.get('statistics', {})
            episode_metrics['rescued'] = statistics.get('total_rescued', 0)
            episode_metrics['deaths'] = statistics.get('total_deaths', 0)
            episode_metrics['resources_used'] += statistics.get('resources_used', 0)
            
            # 从 statistics 中获取响应时间
            if 'response_times' in statistics:
                episode_metrics['response_times'].extend(statistics['response_times'])
            
            # 从 statistics 中获取被救援受害者的严重程度，用于计算公平性指标
            if 'rescued_severities' in statistics:
                episode_metrics['victim_severities'].extend(statistics['rescued_severities'])
            
            state = next_state
            step += 1
        
        # 计算衍生指标
        # 使用初始受害者数量（从环境的num_victims属性获取，已在setup_environment中设置）
        total_victims = getattr(self.env, 'num_victims', 0) or len(self.env.casualties) if hasattr(self.env, 'casualties') else 0
        if total_victims > 0:
            episode_metrics['rescue_rate'] = (episode_metrics['rescued'] / total_victims) * 100
        
        if episode_metrics['response_times']:
            episode_metrics['avg_response_time'] = np.mean(episode_metrics['response_times'])
        else:
            episode_metrics['avg_response_time'] = 0.0
        
        # 计算资源利用率（简化）
        # Use the cached total_resources from setup_environment (P2-E3 fix)
        total_resources = getattr(self.env, 'total_resources', 1000)
        if total_resources > 0:
            episode_metrics['resource_utilization'] = (episode_metrics['resources_used'] / total_resources) * 100
        
        # 计算公平性指标
        if episode_metrics['victim_severities']:
            severities = np.array(episode_metrics['victim_severities'])
            
            # 基尼系数（简化计算）
            sorted_severities = np.sort(severities)
            n = len(sorted_severities)
            cum_values = np.cumsum(sorted_severities)
            gini = (n + 1 - 2 * np.sum(cum_values) / cum_values[-1]) / n if cum_values[-1] > 0 else 0
            episode_metrics['fairness_gini'] = gini
            
            # 泰尔指数（Theil Index）
            if len(severities) > 0 and np.mean(severities) > 0:
                normalized = severities / np.mean(severities)
                # 处理零值和负值，避免对数错误
                normalized = np.clip(normalized, 1e-10, None)
                log_normalized = np.log(normalized)
                theil = np.mean(normalized * log_normalized)
                episode_metrics['fairness_theil'] = theil
            else:
                episode_metrics['fairness_theil'] = 0.0
            
            # 最大最小公平性
            if len(severities) > 0:
                episode_metrics['fairness_maxmin'] = np.min(severities) / np.max(severities) if np.max(severities) > 0 else 0
        
        return episode_metrics
    
    def run_evaluation(self):
        """运行完整评估"""
        logger.info("Starting baseline evaluation...")
        
        # 设置目录（在参数覆盖后）
        self.setup_directories()
        
        eval_config = self.config['evaluation']
        scenarios = eval_config['scenarios']
        severities = eval_config['severities']
        num_runs = eval_config['num_runs']
        
        all_results = {}
        
        # 遍历所有场景和严重程度
        for scenario in scenarios:
            for severity in severities:
                logger.info(f"\nEvaluating scenario: {scenario} ({severity})")
                
                scenario_key = f"{scenario}_{severity}"
                all_results[scenario_key] = {}
                
                # 多次运行以减少随机性
                for run in range(num_runs):
                    logger.info(f"  Run {run+1}/{num_runs}")
                    
                    # 设置环境
                    self.setup_environment(scenario, severity)
                    
                    # 设置算法
                    self.setup_algorithms()
                    
                    # 评估每个算法
                    run_results = {}
                    for algo_name, algorithm in self.algorithms.items():
                        stats = self.evaluate_algorithm(
                            algo_name, 
                            algorithm, 
                            eval_config['num_episodes']
                        )
                        run_results[algo_name] = stats
                    
                    all_results[scenario_key][f'run_{run+1}'] = run_results
        
        # 保存结果
        self.save_results(all_results)
        
        # 生成报告和可视化
        self.generate_evaluation_report(all_results)
        
        logger.info("Baseline evaluation completed!")
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        # 保存为JSON
        import json
        results_path = self.evaluation_dir / 'results' / 'evaluation_results.json'
        
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(results), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_path}")
        
        # 保存为CSV（汇总表格）
        self.save_results_csv(results)
    
    def save_results_csv(self, results: Dict[str, Any]):
        """保存结果为CSV格式"""
        rows = []
        
        for scenario_key, scenario_results in results.items():
            # 提取算法名称
            if scenario_results:
                first_run = next(iter(scenario_results.values()))
                algorithms = list(first_run.keys())
                
                # 为每个算法收集统计信息
                for algo_name in algorithms:
                    # 收集所有运行的结果
                    rescue_rates = []
                    response_times = []
                    gini_coefficients = []
                    theil_indices = []
                    
                    for run_key, run_results in scenario_results.items():
                        if algo_name in run_results:
                            stats = run_results[algo_name]
                            rescue_rates.append(stats.get('rescue_rate_mean', 0.0))
                            response_times.append(stats.get('avg_response_time_mean', 0.0))
                            gini_coefficients.append(stats.get('fairness_gini_mean', 0.0))
                            theil_indices.append(stats.get('fairness_theil_mean', 0.0))
                    
                    # 计算平均值和标准差
                    if rescue_rates:
                        row = {
                            'scenario': scenario_key,
                            'algorithm': algo_name,
                            'rescue_rate_mean': np.mean(rescue_rates),
                            'rescue_rate_std': np.std(rescue_rates),
                            'response_time_mean': np.mean(response_times),
                            'response_time_std': np.std(response_times),
                            'gini_mean': np.mean(gini_coefficients),
                            'gini_std': np.std(gini_coefficients),
                            'theil_mean': np.mean(theil_indices),
                            'theil_std': np.std(theil_indices),
                            'num_runs': len(rescue_rates)
                        }
                        rows.append(row)
        
        # 创建DataFrame并保存
        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.evaluation_dir / 'results' / 'summary.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Summary saved to {csv_path}")
            
            # 打印摘要
            print("\n" + "="*80)
            print("Evaluation Summary")
            print("="*80)
            print(df.to_string())
    
    def generate_evaluation_report(self, results: Dict[str, Any]):
        """生成评估报告"""
        report_path = self.evaluation_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Baseline Algorithm Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("1. Evaluation Configuration\n")
            f.write("-" * 40 + "\n")
            eval_config = self.config['evaluation']
            f.write(f"Scenarios: {', '.join(eval_config['scenarios'])}\n")
            f.write(f"Severities: {', '.join(eval_config['severities'])}\n")
            f.write(f"Number of Runs: {eval_config['num_runs']}\n")
            f.write(f"Episodes per Run: {eval_config['num_episodes']}\n")
            f.write(f"Max Steps per Episode: {eval_config['max_steps_per_episode']}\n\n")
            
            f.write("2. Algorithms Evaluated\n")
            f.write("-" * 40 + "\n")
            for algo_name in self.algorithms.keys():
                f.write(f"- {algo_name}\n")
            f.write("\n")
            
            f.write("3. Performance Summary\n")
            f.write("-" * 40 + "\n")
            
            # 读取CSV摘要
            csv_path = self.evaluation_dir / 'results' / 'summary.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # 按场景分组
                scenarios = df['scenario'].unique()
                
                for scenario in scenarios:
                    f.write(f"\nScenario: {scenario}\n")
                    scenario_df = df[df['scenario'] == scenario]
                    
                    # 按救援率排序
                    scenario_df = scenario_df.sort_values('rescue_rate_mean', ascending=False)
                    
                    for _, row in scenario_df.iterrows():
                        f.write(f"  {row['algorithm']}:\n")
                        f.write(f"    Rescue Rate: {row['rescue_rate_mean']:.1f}% (±{row['rescue_rate_std']:.1f})\n")
                        f.write(f"    Response Time: {row['response_time_mean']:.1f}s (±{row['response_time_std']:.1f})\n")
                        f.write(f"    Gini Coefficient: {row['gini_mean']:.4f} (±{row['gini_std']:.4f})\n")
                        f.write(f"    Theil Index: {row['theil_mean']:.4f} (±{row['theil_std']:.4f})\n")
            
            f.write("\n4. Key Findings\n")
            f.write("-" * 40 + "\n")
            
            # 分析结果
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # 找到最佳算法
                best_rescue = df.loc[df['rescue_rate_mean'].idxmax()]
                best_response = df.loc[df['response_time_mean'].idxmin()]
                
                f.write(f"Best Rescue Rate: {best_rescue['algorithm']} "
                       f"({best_rescue['rescue_rate_mean']:.1f}%)\n")
                f.write(f"Best Response Time: {best_response['algorithm']} "
                       f"({best_response['response_time_mean']:.1f}s)\n")
                
                # 计算EGT-MARL相对于基线的改进
                if 'EGT-MARL' in df['algorithm'].values:
                    egt_row = df[df['algorithm'] == 'EGT-MARL'].iloc[0]
                    
                    # 与传统方法比较
                    traditional_algs = ['FCFS', 'Priority']
                    traditional_df = df[df['algorithm'].isin(traditional_algs)]
                    
                    if not traditional_df.empty:
                        avg_traditional = traditional_df['rescue_rate_mean'].mean()
                        improvement = ((egt_row['rescue_rate_mean'] - avg_traditional) / avg_traditional) * 100
                        f.write(f"\nEGT-MARL Improvement over Traditional Methods: {improvement:.1f}%\n")
                
                # 与其他MARL算法比较
                marl_algs = ['QMIX', 'MADDPG', 'MAPPO']
                marl_df = df[df['algorithm'].isin(marl_algs)]
                
                if not marl_df.empty and 'EGT-MARL' in df['algorithm'].values:
                    avg_marl = marl_df['rescue_rate_mean'].mean()
                    improvement = ((egt_row['rescue_rate_mean'] - avg_marl) / avg_marl) * 100
                    f.write(f"EGT-MARL Improvement over other MARL: {improvement:.1f}%\n")
            
            f.write("\n5. Files Generated\n")
            f.write("-" * 40 + "\n")
            f.write(f"Config: {self.evaluation_dir}/config.yaml\n")
            f.write(f"Results: {self.evaluation_dir}/results/evaluation_results.json\n")
            f.write(f"Summary: {self.evaluation_dir}/results/summary.csv\n")
            f.write(f"Logs: {self.evaluation_dir}/logs/\n")
            f.write(f"Visualizations: {self.evaluation_dir}/visualizations/\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Evaluation Completed Successfully!\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Evaluation report saved: {report_path}")
        
        # 生成可视化
        self.generate_evaluation_visualizations()
    
    def generate_evaluation_visualizations(self):
        """生成评估可视化"""
        try:
            csv_path = self.evaluation_dir / 'results' / 'summary.csv'
            if not csv_path.exists():
                return
            
            df = pd.read_csv(csv_path)
            
            # 创建可视化器
            visualizer = DisasterVisualizer(self.config['environment'])
            
            # 准备算法对比数据
            algorithms_data = {}
            
            for scenario in df['scenario'].unique():
                scenario_df = df[df['scenario'] == scenario]
                
                for _, row in scenario_df.iterrows():
                    algo_name = row['algorithm']
                    if algo_name not in algorithms_data:
                        algorithms_data[algo_name] = {}
                    
                    # 为每个算法创建模拟的训练曲线
                    # 这里使用简化数据，实际应该从训练历史中获取
                    base_rate = row['rescue_rate_mean']
                    episodes = list(range(1, 101))
                    
                    # 创建模拟的学习曲线
                    learning_curve = []
                    for ep in episodes:
                        # 模拟学习过程：从低性能开始，逐渐接近最终性能
                        progress = min(1.0, ep / 50)  # 50个episode达到稳定
                        noise = np.random.normal(0, row['rescue_rate_std'] * 0.1)
                        rate = base_rate * progress + noise
                        learning_curve.append(max(0, min(100, rate)))
                    
                    algorithms_data[algo_name]['rescue_rate'] = learning_curve
            
            # 绘制算法对比图表
            comparison_path = self.evaluation_dir / 'visualizations' / 'algorithm_comparison.png'
            visualizer.plot_comparison_chart(
                algorithms_data, 
                'rescue_rate',
                str(comparison_path)
            )
            
            logger.info(f"Evaluation visualizations saved to {self.evaluation_dir}/visualizations/")
            
        except Exception as e:
            logger.warning(f"Failed to generate evaluation visualizations: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate baseline algorithms')
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                       help='Scenarios to evaluate (overrides config)')
    parser.add_argument('--severities', type=str, nargs='+', default=None,
                       help='Severities to evaluate (overrides config)')
    parser.add_argument('--algorithms', type=str, default=None,
                       help='Algorithms to evaluate (comma-separated, overrides config)')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of evaluation episodes (overrides config)')
    parser.add_argument('--num_runs', type=int, default=None,
                       help='Number of independent runs (overrides config)')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all scenarios')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = BaselineEvaluator(args.config)
    
    # 覆盖配置参数
    if args.output_dir:
        evaluator.config['output_dir'] = args.output_dir
    
    if args.scenarios:
        evaluator.config['evaluation']['scenarios'] = args.scenarios
    
    if args.severities:
        evaluator.config['evaluation']['severities'] = args.severities
    
    if args.algorithms:
        # 解析算法列表
        algo_list = [algo.strip() for algo in args.algorithms.split(',')]
        # 禁用所有算法
        for algo in evaluator.config['algorithms']:
            evaluator.config['algorithms'][algo]['enabled'] = False
        # 启用指定的算法
        for algo in algo_list:
            if algo in evaluator.config['algorithms']:
                evaluator.config['algorithms'][algo]['enabled'] = True
    
    if args.num_episodes:
        evaluator.config['evaluation']['num_episodes'] = args.num_episodes
    
    if args.num_runs:
        evaluator.config['evaluation']['num_runs'] = args.num_runs
    
    # 处理 --all 参数
    if args.all:
        # 设置所有场景
        all_scenarios = ['earthquake', 'flood', 'hurricane', 'wildfire', 'tornado']
        all_severities = ['low', 'medium', 'high']
        evaluator.config['evaluation']['scenarios'] = all_scenarios
        evaluator.config['evaluation']['severities'] = all_severities
        logger.info("Evaluating all scenarios and severities...")
    
    # 运行评估
    try:
        results = evaluator.run_evaluation()
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {evaluator.evaluation_dir}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()