"""
ConfigMerger —— 把 defaults.py / yaml / 命令行合并成最终 TrainingConfig。

依据：设计文档 §8.4。

设计：
1. 优先级（高→低）：CLI > YAML > defaults
2. 用 dataclass 而非 dict，IDE 友好 + 静态检查
3. 加载后立即调用 __post_init__ 做合理性检查
4. 不引入 Hydra 之类的重型依赖（保持零外部）
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple

import yaml


# ============== defaults 模块级常量 ==============

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 32
DEFAULT_BUFFER_SIZE = 10000
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_EPISODES = 1000
DEFAULT_MAX_STEPS_PER_EP = 800
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY_STEPS = 5000
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_SEED = 42


# ============== 配置 dataclass ==============

@dataclass
class TrainingConfig:
    """所有可调超参 + 环境参数。"""
    # 训练循环
    seed: int = DEFAULT_SEED
    num_episodes: int = DEFAULT_NUM_EPISODES
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EP

    # 算法
    learning_rate: float = DEFAULT_LEARNING_RATE
    gamma: float = DEFAULT_GAMMA
    batch_size: int = DEFAULT_BATCH_SIZE
    buffer_size: int = DEFAULT_BUFFER_SIZE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    eps_start: float = DEFAULT_EPS_START
    eps_end: float = DEFAULT_EPS_END
    eps_decay_steps: int = DEFAULT_EPS_DECAY_STEPS
    grad_clip: float = DEFAULT_GRAD_CLIP
    # P2-修复6：Double DQN target_network 硬同步间隔（步数，默认 100）
    target_update_interval: int = 100

    # EGT
    ema_alpha: float = 0.1
    lambda_anchor: float = 0.5
    lambda_anchor_blend: float = 0.0

    # 环境
    num_agents: int = 10
    num_casualties: int = 50
    num_depots: int = 3
    map_size: Tuple[int, int] = (30, 30)
    disaster_severity: str = "medium"
    malicious_ratio: float = 0.0

    # 输出
    save_dir: str = "./checkpoints"
    log_every: int = 10
    eval_every: int = 100

    # 课程（简单版：4 阶段）
    curriculum: list = field(default_factory=lambda: [
        {"end_episode": 200, "num_agents": 5, "num_casualties": 20},
        {"end_episode": 500, "num_agents": 8, "num_casualties": 30},
        {"end_episode": 800, "num_agents": 10, "num_casualties": 50},
        {"end_episode": 1000, "num_agents": 12, "num_casualties": 80},
    ])

    def __post_init__(self):
        assert self.num_episodes > 0
        assert 0.0 < self.learning_rate < 1.0
        assert 0.0 < self.gamma <= 1.0
        assert self.batch_size > 0
        assert self.hidden_dim > 0
        assert 0.0 <= self.eps_end <= self.eps_start <= 1.0
        assert self.disaster_severity in ("low", "medium", "high")
        assert self.map_size[0] > 0 and self.map_size[1] > 0
        assert 0.0 <= self.malicious_ratio <= 1.0


# ============== Merger ==============

class ConfigMerger:
    """YAML + CLI + defaults 合并。"""

    def __init__(
        self,
        defaults: Optional[Dict[str, Any]] = None,
        yaml_path: Optional[str] = None,
        cli_args: Optional[argparse.Namespace] = None,
    ):
        self.defaults = defaults or {}
        self.yaml_path = yaml_path
        self.cli_args = cli_args

    def merge(self) -> TrainingConfig:
        # 1. 从 defaults 起步
        merged: Dict[str, Any] = dict(self.defaults)

        # 2. YAML 覆盖（若提供）
        if self.yaml_path and os.path.exists(self.yaml_path):
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f) or {}
            merged.update(yaml_cfg)

        # 3. CLI 覆盖
        if self.cli_args is not None:
            cli_dict = {k: v for k, v in vars(self.cli_args).items() if v is not None}
            merged.update(cli_dict)

        # 4. 构造 dataclass（__post_init__ 自动断言）
        return TrainingConfig(**merged)

    @staticmethod
    def make_argparser() -> argparse.ArgumentParser:
        """生成可与 ConfigMerger 配合的 CLI parser。"""
        p = argparse.ArgumentParser()
        # 顶层常用项
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--num_episodes", type=int, default=None)
        p.add_argument("--learning_rate", type=float, default=None)
        p.add_argument("--gamma", type=float, default=None)
        p.add_argument("--batch_size", type=int, default=None)
        p.add_argument("--hidden_dim", type=int, default=None)
        p.add_argument("--num_agents", type=int, default=None)
        p.add_argument("--num_casualties", type=int, default=None)
        p.add_argument("--disaster_severity", type=str, default=None)
        p.add_argument("--malicious_ratio", type=float, default=None)
        p.add_argument("--save_dir", type=str, default=None)
        p.add_argument("--yaml", type=str, default=None,
                       help="可选 YAML 配置文件路径")
        return p


# ============== 便捷 CLI ==============

def load_config_from_cli(argv: Optional[list] = None) -> Tuple[TrainingConfig, argparse.Namespace]:
    """从命令行解析并构造 TrainingConfig。"""
    parser = ConfigMerger.make_argparser()
    args = parser.parse_args(argv)
    yaml_path = getattr(args, "yaml", None)
    # 如果指定了 --yaml，把它从 args 中拿掉再 merge
    cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "yaml"}
    cli_ns = argparse.Namespace(**cli_dict)
    merger = ConfigMerger(yaml_path=yaml_path, cli_args=cli_ns)
    cfg = merger.merge()
    return cfg, args


__all__ = [
    "TrainingConfig", "ConfigMerger", "load_config_from_cli",
    "DEFAULT_LEARNING_RATE", "DEFAULT_GAMMA", "DEFAULT_BATCH_SIZE",
    "DEFAULT_HIDDEN_DIM", "DEFAULT_NUM_EPISODES",
    "DEFAULT_MAX_STEPS_PER_EP", "DEFAULT_EPS_START", "DEFAULT_EPS_END",
    "DEFAULT_EPS_DECAY_STEPS", "DEFAULT_GRAD_CLIP", "DEFAULT_SEED",
]