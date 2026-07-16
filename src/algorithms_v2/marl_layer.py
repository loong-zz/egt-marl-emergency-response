"""
MARL Layer (v2) —— 简化版 QMIX-like 网络。

依据：设计文档 §3.3（46→12 → 决策）、§5.2（per-agent Q 网络）。

设计：
1. 每个 agent 一个 DRQN-like 网络：
   - 输入：obs_dim（= DEFAULT_OBS_SPEC.dim = 52）
   - 输出：action_dim（= ACTION_DIM = 12）
2. Greedy / ε-greedy 动作选择
3. 不实现真正的 mixing network（设计文档未要求）；
   训练时通过总奖励做 per-agent credit assignment（简化）
4. 接口暴露 select_actions(obs_dict, eps=...) 和 loss(...) 框架

接口：
    m = MARLLayer(obs_dim=52, action_dim=12, num_agents=10)
    actions = m.select_actions(obs_dict, eps=0.05)   # {aid: action_idx}
    loss = m.compute_td_loss(batch)                  # placeholder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MARLConfig:
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.01                # soft update
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 5000
    # P2-修复6：Double DQN target_network 硬同步间隔（步数）
    target_update_interval: int = 100


class QNetwork(nn.Module):
    """单 agent Q 网络：obs → Q(action)。"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MARLLayer:
    """
    简化 MARL：每个 agent 一个 Q 网络，共享架构但独立参数。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        config: Optional[MARLConfig] = None,
        device: str = "cpu",
    ):
        self.cfg = config or MARLConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device(device)

        # 每 agent 一个网络（保持简单）
        self.q_nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim, self.cfg.hidden_dim).to(self.device)
            for _ in range(num_agents)
        ])
        # target 网络（Double DQN：每隔 target_update_interval 步硬同步）
        self.target_nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim, self.cfg.hidden_dim).to(self.device)
            for _ in range(num_agents)
        ])
        for target_net, q_net in zip(self.target_nets, self.q_nets):
            target_net.load_state_dict(q_net.state_dict())
            for p in target_net.parameters():
                p.requires_grad = False

        # target 网络硬同步间隔（步数）
        self.target_update_interval = self.cfg.target_update_interval
        self._steps_since_target_update = 0

        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=self.cfg.learning_rate)
            for net in self.q_nets
        ]

        self.eps = self.cfg.eps_start
        self.train_step = 0

    # ============== 动作选择 ==============

    def select_actions(
        self,
        obs_dict: Dict[int, np.ndarray],
        eps: Optional[float] = None,
        greedy: bool = False,
    ) -> Dict[int, int]:
        """
        Args:
            obs_dict: {agent_id: np.ndarray shape=(obs_dim,)}
            eps: ε-greedy 探索率；None 表示用 self.eps
            greedy: True 时强制 greedy

        Returns:
            {agent_id: action_idx ∈ [0, action_dim)}
        """
        epsilon = 0.0 if greedy else (eps if eps is not None else self.eps)
        actions: Dict[int, int] = {}

        self.q_nets.eval()
        with torch.no_grad():
            for aid, obs in obs_dict.items():
                if 0 <= aid < self.num_agents:
                    x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    q = self.q_nets[aid](x.unsqueeze(0))   # (1, A)
                    if self.rng.random() < epsilon:
                        actions[aid] = int(self.rng.integers(0, self.action_dim))
                    else:
                        actions[aid] = int(q.argmax(dim=-1).item())
        return actions

    @property
    def rng(self):
        # 复用 numpy rng，每实例独立
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(42)
        return self._rng

    # ============== 学习 ==============

    def compute_td_loss(
        self,
        obs_batch: np.ndarray,        # (B, num_agents, obs_dim)
        actions_batch: np.ndarray,    # (B, num_agents)
        rewards_batch: np.ndarray,    # (B,) 或 (B, num_agents)
        next_obs_batch: np.ndarray,   # (B, num_agents, obs_dim)
        dones_batch: np.ndarray,      # (B,)
    ) -> float:
        """
        简化版 DQN：每个 agent 独立学，team reward 均分。
        返回 scalar loss（用于日志）。
        """
        B = obs_batch.shape[0]
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_batch, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1).expand(-1, self.num_agents)
        dones = torch.as_tensor(dones_batch, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        for aid in range(self.num_agents):
            q_net = self.q_nets[aid]
            opt = self.optimizers[aid]

            q_pred = q_net(obs[:, aid, :]).gather(1, actions[:, aid].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Double DQN：选动作用 online，估值用 target
                q_next_online = q_net(next_obs[:, aid, :])
                next_actions = q_next_online.argmax(dim=-1, keepdim=True)
                q_next_target = self.target_nets[aid](next_obs[:, aid, :])
                q_next = q_next_target.gather(1, next_actions).squeeze(1)
            target = rewards[:, aid] + self.cfg.gamma * (1 - dones) * q_next

            loss = F.mse_loss(q_pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), self.cfg.grad_clip)
            opt.step()
            total_loss += float(loss.item())

        # 周期性硬同步 target 网络
        self._steps_since_target_update += 1
        if self._steps_since_target_update >= self.target_update_interval:
            for target_net, q_net in zip(self.target_nets, self.q_nets):
                target_net.load_state_dict(q_net.state_dict())
            self._steps_since_target_update = 0

        # ε 衰减
        self.train_step += 1
        self.eps = max(
            self.cfg.eps_end,
            self.cfg.eps_start - (self.cfg.eps_start - self.cfg.eps_end)
            * (self.train_step / max(1, self.cfg.eps_decay_steps))
        )
        return total_loss / self.num_agents

    # ============== 序列化 ==============

    def state_dict(self) -> Dict:
        return {
            "q_nets": [net.state_dict() for net in self.q_nets],
            "target_nets": [net.state_dict() for net in self.target_nets],
            "config": self.cfg.__dict__,
            "eps": self.eps,
            "train_step": self.train_step,
        }

    def load_state_dict(self, sd: Dict):
        for net, s in zip(self.q_nets, sd["q_nets"]):
            net.load_state_dict(s)
        if "target_nets" in sd:
            for net, s in zip(self.target_nets, sd["target_nets"]):
                net.load_state_dict(s)
        else:
            # 兼容旧 checkpoint：同步 target = online
            for target_net, q_net in zip(self.target_nets, self.q_nets):
                target_net.load_state_dict(q_net.state_dict())
        self.eps = sd.get("eps", self.eps)
        self.train_step = sd.get("train_step", 0)


__all__ = ["MARLConfig", "MARLLayer", "QNetwork"]