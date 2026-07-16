"""
ReplayBuffer —— 简单环形经验回放。

依据：设计文档 §5.2（MARL 训练）。

设计：
1. 存 (obs_dict, actions_dict, reward, next_obs_dict, done)
2. 每条样本按 agent 维度对齐（用 padding agent_id 占位）
3. 用 numpy 数组加速批量采样

简化：
- 不做 prioritized replay
- 不存 egt_signal / lambda（它们会被注入 env 重新生成）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np


@dataclass
class Transition:
    """单步经验（针对所有 agent）。"""
    obs: Dict[int, np.ndarray]          # {aid: obs_vec}
    actions: Dict[int, int]             # {aid: action_idx}
    reward: float                       # 团队平均奖励
    next_obs: Dict[int, np.ndarray]
    done: bool


class ReplayBuffer:
    """FIFO 经验回放。"""

    def __init__(self, capacity: int, num_agents: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 用 np 数组存，按 agent 维度（缺失 agent 用 padding）
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.agent_mask = np.zeros((capacity, num_agents), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self):
        return self._size

    def add(self, tr: Transition):
        """存一条经验。agent 不全时用 mask 标记。"""
        i = self._idx
        for aid in range(self.num_agents):
            if aid in tr.obs:
                self.obs[i, aid, :] = tr.obs[aid]
                self.next_obs[i, aid, :] = tr.next_obs.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                self.actions[i, aid] = tr.actions.get(aid, 0)
                self.agent_mask[i, aid] = 1.0
            else:
                self.obs[i, aid, :] = 0.0
                self.next_obs[i, aid, :] = 0.0
                self.actions[i, aid] = 0
                self.agent_mask[i, aid] = 0.0
        self.rewards[i] = tr.reward
        self.dones[i] = 1.0 if tr.done else 0.0

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """随机采样 batch_size 条。"""
        if self._size < batch_size:
            raise ValueError(f"buffer size {self._size} < batch {batch_size}")
        idxs = np.random.default_rng().integers(0, self._size, size=batch_size)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
            "agent_mask": self.agent_mask[idxs],
        }


__all__ = ["Transition", "ReplayBuffer"]