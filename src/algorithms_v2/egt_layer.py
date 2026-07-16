"""
EGT (Evolutionary Game Theory) Layer —— 简化版。

依据：设计文档 §1.4（3 策略）、§5.3（λ 公式）。

设计：
1. 3 策略：FAIR / EFFICIENT / BALANCED
2. 状态：策略分布 p = [p_F, p_E, p_B] ∈ Δ³
3. 演化机制（replicator dynamics）：
   - 每步用 fitness(F)=f_F, fitness(E)=f_E, fitness(B)=f_B 更新：
     p_i ← p_i * (f_i / avg_f) ，归一化
4. λ_param 由策略分布加权 + EMA 平滑得到
   λ_t = α · Σ p_i · λ_i^strategy + (1 - α) · λ_{t-1}
5. 提供 get_egt_signal() 给 DisasterSim 注入观测。

接口：
    egt = EGTLayer(seed=0)
    egt.update(rewards={0: r0, 1: r1, ...})
    sig = egt.get_signal()      # shape=(3,)
    lam = egt.get_lambda()      # float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from utils.env_spec import (
    EGT_STRATEGIES, NUM_STRATEGIES,
    STRATEGY_LAMBDA,
)


@dataclass
class EGTConfig:
    """EGT 层超参数。"""
    ema_alpha: float = 0.1                  # λ 的 EMA 平滑系数
    lambda_anchor: float = 0.5              # 课程锚点（§5.3）
    lambda_anchor_blend: float = 0.0        # 课程锚点权重（0 表示不起作用）
    fitness_clip: tuple = (-1.0, 5.0)       # fitness 截断范围
    min_prob: float = 1e-6                  # 策略分布下界（避免全 0）


class EGTLayer:
    """
    演化博弈论元控制器（simplified）。

    设计为"无状态 gym 调用"友好：每 episode 自己 reset。
    """

    def __init__(self, config: Optional[EGTConfig] = None, device: str = "cpu"):
        self.cfg = config or EGTConfig()
        self.device = torch.device(device)

        # 策略参数 λ
        self.strategy_lambda = np.array(
            [STRATEGY_LAMBDA[s] for s in EGT_STRATEGIES], dtype=np.float32
        )   # [0.9, 0.5, 0.7]

        # 策略分布（公平 / 效率 / 均衡）
        self.p = np.ones(NUM_STRATEGIES, dtype=np.float32) / NUM_STRATEGIES
        # λ 历史
        self.lambda_history = [self.cfg.lambda_anchor]

    # ============== 状态管理 ==============

    def reset(self):
        self.p = np.ones(NUM_STRATEGIES, dtype=np.float32) / NUM_STRATEGIES
        self.lambda_history = [self.cfg.lambda_anchor]

    # ============== 演化动力学 ==============

    def update(self, agent_rewards: Dict[int, float], strategy_assign: Optional[Dict[int, int]] = None):
        """
        根据 agent 奖励更新策略分布。

        Args:
            agent_rewards: {agent_id: reward} 单步累计奖励
            strategy_assign: {agent_id: strategy_id} 可选；未提供则按均匀分配
        """
        if not agent_rewards:
            return

        n_agents = len(agent_rewards)
        if strategy_assign is None:
            # 按 ID 轮转分配
            strategy_assign = {aid: aid % NUM_STRATEGIES for aid in agent_rewards}

        # 计算每个策略的 fitness = 该策略下 agent 的平均奖励
        fitness = np.zeros(NUM_STRATEGIES, dtype=np.float32)
        count = np.zeros(NUM_STRATEGIES, dtype=np.int32)
        for aid, r in agent_rewards.items():
            s = strategy_assign.get(aid, 0)
            fitness[s] += r
            count[s] += 1

        for s in range(NUM_STRATEGIES):
            if count[s] > 0:
                fitness[s] /= count[s]
            else:
                # 该策略无 agent：设为平均，避免 NaN
                fitness[s] = float(np.mean(list(agent_rewards.values())))

        # 截断到合理范围
        lo, hi = self.cfg.fitness_clip
        fitness = np.clip(fitness, lo, hi)

        # Replicator dynamics: p_i ← p_i * f_i / avg_f
        avg_f = float(np.dot(self.p, fitness))
        if avg_f <= 0:
            # 全部 fitness 为 0：保持分布不变
            return

        new_p = self.p * fitness / avg_f
        # 归一化 + 下界
        new_p = np.maximum(new_p, self.cfg.min_prob)
        new_p /= new_p.sum()
        self.p = new_p

        # 更新 λ：用策略分布加权 + EMA
        lam_now = float(np.dot(self.p, self.strategy_lambda))
        lam_anchor = self.cfg.lambda_anchor
        lam_target = (
            (1 - self.cfg.lambda_anchor_blend) * lam_now
            + self.cfg.lambda_anchor_blend * lam_anchor
        )
        alpha = self.cfg.ema_alpha
        lam_smoothed = alpha * lam_target + (1 - alpha) * self.lambda_history[-1]
        lam_smoothed = float(np.clip(lam_smoothed, 0.0, 1.0))
        self.lambda_history.append(lam_smoothed)

    # ============== 输出 ==============

    def get_signal(self) -> np.ndarray:
        """策略分布 [p_F, p_E, p_B]，shape=(3,)。"""
        return self.p.copy()

    def get_lambda(self) -> float:
        """当前 λ（公平权重）。"""
        return float(self.lambda_history[-1])

    def get_dominant_strategy(self) -> int:
        """argmax p_i。"""
        return int(np.argmax(self.p))

    def get_state_dict(self) -> Dict:
        """序列化（用于 checkpoint）。"""
        return {
            "p": self.p.tolist(),
            "lambda_history": list(self.lambda_history),
            "config": self.cfg.__dict__,
        }


__all__ = ["EGTConfig", "EGTLayer"]