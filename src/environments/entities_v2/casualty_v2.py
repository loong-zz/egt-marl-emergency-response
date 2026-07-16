"""
Casualty 实体（v2）。

依据：设计文档 §4.1（伤员）、§3.1（伤情）。

设计：
1. Severity 枚举用数值，便于直接放进观测。
2. 生存剩余时间 = Weibull(theta, kappa) 采样得到，或用近似 = theta * 0.7。
3. discovered = True 后才可被治疗（FIND 任务的产物）。
4. treated = True 表示已治愈（不再扣分）。
5. dead = True 表示已死。
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple

import numpy as np


class Severity(IntEnum):
    """伤情等级（数值小 = 重）。"""
    CRITICAL = 0
    SEVERE = 1
    MODERATE = 2
    MILD = 3


# Weibull 参数（与 env_spec.WEIBULL_PARAMS 一致）
# 修复3+：压缩生存时间；MILD 也设上限确保 episode 可自然终止
WEIBULL = {
    Severity.CRITICAL: (200, 1.0),     # max≈140
    Severity.SEVERE:   (240, 1.2),     # max≈168
    Severity.MODERATE: (420, 1.5),     # max≈294
    Severity.MILD:     (800, 2.0),     # max≈560 (episode ≤800 步内可自然终止)
}


@dataclass
class Casualty:
    id: int
    position: Tuple[int, int]
    severity: Severity
    # 初始可救时间（步），由 Weibull 决定
    max_survival_steps: int = 0
    # 当前已等待的步数
    elapsed_steps: int = 0
    # 状态
    discovered: bool = False
    treated: bool = False
    dead: bool = False
    # 谁发现/治疗的（用于信誉计算）
    discovered_by: int = -1
    treated_by: int = -1
    # 所属灾区
    area_id: int = -1
    # 是否已计入死亡统计（去重 marker，避免每帧重复计数 bug）
    _death_counted: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.max_survival_steps <= 0:
            theta, kappa = WEIBULL[self.severity]
            self.max_survival_steps = int(theta * 0.7)   # 取 ~50% 生存期

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_survival_steps - self.elapsed_steps)

    def step(self):
        """每个仿真步推进。死亡判定。"""
        if self.dead or self.treated:
            return
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_survival_steps:
            self.dead = True

    def on_dead_reported(self) -> bool:
        """返回 True 表示本次死亡事件应当被计数（去重保证仅首次返回True）。"""
        if self.dead and not self._death_counted:
            self._death_counted = True
            return True
        return False

    def reset(self):
        """每集初重置。"""
        self._death_counted = False

    def discover(self, agent_id: int) -> bool:
        if self.dead or self.discovered:
            return False
        self.discovered = True
        self.discovered_by = agent_id
        return True

    def treat(self, agent_id: int) -> bool:
        if self.dead or not self.discovered or self.treated:
            return False
        self.treated = True
        self.treated_by = agent_id
        return True


__all__ = ["Severity", "Casualty"]