"""
Rescue Agent 实体（v2）。

依据：设计文档 §3.1（3 种智能体）、§3.4（资源携带）、§3.6（信誉）。

设计：
1. agent_type 用 str（'DRONE' / 'VEHICLE' / 'PERSONNEL'），与 env_spec 对齐。
2. 资源用 dict[str, int]，例 {"medkit": 3, "blood": 0}。
3. 信誉用 dict[str, float] {alpha, beta}（Beta 分布参数）。
4. 行为参数（speed/capacity/vision）从 env_spec.AGENT_* 取，避免重复定义。
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict

from utils.env_spec import (
    AGENT_SPEEDS, AGENT_CAPACITY, AGENT_VISION_RADIUS,
    REPUTATION_ALPHA_INIT, REPUTATION_BETA_INIT,
)


@dataclass
class Agent:
    id: int
    agent_type: str              # 'DRONE' / 'VEHICLE' / 'PERSONNEL'
    position: Tuple[int, int]
    inventory: Dict[str, int] = field(default_factory=dict)
    capacity: int = 0
    speed: int = 1
    vision_radius: int = 4

    # 信誉（Beta 分布）
    rep_alpha: float = REPUTATION_ALPHA_INIT
    rep_beta: float = REPUTATION_BETA_INIT

    # 状态
    alive: bool = True

    # 行为历史（最近 step 多少？）
    recent_actions: list = field(default_factory=list)

    def __post_init__(self):
        self.capacity = AGENT_CAPACITY.get(self.agent_type, 8)
        self.speed = AGENT_SPEEDS.get(self.agent_type, 1)
        self.vision_radius = AGENT_VISION_RADIUS.get(self.agent_type, 4)

    # -------- 资源 --------

    def resources_total(self) -> int:
        return sum(self.inventory.values())

    def has_capacity(self, need: int = 1) -> bool:
        return self.resources_total() + need <= self.capacity

    def add_resource(self, kind: str, qty: int = 1) -> bool:
        if not self.has_capacity(qty):
            return False
        self.inventory[kind] = self.inventory.get(kind, 0) + qty
        return True

    def remove_resource(self, kind: str, qty: int = 1) -> bool:
        if self.inventory.get(kind, 0) < qty:
            return False
        self.inventory[kind] -= qty
        if self.inventory[kind] == 0:
            del self.inventory[kind]
        return True

    # -------- 信誉 --------

    @property
    def reputation(self) -> float:
        """E[Bernoulli] = alpha / (alpha + beta) ∈ [0, 1]"""
        s = self.rep_alpha + self.rep_beta
        if s <= 0:
            return 0.5
        return self.rep_alpha / s

    def update_reputation(self, success: bool, weight_success: float = 1.0, weight_fail: float = 3.0):
        if success:
            self.rep_alpha += weight_success
        else:
            self.rep_beta += weight_fail


__all__ = ["Agent"]