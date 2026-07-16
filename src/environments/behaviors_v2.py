"""
Agent 行为策略（依据设计文档 §4.3）。

设计：
1. 所有行为以 AgentBehavior 抽象类暴露，get_target(agent, env) -> Optional[Position]
   返回 None 时表示 IDLE（不消耗动作）。
2. DRONE 不能治；VEHICLE/PERSONNEL 行为一致，差异在物理参数。
3. 不在类层级复制行为；用工厂函数 + 类继承降低冗余。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from utils.env_spec import CAN_TREAT

# 类型别名
Position = Tuple[int, int]


class AgentBehavior(ABC):
    """所有行为策略的基类。"""

    @abstractmethod
    def get_target(self, agent, env) -> Optional[Position]:
        """
        计算 agent 下一步要走向的目标点。

        Returns:
            (x, y) 或 None（IDLE）。
        """


# ============ 简单可复用行为 ============

class FindCasualtyBehavior(AgentBehavior):
    """走到最近的未发现 / 未指派伤员附近。"""

    def get_target(self, agent, env):
        best = None
        best_dist = float("inf")
        for c in env.casualties.values():
            if c.discovered:
                continue
            d = abs(c.position[0] - agent.position[0]) + abs(c.position[1] - agent.position[1])
            if d < best_dist:
                best_dist = d
                best = c.position
        return best


class TreatCasualtyBehavior(AgentBehavior):
    """走到最近的已发现、未治疗、且需要该 agent 类型的伤员。"""

    def get_target(self, agent, env):
        if not CAN_TREAT.get(agent.agent_type, False):
            return None      # DRONE 不能治
        best = None
        best_dist = float("inf")
        for c in env.casualties.values():
            if not c.discovered or c.treated:
                continue
            d = abs(c.position[0] - agent.position[0]) + abs(c.position[1] - agent.position[1])
            if d < best_dist:
                best_dist = d
                best = c.position
        return best


class ShareResourcesBehavior(AgentBehavior):
    """走到最近的、需要资源的队友旁。

    逻辑参考 V1 DroneBehavior.find_needy_agent：
        - 排除自己和 DRONE（DRONE 不需要任何资源）
        - 寻找 helpers.has_capacity 的队友
        - 优先寻找 resources_total 较小的（更"缺资源"）
        - 在距离和资源缺口之间做 trade-off
    """

    def get_target(self, agent, env):
        # DRONE 也能帮别人送资源，所以不限制自己的类型
        if agent.resources_total() <= 1:
            return None  # 自己资源不足，不分享
        best = None
        best_score = float("inf")
        my_pos = np.array(agent.position, dtype=np.float32)
        for other in env.agents.values():
            if other.id == agent.id or not other.alive:
                continue
            other_type = other.agent_type
            # DRONE 不能被分享物资（不需要救治），但 VEHICLE/PERSONNEL 可接收
            if other_type == "DRONE":
                continue
            # 对方得有容量空间
            if not other.has_capacity(1):
                continue
            d = abs(other.position[0] - agent.position[0]) + abs(other.position[1] - agent.position[1])
            if d == 0:
                continue
            # 资源缺口越大越优先：score = 距离 / (1 + 缺口 + 1)
            other_total = other.resources_total()
            need_score = 1.0 / (1.0 + (other.capacity - other_total))
            score = d / need_score
            if score < best_score:
                best_score = score
                best = other.position
        return best


class RefillResourcesBehavior(AgentBehavior):
    """走到最近的仓库。"""

    def get_target(self, agent, env):
        if not env.depots:
            return None
        best = None
        best_dist = float("inf")
        for d in env.depots.values():
            dd = abs(d.position[0] - agent.position[0]) + abs(d.position[1] - agent.position[1])
            if dd < best_dist:
                best_dist = dd
                best = d.position
        return best


# ============ 工厂（按 task_id 选行为） ============

class BehaviorFactory:
    """根据 task_id 返回对应的行为对象（无状态，可复用）。"""

    _INSTANCES = {
        0: FindCasualtyBehavior(),    # FIND_CASUALTY
        1: TreatCasualtyBehavior(),    # TREAT_CASUALTY
        2: ShareResourcesBehavior(),   # SHARE_RESOURCES
        3: RefillResourcesBehavior(),  # REFILL_RESOURCES
    }

    @classmethod
    def get(cls, task_id: int) -> AgentBehavior:
        if task_id not in cls._INSTANCES:
            raise ValueError(f"unknown task_id={task_id}")
        return cls._INSTANCES[task_id]


__all__ = [
    "AgentBehavior",
    "FindCasualtyBehavior", "TreatCasualtyBehavior",
    "ShareResourcesBehavior", "RefillResourcesBehavior",
    "BehaviorFactory",
]