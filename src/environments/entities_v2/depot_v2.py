"""
Resource Depot 实体（v2）。

依据：设计文档 §4.5.4（仓库职责）。

设计：
1. load_to(agent, kind, qty) —— 仓库 → agent（累加；agent 容量检查）
2. receive_from(agent, kind, qty) —— agent → 仓库（累加）
3. transfer_to(agent, kind, qty) —— 仓库 → agent（覆盖式，agent 残量丢失）
4. 三种语义清晰分开，避免原代码"覆盖式"歧义。
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict

from utils.env_spec import DEPOT_INITIAL_INVENTORY


@dataclass
class Depot:
    id: int
    position: Tuple[int, int]
    inventory: Dict[str, int] = field(default_factory=lambda: DEPOT_INITIAL_INVENTORY.copy())

    def has(self, kind: str, qty: int = 1) -> bool:
        return self.inventory.get(kind, 0) >= qty

    def load_to(self, agent, kind: str, qty: int = 1) -> bool:
        """
        仓库向 agent 累加资源。
        失败条件：仓库库存不足 或 agent 容量不足。
        成功时仓库扣减、agent 累加。
        """
        if not self.has(kind, qty):
            return False
        if not agent.has_capacity(qty):
            return False
        self.inventory[kind] -= qty
        agent.add_resource(kind, qty)
        return True

    def receive_from(self, agent, kind: str, qty: int = 1) -> bool:
        """agent 归还资源到仓库（累加）。"""
        if not agent.remove_resource(kind, qty):
            return False
        self.inventory[kind] = self.inventory.get(kind, 0) + qty
        return True

    def transfer_to(self, agent, kind: str, qty: int = 1) -> bool:
        """
        覆盖式转移：agent 当前 kind 资源被丢弃，换成 qty。
        极少用，但用于"应急补给"场景。
        """
        if not self.has(kind, qty):
            return False
        # 丢弃 agent 现有 kind
        if kind in agent.inventory:
            del agent.inventory[kind]
        # 装入
        agent.inventory[kind] = qty
        self.inventory[kind] -= qty
        return True


__all__ = ["Depot"]