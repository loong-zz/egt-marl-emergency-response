"""
MovementPolicy —— 4 方向移动 + 障碍 + 边界。

依据：设计文档 §4.4.1（4 方向）、§4.4.5（障碍 5–20%）。

设计：
1. 坐标是离散网格 (x, y)，x∈[0, W), y∈[0, H)。
2. 每步最多移动 1 格（由 AGENT_SPEEDS 决定多次迭代）。
3. 障碍物为布尔矩阵 obstacles[y, x] = True 表示该格不可进入。
4. 当目标向量为 0 或目标不可达时，返回 None（隐式 IDLE）。
5. MovementPolicy 不依赖 Agent，仅做几何运算；上层负责调用。
"""

from typing import Optional, Tuple
import numpy as np

from .env_spec import DIRECTION_VECTORS


class MovementPolicy:
    """网格世界的 4 方向移动器。"""

    def __init__(self, map_size: Tuple[int, int], obstacles: np.ndarray):
        """
        Args:
            map_size: (W, H) 网格大小
            obstacles: (H, W) 布尔矩阵，True = 不可进入
        """
        W, H = map_size
        assert obstacles.shape == (H, W), (
            f"obstacles.shape={obstacles.shape} 应为 (H, W)=({H}, {W})"
        )
        self.W = W
        self.H = H
        self.obstacles = obstacles

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def passable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not bool(self.obstacles[y, x])

    def direction_towards(self, src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[int]:
        """
        计算 src → dst 的最近 4 方向之一。

        Returns:
            方向 id ∈ {0,1,2,3}，或 None（已在目标点 / 目标不可达）。
        """
        sx, sy = src
        dx, dy = dst

        # 已在目标点
        if sx == dx and sy == dy:
            return None

        # 选择距离目标更近的轴先走
        dx_diff = abs(dx - sx)
        dy_diff = abs(dy - sy)

        # 候选方向：按"接近目标"的方向选
        # 先尝试 x 方向，再尝试 y 方向（距离更大的优先，避免在同距离时贪心导致死锁）
        candidates = []
        if sx < dx:
            candidates.append(1)   # E
        elif sx > dx:
            candidates.append(3)   # W
        if sy < dy:
            candidates.append(0)   # N（注意：DIRECTION_VECTORS[0]=(0,1)，y 增大为 N）
        elif sy > dy:
            candidates.append(2)   # S

        # 距离更大的轴优先
        if dx_diff >= dy_diff:
            ordered = [d for d in candidates if DIRECTION_VECTORS[d][0] != 0] \
                    + [d for d in candidates if DIRECTION_VECTORS[d][0] == 0]
        else:
            ordered = [d for d in candidates if DIRECTION_VECTORS[d][1] != 0] \
                    + [d for d in candidates if DIRECTION_VECTORS[d][1] == 0]

        # 选第一个可通行的方向
        for d in ordered:
            vx, vy = DIRECTION_VECTORS[d]
            nx, ny = sx + vx, sy + vy
            if self.passable(nx, ny):
                return d
        # 全不可达 → 停留
        return None

    def apply(self, pos: Tuple[int, int], direction: Optional[int]) -> Tuple[int, int]:
        """
        给定当前位置与方向，返回新位置。

        direction 为 None 时返回 pos 不变（隐式 IDLE）。
        方向不可达时返回 pos 不变（兜底）。
        """
        if direction is None:
            return pos
        vx, vy = DIRECTION_VECTORS.get(int(direction), (0, 0))
        nx, ny = pos[0] + vx, pos[1] + vy
        if not self.passable(nx, ny):
            return pos
        return nx, ny


__all__ = ["MovementPolicy"]