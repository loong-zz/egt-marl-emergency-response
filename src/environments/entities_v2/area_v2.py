"""
Affected Area 实体（v2）。

依据：设计文档 §4.6（灾区划分）。

设计：
1. 一个灾区是网格上的一个矩形（x0, y0, x1, y1）含伤员数量。
2. 调度算法用 priority 字段排序（值越小越优先）。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Area:
    id: int
    bbox: Tuple[int, int, int, int]   # (x0, y0, x1, y1)  含右开区间 [x0, x1), [y0, y1)
    priority: int = 0                  # 越小越紧急
    label: str = ""                    # 调试用

    def contains(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.bbox
        return x0 <= x < x1 and y0 <= y < y1


__all__ = ["Area"]