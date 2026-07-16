"""
K_max padding —— 通用最近邻填充。

依据：设计文档 §3.2（局部观测按距离排序，取前 K_max 个，用 0 填充 + mask 标记）。

设计：
1. 输入是一组"对象"，每个对象有自己的特征向量与坐标。
2. 按欧氏距离取最近的 K_max 个。
3. 缺失位置填 0，向量尾部追加 mask 通道（1=真实，0=占位）。
4. 完全可微 / 可数值化，给 MARL 用 numpy 数组即可。
"""

from typing import List, Tuple
import numpy as np


def nearest_k_pad(
    candidates: List[Tuple[np.ndarray, np.ndarray]],
    self_pos: np.ndarray,
    k_max: int,
    feature_dim: int,
    fill_value: float = 0.0,
    return_mask: bool = True,
) -> np.ndarray:
    """
    从 candidates 中挑出距离 self_pos 最近的 k_max 个，按距离升序排列，
    用 fill_value 填充缺失项。

    Args:
        candidates: [(feature_i, pos_i), ...]
            - feature_i: shape=(feature_dim,)
            - pos_i: shape=(2,)
        self_pos: shape=(2,)
        k_max: 最多取多少个
        feature_dim: 每个 feature 的维度
        fill_value: 占位填充值
        return_mask: 是否在末尾追加 mask 通道

    Returns:
        shape=(k_max, feature_dim + (1 if return_mask else 0)) 的 ndarray
        末尾 mask 通道 1=真实 0=占位
    """
    extra = 1 if return_mask else 0
    out = np.full((k_max, feature_dim + extra), fill_value, dtype=np.float32)

    if k_max <= 0 or not candidates:
        return out

    # 计算距离
    positions = np.stack([c[1] for c in candidates], axis=0)  # (n, 2)
    diffs = positions - self_pos[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    order = np.argsort(dists)
    n_take = min(k_max, len(candidates))
    for i in range(n_take):
        idx = int(order[i])
        feat = candidates[idx][0]
        out[i, :feature_dim] = feat
        if return_mask:
            out[i, feature_dim] = 1.0   # mask=1 表示真实

    return out


def nearest_k_pad_flat(
    candidates: List[Tuple[np.ndarray, np.ndarray]],
    self_pos: np.ndarray,
    k_max: int,
    feature_dim: int,
    fill_value: float = 0.0,
    return_mask: bool = True,
) -> np.ndarray:
    """
    同 nearest_k_pad，但返回一维向量 (k_max * (feature_dim + 1),)
    便于直接拼接到 ObservationSpec.dim 那种扁平结构。
    """
    block = nearest_k_pad(candidates, self_pos, k_max, feature_dim,
                          fill_value=fill_value, return_mask=return_mask)
    return block.reshape(-1)


__all__ = ["nearest_k_pad", "nearest_k_pad_flat"]