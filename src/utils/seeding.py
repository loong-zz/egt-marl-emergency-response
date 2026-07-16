"""
Seed 管理：env / numpy / torch 三方种子统一设置。

依据：设计文档 §9.1（可复现）、§8.5。
"""

import os
import random
import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    设置 Python / NumPy / PyTorch 种子。

    Args:
        seed: 主种子
        deterministic_torch: 若 True，启用 torch 确定性算法（性能更慢，但跨 run 完全可复现）
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


__all__ = ["set_seed"]