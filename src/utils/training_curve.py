"""
training_curve.py —— 从 training.jsonl 画出 4 子图 dashboard。

依据：设计文档 §6.6 训练日志 schema。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")   # 无 display 环境也能画
import matplotlib.pyplot as plt
import numpy as np


def _smooth(y: List[float], window: int = 10) -> np.ndarray:
    """滑动平均（中心对齐）。window 偶数自动 +1。"""
    if window % 2 == 0:
        window += 1
    if window <= 1 or len(y) < window:
        return np.asarray(y, dtype=np.float32)
    k = window // 2
    pad = np.pad(y, (k, k), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(pad, kernel, mode="valid")[: len(y)]


def load_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    """读 training.jsonl。"""
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def plot_dashboard(records: List[Dict[str, Any]], save_path: Path,
                   window: int = 10, title: str = "EGT-MARL v2 Training"):
    """
    画 4 子图 dashboard：
        [0,0] rescue_rate vs episode
        [0,1] reward_sum vs episode
        [1,0] lambda + eps vs episode (双 y 轴)
        [1,1] dominant_strategy scatter
    """
    if not records:
        raise ValueError("empty records")
    eps = np.asarray([r["episode"] for r in records])
    rr = np.asarray([r["rescue_rate"] for r in records], dtype=np.float32)
    rew = np.asarray([r["reward_sum"] for r in records], dtype=np.float32)
    lam = np.asarray([r["lambda"] for r in records], dtype=np.float32)
    eps_v = np.asarray([r["eps"] for r in records], dtype=np.float32)
    dom = np.asarray([r["dominant_strategy"] for r in records], dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- [0,0] rescue rate ---
    ax = axes[0, 0]
    ax.plot(eps, rr, color="steelblue", alpha=0.35, label="raw")
    ax.plot(eps, _smooth(rr.tolist(), window), color="steelblue", lw=2, label=f"smoothed({window})")
    ax.set_xlabel("episode"); ax.set_ylabel("rescue_rate")
    ax.set_title("Rescue Rate")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    ax.set_ylim(0, max(0.05, float(rr.max()) * 1.2))

    # --- [0,1] reward sum ---
    ax = axes[0, 1]
    ax.plot(eps, rew, color="seagreen", alpha=0.35, label="raw")
    ax.plot(eps, _smooth(rew.tolist(), window), color="seagreen", lw=2, label=f"smoothed({window})")
    ax.set_xlabel("episode"); ax.set_ylabel("reward_sum")
    ax.set_title("Episode Reward")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    # --- [1,0] lambda + eps (双 y 轴) ---
    ax = axes[1, 0]
    ax.plot(eps, _smooth(lam.tolist(), window), color="darkorange", lw=2, label="lambda (smoothed)")
    ax.set_xlabel("episode"); ax.set_ylabel("lambda", color="darkorange")
    ax.tick_params(axis="y", labelcolor="darkorange")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(eps, _smooth(eps_v.tolist(), window), color="purple", lw=1.5, label="eps (smoothed)")
    ax2.set_ylabel("epsilon", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.set_ylim(0, 1.05)
    ax.set_title("Lambda & Epsilon")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # --- [1,1] dominant strategy ---
    ax = axes[1, 1]
    strategy_labels = {0: "Fair", 1: "Eff", 2: "Bal"}
    for s in range(3):
        mask = (dom == s)
        if mask.any():
            ax.scatter(eps[mask], dom[mask], s=14, alpha=0.6,
                       label=strategy_labels.get(s, str(s)))
    ax.set_xlabel("episode"); ax.set_ylabel("dominant strategy")
    ax.set_title("Dominant Strategy")
    ax.set_yticks(list(range(3)))
    ax.set_yticklabels([strategy_labels.get(s, str(s)) for s in range(3)])
    ax.grid(True, alpha=0.3); ax.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_summary(records: List[Dict[str, Any]], save_path: Path, window: int = 10):
    """单图：rescue rate + reward 画一张，便于论文。"""
    eps = np.asarray([r["episode"] for r in records])
    rr = np.asarray([r["rescue_rate"] for r in records], dtype=np.float32)
    rew = np.asarray([r["reward_sum"] for r in records], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=110)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("rescue_rate", color="steelblue")
    ax1.plot(eps, _smooth(rr.tolist(), window), color="steelblue", lw=2, label="rescue_rate")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, max(0.05, float(rr.max()) * 1.2))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("reward_sum", color="seagreen")
    ax2.plot(eps, _smooth(rew.tolist(), window), color="seagreen", lw=2, label="reward_sum")
    ax2.tick_params(axis="y", labelcolor="seagreen")

    fig.suptitle("Training Summary (smoothed, window={})".format(window))
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return save_path


__all__ = ["load_records", "plot_dashboard", "plot_summary", "_smooth"]


if __name__ == "__main__":   # 自测
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m utils.training_curve <path/to/training.jsonl>")
        sys.exit(1)
    p = Path(sys.argv[1])
    recs = load_records(p)
    print(f"loaded {len(recs)} records")
    out_dash = p.parent / "dashboard.png"
    out_sum = p.parent / "summary.png"
    print(plot_dashboard(recs, out_dash))
    print(plot_summary(recs, out_sum))