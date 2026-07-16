"""
visualize_training.py —— 读 training.jsonl，输出 dashboard.png + summary.png。

用法：
    python -m experiments.visualize_training train_v2_run300b/
    python -m experiments.visualize_training train_v2_run300b/training.jsonl --smooth 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.training_curve import load_records, plot_dashboard, plot_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="training.jsonl 或其所在目录")
    p.add_argument("--smooth", type=int, default=10,
                   help="滑动平均窗口大小")
    p.add_argument("--title", type=str, default="EGT-MARL v2 Training",
                   help="图表主标题")
    p.add_argument("--only-summary", action="store_true",
                   help="只画 summary.png，不画 dashboard.png")
    args = p.parse_args()

    target = Path(args.path)
    if target.is_dir():
        target = target / "training.jsonl"
    if not target.exists():
        raise FileNotFoundError(f"not found: {target}")

    records = load_records(target)
    print(f"loaded {len(records)} records from {target}")
    if not records:
        raise ValueError("empty records")

    save_dir = target.parent
    if not args.only_summary:
        dash = save_dir / "dashboard.png"
        plot_dashboard(records, dash, window=args.smooth, title=args.title)
        print(f"saved {dash}")

    summary = save_dir / "summary.png"
    plot_summary(records, summary, window=args.smooth)
    print(f"saved {summary}")


if __name__ == "__main__":
    main()