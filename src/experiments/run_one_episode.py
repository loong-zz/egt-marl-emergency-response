"""
run_one_episode.py —— 单 episode 端到端验收脚本。

用途：阶段 8 验收目标，确认从 DisasterSim → MARL → EGT → 反馈 → step 的
完整闭环能跑通。

用法：
    python -m experiments.run_one_episode \\
        --num_agents 10 --num_casualties 30 --max_steps 100 \\
        --mode random     # or "marl"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 让 `python -m experiments.run_one_episode` 在 src/ 下能找到包
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from environments.disaster_sim_v2 import DisasterSim
from algorithms_v2.egt_layer import EGTLayer
from algorithms_v2.marl_layer import MARLLayer
from utils.env_spec import ACTION_DIM, DEFAULT_OBS_SPEC


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_one_episode")


def run_episode(
    env: DisasterSim,
    marl: MARLLayer | None,
    egt: EGTLayer,
    max_steps: int,
    mode: str = "random",
    rng: np.random.Generator | None = None,
    log_every: int = 20,
) -> dict:
    """
    跑单 episode。

    Args:
        mode: 'random' 或 'marl'（marl 模式必须传入 marl）
    """
    if rng is None:
        rng = np.random.default_rng(0)

    obs_dict, info = env.reset()
    egt.reset()

    total_reward = {aid: 0.0 for aid in env.agents}
    steps_taken = 0
    t0 = time.time()

    for step in range(max_steps):
        # 选动作
        if mode == "random":
            actions = {
                aid: int(rng.integers(0, ACTION_DIM))
                for aid in env.agents if env.agents[aid].alive
            }
        elif mode == "marl":
            assert marl is not None
            actions = marl.select_actions(obs_dict, eps=0.05)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # step
        obs_dict, reward_dict, terminated, truncated, info = env.step(actions)
        steps_taken += 1
        for aid, r in reward_dict.items():
            total_reward[aid] = total_reward.get(aid, 0.0) + r

        # EGT 更新（按 agent 奖励）
        egt.update(agent_rewards=reward_dict)

        # 把 EGT 注入环境（下一帧 obs 会用到）
        env.egt_signal_fn = egt.get_signal
        env.egt_lambda_fn = egt.get_lambda

        if step % log_every == 0:
            stats = info["statistics"]
            logger.info(
                f"step={step} rescued={stats['total_rescued']} "
                f"deaths={stats['total_deaths']} lam={egt.get_lambda():.3f}"
            )

        if terminated or truncated:
            logger.info(
                f"episode done @ step={step} reason="
                f"{'terminated' if terminated else 'truncated'}"
            )
            break

    elapsed = time.time() - t0
    final_stats = env.statistics
    summary = {
        "steps": steps_taken,
        "elapsed_sec": round(elapsed, 3),
        "total_reward_sum": round(sum(total_reward.values()), 3),
        "total_rescued": final_stats["total_rescued"],
        "total_deaths": final_stats["total_deaths"],
        "total_reports": final_stats["total_reports"],
        "total_shares": final_stats["total_shares"],
        "final_lambda": round(egt.get_lambda(), 4),
        "dominant_strategy": egt.get_dominant_strategy(),
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_agents", type=int, default=10)
    p.add_argument("--num_casualties", type=int, default=30)
    p.add_argument("--num_depots", type=int, default=2)
    p.add_argument("--map_size", type=int, nargs=2, default=[30, 30])
    p.add_argument("--disaster_severity", type=str, default="medium")
    p.add_argument("--malicious_ratio", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--mode", choices=["random", "marl"], default="random")
    p.add_argument("--log_every", type=int, default=20)
    args = p.parse_args()

    logger.info("=" * 60)
    logger.info("Single-episode acceptance run")
    logger.info(f"  num_agents={args.num_agents}  num_casualties={args.num_casualties}")
    logger.info(f"  map_size={args.map_size}  severity={args.disaster_severity}")
    logger.info(f"  mode={args.mode}  max_steps={args.max_steps}")
    logger.info("=" * 60)

    env = DisasterSim(
        seed=args.seed,
        map_size=tuple(args.map_size),
        num_agents=args.num_agents,
        num_casualties=args.num_casualties,
        num_depots=args.num_depots,
        disaster_severity=args.disaster_severity,
        malicious_ratio=args.malicious_ratio,
    )

    egt = EGTLayer()

    marl = None
    if args.mode == "marl":
        marl = MARLLayer(
            obs_dim=DEFAULT_OBS_SPEC.dim,
            action_dim=ACTION_DIM,
            num_agents=args.num_agents,
        )
        # 让 MARL 自带的 rng 用同样的 seed
        marl._rng = np.random.default_rng(args.seed)

    summary = run_episode(
        env=env, marl=marl, egt=egt,
        max_steps=args.max_steps, mode=args.mode, log_every=args.log_every,
    )

    logger.info("=" * 60)
    logger.info("SUMMARY")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)