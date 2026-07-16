"""
train_v2.py —— 用 v2 模块做端到端训练（不破坏 train_egt_marl.py）。

依据：设计文档 §5（训练）。

设计：
1. 用 DisasterSim v2 + EGTLayer v2 + MARLLayer v2
2. ReplayBuffer 存 (obs, action, reward, next_obs, done)
3. 每 update_freq 步做一次 TD 更新
4. 每 checkpoint_interval 集保存一次
5. 输出 JSONL 训练日志

用法：
    python -m experiments.train_v2 \\
        --num_episodes 50 --num_agents 6 --num_casualties 20 \\
        --buffer_size 5000 --batch_size 32 \\
        --checkpoint_interval 25 --log_every 5 \\
        --save_dir ./train_v2_out
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 让脚本可独立运行
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from environments.disaster_sim_v2 import DisasterSim
from algorithms_v2.egt_layer import EGTLayer, EGTConfig
from algorithms_v2.marl_layer import MARLLayer, MARLConfig
from utils.env_spec import ACTION_DIM, DEFAULT_OBS_SPEC
from utils.config import TrainingConfig
from utils.replay_buffer import ReplayBuffer, Transition


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_v2")


def save_checkpoint(
    marl: MARLLayer, egt: EGTLayer, episode: int, save_dir: Path, metrics: dict
):
    """保存 checkpoint（直接用 torch.save）。"""
    import torch
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / f"checkpoint_ep_{episode}.pt"
    torch.save({
        "marl_state": marl.state_dict(),
        "egt_state": egt.get_state_dict(),
        "episode": episode,
        "metrics": metrics,
    }, ckpt)
    logger.info(f"saved checkpoint -> {ckpt}")
    return ckpt


def load_checkpoint(marl: MARLLayer, egt: EGTLayer, ckpt_path: Path):
    """加载 checkpoint。"""
    import torch
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    marl.load_state_dict(blob["marl_state"])
    egt_state = blob["egt_state"]
    egt.p = np.array(egt_state["p"], dtype=np.float32)
    egt.lambda_history = list(egt_state["lambda_history"])
    ep = int(blob["episode"])
    logger.info(f"loaded checkpoint from {ckpt_path} @ ep={ep}")
    return ep


def train(
    cfg: TrainingConfig,
    save_dir: Path,
    resume: str | None = None,
    checkpoint_interval: int = 25,
    log_every: int = 5,
):
    """主训练循环。"""
    # 1. 环境 + EGT + MARL
    env = DisasterSim(
        seed=cfg.seed,
        map_size=cfg.map_size,
        num_agents=cfg.num_agents,
        num_casualties=cfg.num_casualties,
        num_depots=cfg.num_depots,
        disaster_severity=cfg.disaster_severity,
        malicious_ratio=cfg.malicious_ratio,
        max_steps=cfg.max_steps_per_episode,
    )

    egt = EGTLayer(EGTConfig(
        ema_alpha=cfg.ema_alpha,
        lambda_anchor=cfg.lambda_anchor,
        lambda_anchor_blend=cfg.lambda_anchor_blend,
    ))

    # hidden_dim 通过 MARLConfig 传入
    marl_cfg_obj = MARLConfig(
        hidden_dim=cfg.hidden_dim,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay_steps=cfg.eps_decay_steps,
        grad_clip=cfg.grad_clip,
        target_update_interval=cfg.target_update_interval,
    )
    marl = MARLLayer(
        obs_dim=DEFAULT_OBS_SPEC.dim,
        action_dim=ACTION_DIM,
        num_agents=cfg.num_agents,
        config=marl_cfg_obj,
    )

    # 把 env 的 egt 钩子接到 egt 实例上
    env.egt_signal_fn = egt.get_signal
    env.egt_lambda_fn = egt.get_lambda

    # 2. Replay buffer
    buffer = ReplayBuffer(
        capacity=cfg.buffer_size,
        num_agents=cfg.num_agents,
        obs_dim=DEFAULT_OBS_SPEC.dim,
        action_dim=ACTION_DIM,
    )

    # 3. 恢复（可选）
    start_ep = 0
    if resume:
        start_ep = load_checkpoint(marl, egt, Path(resume))

    # 4. 日志 JSONL
    log_path = save_dir / "training.jsonl"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")

    best_rescue_rate = 0.0
    t0 = time.time()

    # 5. 主循环
    for ep in range(start_ep, cfg.num_episodes):
        obs_dict, info = env.reset()
        egt.reset()

        ep_reward_sum = 0.0
        ep_steps = 0
        last_obs_dict = obs_dict   # 用于 buffer 存"最后一步"的 next_obs=obs_dict
        rescued_this_ep = 0
        deaths_this_ep = 0
        rescue_rate = 0.0      # 初始化防止 UnboundLocalError
        death_rate = 0.0

        for step in range(cfg.max_steps_per_episode):
            # 选动作
            actions = marl.select_actions(obs_dict, eps=marl.eps)
            # step
            next_obs_dict, reward_dict, terminated, truncated, info = env.step(actions)
            ep_steps += 1

            # 团队奖励
            team_reward = float(np.mean(list(reward_dict.values()))) if reward_dict else 0.0
            ep_reward_sum += team_reward

            # 入 buffer
            if ep_steps >= 2:
                buffer.add(Transition(
                    obs=last_obs_dict,
                    actions=actions,
                    reward=team_reward,
                    next_obs=next_obs_dict,
                    done=(terminated or truncated),
                ))

            # EGT 更新
            egt.update(agent_rewards=reward_dict)
            env.egt_signal_fn = egt.get_signal
            env.egt_lambda_fn = egt.get_lambda

            # TD 更新（每 update_freq 步 + buffer 足够）
            if (len(buffer) >= cfg.batch_size and
                step % 10 == 0):
                batch = buffer.sample(cfg.batch_size)
                marl.compute_td_loss(
                    batch["obs"], batch["actions"], batch["rewards"],
                    batch["next_obs"], batch["dones"],
                )

            last_obs_dict = next_obs_dict
            obs_dict = next_obs_dict

            if terminated or truncated:
                rescued_this_ep = env.statistics["total_rescued"]
                deaths_this_ep = env.statistics["total_deaths"]
                break

        # 6. 集末日志
        elapsed = time.time() - t0
        rescue_rate = rescued_this_ep / max(1, cfg.num_casualties)
        death_rate = deaths_this_ep / max(1, cfg.num_casualties)
        lam = egt.get_lambda()
        dom_strat = egt.get_dominant_strategy()

        record = {
            "episode": ep,
            "steps": ep_steps,
            "reward_sum": round(ep_reward_sum, 3),
            "rescued": rescued_this_ep,
            "deaths": deaths_this_ep,
            "rescue_rate": round(rescue_rate, 4),
            "death_rate": round(death_rate, 4),
            "lambda": round(lam, 4),
            "dominant_strategy": dom_strat,
            "eps": round(marl.eps, 4),
            "elapsed_sec": round(elapsed, 1),
            "buffer_size": len(buffer),
        }
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()

        if (ep + 1) % cfg.log_every == 0:
            logger.info(
                f"ep={ep+1}/{cfg.num_episodes} RR={rescue_rate:.2%} "
                f"DR={death_rate:.2%} lam={lam:.3f} dom={dom_strat} "
                f"eps={marl.eps:.3f} rew={ep_reward_sum:.1f} "
                f"buf={len(buffer)} t={elapsed:.0f}s"
            )

        if (ep + 1) % checkpoint_interval == 0:
            save_checkpoint(marl, egt, ep + 1, save_dir, record)
            if rescue_rate > best_rescue_rate:
                best_rescue_rate = rescue_rate
                import torch
                best_path = save_dir / "best.pt"
                torch.save({
                    "marl_state": marl.state_dict(),
                    "egt_state": egt.get_state_dict(),
                    "episode": ep + 1,
                    "rescue_rate": rescue_rate,
                }, best_path)
                logger.info(f"new best RR={rescue_rate:.2%} -> {best_path}")

    log_f.close()
    # 终态 checkpoint
    final_rescue_rate = locals().get("rescue_rate", best_rescue_rate)
    save_checkpoint(marl, egt, cfg.num_episodes, save_dir,
                   {"final": True, "rescue_rate": final_rescue_rate})
    logger.info(f"training done. best_rescue_rate={best_rescue_rate:.2%}")

    # 自动生成 dashboard.png + summary.png
    try:
        from utils.training_curve import load_records, plot_dashboard, plot_summary
        recs = load_records(save_dir / "training.jsonl")
        dash = save_dir / "dashboard.png"
        sm = save_dir / "summary.png"
        plot_dashboard(recs, dash, window=10, title="EGT-MARL v2 (auto)")
        plot_summary(recs, sm, window=10)
        logger.info(f"saved {dash}")
        logger.info(f"saved {sm}")
    except Exception as e:
        logger.warning(f"visualization skipped: {e}")

    return best_rescue_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--num_agents", type=int, default=None)
    p.add_argument("--num_casualties", type=int, default=None)
    p.add_argument("--map_size", type=int, nargs=2, default=None)
    p.add_argument("--buffer_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--target_update_interval", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--eps_end", type=float, default=None)
    p.add_argument("--eps_decay_steps", type=int, default=None)
    p.add_argument("--ema_alpha", type=float, default=None, help="EGT lambda EMA平滑系数，越小越稳定")
    p.add_argument("--max_steps_per_episode", type=int, default=None)
    p.add_argument("--checkpoint_interval", type=int, default=25)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="./train_v2_out")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    # 训练脚本自己的参数（不在 TrainingConfig）
    script_args = {
        "checkpoint_interval": args.checkpoint_interval,
        "log_every": args.log_every,
    }

    # 算法超参（传到 TrainingConfig）
    algo_args = {
        k: v for k, v in vars(args).items()
        if v is not None
        and k not in ("save_dir", "resume", "checkpoint_interval", "log_every")
    }
    cfg = TrainingConfig(**algo_args)
    save_dir = Path(args.save_dir)

    logger.info("=" * 60)
    logger.info("Training v2 (DisasterSim v2 + EGT v2 + MARL v2)")
    logger.info(f"  num_episodes={cfg.num_episodes}  num_agents={cfg.num_agents}  "
                f"num_casualties={cfg.num_casualties}")
    logger.info(f"  obs_dim={DEFAULT_OBS_SPEC.dim}  action_dim={ACTION_DIM}")
    logger.info(f"  save_dir={save_dir}")
    logger.info(f"  checkpoint_interval={args.checkpoint_interval}")
    logger.info("=" * 60)

    train(cfg, save_dir, resume=args.resume,
          checkpoint_interval=args.checkpoint_interval,
          log_every=args.log_every)


if __name__ == "__main__":
    main()