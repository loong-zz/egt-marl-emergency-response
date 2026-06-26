"""
统一评估脚本：EGT-MARL (final/best) + 全部基线 + 决策时延
输出 JSON + CSV 到 evaluation_results/unified_eval/
"""
import os
import sys
import json
import time
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import importlib.util as _ilu
import types
import warnings
warnings.filterwarnings('ignore')

# Stub out tensorboard to avoid the optional dependency
try:
    import tensorboard  # noqa: F401
except ImportError:
    _tb = types.ModuleType('tensorboard')
    _tb_summary = types.ModuleType('tensorboard.summary')
    _tb_summary.Writer = object
    _tb.summary = _tb_summary
    sys.modules['tensorboard'] = _tb
    sys.modules['tensorboard.summary'] = _tb_summary
    _torch_tb = types.ModuleType('torch.utils.tensorboard')
    _torch_tb.SummaryWriter = object
    sys.modules['torch.utils.tensorboard'] = _torch_tb

# Path setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Direct module load for evaluate_baselines to avoid the experiments package __init__
# (which imports train_egt_marl.py -> tensorboard -> missing module)
_eval_spec = _ilu.spec_from_file_location(
    '_eval_baselines', str(Path(__file__).parent / 'evaluate_baselines.py')
)
_eval_mod = _ilu.module_from_spec(_eval_spec)
sys.modules['_eval_baselines'] = _eval_mod
_eval_spec.loader.exec_module(_eval_mod)
BaselineEvaluator = _eval_mod.BaselineEvaluator

from environments.disaster_sim import DisasterSim
from environments.config.constants import SimulationConfig, NUM_STRATEGIES
from algorithms.egt_marl import EGTMARL
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_env(seed: int, config: dict):
    env_config = config.get('environment', {})
    sim_config = SimulationConfig()
    sim_config.num_agents = env_config.get('num_agents', 20)
    sim_config.num_victims = env_config.get('num_victims', 200)
    sim_config.num_resources = env_config.get('num_resources', 10)
    sim_config.num_hospitals = env_config.get('num_hospitals', 3)
    sim_config.map_size = tuple(env_config.get('map_size', [1000, 1000]))
    sim_config.disaster_type = env_config.get('disaster_type', 'earthquake')
    sim_config.severity = env_config.get('severity', 'medium')

    env = DisasterSim(
        scenario=env_config.get('scenario', 'earthquake_standard'),
        map_size=sim_config.map_size,
        num_agents=sim_config.num_agents,
        num_victims=sim_config.num_victims,
        num_resources=sim_config.num_resources,
        num_hospitals=sim_config.num_hospitals,
        disaster_type=sim_config.disaster_type,
        severity=sim_config.severity,
        config=sim_config,
    )
    return env


def make_algorithm(env, config: dict, checkpoint_path: str):
    algo_config = config.get('algorithm', {})
    training_config = config.get('training', {})
    algo_config_dict = {
        'marl': {
            'state_dim': env.get_state_dimension(),
            'action_dim': 32,
            'num_agents': len(env.rescue_agents),
            'hidden_dim': algo_config.get('hidden_dim', 64),
            'mixing_hidden_dim': algo_config.get('mixing_hidden_dim', 64),
            'attention_heads': algo_config.get('attention_heads', 4),
            'learning_rate': training_config.get('learning_rate', 0.0001),
            'batch_size': training_config.get('batch_size', 32),
            'buffer_size': training_config.get('buffer_size', 10000),
        },
        'egt': {'num_strategies': NUM_STRATEGIES, 'learning_rate': 0.01},
        'anti_spoofing': {'observation_dim': env.get_state_dimension(), 'action_dim': 32},
        'dynamic_frontier': {
            'alpha': algo_config.get('pareto_weight_alpha', 0.3),
            'beta': algo_config.get('pareto_weight_beta', 0.4),
            'gamma': algo_config.get('pareto_weight_gamma', 0.3),
        },
    }
    algorithm = EGTMARL(env=env, config=algo_config_dict, hidden_dim=algo_config.get('hidden_dim', 64))
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    if 'marl_layer_state' in ckpt:
        try:
            algorithm.marl_layer.load_state_dict(ckpt['marl_layer_state'])
        except RuntimeError:
            algorithm.marl_layer.load_state_dict(ckpt['marl_layer_state'], strict=False)
    if 'egt_layer_state' in ckpt:
        try:
            algorithm.egt_layer.load_state_dict(ckpt['egt_layer_state'])
        except RuntimeError:
            algorithm.egt_layer.load_state_dict(ckpt['egt_layer_state'], strict=False)
    if 'anti_spoofing_state' in ckpt:
        try:
            if hasattr(algorithm.anti_spoofing, 'load_state_dict'):
                algorithm.anti_spoofing.load_state_dict(ckpt['anti_spoofing_state'])
        except Exception:
            pass
    return algorithm


def run_episode_egt(algo, env, seed: int, max_steps: int = 1200):
    """EGT-MARL episode with per-step latency."""
    np.random.seed(seed)
    try:
        if hasattr(env, 'np_random'):
            env.np_random = np.random.RandomState(seed)
    except Exception:
        pass
    state, info = env.reset()
    ep_reward = 0.0
    ep_latency_ms = []
    done = False
    step = 0
    while not done and step < max_steps:
        t0 = time.perf_counter()
        actions = algo.select_action(state, training=False, epsilon=0.0)
        t1 = time.perf_counter()
        ep_latency_ms.append((t1 - t0) * 1000.0)
        next_state, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        ep_reward += sum(rewards.values()) if isinstance(rewards, dict) else float(rewards)
        state = next_state
        step += 1
    return _extract_metrics(env, info, ep_reward, ep_latency_ms, step)


def run_episode_baseline(policy, env, seed: int, max_steps: int = 1200):
    """Baseline policy episode."""
    np.random.seed(seed)
    try:
        if hasattr(env, 'np_random'):
            env.np_random = np.random.RandomState(seed)
    except Exception:
        pass
    state, info = env.reset()
    ep_reward = 0.0
    done = False
    step = 0
    agent_ids = sorted(env.rescue_agents.keys())
    while not done and step < max_steps:
        actions_list = policy.select_actions(state, epsilon=0.0)
        # Convert list[int] -> dict[agent_id, {'tactical': int}]
        actions_dict = {aid: {'tactical': int(actions_list[i])} for i, aid in enumerate(agent_ids)}
        next_state, rewards, terminated, truncated, info = env.step(actions_dict)
        done = terminated or truncated
        ep_reward += sum(rewards.values()) if isinstance(rewards, dict) else float(rewards)
        state = next_state
        step += 1
    return _extract_metrics(env, info, ep_reward, [], step)


def _extract_metrics(env, info, ep_reward, ep_latency_ms, step):
    stats = info.get('statistics', {}) if info else {}
    rescued = stats.get('total_rescued', 0)
    deaths = stats.get('total_deaths', 0)
    total = len(env.casualties)
    rr = (rescued / total * 100.0) if total > 0 else 0.0
    rt_list = stats.get('response_times', [])
    rt = float(np.mean(rt_list)) if rt_list else 0.0
    util_list = stats.get('resource_utilization', [])
    util = float(np.mean(util_list)) if util_list else 0.0
    return {
        'rescue_rate': rr,
        'rescued': rescued,
        'deaths': deaths,
        'total': total,
        'response_time': rt,
        'resource_util': util,
        'reward': ep_reward,
        'steps': step,
        'latency_ms': np.mean(ep_latency_ms) if ep_latency_ms else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--out_dir', default='evaluation_results/unified_eval')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--episodes_per_seed', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=1200)
    p.add_argument('--models', default='final,best')
    p.add_argument('--baselines', default='FCFS,Priority,GreedyLocal,ProportionalFair,CentralizedMPC,GameTheoretic,GNNBased,TransformerBased,StandardMARL')
    p.add_argument('--baseline_config', default='src/experiments/configs/evaluate_baselines.yaml')
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    models = [m.strip() for m in args.models.split(',')]
    baselines = [b.strip() for b in args.baselines.split(',')]

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_subdir = out_dir / f'eval_{ts}'
    out_subdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output to: {out_subdir}")

    all_results = []

    # === EGT-MARL models ===
    for mname in models:
        ckpt_path = Path(args.checkpoint_dir) / 'models' / f'{mname}_model.pt'
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}, skip {mname}")
            continue
        logger.info(f"=== Loading {mname} model: {ckpt_path} ===")
        env = make_env(0, config)
        algo = make_algorithm(env, config, str(ckpt_path))
        if hasattr(algo, 'eval_mode'):
            algo.eval_mode()
        elif hasattr(algo, 'set_eval_mode'):
            algo.set_eval_mode()
        for seed in seeds:
            for ep in range(args.episodes_per_seed):
                r = run_episode_egt(algo, env, seed=seed * 1000 + ep, max_steps=args.max_steps)
                r['method'] = f'EGT-MARL({mname})'
                r['seed'] = seed
                r['episode'] = ep
                all_results.append(r)
                if (ep + 1) % 5 == 0:
                    logger.info(f"  [{mname}] seed={seed} ep={ep+1}/{args.episodes_per_seed}: RR={r['rescue_rate']:.1f}%")
        del algo

    # === Baselines via BaselineEvaluator ===
    if baselines:
        logger.info(f"=== Initializing BaselineEvaluator with {args.baseline_config} ===")
        evaluator = BaselineEvaluator(args.baseline_config)
        evaluator.setup_directories = lambda: None  # disable directory creation
        # Map baseline name -> factory method suffix
        BNAME_TO_FACTORY = {
            'FCFS': '_create_fcfs_policy',
            'Priority': '_create_priority_policy',
            'GNNBased': '_create_gnn_policy',
            'TransformerBased': '_create_transformer_policy',
            'GreedyLocal': '_create_greedy_policy',
            'ProportionalFair': '_create_proportional_fair_policy',
            'StandardMARL': '_create_standard_marl_policy',
            'CentralizedMPC': '_create_mpc_policy',
            'GameTheoretic': '_create_game_theoretic_policy',
        }
        # Override config to use our larger env
        evaluator.env_config = config.get('environment', {})
        for bname in baselines:
            logger.info(f"=== Baseline: {bname} ===")
            for seed in seeds:
                for ep in range(args.episodes_per_seed):
                    env = make_env(seed * 1000 + ep, config)
                    np.random.seed(seed * 1000 + ep)
                    try:
                        if hasattr(env, 'np_random'):
                            env.np_random = np.random.RandomState(seed * 1000 + ep)
                    except Exception:
                        pass
                    state, _ = env.reset()
                    factory_name = BNAME_TO_FACTORY.get(bname)
                    try:
                        factory = getattr(evaluator, factory_name) if factory_name else None
                        if factory is None:
                            raise ValueError(f"No factory for {bname}")
                        # Some factories read self.env (e.g. StandardMARL, TransformerBased)
                        evaluator.env = env
                        # DisasterSim doesn't expose num_agents; derive from rescue_agents
                        if not hasattr(env, 'num_agents'):
                            env.num_agents = len(env.rescue_agents)
                        policy = factory()
                    except Exception as e:
                        logger.warning(f"  [skip] {bname} seed={seed} ep={ep}: {e}")
                        continue
                    r = run_episode_baseline(policy, env, seed=seed * 1000 + ep, max_steps=args.max_steps)
                    r['method'] = bname
                    r['seed'] = seed
                    r['episode'] = ep
                    all_results.append(r)
                    if (ep + 1) % 5 == 0:
                        logger.info(f"  [{bname}] seed={seed} ep={ep+1}/{args.episodes_per_seed}: RR={r['rescue_rate']:.1f}%")

    # === Aggregate ===
    methods = sorted(set(r['method'] for r in all_results))
    summary = []
    for m in methods:
        rows = [r for r in all_results if r['method'] == m]
        seeds_set = sorted(set(r['seed'] for r in rows))
        per_seed_rr = []
        per_seed_rt = []
        per_seed_util = []
        latencies = [r['latency_ms'] for r in rows if r.get('latency_ms') is not None]
        for s in seeds_set:
            srows = [r for r in rows if r['seed'] == s]
            per_seed_rr.append(np.mean([r['rescue_rate'] for r in srows]))
            per_seed_rt.append(np.mean([r['response_time'] for r in srows]))
            per_seed_util.append(np.mean([r['resource_util'] for r in srows]))
        summary.append({
            'method': m,
            'n_episodes': len(rows),
            'rescue_rate_mean': float(np.mean(per_seed_rr)),
            'rescue_rate_std': float(np.std(per_seed_rr)),
            'rescue_rate_min': float(min(per_seed_rr)),
            'rescue_rate_max': float(max(per_seed_rr)),
            'response_time_mean': float(np.mean(per_seed_rt)),
            'resource_util_mean': float(np.mean(per_seed_util)),
            'reward_mean': float(np.mean([r['reward'] for r in rows])),
            'deaths_mean': float(np.mean([r['deaths'] for r in rows])),
            'latency_ms_mean': float(np.mean(latencies)) if latencies else None,
            'latency_ms_p50': float(np.percentile(latencies, 50)) if latencies else None,
            'latency_ms_p95': float(np.percentile(latencies, 95)) if latencies else None,
            'latency_ms_p99': float(np.percentile(latencies, 99)) if latencies else None,
        })

    summary_sorted = sorted(summary, key=lambda x: -x['rescue_rate_mean'])

    with open(out_subdir / 'raw_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(out_subdir / 'summary.json', 'w') as f:
        json.dump(summary_sorted, f, indent=2)

    keys = ['method', 'n_episodes', 'rescue_rate_mean', 'rescue_rate_std',
            'rescue_rate_min', 'rescue_rate_max', 'response_time_mean',
            'resource_util_mean', 'reward_mean', 'deaths_mean',
            'latency_ms_mean', 'latency_ms_p50', 'latency_ms_p95', 'latency_ms_p99']
    with open(out_subdir / 'summary.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for s in summary_sorted:
            row = []
            for k in keys:
                v = s.get(k)
                if v is None:
                    row.append('')
                elif isinstance(v, float):
                    row.append(f'{v:.4f}')
                else:
                    row.append(str(v))
            f.write(','.join(row) + '\n')

    print('\n' + '=' * 100)
    print('UNIFIED EVALUATION SUMMARY (sorted by Rescue Rate)')
    print('=' * 100)
    print(f"{'Method':<22} {'RR%':>7} {'±':>5} {'min':>5} {'max':>5} {'RT(s)':>7} {'Util%':>6} {'Reward':>9} {'Deaths':>7} {'Lat(ms)':>9}")
    print('-' * 100)
    for s in summary_sorted:
        rr = f"{s['rescue_rate_mean']:.2f}"
        sd = f"{s['rescue_rate_std']:.2f}"
        mn = f"{s['rescue_rate_min']:.2f}"
        mx = f"{s['rescue_rate_max']:.2f}"
        rt = f"{s['response_time_mean']:.1f}"
        ut = f"{s['resource_util_mean']:.1f}"
        rw = f"{s['reward_mean']:.1f}"
        dh = f"{s['deaths_mean']:.1f}"
        lt = f"{s['latency_ms_mean']:.2f}" if s['latency_ms_mean'] else "N/A"
        print(f"{s['method']:<22} {rr:>7} {sd:>5} {mn:>5} {mx:>5} {rt:>7} {ut:>6} {rw:>9} {dh:>7} {lt:>9}")
    print('=' * 100)
    print(f"\nResults saved to: {out_subdir}")


if __name__ == '__main__':
    main()