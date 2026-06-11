"""
Dynamic Pareto Frontier Parameter Tuning Script

Tune the Dynamic Pareto Frontier component to optimize performance.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.disaster_sim import DisasterSim
from algorithms.egt_marl import EGTMARL
from algorithms.dynamic_frontier import DynamicParetoFrontier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicFrontierTuner:
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results = []

    def create_frontier_config(self, param_combination: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'frontier_size': param_combination.get('frontier_size', 50),
            'weight_adaptation_rate': param_combination.get('weight_adaptation_rate', 0.05),
            'min_weight': param_combination.get('min_weight', 0.1),
            'max_weight': param_combination.get('max_weight', 0.8),
            'mutation_strength': param_combination.get('mutation_strength', 0.1),
            'crossover_rate': param_combination.get('crossover_rate', 0.7),
            'elitism_rate': param_combination.get('elitism_rate', 0.1),
            'population_size': param_combination.get('population_size', 100),
        }

    def evaluate_config(self, config: Dict[str, Any], num_episodes: int = 20) -> Dict[str, float]:
        env_config = self.base_config['environment']
        env = DisasterSim(
            num_agents=env_config['num_agents'],
            num_victims=env_config['num_victims'],
            map_size=tuple(env_config['map_size']),
            num_resources=env_config.get('num_resources', 10),
            num_hospitals=env_config.get('num_hospitals', 3),
            disaster_type=env_config.get('disaster_type', 'earthquake'),
            severity=env_config.get('severity', 'medium')
        )

        frontier_config = self.create_frontier_config(config)
        full_config = {
            'marl': {
                'hidden_dim': 64,
                'mixing_hidden_dim': 32,
                'attention_heads': 4,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'tau': 0.01,
                'batch_size': 32,
                'buffer_size': 5000,
            },
            'egt': {'lambda': config.get('egt_lambda', 0.5), 'num_strategies': 5, 'learning_rate': 0.01},
            'dynamic_frontier': frontier_config,
            'anti_spoofing': {'enabled': config.get('anti_spoofing', True), 'observation_dim': 64},
        }
        algorithm = EGTMARL(
            state_dim=env.get_state_dimension(),
            action_dim=env.get_action_dimension(),
            num_agents=env_config['num_agents'],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            config=full_config
        )

        episode_rewards = []
        episode_rescue_rates = []

        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 200

            while steps < max_steps:
                actions = algorithm.select_actions(state, epsilon=0.1)
                next_state, reward, done, info = env.step(actions)

                algorithm.store_transition(state, actions, reward, next_state, done)
                algorithm.train()

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            rescue_rate = info.get('rescued', 0) / env_config['num_victims'] * 100
            episode_rescue_rates.append(rescue_rate)

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_rescue_rate': np.mean(episode_rescue_rates),
            'std_rescue_rate': np.std(episode_rescue_rates),
        }

    def run_tuning(self, output_dir: str = 'dynamic_frontier_tuning'):
        param_grid = [
            {'name': 'baseline', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'large_frontier', 'frontier_size': 100, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'small_frontier', 'frontier_size': 25, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'fast_adaptation', 'frontier_size': 50, 'weight_adaptation_rate': 0.15, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'slow_adaptation', 'frontier_size': 50, 'weight_adaptation_rate': 0.01, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'high_mutation', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.3, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'low_mutation', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.05, 'elitism_rate': 0.1, 'egt_lambda': 0.5},
            {'name': 'high_elitism', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.3, 'egt_lambda': 0.5},
            {'name': 'low_elitism', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.05, 'egt_lambda': 0.5},
            {'name': 'lambda_0.3', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.3},
            {'name': 'lambda_0.7', 'frontier_size': 50, 'weight_adaptation_rate': 0.05, 'mutation_strength': 0.1, 'elitism_rate': 0.1, 'egt_lambda': 0.7},
            {'name': 'balanced', 'frontier_size': 75, 'weight_adaptation_rate': 0.1, 'mutation_strength': 0.15, 'elitism_rate': 0.15, 'egt_lambda': 0.5},
            {'name': 'aggressive', 'frontier_size': 50, 'weight_adaptation_rate': 0.2, 'mutation_strength': 0.25, 'elitism_rate': 0.2, 'egt_lambda': 0.5},
            {'name': 'conservative', 'frontier_size': 50, 'weight_adaptation_rate': 0.02, 'mutation_strength': 0.05, 'elitism_rate': 0.05, 'egt_lambda': 0.5},
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(output_dir) / f"tuning_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Dynamic Frontier parameter tuning...")
        logger.info(f"Testing {len(param_grid)} parameter combinations")

        for params in param_grid:
            logger.info(f"Evaluating: {params['name']}")
            try:
                metrics = self.evaluate_config(params)
                result = {**params, **metrics}
                self.results.append(result)
                logger.info(f"  Rescue Rate: {metrics['mean_rescue_rate']:.2f}% (±{metrics['std_rescue_rate']:.2f})")
                logger.info(f"  Mean Reward: {metrics['mean_reward']:.2f} (±{metrics['std_reward']:.2f})")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                self.results.append({**params, 'error': str(e)})

        df = self._create_results_dataframe()
        summary_path = results_dir / 'tuning_summary.csv'
        df.to_csv(summary_path, index=False)

        best_idx = df['mean_rescue_rate'].idxmax()
        best_config = df.iloc[best_idx]

        report_path = results_dir / 'tuning_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Dynamic Pareto Frontier Parameter Tuning Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Tuning Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Configurations Tested: {len(param_grid)}\n\n")

            f.write("Results Summary:\n")
            f.write("-" * 60 + "\n")
            f.write(df[['name', 'mean_rescue_rate', 'std_rescue_rate', 'mean_reward', 'std_reward']].to_string(index=False))
            f.write("\n\n")

            f.write("Best Configuration:\n")
            f.write("-" * 60 + "\n")
            for key, value in best_config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nKey Findings:\n")
            f.write("-" * 60 + "\n")

            baseline = df[df['name'] == 'baseline']['mean_rescue_rate'].values[0]
            for _, row in df.iterrows():
                diff = row['mean_rescue_rate'] - baseline
                if abs(diff) > 1.0:
                    f.write(f"  - {row['name']}: {diff:+.2f}% vs baseline\n")

            f.write("\nRecommendations:\n")
            f.write("-" * 60 + "\n")

            best_row = df.loc[df['mean_rescue_rate'].idxmax()]
            f.write(f"1. Optimal frontier_size: {int(best_row['frontier_size'])}\n")
            f.write(f"2. Optimal weight_adaptation_rate: {best_row['weight_adaptation_rate']}\n")
            f.write(f"3. Optimal mutation_strength: {best_row['mutation_strength']}\n")
            f.write(f"4. Optimal elitism_rate: {best_row['elitism_rate']}\n")
            f.write(f"5. Optimal egt_lambda: {best_row['egt_lambda']}\n")

        logger.info(f"\nTuning completed! Results saved to {results_dir}")
        logger.info(f"Best configuration: {best_config['name']} with {best_config['mean_rescue_rate']:.2f}% rescue rate")

        return df, best_config

    def _create_results_dataframe(self):
        import pandas as pd
        data = []
        for result in self.results:
            row = {
                'name': result.get('name', 'unknown'),
                'frontier_size': result.get('frontier_size', 0),
                'weight_adaptation_rate': result.get('weight_adaptation_rate', 0),
                'mutation_strength': result.get('mutation_strength', 0),
                'elitism_rate': result.get('elitism_rate', 0),
                'egt_lambda': result.get('egt_lambda', 0),
            }
            if 'mean_rescue_rate' in result:
                row.update({
                    'mean_rescue_rate': result['mean_rescue_rate'],
                    'std_rescue_rate': result['std_rescue_rate'],
                    'mean_reward': result['mean_reward'],
                    'std_reward': result['std_reward'],
                })
            else:
                row.update({
                    'mean_rescue_rate': 0,
                    'std_rescue_rate': 0,
                    'mean_reward': 0,
                    'std_reward': 0,
                    'error': result.get('error', 'unknown')
                })
            data.append(row)
        return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Tune Dynamic Pareto Frontier parameters')
    parser.add_argument('--output_dir', type=str, default='dynamic_frontier_tuning',
                       help='Output directory for tuning results')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes per configuration')
    args = parser.parse_args()

    base_config = {
        'environment': {
            'num_agents': 5,
            'num_victims': 20,
            'map_size': [100, 100],
            'num_resources': 10,
            'num_hospitals': 3,
            'disaster_type': 'earthquake',
            'severity': 'medium'
        }
    }

    tuner = DynamicFrontierTuner(base_config)
    df, best_config = tuner.run_tuning(output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best Configuration: {best_config['name']}")
    print(f"Best Rescue Rate: {best_config['mean_rescue_rate']:.2f}%")


if __name__ == '__main__':
    main()
