# EGT-MARL: Evolutionary Game Theory - Multi-Agent Reinforcement Learning for Disaster Medical Resource Allocation

This repository contains the Python implementation of the EGT-MARL framework for dynamic allocation of medical resources during disasters, as described in the paper "Lifelines in a Zero-Sum Dilemma: Dynamic Allocation of Medical Resources in Disasters via Evolutionary Game Theory and Multi-Agent Reinforcement Learning".

## Project Structure

```
src/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── configs/                     # Configuration files
│   ├── disaster_sim.yaml       # Disaster simulation parameters
│   ├── egt_marl.yaml           # EGT-MARL algorithm parameters
│   └── training.yaml           # Training configuration
├── environments/               # Simulation environments
│   ├── __init__.py
│   ├── disaster_sim.py         # Main DisasterSim-2026 environment
│   ├── disaster_scenarios.py   # Pre-defined disaster scenarios
│   └── visualization.py        # Environment visualization
├── algorithms/                 # Algorithm implementations
│   ├── __init__.py
│   ├── egt_marl.py            # Main EGT-MARL algorithm
│   ├── marl_layer.py          # Multi-agent RL layer
│   ├── egt_layer.py           # Evolutionary game theory layer
│   ├── anti_spoofing.py       # Anti-spoofing mechanism
│   ├── qmix_improved.py       # Improved QMIX implementation
│   └── dynamic_frontier.py    # Dynamic Pareto frontier
├── agents/                     # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── rescue_agent.py        # Rescue agent (drone, ambulance, mobile hospital)
│   └── malicious_agent.py     # Malicious agent for robustness testing
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── metrics.py             # Evaluation metrics
│   ├── fairness.py            # Fairness metrics (Gini, Theil, etc.)
│   ├── visualization.py       # Result visualization
│   └── data_processing.py     # Data processing utilities
├── experiments/               # Experiment scripts
│   ├── __init__.py
│   ├── train_egt_marl.py      # Training script
│   ├── evaluate_baselines.py  # Baseline evaluation
│   ├── ablation_study.py      # Ablation studies
│   └── robustness_test.py     # Robustness testing
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_algorithms.py
│   └── test_metrics.py
└── notebooks/                 # Jupyter notebooks for exploration
    ├── 01_environment_demo.ipynb
    ├── 02_algorithm_demo.ipynb
    └── 03_results_analysis.ipynb
```

## Key Features

### 1. DisasterSim-2026 High-Fidelity Simulation Environment
- Realistic disaster modeling (earthquake, flood, etc.)
- Dynamic resource constraints and time-critical survival probabilities
- Multiple rescue agent types (drones, ambulances, mobile hospitals)
- Communication delays and failures
- Malicious agent behavior modeling

### 2. EGT-MARL Algorithm Framework
- **Two-layer architecture**: MARL execution layer + EGT regulation layer
- **Improved QMIX**: Enhanced reward structure and hierarchical action space
- **Dynamic fairness-efficiency trade-off**: Evolutionary game theory for adaptive weighting
- **Anti-spoofing mechanism**: Bayesian truthfulness verification and reputation system
- **Dynamic Pareto frontier**: Multi-objective optimization with adaptive weights

### 3. Comprehensive Evaluation Metrics
- Efficiency metrics: Total survivors, mean response time, resource utilization
- Fairness metrics: Gini coefficient, max-min fairness, Theil index
- Robustness metrics: Performance under attacks, system recovery time
- Practicality metrics: Decision time, communication overhead

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd src

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Run a simple simulation
```python
from environments.disaster_sim import DisasterSim
from algorithms.egt_marl import EGTMARL

# Create environment
env = DisasterSim(scenario="earthquake_standard")

# Create algorithm
algorithm = EGTMARL(env)

# Run simulation
results = algorithm.run_episode()
print(f"Total survivors: {results['total_survivors']}")
print(f"Gini coefficient: {results['gini_coefficient']}")
```

### 2. Train the EGT-MARL algorithm
```bash
python experiments/train_egt_marl.py --config configs/training.yaml
```

### 3. Evaluate against baselines
```bash
python experiments/evaluate_baselines.py --scenario earthquake_standard
```

## Configuration

The system is highly configurable through YAML files:

- `configs/disaster_sim.yaml`: Disaster parameters, agent types, resource constraints
- `configs/egt_marl.yaml`: Algorithm hyperparameters, network architectures
- `configs/training.yaml`: Training schedules, optimization parameters

## Results Reproduction

To reproduce the paper results:

1. **Standard performance comparison**:
   ```bash
   python experiments/evaluate_baselines.py --all
   ```

2. **Ablation studies**:
   ```bash
   python experiments/ablation_study.py --components egt anti_spoofing dynamic_frontier
   ```

3. **Robustness testing**:
   ```bash
   python experiments/robustness_test.py --attack_levels 0.1 0.2 0.3
   ```

## Citation

If you use this code in your research, please cite:

```
@article{egtmarl2026,
  title={Lifelines in a Zero-Sum Dilemma: Dynamic Allocation of Medical Resources in Disasters via Evolutionary Game Theory and Multi-Agent Reinforcement Learning},
  author={Your Name},
  journal={Journal of Artificial Intelligence Research},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].