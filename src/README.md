# EGT-MARL: 基于演化博弈论和多智能体强化学习的灾难医疗资源动态分配

本仓库包含 EGT-MARL 框架的 Python 实现，用于灾难期间医疗资源的动态分配。该框架结合了演化博弈论和多智能体强化学习，以实现高效、公平的资源分配策略。

## 项目结构

```
src/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖项
├── setup.py                     # 包安装文件
├── configs/                     # 配置文件
│   ├── disaster_sim.yaml       # 灾难模拟参数
│   ├── egt_marl.yaml           # EGT-MARL 算法参数
│   └── training.yaml           # 训练配置
├── environments/               # 模拟环境
│   ├── __init__.py
│   ├── disaster_sim.py         # 主 DisasterSim-2026 环境
│   ├── disaster_scenarios.py   # 预定义灾难场景
│   └── visualization.py        # 环境可视化
├── algorithms/                 # 算法实现
│   ├── __init__.py
│   ├── egt_marl.py            # 主 EGT-MARL 算法
│   ├── marl_layer.py          # 多智能体强化学习层
│   ├── egt_layer.py           # 演化博弈论层
│   ├── anti_spoofing.py       # 反欺骗机制
│   ├── qmix_improved.py       # 改进的 QMIX 实现
│   └── dynamic_frontier.py    # 动态帕累托前沿
├── agents/                     # 智能体实现
│   ├── __init__.py
│   ├── base_agent.py          # 基础智能体类
│   ├── rescue_agent.py        # 救援智能体（无人机、救护车、移动医院）
│   └── malicious_agent.py     # 用于鲁棒性测试的恶意智能体
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── metrics.py             # 评估指标
│   ├── fairness.py            # 公平性指标（基尼系数、泰尔指数等）
│   ├── visualization.py       # 结果可视化
│   └── data_processing.py     # 数据处理工具
├── experiments/               # 实验脚本
│   ├── __init__.py
│   ├── train_egt_marl.py      # 训练脚本
│   ├── evaluate_baselines.py  # 基准算法评估
│   ├── ablation_study.py      # 消融研究
│   ├── robustness_test.py     # 鲁棒性测试
│   └── run_system_test.py     # 系统集成测试
├── tests/                     # 单元测试
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_algorithms.py
│   └── test_metrics.py
└── notebooks/                 # 用于探索的 Jupyter 笔记本
    ├── 01_environment_demo.ipynb
    ├── 02_algorithm_demo.ipynb
    └── 03_results_analysis.ipynb
```

## 核心功能

### 1. DisasterSim-2026 高保真模拟环境
- 真实灾难建模（地震、洪水、飓风等）
- 动态资源约束和时间关键的生存概率
- 多种救援智能体类型（无人机、救护车、移动医院）
- 通信延迟和故障
- 恶意智能体行为建模

### 2. EGT-MARL 算法框架
- **双层架构**：MARL 执行层 + EGT 调节层
- **改进的 QMIX**：增强的奖励结构和分层动作空间
- **动态公平-效率权衡**：使用演化博弈论进行自适应权重调整
- **反欺骗机制**：贝叶斯真实性验证和声誉系统
- **动态帕累托前沿**：具有自适应权重的多目标优化

### 3. 综合评估指标
- 效率指标：总幸存者数、平均响应时间、资源利用率
- 公平性指标：基尼系数、最大最小公平性、泰尔指数
- 鲁棒性指标：攻击下的性能、系统恢复时间
- 实用性指标：决策时间、通信开销

## 安装指南

### 1. 基本安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/egt-marl-emergency-response.git
cd egt-marl-emergency-response/src

# 创建虚拟环境
python -m venv venv
# Windows 系统：venv\Scripts\activate
# Linux/Mac 系统：source venv/bin/activate

# 安装依赖（使用国内源加速）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 以开发模式安装包
pip install -e .
```

### 2. 解决依赖问题

#### GitHub 依赖问题
如果遇到 GitHub 连接问题，可以使用以下方法：

```bash
# 方法1：使用代理安装
pip install git+https://ghproxy.com/https://github.com/uoe-agents/epymarl.git
pip install git+https://ghproxy.com/https://github.com/facebookresearch/marl_benchmark.git

# 方法2：先安装其他依赖
pip install numpy scipy pandas matplotlib seaborn scikit-learn torch torchvision torchaudio gym gymnasium pettingzoo supersuit nashpy plotly networkx pygame tqdm pyyaml joblib wandb pytest pytest-cov hypothesis black flake8 mypy isort pre-commit -i https://mirrors.aliyun.com/pypi/simple/
```

#### egttools 编译问题
如果 egttools 安装失败（需要 Visual Studio），可以尝试：

```bash
# 尝试安装预编译版本
pip install egttools --only-binary :all:
```

## 快速开始

### 1. 运行简单模拟

```python
from environments.disaster_sim import DisasterSim
from algorithms.egt_marl import EGTMARL

# 创建环境（可选场景：earthquake_standard, flood_standard, hurricane_standard）
env = DisasterSim(scenario="earthquake_standard")

# 创建算法实例
algorithm = EGTMARL(env)

# 运行单个模拟 episode
results = algorithm.run_episode()

# 查看结果
print(f"总幸存者数: {results['total_survivors']}")
print(f"基尼系数（公平性）: {results['gini_coefficient']}")
print(f"平均响应时间: {results['mean_response_time']}")
print(f"资源利用率: {results['resource_utilization']}")
```

### 2. 训练 EGT-MARL 算法

```bash
# 使用默认配置训练
python experiments/train_egt_marl.py --config configs/training.yaml

# 可选参数
# --epochs: 训练轮数
# --batch-size: 批次大小
# --learning-rate: 学习率
python experiments/train_egt_marl.py --config configs/training.yaml --epochs 100 --batch-size 64 --learning-rate 0.0001
```

### 3. 与基准算法比较

```bash
# 评估特定场景
python experiments/evaluate_baselines.py --scenario earthquake_standard

# 评估所有场景
python experiments/evaluate_baselines.py --all

# 可选参数
# --baselines: 指定基准算法（qmix, vdn, maddpg）
# --runs: 运行次数
python experiments/evaluate_baselines.py --scenario flood_standard --baselines qmix vdn --runs 5
```

### 4. 使用 Jupyter Notebook 进行探索

```bash
# 启动 Jupyter Notebook
jupyter notebook

# 打开以下 notebook 进行交互式探索：
# - notebooks/01_environment_demo.ipynb  # 环境演示
# - notebooks/02_algorithm_demo.ipynb   # 算法演示
# - notebooks/03_results_analysis.ipynb  # 结果分析
```

### 5. 运行系统集成测试

```bash
# 运行完整的系统集成与测试流程
python experiments/run_system_test.py --run-all 

# 可选参数
# --run-all: 运行所有任务
# --run-tests: 仅运行系统测试
# --train-egt-marl: 仅训练 EGT-MARL 算法
# --evaluate-baselines: 仅评估基线算法
# --run-ablation: 仅运行消融研究
# --test-robustness: 仅测试鲁棒性
# --test-mode: 测试模式（quick, integration, comprehensive）
# --quick-mode: 快速测试模式，减少episode数量以加快测试速度
# --num-episodes: 每个任务的 episode 数量
# --stop-on-failure: 任务失败时停止执行

# 注：参数可以自由组合，不互斥

# 示例1：仅运行测试和训练
python experiments/run_system_test.py --run-tests --train-egt-marl --test-mode quick

# 示例2：使用快速测试模式运行所有任务
python experiments/run_system_test.py --run-all --quick-mode

# 示例3：使用快速测试模式并指定episode数量
python experiments/run_system_test.py --run-all --quick-mode --num-episodes 5
```

## 配置说明

系统通过 YAML 文件高度可配置，主要配置文件如下：

### 1. 灾难模拟配置 (`configs/disaster_sim.yaml`)

**核心参数**：
- `scenario`: 灾难场景（earthquake, flood, hurricane）
- `grid_size`: 模拟区域大小
- `population_density`: 人口密度
- `resource_initial`: 初始资源配置
- `agent_types`: 救援智能体类型和数量
- `survival_probability`: 生存概率模型参数
- `communication_delay`: 通信延迟设置

**示例配置**：
```yaml
scenario: earthquake_standard
grid_size: [50, 50]
population_density: 0.8
resource_initial:
  medical_supplies: 1000
  personnel: 50
agent_types:
  drones: 10
  ambulances: 5
  mobile_hospitals: 2
```

### 2. EGT-MARL 算法配置 (`configs/egt_marl.yaml`)

**核心参数**：
- `marl_algorithm`: MARL 算法选择（improved_qmix, qmix, vdn）
- `network_architecture`: 神经网络架构参数
- `egt_parameters`: 演化博弈论参数
- `anti_spoofing`: 反欺骗机制配置
- `dynamic_frontier`: 动态帕累托前沿配置

**示例配置**：
```yaml
marl_algorithm: improved_qmix
network_architecture:
  hidden_dim: 64
  num_layers: 2
egt_parameters:
  population_size: 100
  mutation_rate: 0.1
anti_spoofing:
  enabled: true
  reputation_weight: 0.7
```

### 3. 训练配置 (`configs/training.yaml`)

**核心参数**：
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `gamma`: 折扣因子
- `epsilon_start`: ε-贪婪策略初始值
- `epsilon_end`: ε-贪婪策略最终值
- `target_update_freq`: 目标网络更新频率

**示例配置**：
```yaml
epochs: 500
batch_size: 64
learning_rate: 0.0005
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.01
target_update_freq: 100
```

## 实验与结果

### 1. 标准性能比较

```bash
# 评估所有场景和基准算法
python experiments/evaluate_baselines.py --all

# 运行结果将保存在 `experiment_results/performance_comparison/` 目录
# 包含效率、公平性、鲁棒性等多维度指标
```

### 2. 消融实验

```bash
# 评估不同组件的贡献
python experiments/ablation_study.py --components egt anti_spoofing dynamic_frontier

# 可选参数
# --scenarios: 指定场景
# --runs: 运行次数
python experiments/ablation_study.py --components egt anti_spoofing --scenarios earthquake_standard flood_standard --runs 3

# 结果将保存在 `experiment_results/ablation_studies/` 目录
```

### 3. 鲁棒性测试

```bash
# 测试不同攻击强度下的性能
python experiments/robustness_test.py --attack_levels 0.1 0.2 0.3

# 可选参数
# --attack_types: 攻击类型（spoofing, jamming, collusion）
# --defense_enabled: 是否启用防御机制
python experiments/robustness_test.py --attack_levels 0.1 0.2 0.3 --attack_types spoofing jamming --defense_enabled true

# 结果将保存在 `experiment_results/robustness_tests/` 目录
```

### 4. 结果可视化

```bash
# 使用工具脚本生成可视化结果
python utils/visualization.py --results_dir experiment_results/performance_comparison/

# 生成的图表将保存在 `experiment_results/visualizations/` 目录
```

## 常见问题与解决方案

### 1. 依赖安装问题
- **GitHub 连接失败**：使用代理安装或先安装其他依赖
- **egttools 编译失败**：尝试使用 `pip install egttools --only-binary :all:`
- **权限错误**：使用管理员权限运行命令行

### 2. 运行问题
- **CUDA 错误**：确保安装了正确版本的 CUDA 或使用 CPU 模式
- **内存不足**：减小 batch size 或使用更小的网格大小
- **导入错误**：确保在正确的虚拟环境中运行

### 3. 结果解释
- **基尼系数**：值越小表示资源分配越公平（0 表示完全公平）
- **总幸存者数**：直接反映救援效率
- **平均响应时间**：越短越好，反映救援速度
- **资源利用率**：越高越好，反映资源使用效率

## 贡献

欢迎通过以下方式贡献本项目：
- 提交问题和功能请求
- 提交代码改进
- 改进文档
- 分享使用案例

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 联系方式

如有问题或建议，请在 GitHub 上打开 issue 或联系项目维护者。