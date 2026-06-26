# AGENTS.md

> EGT-MARL 项目 AI Agent 工作指南
>
> 目的:让新接手的 AI 编码 Agent 在 5 分钟内掌握项目骨架、配置/常量体系、运行命令与环境,避免在训练/评估流程上踩坑浪费算力。

---

## 1. 一句话项目说明

基于演化博弈论(EGT)与多智能体强化学习(MARL)双层架构的灾难医疗资源动态分配框架。MARL 执行层(17 智能体,drones + ambulances + hospitals)走 `ImprovedQMIX + HierarchicalActionSpace`,EGT 调节层用演化博弈动态调整公平-效率权重与策略分布,影响 QMIX 奖励塑形。论文提交物在 `doc/基于演化博弈与多智能体强化的灾时医疗物资动态分配机制.md`。

---

## 2. 目录速查

```
egt-marl-emergency-response/
├── src/
│   ├── algorithms/        # 核心算法 (egt_marl.py 是入口,其它都是子模块)
│   │   ├── egt_marl.py          # EGT-MARL 主类
│   │   ├── marl_layer.py        # 轻量级 MARL 备选路径
│   │   ├── egt_layer.py         # 演化博弈层 (payoff/replicator dynamics)
│   │   ├── qmix_improved.py     # 改进 QMIX (5-component reward, AttentionMixing)
│   │   ├── anti_spoofing.py     # 反欺骗
│   │   └── dynamic_frontier.py  # 帕累托前沿
│   ├── environments/      # DisasterSim-2026 + managers + 实体
│   │   └── config/
│   │       └── constants.py     # ★ 项目级硬常量集中地 (仿真 + 算法)
│   ├── agents/            # RescueAgent 三种类型
│   ├── experiments/       # 训练/评估/消融/鲁棒性脚本 + 实验用 yaml
│   │   └── configs/
│   ├── configs/           # 默认配置 (egt_marl.yaml / training.yaml / evaluation.yaml)
│   ├── utils/             # 指标/公平性/可视化
│   ├── .venv/             # ★ Python 虚拟环境 (WSL 风格 bin/python)
│   ├── requirements.txt
│   └── README.md
├── doc/                   # 论文/操作手册
├── AGENTS.md              # 本文件
└── README.md
```

---

## 3. 常量与配置:谁覆盖谁

⚠️ 修改前先看这张表。值可以从三个地方来,**优先级从高到低**:

```
优先级(高 → 低):
  ① YAML 配置文件  ← 实际生效的值;yaml 里写什么就是什么
  ② 命令行参数    ← 只在脚本明确解析时才生效(--config/--checkpoint/--num_episodes)
  ③ constants.py   ← yaml 缺字段时的兜底默认值
```

**优先级带来的直接含义**:
- yaml 显式给值 → **永远优先于 constants**,改 yaml 即生效,改 constants 没用
- yaml 缺字段 → 回退到 constants,改 constants 才生效
- 一处生效一处即可:常量化改 constants,场景化改 yaml。不要"两边都改以求保险"

### 3.1 硬常量集中地:`src/environments/config/constants.py`

| 分组 | 典型常量 | 含义 |
|------|---------|------|
| **仿真侧** | `WEIBULL_PARAMS`、`TREATMENT_DURATION`、`AGENT_SPEEDS`、`NUM_REGIONS=4` | 物理仿真参数,改它影响场景 |
| **仿真配置 dataclass** | `SimulationConfig(...)` | 灾难规模/资源/地图,可在 yaml 里覆盖 |
| **Manager 配置** | `EGT_CONFIG`、`REPUTATION_CONFIG`、`PARETO_CONFIG`、`COMMUNICATION_CONFIG`、`INTERFERENCE_CONFIG` | 灾难救援 Manager 参数 |
| **★ 算法侧** | `NUM_STRATEGIES=3`、`STRATEGY_NAMES=['Fairness','Efficiency','Balanced']` | EGT 策略数量与命名;yaml 没给 `egt.num_strategies` 时回退到这里 |

### 3.2 配置 yaml 的两套目录

| 目录 | 性质 | 典型文件 |
|------|------|---------|
| `src/configs/` | **默认主配置**,脚本 `argparse` 默认指向这里 | `training.yaml`(四阶段课程)、`egt_marl.yaml`(算法核心)、`evaluation.yaml`、`disaster_sim.yaml` |
| `src/experiments/configs/` | **实验场景配置**,按场景切分 | `quick_train.yaml`(快速冒烟)、`evaluate_small/medium/large.yaml`(三规模评估)、`evaluate_baselines.yaml`(基线对比) |

### 3.3 现有 yaml 一览(2026-06-26 核实)

| yaml | 用途 | 集数 | 备注 |
|------|------|------|------|
| `src/configs/training.yaml` | 主训练,含四阶段课程 | 1500 | 含 lambda/Epsilon 阶段切换 |
| `src/configs/egt_marl.yaml` | 算法核心(marl/egt/anti_spoofing/dynamic_frontier) | - | 与 `constants.NUM_STRATEGIES` 必须同步 |
| `src/configs/evaluation.yaml` | 基线评估 | 50×5 | 17 agents,200 victims |
| `src/configs/disaster_sim.yaml` | 场景配置 | - | - |
| `src/experiments/configs/quick_train.yaml` | 快速 smoke test | 300 | 地图 1500×1500 |
| `src/experiments/configs/evaluate_small.yaml` | 小规模评估 | 10 | 地图 50×50 |
| `src/experiments/configs/evaluate_medium.yaml` | 中规模评估 | 50 | 地图 200×200 |
| `src/experiments/configs/evaluate_large.yaml` | 大规模评估 | 10 | 地图 4500×4500 |
| `src/experiments/configs/evaluate_baselines.yaml` | 9 baseline + EGT-MARL 对比 | - | - |
| `src/experiments/configs/ablation.yaml` | 消融(EGT/anti-spoof/dynamic-frontier/attention-heads/mixing-net) | 50×3 | 5 组件独立 ablation |

---

## 4. 命令模板

### 4.1 AI Agent 可用环境

| 环境 | AI Agent 是否可用 | 备注 |
|------|-------------------|------|
| PowerShell(原生,通过 RunCommand) | ✅ | AI Agent 的默认 shell |
| WSL(通过 PowerShell 调起) | ✅ | 推荐用于训练/评估长任务 |
| Git Bash | ❌ | AI Agent 无法直接使用;人类开发者可在自己机器上用 |

**WSL 推荐用法**(用于后台训练任务):
```powershell
wsl --exec bash -c "cd /mnt/e/studio/workspace/egt-marl-emergency-response && \
  source src/.venv/bin/activate && \
  python -u src/experiments/train_egt_marl.py --config src/experiments/configs/quick_train.yaml"
```

### 4.2 激活虚拟环境

```powershell
# PowerShell(原生,但 venv 是 WSL 风格,需要 wsl 调起)
wsl --exec bash -c "cd /mnt/e/studio/workspace/egt-marl-emergency-response && source src/.venv/bin/activate && <your command>"

# 或者手动进入 WSL 交互 bash 后操作
wsl
# 进了 WSL 后:
cd /mnt/e/studio/workspace/egt-marl-emergency-response
source src/.venv/bin/activate
```

工作目录是**项目根**(不是 `src/`),venv 在 `src/.venv/` 下。

### 4.3 训练

```bash
# 1500 ep 全量训练(主训练配置,四阶段课程)
# 注意:此任务耗时数小时-数天,务必用 nohup 后台 + 日志
wsl --exec bash -c "cd /mnt/e/studio/workspace/egt-marl-emergency-response && \
  source src/.venv/bin/activate && \
  nohup python -u src/experiments/train_egt_marl.py \
    --config src/configs/training.yaml \
  > training.log 2>&1 & \
  echo PID:\$!"

# 快速 smoke test(300 集, ~ 数小时完成)
python src/experiments/train_egt_marl.py \
  --config src/experiments/configs/quick_train.yaml

# 从检查点恢复
python src/experiments/train_egt_marl.py \
  --config src/configs/training.yaml \
  --resume src/experiment_results/egt_marl_TIMESTAMP/checkpoints/checkpoint_ep_1000.pt
```

### 4.4 评估

```bash
# 标准评估
python src/experiments/evaluate_model.py \
  --config src/experiments/configs/evaluate_medium.yaml \
  --checkpoint src/experiment_results/egt_marl_TIMESTAMP/models/best_model.pt

# 9 baseline 对比
python src/experiments/evaluate_baselines.py \
  --config src/experiments/configs/evaluate_baselines.yaml

# 小/中/大规模评估(不同 yaml,不要命令行覆盖)
python src/experiments/evaluate_model.py --config src/experiments/configs/evaluate_small.yaml --checkpoint src/experiment_results/egt_marl_TIMESTAMP/models/best_model.pt
python src/experiments/evaluate_model.py --config src/experiments/configs/evaluate_large.yaml --checkpoint src/experiment_results/egt_marl_TIMESTAMP/models/best_model.pt
```

### 4.5 命令风格约束

- **能用 yaml 配置的就不要用命令行参数**。事后好复盘、便于 diff
- 命令行只用来:① 选 yaml ② 选 checkpoint ③ 选 num_episodes
- 训练/评估的命令**入口、参数、输出结构统一**,不同场景用不同 yaml 区分
- 入口都走 `src/experiments/<script>.py`,没有顶层脚本

---

## 5. 进程管理(避免重复启动浪费算力)

⚠️ 训练启动慢(几十秒才有日志),**不要**仅凭"目录中没有 best_model.pt"就判定任务失败。

### 5.1 启动前检查

```bash
# WSL 内
ps aux | grep -E "train_egt_marl|evaluate_model|evaluate_baselines" | grep -v grep

# PowerShell 侧(查 WSL 内进程)
wsl --exec bash -c "ps aux | grep -E 'train_egt_marl|evaluate_model|evaluate_baselines' | grep -v grep"
```

### 5.2 启动训练(后台)

```bash
# WSL 后台,工作目录是项目根
nohup python -u src/experiments/train_egt_marl.py \
  --config src/experiments/configs/quick_train.yaml \
  > training.log 2>&1 &
echo "PID: $!"
```

启动后:
1. 等待 10-20 秒
2. `tail -f training.log` 跟踪日志
3. 确认训练启动成功的标志:**日志中开始周期性打印 episode 指标**(如 `Episode X/1500: Total Reward: ...`)

### 5.3 进程清理

```bash
# 找到所有 EGT-MARL 相关 python 进程
pgrep -af "train_egt_marl|evaluate_model|evaluate_baselines"

# 优雅退出
pkill -f train_egt_marl
```

---

## 6. 四阶段训练指标预期

| 阶段 | Episodes | Lambda 期望 | Epsilon 期望 |
|------|----------|------------|--------------|
| Warmup | 1-400 | ~0.9 | 0.8 → 衰减 |
| Transition | 401-600 | ~0.8 | 继续衰减 |
| MainTrain | 601-1100 | ~0.7 | 继续衰减 |
| FineTune | 1101-1500 | ~0.6 | ~0.01 |

异常信号:
- `Lambda 长时间停在 0/1` → EGT 层故障,查 `egt_layer.update_with_weights` 日志
- `MARL Loss = NaN` → 学习率过大,检查 gradient clipping
- `Gini 始终不变` → `_extract_performance_metrics` 失败,检查 `fairness_score` 是否被硬编码

---

## 7. 输出目录结构(统一在 `src/` 下)

```
src/
├── experiment_results/
│   └── egt_marl_YYYYMMDD_HHMMSS/
│       ├── models/
│       │   ├── best_model.pt           # 按 rescue_rate 最优
│       │   └── final_model.pt          # 最终
│       ├── checkpoints/
│       │   ├── checkpoint_ep_100.pt
│       │   └── ...
│       ├── logs/
│       │   └── training.log
│       ├── metrics/
│       │   └── metrics.json
│       ├── training_report.txt         # 自动生成
│       └── visualizations/             # 自动生成 dashboard.png
├── evaluation_results/                 # evaluate_baselines / evaluate_model 产物
│   └── baseline_evaluation_YYYYMMDD_HHMMSS/
├── ablation_results/                   # ablation_study 产物
│   └── ablation_study_YYYYMMDD_HHMMSS/
└── robustness_results/                 # robustness_test 产物
    └── robustness_test_YYYYMMDD_HHMMSS/
```

训练/评估/消融/鲁棒性产物**全部统一在 `src/` 下**,与代码同仓库。

### 7.1 训练完成后的标准流程

训练脚本会自动写 `training_report.txt` + `visualizations/training_dashboard.png`,但下列任务**不会自动执行**,AI Agent 必须按顺序手动跑:

```bash
# Step 1: 评估 best_model.pt(必做)
python src/experiments/evaluate_model.py \
  --config src/experiments/configs/evaluate_medium.yaml \
  --checkpoint src/experiment_results/egt_marl_TIMESTAMP/models/best_model.pt

# Step 2: 跑 9 baseline + EGT-MARL 对比(必做,产出论文表)
python src/experiments/evaluate_baselines.py \
  --config src/experiments/configs/evaluate_baselines.yaml

# Step 3: 大规模压测(可选,论文扩展性章节用)
python src/experiments/evaluate_model.py \
  --config src/experiments/configs/evaluate_large.yaml \
  --checkpoint src/experiment_results/egt_marl_TIMESTAMP/models/best_model.pt

# Step 4: 决策时延测试(可选,论文决策时间章节用)
python src/experiments/decision_time_evaluation.py \
  --checkpoint src/experiment_results/egt_marl_TIMESTAMP/models/best_model.pt

# Step 4.5: 消融实验(可选,论文消融章节用)
python src/experiments/ablation_study.py \
  --config src/experiments/configs/ablation.yaml

# Step 5: 与论文目标对照(必做)
#   - rescue_rate > 70% ?
#   - gini ≈ 0.2 ?
#   - lambda 四阶段切换曲线正常 ?
# 不达标 → 回到 §4.3 调整 yaml 继续训练 / 调参
# 达标 → 归档 src/experiment_results/ 后开始写报告
```

---

## 8. 修改代码前的习惯

1. **先 `git log -p <file>`** 看最近 5-10 次 commit 改了什么:
   - bug 修复会带 `P\d+ fix` / `B\d+ fix` 标记
   - 配置变更会带 `[config]` / `[refactor]` 标记
   - 不要 rebase 已合并的 fix commit
2. **不要"优化"已修复的位置**。`grep "P\d\+ fix\|B\d\+ fix" src/algorithms/` 快速锁定审计修复点
3. **修改 yaml 的字段前先确认优先级**:改 yaml 就够了,不需要同步改 constants
4. **改 constants 时同步看 yaml 里引用此常量的字段**,避免后续 yaml 显式给值时仍按旧值生效

---

## 9. 进一步阅读

- 操作手册:[`doc/operations_guide.md`](file:///e:/studio/workspace/egt-marl-emergency-response/doc/operations_guide.md)
- 项目结构 + 论文摘要:[`src/README.md`](file:///e:/studio/workspace/egt-marl-emergency-response/src/README.md)
- 硬常量集中地:[`src/environments/config/constants.py`](file:///e:/studio/workspace/egt-marl-emergency-response/src/environments/config/constants.py)
- 论文正文:[`doc/基于演化博弈与多智能体强化的灾时医疗物资动态分配机制.md`](file:///e:/studio/workspace/egt-marl-emergency-response/doc/基于演化博弈与多智能体强化的灾时医疗物资动态分配机制.md)

---

**最后更新**: 2026-06-26