"""
Environment Specification — single source of truth for obs/action shape.

依据：doc/系统设计文档.md §3.2（观测）、§3.3（动作）、§8（配置）。

设计原则：
1. 这里只放"规格"，不放实现。任何环境/算法只要 import 这个模块，
   就能拿到一致的维度，硬约束由 `__post_init__` 断言守住。
2. 不动现有 disaster_sim.py / qmix_improved.py —— 它们继续按自己的
   32 维/全局 state 跑；新代码引用本模块即可获得"目标"维度。
3. K_max 是业务常量（不允许 yaml 覆盖）；K 是实际考虑的 K（≤ K_max），
   yaml 可调。
4. action_dim 必须 == num_tasks × num_comms（笛卡尔积硬约束）。

详见设计文档 §3.2 / §3.3 / §8.4 / §9.1。
"""

from dataclasses import dataclass, field
from typing import Tuple


# ============== 业务常量（不允许 yaml 覆盖） ==============

#: 任务枚举（4 个，IDLE 为隐式行为）
TASK_NAMES = (
    "FIND_CASUALTY",      # 找伤员
    "TREAT_CASUALTY",     # 救治伤员
    "SHARE_RESOURCES",    # 分享物资
    "REFILL_RESOURCES",   # 补充物资
)
NUM_TASKS: int = 4

#: 通信枚举（3 个）
COMM_NAMES = (
    "REPORT_CASUALTY",    # 报告伤员
    "REQUEST_RESOURCE",   # 请求资源
    "SILENT",             # 静默
)
NUM_COMMS: int = 3

#: 联合动作维度 = 4 任务 × 3 通信（笛卡尔积）
ACTION_DIM: int = NUM_TASKS * NUM_COMMS

#: K_max —— 观测里最多考虑多少个最近邻（业务常量）
K_MAX_CASUALTIES: int = 3
K_MAX_AGENTS: int = 3
K_MAX_DEPOTS: int = 3

#: 智能体类型枚举
AGENT_TYPES = ("DRONE", "VEHICLE", "PERSONNEL")
NUM_AGENT_TYPES: int = 3

#: 智能体类型比例（§3.1）
DRONE_RATIO = 0.10
VEHICLE_RATIO = 0.50
PERSONNEL_RATIO = 0.40

#: 移动方向（4 方向：N/E/S/W）
NUM_DIRECTIONS = 4
DIRECTION_VECTORS = {
    0: (0, 1),    # N
    1: (1, 0),    # E
    2: (0, -1),   # S
    3: (-1, 0),   # W
}
DIRECTION_NAMES = ("N", "E", "S", "W")

#: 物理参数（§3.1, §3.5）
AGENT_SPEEDS = {"DRONE": 3, "VEHICLE": 2, "PERSONNEL": 1}
AGENT_CAPACITY = {"DRONE": 5, "VEHICLE": 20, "PERSONNEL": 8}
AGENT_VISION_RADIUS = {"DRONE": 8, "VEHICLE": 4, "PERSONNEL": 6}
CAN_TREAT = {"DRONE": False, "VEHICLE": True, "PERSONNEL": True}

#: 通信失败率（按灾害阶段）
COMM_FAILURE_RATE = {"low": 0.05, "medium": 0.20, "high": 0.40}

#: EGT 策略（§1.4, §5.3）
EGT_STRATEGIES = ("FAIR", "EFFICIENT", "BALANCED")
NUM_STRATEGIES = 3
STRATEGY_LAMBDA = {"FAIR": 0.9, "EFFICIENT": 0.5, "BALANCED": 0.7}

#: 信誉阈值（§3.6.4）
REPUTATION_ALPHA_INIT = 1.0
REPUTATION_BETA_INIT = 1.0
REPUTATION_SUCCESS_WEIGHT = 1.0
REPUTATION_FAILURE_WEIGHT = 3.0
REPUTATION_TRUST_THRESHOLD = 0.7
REPUTATION_DISTRUST_THRESHOLD = 0.3

#: 奖励默认值（§3.4）
REWARD_RESCUE = 25.0
REWARD_DEATH = -2.0           # 减轻：避免未指派死亡压垮 reward 信号
REWARD_REPORT = 0.2
REWARD_REPORT_RECEIVED = 0.5
REWARD_SHARE = 2.0
REWARD_REQUEST_RESPONDED = 3.0
PENALTY_SPOOFING = -2.0
PENALTY_HOARDING = -1.5

#: 接近 shaping（设计文档 §5.2 奖励塑形）
REWARD_PROXIMITY = 0.02        # 朝目标每近 1 格的奖励（降低防止被利用）
REWARD_PROXIMITY_CAP = 0.1     # 单步最多 0.1（降低防止被利用）

#: 伤员生存 Weibull 参数（按伤情）
#: 由 theta*0.7 计算 max_survival_steps，需保证在 max_steps_per_episode(300) 内
#: 至少 70% 伤员可能在 episode 内死亡，给 agent 紧迫感：
#:   CRITICAL: 200×0.7=140步 ✅ 在中段会死
#: CRITICAL: 200→200×0.7=140 早期就有死亡压力
#:   SEVERE:   240→240×0.7=168 一定时间会死
#:   MODERATE: 420→420×0.7=294 末段可能死
#:   MILD:     800→800×0.7=560 episode 内可自然死亡 (整体≤800 步兜底)
WEIBULL_PARAMS = {
    "CRITICAL": (200, 1.0),
    "SEVERE":   (240, 1.2),
    "MODERATE": (420, 1.5),
    "MILD":     (800, 2.0),
}

#: 伤员按伤情等级的物资需求
#: v2仅使用 medkit + blood；food已被废弃（历史遗留）
RESOURCES_NEEDED_PER_SEVERITY = {
    "CRITICAL": {"medkit": 2, "blood": 1},   # 总 3 单位
    "SEVERE":   {"medkit": 1, "blood": 1},   # 总 2 单位
    "MODERATE": {"medkit": 1},                # 总 1 单位
    "MILD":     {},                           # 总 0 单位
}

#: 伤情分布（用于按规模动态计算仓库库存）
CASUALTY_SEVERITY_DIST = {
    "CRITICAL": 0.10,
    "SEVERE":   0.20,
    "MODERATE": 0.40,
    "MILD":     0.30,
}

#: 仓库供给比例（仓库库存 = 实际需求量 × 此系数）
#: 0.75 设计意图：构造 "资源约束" 场景，避免过早富余状态
DEPOT_SUPPLY_RATIO: float = 0.75

#: 默认仓库初始物资（按 num_depots=1 时的单点数量，作为兜底默认值）
#: 实际场景下由 DisasterSim._compute_depot_inventory() 动态计算
DEPOT_INITIAL_INVENTORY = {"medkit": 100, "blood": 50}


# ============== 观测规格 ==============

@dataclass(frozen=True)
class ObservationSpec:
    """
    局部观测规格（按设计文档 §3.2 + padding 实操修订）。

    各字段维度：
      self_state       = 5
      casualties       = (per_feat=4 + mask=1) * K_max_casualties
      other_agents     = (per_feat=4 + mask=1) * K_max_agents
      depots           = (per_feat=3 + mask=1) * K_max_depots
      egt_signal       = 3   (p_F, p_E, p_B)
      time_features    = 2   (step_norm, lambda_fairness)

    总维度 = 10 + 5*K_max_casualties + 5*K_max_agents + 4*K_max_depots
    默认 K_max = 3 → 10 + 15 + 15 + 12 = 52 维

    [修订说明] 设计文档 §3.2 写作 13 + 4·K1 + 4·K2 + 3·K3 = 46，
    其中 mask_channel=3 假设为"类别级 mask"，与 padding 实现中的
    "per-element mask" 重复。代码以 per-element mask 为准（更具表达力），
    删除冗余 category mask，实测 obs 维度为 52。文档下一轮修订时同步。
    """
    self_state_dim: int = 5
    per_casualty_dim: int = 6       # (x, y, severity, remaining_time, needs_medkit, needs_blood)
    per_agent_dim: int = 4          # (x, y, type, reputation)
    per_depot_dim: int = 3          # (x, y, inventory)
    egt_signal_dim: int = 3         # [p_F, p_E, p_B]
    time_feature_dim: int = 2       # [step_norm, lambda_fairness]

    k_max_casualties: int = K_MAX_CASUALTIES
    k_max_agents: int = K_MAX_AGENTS
    k_max_depots: int = K_MAX_DEPOTS

    def __post_init__(self):
        assert self.k_max_casualties >= 0
        assert self.k_max_agents >= 0
        assert self.k_max_depots >= 0

    @property
    def dim(self) -> int:
        """单 agent 局部观测总维度。"""
        return (
            self.self_state_dim
            + (self.per_casualty_dim + 1) * self.k_max_casualties
            + (self.per_agent_dim + 1) * self.k_max_agents
            + (self.per_depot_dim + 1) * self.k_max_depots
            + self.egt_signal_dim
            + self.time_feature_dim
        )

    def slice_index(self) -> dict:
        """返回各字段在拼接向量里的起止索引（便于实现层切片）。"""
        idx = {}
        cur = 0
        idx["self_state"] = (cur, cur + self.self_state_dim)
        cur += self.self_state_dim
        idx["casualties"] = (
            cur,
            cur + (self.per_casualty_dim + 1) * self.k_max_casualties,
        )
        cur += (self.per_casualty_dim + 1) * self.k_max_casualties
        idx["agents"] = (
            cur,
            cur + (self.per_agent_dim + 1) * self.k_max_agents,
        )
        cur += (self.per_agent_dim + 1) * self.k_max_agents
        idx["depots"] = (
            cur,
            cur + (self.per_depot_dim + 1) * self.k_max_depots,
        )
        cur += (self.per_depot_dim + 1) * self.k_max_depots
        idx["egt"] = (cur, cur + self.egt_signal_dim)
        cur += self.egt_signal_dim
        idx["time"] = (cur, cur + self.time_feature_dim)
        cur += self.time_feature_dim
        assert cur == self.dim, f"slice_index mismatch: cur={cur} dim={self.dim}"
        return idx


# ============== 动作规格 ==============

@dataclass(frozen=True)
class ActionSpec:
    """
    联合动作规格（按设计文档 §3.3）。

    笛卡尔积编码：action_idx ∈ [0, num_tasks * num_comms)
      task_id = action_idx % num_tasks
      comm_id = action_idx // num_tasks

    隐式空闲：当 MovementPolicy.get_target_for_task 返回 None 时，
    agent 停留 —— 不占动作维度。
    """
    num_tasks: int = NUM_TASKS
    num_comms: int = NUM_COMMS

    def __post_init__(self):
        assert self.num_tasks == NUM_TASKS, (
            f"num_tasks={self.num_tasks} 与设计文档不符，应为 {NUM_TASKS}"
        )
        assert self.num_comms == NUM_COMMS, (
            f"num_comms={self.num_comms} 与设计文档不符，应为 {NUM_COMMS}"
        )

    @property
    def dim(self) -> int:
        return self.num_tasks * self.num_comms

    def decode(self, action_idx: int) -> Tuple[int, int]:
        """action_idx → (task_id, comm_id)"""
        if not (0 <= action_idx < self.dim):
            raise ValueError(
                f"action_idx={action_idx} 越界，应在 [0, {self.dim})"
            )
        return action_idx % self.num_tasks, action_idx // self.num_tasks

    def encode(self, task_id: int, comm_id: int) -> int:
        """(task_id, comm_id) → action_idx"""
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(f"task_id={task_id} 越界")
        if not (0 <= comm_id < self.num_comms):
            raise ValueError(f"comm_id={comm_id} 越界")
        return task_id + comm_id * self.num_tasks


# ============== 默认实例（与设计文档默认值一致） ==============

DEFAULT_OBS_SPEC = ObservationSpec()
DEFAULT_ACTION_SPEC = ActionSpec()


# ============== 便捷断言 ==============

def assert_action_dim(value: int) -> None:
    """断言 action_dim == num_tasks * num_comms（用于 yaml 加载后立刻校验）。"""
    assert value == ACTION_DIM, (
        f"action_dim={value} 与设计文档硬约束不符，应为 {ACTION_DIM} "
        f"(= num_tasks({NUM_TASKS}) × num_comms({NUM_COMMS}))"
    )


__all__ = [
    # 任务 / 通信 / 动作
    "TASK_NAMES", "NUM_TASKS",
    "COMM_NAMES", "NUM_COMMS",
    "ACTION_DIM",
    # K_max
    "K_MAX_CASUALTIES", "K_MAX_AGENTS", "K_MAX_DEPOTS",
    # Agent
    "AGENT_TYPES", "NUM_AGENT_TYPES",
    "DRONE_RATIO", "VEHICLE_RATIO", "PERSONNEL_RATIO",
    "AGENT_SPEEDS", "AGENT_CAPACITY", "AGENT_VISION_RADIUS", "CAN_TREAT",
    # 移动
    "NUM_DIRECTIONS", "DIRECTION_VECTORS", "DIRECTION_NAMES",
    # 通信
    "COMM_FAILURE_RATE",
    # EGT
    "EGT_STRATEGIES", "NUM_STRATEGIES", "STRATEGY_LAMBDA",
    # 信誉
    "REPUTATION_ALPHA_INIT", "REPUTATION_BETA_INIT",
    "REPUTATION_SUCCESS_WEIGHT", "REPUTATION_FAILURE_WEIGHT",
    "REPUTATION_TRUST_THRESHOLD", "REPUTATION_DISTRUST_THRESHOLD",
    # 奖励
    "REWARD_RESCUE", "REWARD_DEATH", "REWARD_REPORT",
    "REWARD_REPORT_RECEIVED", "REWARD_SHARE", "REWARD_REQUEST_RESPONDED",
    "PENALTY_SPOOFING", "PENALTY_HOARDING",
    "REWARD_PROXIMITY", "REWARD_PROXIMITY_CAP",
    # 伤员
    "WEIBULL_PARAMS",
    # 仓库
    "RESOURCES_NEEDED_PER_SEVERITY",
    "CASUALTY_SEVERITY_DIST",
    "DEPOT_SUPPLY_RATIO",
    "DEPOT_INITIAL_INVENTORY",
    # dataclass
    "ObservationSpec", "ActionSpec",
    # 默认实例
    "DEFAULT_OBS_SPEC", "DEFAULT_ACTION_SPEC",
    # 工具
    "assert_action_dim",
]