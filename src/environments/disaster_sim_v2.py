"""
DisasterSim v2 —— 完全按设计文档 §3 重写。

关键点：
1. 观测 = 46 维局部观测（utils.env_spec.DEFAULT_OBS_SPEC）
2. 动作 = 12 维（4 task × 3 comm）笛卡尔积
3. 4 方向移动（DIRECTION_VECTORS）+ 障碍 + 边界
4. 行为策略 = environments.behaviors.BehaviorFactory
5. K_max = 3，padding 用 utils.padding.nearest_k_pad
6. 终止条件 = 全 agent 死 / 全救援 / 全死亡 / 超时（§6.7）
7. step() 返回 Gymnasium 风格 (obs_dict, reward_dict, terminated, truncated, info)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional
import logging

import numpy as np

from utils.env_spec import (
    ACTION_DIM, NUM_TASKS, NUM_COMMS,
    K_MAX_CASUALTIES, K_MAX_AGENTS, K_MAX_DEPOTS,
    AGENT_VISION_RADIUS, AGENT_CAPACITY,
    COMM_FAILURE_RATE, CAN_TREAT,
    REWARD_RESCUE, REWARD_DEATH, REWARD_REPORT, REWARD_SHARE,
    PENALTY_SPOOFING,
    REWARD_PROXIMITY, REWARD_PROXIMITY_CAP,
    DEFAULT_OBS_SPEC,
    RESOURCES_NEEDED_PER_SEVERITY, CASUALTY_SEVERITY_DIST,
    DEPOT_SUPPLY_RATIO,
)
from utils.padding import nearest_k_pad_flat
from utils.movement_policy import MovementPolicy
from environments.entities_v2 import Agent, Casualty, Severity, Depot, Area
from environments.behaviors_v2 import BehaviorFactory

logger = logging.getLogger(__name__)


# 每类伤情需要的物资（数量）
# 注意：与 env_spec.RESOURCES_NEEDED_PER_SEVERITY 保持一致，使用 str key
RESOURCES_NEEDED = {
    Severity.CRITICAL: RESOURCES_NEEDED_PER_SEVERITY["CRITICAL"],
    Severity.SEVERE:   RESOURCES_NEEDED_PER_SEVERITY["SEVERE"],
    Severity.MODERATE: RESOURCES_NEEDED_PER_SEVERITY["MODERATE"],
    Severity.MILD:     RESOURCES_NEEDED_PER_SEVERITY["MILD"],
}


class DisasterSim:
    """
    单进程、可复现的灾场仿真器。

    API（与训练脚本约定）：
        env = DisasterSim(seed=42, num_agents=10, num_casualties=50, ...)
        obs_dict, info = env.reset()
        for _ in range(max_steps):
            action_dict = {...}    # {agent_id: action_idx in [0, 12)}
            obs_dict, reward_dict, terminated, truncated, info = env.step(action_dict)
            if terminated or truncated:
                break
    """

    def __init__(
        self,
        seed: int = 42,
        map_size: Tuple[int, int] = (50, 50),
        obstacle_ratio: float = 0.10,
        num_agents: int = 10,
        num_casualties: int = 50,
        num_depots: int = 3,
        num_areas: int = 3,
        max_steps: int = 300,
        disaster_severity: str = "medium",
        malicious_ratio: float = 0.0,
        egt_signal_fn=None,                # callable() -> np.ndarray shape=(3,) 策略分布
        egt_lambda_fn=None,                # callable() -> float λ ∈ [0,1]
    ):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.W, self.H = map_size
        self.obstacle_ratio = obstacle_ratio
        self.num_agents = num_agents
        self.num_casualties = num_casualties
        self.num_depots = num_depots
        self.num_areas = num_areas
        self.max_steps = max_steps
        self.disaster_severity = disaster_severity
        self.malicious_ratio = malicious_ratio
        self.comm_failure_rate = COMM_FAILURE_RATE[disaster_severity]

        # EGT 钩子
        self.egt_signal_fn = egt_signal_fn
        self.egt_lambda_fn = egt_lambda_fn

        # 状态（在 reset 中填充）
        self.step_count = 0
        self.obstacles: np.ndarray
        self.agents: Dict[int, Agent] = {}
        self.casualties: Dict[int, Casualty] = {}
        self.depots: Dict[int, Depot] = {}
        self.areas: Dict[int, Area] = {}
        self.movement: MovementPolicy

        # 共享信息（FIND 任务的报告，会被同 area 队友收到）
        self.shared_info: Dict[int, set] = {}     # agent_id -> set(casualty_id)

        # 全局统计
        self.statistics = {
            "total_rescued": 0,
            "total_deaths": 0,
            "total_reports": 0,
            "total_shares": 0,
            "total_spoofing": 0,
        }

    # ============== 初始化辅助 ==============

    def _gen_obstacles(self) -> np.ndarray:
        obs = self.rng.random((self.H, self.W)) < self.obstacle_ratio
        # 至少留出边缘可通行
        obs[0, :] = obs[-1, :] = obs[:, 0] = obs[:, -1] = False
        return obs

    def _gen_position(self, exclude_obstacle: bool = True) -> Tuple[int, int]:
        for _ in range(200):
            x = int(self.rng.integers(0, self.W))
            y = int(self.rng.integers(0, self.H))
            if not exclude_obstacle or not self.obstacles[y, x]:
                return x, y
        # 兜底：左上角
        return 0, 0

    def _gen_areas(self):
        # 把地图均分为 num_areas 块
        n = max(1, self.num_areas)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        w = self.W // cols
        h = self.H // rows
        areas = {}
        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= n:
                    break
                x0 = c * w
                y0 = r * h
                x1 = (c + 1) * w if c < cols - 1 else self.W
                y1 = (r + 1) * h if r < rows - 1 else self.H
                # priority：灾区编号越小越紧急（演示用，可调整）
                areas[k] = Area(id=k, bbox=(x0, y0, x1, y1), priority=k, label=f"A{k}")
                k += 1
        self.areas = areas

    def _gen_agents(self):
        # 按比例分配 DRONE / VEHICLE / PERSONNEL
        n = self.num_agents
        nd = max(1, int(round(n * 0.10))) if n >= 10 else (1 if n >= 1 else 0)
        nv = max(1, int(round(n * 0.50))) if n >= 2 else 0
        np_ = n - nd - nv
        types = (["DRONE"] * nd) + (["VEHICLE"] * nv) + (["PERSONNEL"] * np_)
        self.rng.shuffle(types)
        agents = {}
        for i, t in enumerate(types):
            pos = self._gen_position()
            a = Agent(id=i, agent_type=t, position=pos)
            agents[i] = a
        self.agents = agents

    def _gen_casualties(self):
        sev_dist = [0.10, 0.20, 0.40, 0.30]   # CRITICAL/SEVERE/MODERATE/MILD
        sevs = self.rng.choice(4, size=self.num_casualties, p=sev_dist)
        c = 0
        for s in sevs:
            pos = self._gen_position()
            area = self._locate_area(*pos)
            self.casualties[c] = Casualty(
                id=c, position=pos, severity=Severity(int(s)), area_id=area.id
            )
            c += 1

    def _gen_depots(self):
        """按灾害规模动态计算每个仓库的初始物资，避免硬编码导致供需失衡。

        公式：
            total_demand[kind] = sum_{severity} 需求量(severity) * 人数(severity)
            total_supply[kind] = total_demand[kind] * DEPOT_SUPPLY_RATIO
            per_depot[kind]    = total_supply[kind] / num_depots
        """
        # 1. 按人数计算各种物资的总需求量
        total_demand: Dict[str, float] = {"medkit": 0.0, "blood": 0.0}
        for sev_name, ratio in CASUALTY_SEVERITY_DIST.items():
            count = int(self.num_casualties * ratio)
            for kind, qty in RESOURCES_NEEDED_PER_SEVERITY[sev_name].items():
                total_demand[kind] = total_demand.get(kind, 0.0) + qty * count

        # 2. 应用供给比例（设计文档 §4.5.4：构造"资源约束"场景）
        #    blood 用 1.0（保证 CRITICAL/SEVERE 不会因 blood 不足而死亡）
        #    medkit 用 0.75（保留适度资源约束）
        RESOURCE_SUPPLY_RATIO = {"blood": 1.0, "medkit": DEPOT_SUPPLY_RATIO}
        total_supply = {
            kind: qty * RESOURCE_SUPPLY_RATIO.get(kind, DEPOT_SUPPLY_RATIO)
            for kind, qty in total_demand.items()
        }

        # 3. 平均分配到各 depot
        if self.num_depots <= 0:
            per_depot_inv: Dict[str, float] = dict(total_supply)
        else:
            per_depot_inv = {
                kind: qty / self.num_depots
                for kind, qty in total_supply.items()
            }

        # 4. 创建 depot
        self.depots = {}
        for i in range(self.num_depots):
            pos = self._gen_position()
            self.depots[i] = Depot(
                id=i,
                position=pos,
                inventory={k: int(round(v)) for k, v in per_depot_inv.items()},
            )

        logger.info(
            f"[DEPOT_INIT] num_casualties={self.num_casualties} "
            f"num_depots={self.num_depots} ratio={DEPOT_SUPPLY_RATIO} "
            f"demand={total_demand} supply={total_supply} per_depot={per_depot_inv}"
        )

    def _locate_area(self, x: int, y: int) -> Area:
        for a in self.areas.values():
            if a.contains(x, y):
                return a
        # 兜底：返回第 0 个
        return next(iter(self.areas.values()))

    # ============== API ==============

    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        self.rng = np.random.default_rng(self.seed)
        self.step_count = 0
        self.statistics = {
            "total_rescued": 0, "total_deaths": 0,
            "total_reports": 0, "total_shares": 0, "total_spoofing": 0,
        }
        self.shared_info = {aid: set() for aid in range(self.num_agents)}
        self.obstacles = self._gen_obstacles()
        self._gen_areas()
        self._gen_depots()
        self._gen_agents()
        self._gen_casualties()
        self.movement = MovementPolicy((self.W, self.H), self.obstacles)
        obs = self._build_observations()
        info = {"statistics": dict(self.statistics)}
        return obs, info

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, bool, Dict[str, Any]]:
        """
        actions: {agent_id: action_idx ∈ [0, 12)}
            action_idx = task_id + comm_id * NUM_TASKS
        """
        self.step_count += 1
        reward: Dict[int, float] = {aid: 0.0 for aid in self.agents}
        info: Dict[str, Any] = {"events": []}

        # 1. 解码 + 应用通信（先通信，后移动，保证 agent 能共享刚发现的信息）
        reports_this_step: list = []  # (reporter_id, casualty_id)
        requests_this_step: list = []  # (requester_id, kind)
        spoofs_this_step: list = []

        for aid, action_idx in actions.items():
            if aid not in self.agents or not self.agents[aid].alive:
                continue
            if not (0 <= action_idx < ACTION_DIM):
                continue
            task_id, comm_id = action_idx % NUM_TASKS, action_idx // NUM_TASKS
            agent = self.agents[aid]

            # 通信（按 comm_failure_rate 模拟丢包）
            if self.rng.random() < self.comm_failure_rate:
                comm_id = 2  # 视为 SILENT

            if comm_id == 0 and task_id == 0:  # 报告 + 找 → 实际发现
                # 视野内的未发现伤员都会被"发现"
                discovered_any = False
                for c in self.casualties.values():
                    if c.discovered or c.dead:
                        continue
                    d = abs(c.position[0] - agent.position[0]) \
                      + abs(c.position[1] - agent.position[1])
                    if d <= agent.vision_radius:
                        if c.discover(aid):
                            discovered_any = True
                            reports_this_step.append((aid, c.id))
                if discovered_any:
                    reward[aid] += REWARD_REPORT
                    self.statistics["total_reports"] += 1
                else:
                    # 修复：report失败不应当 spoofing
                    # 真正 "空报/spoofing" 仅当：视野范围内"无未被发现"伤员，
                    # 即所有视野内的伤员要么已发现、要么已死、要么已治愈，
                    # 但任务栏里"已发现"过伤员还亮着 → 这才是浪费通信带宽。
                    has_unhandled_in_view = False
                    for c in self.casualties.values():
                        d = abs(c.position[0] - agent.position[0]) \
                          + abs(c.position[1] - agent.position[1])
                        if d > agent.vision_radius:
                            continue
                        if not c.discovered and not c.dead and not c.treated:
                            has_unhandled_in_view = True
                            break
                    if not has_unhandled_in_view:
                        # 视野内已无可报告内容 → 算 spoofs
                        reward[aid] += PENALTY_SPOOFING
                        self.statistics["total_spoofing"] += 1
                        spoofs_this_step.append(aid)
            elif comm_id == 1:
                # 请求资源
                requests_this_step.append((aid, "medkit"))

        # 2. 信息共享：把 reports 推送给同灾区 + 视野内的其他 agent
        for reporter, cid in reports_this_step:
            for other in self.agents.values():
                if other.id == reporter or not other.alive:
                    continue
                if cid not in self.shared_info[other.id]:
                    self.shared_info[other.id].add(cid)

        # 3. 应用任务（移动 + 副作用）
        for aid, action_idx in actions.items():
            if aid not in self.agents or not self.agents[aid].alive:
                continue
            task_id, _ = action_idx % NUM_TASKS, action_idx // NUM_TASKS
            agent = self.agents[aid]
            old_pos = agent.position

            # 行为策略选目标
            behavior = BehaviorFactory.get(task_id)
            target = behavior.get_target(agent, self)

            if target is None:
                continue  # IDLE

            direction = self.movement.direction_towards(agent.position, target)
            new_pos = self.movement.apply(agent.position, direction)
            agent.position = new_pos

            # 接近 shaping（每近 1 格 → +0.1，上限 0.5）
            # 限制：仅当 agent 能够实际完成目标任务时才给奖励
            can_perform_task = False
            if task_id == 0:
                for c in self.casualties.values():
                    if c.position == target and not c.discovered and not c.dead:
                        d = abs(c.position[0] - agent.position[0]) + abs(c.position[1] - agent.position[1])
                        if d <= agent.vision_radius:
                            can_perform_task = True
                            break
            elif task_id == 1 and CAN_TREAT.get(agent.agent_type, False):
                for c in self.casualties.values():
                    if c.position == target and c.discovered and not c.dead and not c.treated:
                        for kind, qty in RESOURCES_NEEDED[c.severity].items():
                            if agent.inventory.get(kind, 0) < qty:
                                can_perform_task = False
                                break
                        else:
                            can_perform_task = True
                        break
            elif task_id == 2:
                can_perform_task = agent.resources_total() > 1
            elif task_id == 3:
                can_perform_task = True

            if target is not None and new_pos != old_pos and can_perform_task:
                d_old = abs(target[0] - old_pos[0]) + abs(target[1] - old_pos[1])
                d_new = abs(target[0] - new_pos[0]) + abs(target[1] - new_pos[1])
                delta = d_old - d_new
                if delta > 0:
                    distance_factor = max(0, 1.0 - d_new / 10.0)
                    reward[aid] += min(delta * REWARD_PROXIMITY * distance_factor, REWARD_PROXIMITY_CAP)

            # 副作用：任务执行成功的奖励（按 §3.4）
            if task_id == 0:  # FIND（视野内发现伤员已被统计）
                pass
            elif task_id == 1 and CAN_TREAT.get(agent.agent_type, False):
                # 治疗：若与某已发现未治疗伤员距离 ≤ 1
                for c in self.casualties.values():
                    if not c.discovered or c.dead or c.treated:
                        continue
                    d = abs(c.position[0] - agent.position[0]) \
                      + abs(c.position[1] - agent.position[1])
                    if d <= 1:
                        # 资源检查
                        ok = True
                        for kind, qty in RESOURCES_NEEDED[c.severity].items():
                            if agent.inventory.get(kind, 0) < qty:
                                ok = False
                                break
                        if ok:
                            # 扣资源 + 治愈
                            for kind, qty in RESOURCES_NEEDED[c.severity].items():
                                agent.remove_resource(kind, qty)
                            if c.treat(aid):
                                # 修复5：救援紧迫度奖励（越接近死亡奖励越高）
                                # remaining_ratio: 1.0=刚开始, 0.0=即将死亡
                                # 紧迫度倍率 = 1 + 2*(1 - remaining_ratio)
                                #   - 刚发现时就救：1×（无加成）
                                #   - 临近死亡时救：3×（高激励）
                                if c.max_survival_steps > 0:
                                    remaining_ratio = c.remaining_steps / c.max_survival_steps
                                    urgency = 1.0 + 2.0 * max(0.0, 1.0 - remaining_ratio)
                                else:
                                    urgency = 1.0
                                reward[aid] += REWARD_RESCUE * urgency
                                self.statistics["total_rescued"] += 1
                                agent.update_reputation(success=True)
                        break
            elif task_id == 2:
                # 分享资源：找最近的需要资源的队友，把自己多余的任意资源给出
                # 参考 V1 DroneBehavior.deliver_resources：
                #   - DRONE 可视为"运输车"，把自己载的多余资源送给缺资源的agent
                #   - VEHICLE/PERSONNEL 也能互相分享，不限于 medkit
                # 匹配条件：
                #   1) 队友 alive，位置 ≤ 1
                #   2) 自己持有队友能用上的资源（任何 kind）
                #   3) 队友有剩余容量
                for other in self.agents.values():
                    if other.id == aid or not other.alive:
                        continue
                    d = abs(other.position[0] - agent.position[0]) \
                      + abs(other.position[1] - agent.position[1])
                    if d > 1:
                        continue
                    # 遍历自己身上的资源，找出对方能用且自己有富余的 kind
                    transferred = False
                    for kind, qty in list(agent.inventory.items()):
                        if qty <= 0:
                            continue
                        if not other.has_capacity(1):
                            break
                        # 只在自己"资源过剩"时才分享（即至少留 1 单位给自己）
                        if qty <= 1:
                            continue
                        agent.remove_resource(kind, 1)
                        other.add_resource(kind, 1)
                        reward[aid] += REWARD_SHARE
                        self.statistics["total_shares"] += 1
                        transferred = True
                        break   # 一次给 1 单位，避免震荡
                    if transferred:
                        break
            elif task_id == 3:
                # 补充资源：走到 depot，按自身容量填满（与 V1 ResourceManager.refill_from_depot 一致）
                # 不引入"按需求取""半满不取"等额外约束——
                #   视野盲区下 agent 不知道伤员在哪，必须满装出发才能单循环完成救援，
                #   HOARDING 惩罚应在行为/信誉层处理，不由补给策略硬限制。
                for depot in self.depots.values():
                    d = abs(depot.position[0] - agent.position[0]) \
                      + abs(depot.position[1] - agent.position[1])
                    if d <= 1:
                        self._refill_from_depot(agent, depot)
                        break

        # 4. 推进伤员时间 + 判定死亡（修复：去重计数，避免每帧重复累加）
        for c in self.casualties.values():
            c.step()
            # 第一次死亡时计数 + 扣分；后续帧不再重复
            if c.on_dead_reported():
                if c.discovered_by >= 0:
                    reward[c.discovered_by] = reward.get(c.discovered_by, 0.0) + REWARD_DEATH / 5
                self.statistics["total_deaths"] += 1

        # 5. 终止条件
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        # 6. 构建新观测
        obs = self._build_observations()
        info["statistics"] = dict(self.statistics)
        info["step"] = self.step_count
        return obs, reward, terminated, truncated, info

    # ============== 补给策略 ==============

    def _refill_from_depot(self, agent, depot) -> None:
        """按自身容量填满（对齐 V1 ResourceManager.refill_from_depot）。

        V1 原版逻辑：
            for resource_type in ResourceType:
                needed = max_capacity - capacity
                if needed > 0:
                    actual = depot.consume(resource_type, needed)
                    capacity += actual
        本函数完全照搬该行为：每种 kind 一律填到 agent 容量上限（或 depot 耗尽）为止。
        """
        PRIORITY_KINDS = ("blood", "medkit")
        for kind in PRIORITY_KINDS:
            # 填到自身容量上限（与 V1 行为完全一致）
            while depot.has(kind, 1) and agent.has_capacity(1):
                depot.load_to(agent, kind, 1)

    # ============== 终止判定 ==============

    def _check_terminated(self) -> bool:
        """朴素的 episode 终止条件：
        1. 全 agent 死
        2. 所有伤员都被救治或死亡（treated_or_dead 覆盖所有）
        3. 所有伤员都死亡
        不再有额外的"修复4 提前终止"——MILD max_surv 已被调到足以在 episode
        末尾自然死亡，故 episode 必然会在 max_steps 内自然结束。
        """
        alive = [a for a in self.agents.values() if a.alive]
        # 全 agent 死
        if not alive:
            return True
        # 全救援完（所有伤员都被治或死）
        if all(c.treated or c.dead for c in self.casualties.values()):
            return True
        # 全死亡（全部 episode 没救成）
        if all(c.dead for c in self.casualties.values()):
            return True
        return False

    # ============== 观测构建（46 维） ==============

    def _build_observations(self) -> Dict[int, np.ndarray]:
        """为每个 agent 生成 46 维局部观测。"""
        egt_signal = self._get_egt_signal()              # (3,)
        lambda_fairness = self._get_lambda()             # float

        obs_dict: Dict[int, np.ndarray] = {}
        for aid, agent in self.agents.items():
            if not agent.alive:
                # 死亡 agent 返回全 0
                obs_dict[aid] = np.zeros(DEFAULT_OBS_SPEC.dim, dtype=np.float32)
                continue

            parts = []

            # --- self_state (5): (x_norm, y_norm, type_id, capacity_norm, alive) ---
            parts.append(self._self_state(agent))

            # --- casualties (4*K): (x, y, severity, remaining_norm) + mask ---
            parts.append(self._casualty_block(agent))

            # --- other_agents (4*K): (x, y, type_id, reputation) + mask ---
            parts.append(self._agent_block(agent))

            # --- depots (3*K): (x, y, inventory_norm) + mask ---
            parts.append(self._depot_block(agent))

            # --- egt signal (3): (p_F, p_E, p_B) ---
            parts.append(egt_signal.astype(np.float32))

            # --- time (2): (step_norm, lambda_fairness) ---
            parts.append(np.array([
                self.step_count / max(1, self.max_steps),
                lambda_fairness,
            ], dtype=np.float32))

            obs = np.concatenate(parts)
            assert obs.shape[0] == DEFAULT_OBS_SPEC.dim, (
                f"agent {aid} obs shape {obs.shape[0]} != {DEFAULT_OBS_SPEC.dim}"
            )
            obs_dict[aid] = obs

        return obs_dict

    def _self_state(self, agent: Agent) -> np.ndarray:
        return np.array([
            agent.position[0] / max(1, self.W),
            agent.position[1] / max(1, self.H),
            {"DRONE": 0.0, "VEHICLE": 0.5, "PERSONNEL": 1.0}[agent.agent_type],
            agent.resources_total() / max(1, agent.capacity),
            1.0 if agent.alive else 0.0,
        ], dtype=np.float32)

    def _casualty_block(self, agent: Agent) -> np.ndarray:
        # 视野内的所有伤员
        cands = []
        for c in self.casualties.values():
            if c.dead or c.treated:
                continue
            d = abs(c.position[0] - agent.position[0]) \
              + abs(c.position[1] - agent.position[1])
            if d > agent.vision_radius:
                continue
            needs = RESOURCES_NEEDED.get(c.severity, {})
            feat = np.array([
                c.position[0] / max(1, self.W),
                c.position[1] / max(1, self.H),
                c.severity / 3.0,            # 0=CRITICAL, 1=MILD
                c.remaining_steps / max(1, self.max_steps),
                float(needs.get("medkit", 0) > 0),  # 是否需要medkit
                float(needs.get("blood", 0) > 0),   # 是否需要blood
            ], dtype=np.float32)
            pos = np.array(c.position, dtype=np.float32)
            cands.append((feat, pos))
        return nearest_k_pad_flat(
            cands, np.array(agent.position, dtype=np.float32),
            k_max=K_MAX_CASUALTIES, feature_dim=6,
        )

    def _agent_block(self, agent: Agent) -> np.ndarray:
        cands = []
        for other in self.agents.values():
            if other.id == agent.id or not other.alive:
                continue
            feat = np.array([
                other.position[0] / max(1, self.W),
                other.position[1] / max(1, self.H),
                {"DRONE": 0.0, "VEHICLE": 0.5, "PERSONNEL": 1.0}[other.agent_type],
                other.reputation,
            ], dtype=np.float32)
            pos = np.array(other.position, dtype=np.float32)
            cands.append((feat, pos))
        return nearest_k_pad_flat(
            cands, np.array(agent.position, dtype=np.float32),
            k_max=K_MAX_AGENTS, feature_dim=4,
        )

    def _depot_block(self, agent: Agent) -> np.ndarray:
        cands = []
        for d in self.depots.values():
            total = sum(d.inventory.values()) or 1
            feat = np.array([
                d.position[0] / max(1, self.W),
                d.position[1] / max(1, self.H),
                d.inventory.get("medkit", 0) / total,
            ], dtype=np.float32)
            pos = np.array(d.position, dtype=np.float32)
            cands.append((feat, pos))
        return nearest_k_pad_flat(
            cands, np.array(agent.position, dtype=np.float32),
            k_max=K_MAX_DEPOTS, feature_dim=3,
        )

    # ============== EGT 钩子 ==============

    def _get_egt_signal(self) -> np.ndarray:
        if self.egt_signal_fn is not None:
            sig = np.asarray(self.egt_signal_fn(), dtype=np.float32)
            assert sig.shape == (3,), f"EGT signal must be shape (3,), got {sig.shape}"
            return sig
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def _get_lambda(self) -> float:
        if self.egt_lambda_fn is not None:
            return float(self.egt_lambda_fn())
        return 0.5

    # ============== 暴露给训练脚本的辅助 ==============

    def get_observation_dimension(self) -> int:
        return DEFAULT_OBS_SPEC.dim

    def get_action_dimension(self) -> int:
        return ACTION_DIM


__all__ = ["DisasterSim"]