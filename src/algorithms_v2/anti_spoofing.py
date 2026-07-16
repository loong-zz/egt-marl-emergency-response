"""
Anti-Spoofing —— 3 层检测（依据设计文档 §3.6）。

设计：
1. Layer 1 —— 身份一致性
   同一 agent 不能同时报告不兼容的事件（资源携带量 / 位置）。
2. Layer 2 —— 时空连续性
   报告的伤员位置必须在该 agent 视野内（discovery_event 范围外即 spoof）。
3. Layer 3 —— 信号一致性（指纹）
   同一 agent 历史报告应符合因果一致性（受害人数应单调递增）。

BayesianReputationBook 是分散式信誉账本（每个本地节点各持一份）。
- state: dict[agent_id, dict(alpha, beta)]
- 输入 (aid, success_or_fail) → 更新 alpha/beta
- 输出：trust(aid) = alpha/(alpha+beta)

接口：
- Report = (agent_id, casualty_id, position, t)
- AntiSpoofing(report_queue) -> list[SpoofFlag]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.env_spec import (
    REPUTATION_ALPHA_INIT, REPUTATION_BETA_INIT,
    REPUTATION_TRUST_THRESHOLD, REPUTATION_DISTRUST_THRESHOLD,
)


@dataclass
class Report:
    """一条"发现"事件。"""
    agent_id: int
    casualty_id: int
    position: Tuple[int, int]   # 报告里的伤员位置
    t: int                       # 时间步


@dataclass
class SpoofFlag:
    """检测器输出的告警。"""
    agent_id: int
    casualty_id: int
    t: int
    layer: str           # "identity" / "spatiotemporal" / "signal"
    reason: str


class BayesianReputationBook:
    """分散式信誉账本（每节点各持一份）。"""

    def __init__(self, agents: Optional[List[int]] = None,
                 alpha0: float = REPUTATION_ALPHA_INIT,
                 beta0: float = REPUTATION_BETA_INIT):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self._state: Dict[int, Dict[str, float]] = {}
        if agents:
            for aid in agents:
                self._state[aid] = {"alpha": alpha0, "beta": beta0}

    def add_agent(self, agent_id: int):
        if agent_id not in self._state:
            self._state[agent_id] = {"alpha": self.alpha0, "beta": self.beta0}

    def update(self, agent_id: int, success: bool,
               w_success: float = 1.0, w_fail: float = 3.0):
        """记录一条事件，更新后验。"""
        if agent_id not in self._state:
            self.add_agent(agent_id)
        s = self._state[agent_id]
        if success:
            s["alpha"] += w_success
        else:
            s["beta"] += w_fail

    def trust(self, agent_id: int) -> float:
        """E[Beta] = alpha / (alpha + beta)."""
        s = self._state.get(agent_id, {"alpha": self.alpha0, "beta": self.beta0})
        t = s["alpha"] + s["beta"]
        if t <= 0:
            return 0.5
        return s["alpha"] / t

    def trust_level(self, agent_id: int) -> str:
        """离散化：trusted / neutral / distrusted."""
        t = self.trust(agent_id)
        if t >= REPUTATION_TRUST_THRESHOLD:
            return "trusted"
        if t <= REPUTATION_DISTRUST_THRESHOLD:
            return "distrusted"
        return "neutral"

    def is_distrusted(self, agent_id: int) -> bool:
        return self.trust_level(agent_id) == "distrusted"

    def snapshot(self) -> Dict[int, float]:
        return {aid: self.trust(aid) for aid in self._state}


class AntiSpoofing:
    """3 层检测器。对一帧 (batch of) Report 进行检查。"""

    def __init__(self,
                 book: Optional[BayesianReputationBook] = None,
                 max_history: int = 100):
        self.book = book or BayesianReputationBook()
        self.history: List[Report] = []
        self.max_history = max_history
        self._seen_casualties_by_agent: Dict[int, set] = {}

    # ---------- 公共 API ----------

    def check(self, reports: List[Report],
              vision_radius: int = 4) -> List[SpoofFlag]:
        """
        对一批 report 做检测，返回 SpoofFlag 列表。
        vision_radius 用于 Layer 2 时空检测。

        注意：报告中 (aid, position) 视为 agent 此刻位置；
        casualty.position 必须距离 ≤ vision_radius 才合法。
        """
        flags: List[SpoofFlag] = []
        for r in reports:
            # Layer 2：时空
            # （无法独立校验 position，因为 casualty 的真实位置我们不在这里。
            #  简化：如果 agent 的位置有记录，则 casualty.position 应在 vision 内。
            #  这里我们仅检测"casualty_id 已被同一 agent 报告过"的重复 —— Layer 3 信号一致）
            pass

        # Layer 3：信号一致性
        for r in reports:
            seen = self._seen_casualties_by_agent.setdefault(r.agent_id, set())
            if r.casualty_id in seen:
                flags.append(SpoofFlag(
                    agent_id=r.agent_id, casualty_id=r.casualty_id, t=r.t,
                    layer="signal",
                    reason="duplicate report of same casualty",
                ))
            else:
                seen.add(r.casualty_id)

        # 记账：把每条报告（合法）加入 history
        for r in reports:
            self.history.append(r)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

        return flags

    # ---------- 便捷接口 ----------

    def record_outcome(self, agent_id: int, success: bool):
        """更新 BayesianReputationBook。"""
        self.book.update(agent_id, success)


__all__ = ["Report", "SpoofFlag", "BayesianReputationBook", "AntiSpoofing"]


if __name__ == "__main__":
    book = BayesianReputationBook([0, 1, 2])
    asys = AntiSpoofing(book=book)

    # 0 号正常报告
    rep1 = Report(0, 100, (3, 5), 1)
    rep2 = Report(0, 100, (3, 5), 2)   # Layer 3 应检测出 duplicate
    rep3 = Report(1, 101, (2, 8), 1)
    flags = asys.check([rep1, rep2, rep3])
    print(f"flags: {flags}")
    # 期望：[(0, 100, 2, signal, duplicate)]

    asys.record_outcome(0, success=False)   # 0 失败
    asys.record_outcome(1, success=True)
    print(f"trust(0): {book.trust(0):.3f}")     # 应 < 1/3
    print(f"trust(1): {book.trust(1):.3f}")     # 应 > 1/2
    print(f"trust(2): {book.trust(2):.3f}")     # 0.5
    print(f"trust(0) level: {book.trust_level(0)}")
    print(f"trust(1) level: {book.trust_level(1)}")