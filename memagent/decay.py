"""遗忘曲线与记忆评分：Ebbinghaus 曲线 + 测试效应 + 重要性加权。

score(m) = w_recency · exp(−Δt/τ)          ← 遗忘曲线：越久越淡
         + w_freq    · (1 − exp(−n/κ))     ← 测试效应：检索越多越牢（饱和）
         + w_importance · importance       ← 重要性：一次重大事件也能刻进记忆

所有时间单位统一为秒；τ 越大衰减越慢。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScorerConfig:
    tau_seconds: float = 7 * 24 * 3600      # 遗忘曲线时间常数 τ
    kappa: float = 5.0                      # 检索次数饱和常数 κ
    w_recency: float = 1.0
    w_freq: float = 0.6
    w_importance: float = 1.2


@dataclass(frozen=True)
class MemoryScore:
    recency: float
    frequency: float
    importance: float
    total: float


class DecayScorer:
    """对单条记忆计算遗忘曲线得分。"""

    def __init__(self, cfg: ScorerConfig | None = None):
        self.cfg = cfg or ScorerConfig()

    def score(
        self,
        *,
        last_access: float,
        access_count: int,
        importance: float,
        now: float | None = None,
        tau_seconds: float | None = None,
    ) -> MemoryScore:
        """tau_seconds: 按记忆类型覆盖全局 τ（技能慢、情景快）。"""
        now = now if now is not None else time.time()
        tau = tau_seconds or self.cfg.tau_seconds
        dt = max(0.0, now - last_access)
        recency = math.exp(-dt / tau)
        frequency = 1.0 - math.exp(-access_count / self.cfg.kappa)
        total = (
            self.cfg.w_recency * recency
            + self.cfg.w_freq * frequency
            + self.cfg.w_importance * importance
        )
        return MemoryScore(recency, frequency, importance, total)

    def decay_only(
        self,
        *,
        last_access: float,
        now: float | None = None,
        tau_seconds: float | None = None,
    ) -> float:
        """只算时间衰减项（用于 Hot 层判定是否需要降级）。"""
        now = now if now is not None else time.time()
        tau = tau_seconds or self.cfg.tau_seconds
        return math.exp(-max(0.0, now - last_access) / tau)
