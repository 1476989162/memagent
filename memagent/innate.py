"""出厂配置（innate defaults）+ 全局时间缩放。

进化硬约束：学习器只能在出厂上下界内微调，不可翻转方向。
时间缩放：出厂参数在人类级秒数上乘以 TIME_SCALE，使 agent 认知时间远快于
真实时间——一个 agent 秒可代表一个人的一天，测试可行。

设计原则：
- 出厂值是量级估计（方向正确），来自文献与认知科学共识
- 学习器只能在出厂上下界内微调，不能翻转方向
- 出厂值 = 基因；经验数据 = 成长环境——基因决定方向，环境只微调
- TIME_SCALE 控制认知成长速度：默认为 1/86400（1 agent-秒 = 1 人类-天）
  可通过环境变量 MEMAGENT_TIME_SCALE 覆盖

TIME_SCALE 用法：
  # 默认：1 agent-秒 = 1 人类-天（技能 τ_max ≈ 6 分钟真实时间）
  import memagent.innate as innate
  innate.TIME_SCALE  # 1.157e-05

  # 更快：1 agent-秒 = 1 人类-周（测试速度约 7 倍）
  # 设环境变量后重启进程
  MEMAGENT_TIME_SCALE=1.438e-06  # 1 / (7*86400)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .memory import MemType

# ============================================================
# 时间缩放：1 agent-秒 = N 人类-秒（N = 1/TIME_SCALE）
# 默认：1 agent-秒 ≈ 1 人类-天，学习器几秒完成完整遗忘周期
# ============================================================
# 环境变量 MEMAGENT_TIME_SCALE 优先，否则用默认值
_DEFAULT_TIME_SCALE = 1.0 / 86400  # 1 agent-秒 = 1 人类-天

_time_scale_raw = os.environ.get("MEMAGENT_TIME_SCALE")
TIME_SCALE: float = (
    float(_time_scale_raw)
    if _time_scale_raw
    else _DEFAULT_TIME_SCALE
)


@dataclass(frozen=True)
class InnateBounds:
    """某类型记忆的出厂参数上下界。

    所有 τ 参数已在人类级秒数基础上乘以 TIME_SCALE，是 agent 内部时间尺度下的值。
    """

    tau_min: float = 0.0         # τ 出厂下限（遗忘速度不可快于此）
    tau_max: float = float("inf")  # τ 出厂上限（遗忘速度不可慢于此）
    drift_min: float = 0.0       # 再巩固漂移因子下限（无量纲，不受时间缩放）
    drift_max: float = 1e9       # 再巩固漂移因子上限
    importance_min: float = 0.0  # 重要性漂移因子下限
    importance_max: float = 1e9  # 重要性漂移因子上限
    frozen: bool = False         # 出厂冻结（完全不可学习，经验数据无效）


# ============================================================
# 人类级参数（秒）：文献量级估计，未经时间缩放
# 依据：
#   - skill: 程序性记忆，杏仁核-小脑回路，几乎不可遗忘，极难改写
#   - semantic: 陈述性语义记忆，皮层分布存储，中等衰减
#   - episodic: 海马体情景记忆，快衰减，高可塑（易受情境改写）
# ============================================================
_HUMAN_SPEC: dict[MemType, dict] = {
    MemType.SKILL: {
        "tau_min": 10 * 86400,       # 10 天
        "tau_max": 365 * 86400,      # 1 年
        "drift_min": 0.0,
        "drift_max": 0.5,
        "importance_min": 0.1,
        "importance_max": 0.3,
        "frozen": False,
    },
    MemType.SEMANTIC: {
        "tau_min": 1 * 86400,        # 1 天
        "tau_max": 90 * 86400,       # 90 天
        "drift_min": 0.0,
        "drift_max": 2.0,
        "importance_min": 0.0,
        "importance_max": 1.5,
        "frozen": False,
    },
    MemType.EPISODIC: {
        "tau_min": 60.0,             # 60 秒
        "tau_max": 3 * 86400,        # 3 天
        "drift_min": 0.0,
        "drift_max": 5.0,
        "importance_min": 0.0,
        "importance_max": 3.0,
        "frozen": False,
    },
}


def make_innate_bounds(mtype: MemType, scale: float | None = None) -> InnateBounds:
    """从人类级参数生成缩放后的 InnateBounds。

    scale: 时间缩放系数，默认 TIME_SCALE。drift/importance 无量纲，不缩放。
    """
    s = scale if scale is not None else TIME_SCALE
    spec = _HUMAN_SPEC[mtype]
    return InnateBounds(
        tau_min=spec["tau_min"] * s,
        tau_max=spec["tau_max"] * s,
        drift_min=spec["drift_min"],
        drift_max=spec["drift_max"],
        importance_min=spec["importance_min"],
        importance_max=spec["importance_max"],
        frozen=spec["frozen"],
    )


def default_innate_bounds(scale: float | None = None) -> dict[MemType, InnateBounds]:
    """生成全部类型的出厂边界（带时间缩放），供 AgentConfig 默认值使用。"""
    return {t: make_innate_bounds(t, scale) for t in MemType}


# 向后兼容：模块级常量 INNATE_DEFAULTS = 默认缩放后的出厂值
INNATE_DEFAULTS: dict[MemType, InnateBounds] = default_innate_bounds()