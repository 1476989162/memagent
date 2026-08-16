"""记忆类型画像：把每类型的遗忘、可塑性、压缩参数统一成一张配置表。

三个正交的行为轴——遗忘多快（τ）、回忆改多狠（再巩固 drift/importance 因子）、
闲置多久埋藏（压缩阈值）——在真实大脑里是同一类记忆的内在属性，这里聚合成
一份"画像"，供 CLI（/types）与仪表盘面板展示。
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory import MemType
from .visualize import fmt_duration

from .innate import INNATE_DEFAULTS, InnateBounds

TYPE_LABELS = {
    MemType.SKILL: "技能",
    MemType.SEMANTIC: "语义",
    MemType.EPISODIC: "情景",
}


@dataclass
class TypeProfile:
    """某类型记忆的完整行为画像。"""

    mtype: str                 # skill | semantic | episodic
    label: str                 # 中文标签
    tau_seconds: float         # 遗忘时间常数（含学习器校准后的有效值）
    drift_factor: float        # 再巩固语义漂移因子（技能低=稳、情景高=易改写）
    importance_factor: float   # 再巩固重要性漂移因子
    cold_after_seconds: float  # 闲置多久可压缩进 Cold（绝对秒数）
    cold_after_tau: float      # 压缩阈值 = cold_after / τ（几倍 τ 后埋藏）
    awakening_signal: dict | None = None  # 实测唤醒信号（awakening_signal_stats 单类型输出，None=未提供）
    innate_range: str | None = None       # 出厂 τ 边界文字（如 "出厂 10天~1年"），无出厂边界则为 None

    def to_dict(self) -> dict:
        return {
            "mtype": self.mtype,
            "label": self.label,
            "tau_seconds": round(self.tau_seconds, 4),
            "tau_text": fmt_duration(self.tau_seconds),
            "drift_factor": round(self.drift_factor, 3),
            "importance_factor": round(self.importance_factor, 3),
            "cold_after_seconds": round(self.cold_after_seconds, 4),
            "cold_after_text": fmt_duration(self.cold_after_seconds),
            "cold_after_tau": round(self.cold_after_tau, 2),
            "awakening_signal": self.awakening_signal,
            "signal_text": signal_text(self.awakening_signal),
            "innate_range": self.innate_range,
        }


def signal_text(signal: dict | None) -> str:
    """唤醒信号列文本：方向箭头 + 一致性 + 事件数。"""
    if not signal or signal.get("events", 0) == 0:
        return "无观测"
    arrows = {"up": "↑", "down": "↓", "flat": "="}
    labels = {"up": "上调", "down": "下调", "flat": "持平"}
    dom = signal["dominant"]
    return f"{arrows[dom]}{labels[dom]}·{signal['consistency']:.0%}（{signal['events']}条）"


def type_profiles(cfg: "object",
                  awakening_signal: dict | None = None) -> list[TypeProfile]:
    """从 AgentConfig 生成全部类型的画像（按 技能/语义/情景 顺序）。

    awakening_signal 为 awakening_signal_stats(agent) 的输出（按 mtype 键）时，
    每个画像带上对应类型的实测唤醒信号（方向 + 一致性）。"""
    out: list[TypeProfile] = []
    for t in (MemType.SKILL, MemType.SEMANTIC, MemType.EPISODIC):
        tau = cfg.tau_for(t)
        if cfg.cold_after_seconds is not None:
            cold_after = cfg.cold_after_seconds
        else:
            cold_after = cfg.cold_after_tau * tau
        # 出厂边界文字：从实例级 cfg.innate_bounds 取 τ 上下界（实例可能修改过
        # 默认值，例如把某类型设为 frozen），格式化为可读区间
        innate: InnateBounds | None = cfg.innate_bounds.get(t) if hasattr(cfg, "innate_bounds") else INNATE_DEFAULTS.get(t)
        innate_range: str | None = None
        if innate:
            innate_range = f"出厂 {fmt_duration(innate.tau_min)}~{fmt_duration(innate.tau_max)}"
            if innate.frozen:
                innate_range += "（冻结）"
        out.append(
            TypeProfile(
                mtype=t.value,
                label=TYPE_LABELS[t],
                tau_seconds=tau,
                drift_factor=cfg.reconsolidation_factor(t, "drift"),
                importance_factor=cfg.reconsolidation_factor(t, "importance"),
                cold_after_seconds=cold_after,
                cold_after_tau=(cold_after / tau if tau else 0.0),
                awakening_signal=(awakening_signal or {}).get(t.value),
                innate_range=innate_range,
            )
        )
    return out


def format_profiles(cfg: "object",
                    awakening_signal: dict | None = None,
                    health: dict | None = None) -> str:
    """人类可读的画像表格（CLI /types 用）。

    awakening_signal 传入时追加「唤醒信号」列（方向 + 一致性 + 事件数）；
    health（tau_learner_health 输出）传入时再追加「干净段 / 唤醒 / 一致性」
    三列——与仪表盘画像面板、--export-signals CSV 合表同源，终端/仪表盘/CSV
    三处输出一致。"""
    title = ("记忆类型画像（遗忘 τ / 再巩固因子 / 压缩阈值 / 唤醒信号"
             + (" / τ 两路一致性）" if health else "）"))
    # 出厂边界列：展示每类型的 τ 出厂上下界，与当前值对比看是否接近边界
    hdr = (f"{'类型':<6}{'τ（遗忘速度）':<32}{'drift':<8}{'importance':<11}"
           f"{'压缩阈值':<16}唤醒信号（实测）")
    if health:
        hdr += f"{' 干净段':<8}{' 唤醒':<8} 一致性"
    lines = [title, hdr]
    for p in type_profiles(cfg, awakening_signal):
        tau_text = fmt_duration(p.tau_seconds)
        if p.innate_range:
            tau_cell = f"{tau_text}  {p.innate_range}"
        else:
            tau_cell = tau_text
        row = (f"{p.label:<6}{tau_cell:<32}{p.drift_factor:<8.2f}"
               f"{p.importance_factor:<11.2f}{fmt_duration(p.cold_after_seconds)}（{p.cold_after_tau:.1f}×τ）"
               f"  {signal_text(p.awakening_signal)}")
        if health:
            cl, aw, cs = _health_cells(health, p.mtype)
            row += f"{cl:>7}{aw:>7}{cs:>7}"
        lines.append(row)
    return "\n".join(lines)


_CS_TEXT = {"agree": "✔一致", "conflict": "✘冲突", "one_sided": "△单源",
            "no_data": "—无信号"}
_DIR_ARROW = {"down": "↓", "up": "↑", "flat": "="}


def _health_cells(health: dict, mtype: str) -> tuple[str, str, str]:
    """从 health 合表取该类型的 干净段方向 / 唤醒方向 / 一致性 三格（与仪表盘一致）。"""
    h = (health.get("by_type") or {}).get(mtype) or {}
    cl_dir = (h.get("clean") or {}).get("direction")
    aw_dir = (h.get("awakening") or {}).get("direction")
    cl = _DIR_ARROW.get(cl_dir, "—") if cl_dir else "—"
    aw = _DIR_ARROW.get(aw_dir, "—") if aw_dir else "—"
    return cl, aw, _CS_TEXT.get(h.get("consistency", "no_data"))
