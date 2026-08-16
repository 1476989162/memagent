"""记忆强度曲线可视化：纯 Python 生成 SVG（零依赖），另导出 CSV / JSON。

曲线画法：每条记忆画「预测曲线」（按遗忘曲线公式外推，实线）
+「实际采样点」（Agent 在创建/检索/升级时记录的历史，圆点）。
预测部分延伸到 now + horizon，能看到未来的遗忘趋势。
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .memory import MemType, Tier

TIER_COLORS = {
    Tier.HOT: "#e34a2f",   # 红：工作记忆
    Tier.WARM: "#2f6fd6",  # 蓝：长时记忆
    Tier.COLD: "#8a8f98",  # 灰：深藏记忆
}
TIER_LABELS = {Tier.HOT: "Hot 工作记忆", Tier.WARM: "Warm 长时记忆", Tier.COLD: "Cold 深藏记忆"}
# 类型颜色：用于主图叠加的"典型遗忘参考曲线"（与分布面板/类型对比视图的类型色系一致）
MTYPE_COLORS = {
    MemType.SKILL: "#2f9e44",     # 绿
    MemType.SEMANTIC: "#7048e8",  # 紫
    MemType.EPISODIC: "#f08c00",  # 橙
}

W, H = 1100, 640
ML, MR, MT, MB = 70, 30, 64, 64  # 边距


def default_horizon(agent: "object") -> float:
    """预测窗：演示用小 τ 时给 6×最大τ，生产用 14 天。"""
    taus = {agent._tau_for(m) for m in agent.store.all()}
    tau = max(taus) if taus else agent.cfg.tau_seconds
    return 6 * tau if tau < 3600 else 14 * 86400


def fmt_duration(sec: float) -> str:
    """把时长格式化为中文（90秒 / 3天）。"""
    if sec < 60:
        return f"{sec:.0f}秒"
    if sec < 3600:
        return f"{sec / 60:.0f}分钟"
    if sec < 86400:
        return f"{sec / 3600:.0f}小时"
    return f"{sec / 86400:.0f}天"


def tau_summary(cfg: "object") -> str:
    """按类型的 τ 摘要，如：τ:技能90秒/语义30秒/情景8秒。"""
    m = cfg.tau_by_type
    if not m:
        return f"τ={cfg.tau_seconds:.0f}s"
    return "τ:" + "/".join(f"{t.value}{fmt_duration(v)}" for t, v in m.items())


# ---------- 曲线数据 ----------

def strength_series(agent, mem, now: float, horizon: float, samples: int = 200) -> list[list]:
    """预测曲线：从记忆创建时刻到 now+horizon 的等间距强度采样。"""
    t_end = now + horizon
    t_start = min(mem.created_at, now)
    if t_end <= t_start:
        t_end = t_start + 1
    step = (t_end - t_start) / (samples - 1)
    return [[t, agent._strength_at(mem, t)] for t in (t_start + step * i for i in range(samples))]


def _nice_step(span: float, target: int = 9) -> float:
    """在 1/2/5×10^k 序列里选一个接近 span/target 的时间刻度。"""
    if span <= 0:
        return 1.0
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def fmt_delta(sec: float, now: float) -> str:
    """把时间戳格式化为相对现在的文本（-3天 / -45分钟 / +2小时）。"""
    d = sec - now
    sign = "+" if d >= 0 else "-"
    a = abs(d)
    if a < 60:
        return f"{sign}{a:.0f}秒"
    if a < 3600:
        return f"{sign}{a / 60:.0f}分钟"
    if a < 86400:
        return f"{sign}{a / 3600:.1f}小时"
    return f"{sign}{a / 86400:.1f}天"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def access_events(mem) -> list[float]:
    """从观测快照中提取检索事件时间戳：相邻采样间 access_count 增大即发生过检索。"""
    events: list[float] = []
    prev_n: int | None = None
    for row in mem.history:
        n = row[3]
        if prev_n is not None and n > prev_n:
            events.append(row[0])
        prev_n = n
    return events


def strength_at(history: list, ts: float) -> float:
    """观测历史里某时刻的强度（找不到返回 0.3 占位）。"""
    for row in history:
        if abs(row[0] - ts) < 1e-6:
            return row[1]
    return 0.3


# ---------- 预测贴合度（持续观测验证） ----------

RESIDUAL_TOL = 0.03  # 预测命中容差（强度绝对值）
FLOOR_STRENGTH = 0.2  # 强度下限（与 agent.STRENGTH_FLOOR 一致）；触底段无衰减信息


def fit_report(agent, now: float | None = None) -> dict:
    """预测 vs 真实遗忘的贴合度。

    对每条记忆的相邻观测段做分析：
    - 若段内发生过检索（last_access 前进）或再巩固（重要性变化）→ 干扰段；
    - 否则为"干净衰减段"：用段首状态回放预测段末强度，同时从实际衰减
      反推该段的实测 τ（empirical tau）。
    按类型聚合实测 τ，与配置 τ 对比得到贴合度 fit = 1 − |Δτ|/τ_cfg。

    注意：观测采样使用"真实"τ（true_tau_by_type，未配置时同模型 τ），
    预测使用模型 τ——所以 τ 配置失准会被真实地暴露出来。
    """
    cfg = agent.scorer.cfg
    denom = cfg.w_recency + cfg.w_freq + cfg.w_importance
    by_type: dict = defaultdict(lambda: {"obs": 0, "interf": 0, "clean": 0, "tau_ests": [], "dts": []})
    overall = {"obs": 0, "interf": 0}
    memories: list[dict] = []

    for mem in agent.store.all():
        hist = mem.history
        interf = clean = 0
        residuals: list[float] = []
        for (t0, s0, la0, n0, imp0), (t1, s1, la1, n1, imp1) in zip(hist, hist[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            overall["obs"] += 1
            by_type[mem.mtype]["obs"] += 1
            if la1 > t0 or imp1 != imp0:
                interf += 1
                by_type[mem.mtype]["interf"] += 1
                continue
            clean += 1
            by_type[mem.mtype]["clean"] += 1
            # 段首状态回放：模型预测的段末强度
            pred = agent.strength_at_state(mem.mtype, la0, n0, imp0, t1)
            residuals.append(abs(s1 - pred))
            # 从实际衰减反推实测 τ（剥离频率与重要性常数项）
            freq0 = 1.0 - math.exp(-n0 / cfg.kappa)
            const = cfg.w_freq * freq0 + cfg.w_importance * imp0
            a = s0 * denom - const
            b = s1 * denom - const
            if a > 1e-9 and b > 1e-9 and a > b:
                # 触底段（观测值在强度下限）已被钳制，不含真实衰减信息，跳过
                if s0 <= FLOOR_STRENGTH + 1e-6 or s1 <= FLOOR_STRENGTH + 1e-6:
                    continue
                try:
                    tau_est = dt / math.log(a / b)
                except ZeroDivisionError:
                    continue
                if 0 < tau_est < 1e7:
                    by_type[mem.mtype]["tau_ests"].append(tau_est)
                    by_type[mem.mtype]["dts"].append(dt)
        overall["interf"] += interf
        memories.append(
            {
                "id": mem.id,
                "tier": mem.tier.value,
                "mtype": mem.mtype.value,
                "segments": max(0, len(hist) - 1),
                "interference": interf,
                "clean": clean,
                "mean_residual": round(sum(residuals) / len(residuals), 4) if residuals else None,
            }
        )

    by_type_out: dict = {}
    for t in MemType:
        d = by_type[t]
        tau_cfg = agent.cfg.tau_for(t)
        tau_est = fit = None
        if d["tau_ests"]:
            wsum = sum(d["dts"]) or 1.0
            tau_est = sum(te * dt for te, dt in zip(d["tau_ests"], d["dts"])) / wsum
            fit = max(0.0, 1.0 - abs(tau_est - tau_cfg) / tau_cfg)
        by_type_out[t.value] = {
            "tau_cfg": tau_cfg,
            "tau_est": round(tau_est, 2) if tau_est is not None else None,
            "fit": round(fit, 4) if fit is not None else None,
            "observations": d["obs"],
            "interference": d["interf"],
            "clean": d["clean"],
            "clean_seconds": round(sum(d["dts"]), 2),
        }
    return {"overall": overall, "by_type": by_type_out, "memories": memories}


def format_fit_report(agent) -> str:
    """贴合度报告的中文摘要（CLI /plot 与 /observe 使用）。"""
    r = fit_report(agent)
    lines = [
        f"预测贴合度报告（观测 {r['overall']['obs']} 段，"
        f"其中检索/再巩固干扰 {r['overall']['interf']} 段）"
    ]
    for t in MemType:
        d = r["by_type"][t.value]
        if d["tau_est"] is not None:
            lines.append(
                f"  {t.value}: 配置τ={fmt_duration(d['tau_cfg'])} "
                f"实测τ≈{fmt_duration(d['tau_est'])} 贴合度{d['fit'] * 100:.0f}% "
                f"｜ 观测{d['observations']}段 干扰{d['interference']}段"
            )
        else:
            lines.append(
                f"  {t.value}: 观测不足（有效衰减段太少）"
            )
    return "\n".join(lines)


# ---------- SVG 渲染 ----------

@dataclass
class _ChartCtx:
    """图表上下文：坐标映射与静态骨架。"""
    t0: float
    t1: float
    px: Callable[[float], float]
    py: Callable[[float], float]
    memories: list
    counts: dict
    fit: dict


def _scaffold(agent, now: float, horizon: float, samples: int = 200):
    """绘制图表静态骨架（标题/图例/网格/坐标轴），返回 (静态部分, 绘图区部分, 上下文)。

    t0 取创建时刻、now 与**全部唤醒事件最早时刻**的最小值——唤醒历史在记忆
    创建之后发生、但早于 now（多次 Cold↔Warm 往返的曲线连续性验证），窗口
    必须覆盖它们，否则主图/仪表盘的唤醒标注（dev vs expected 双条 + 信号
    徽章）会随窗口右移被截掉。"""
    memories = sorted(agent.store.all(), key=lambda m: (-m.importance, m.created_at))
    t1 = now + horizon
    aw_times = [aw[0] for m in memories for aw in m.awakenings if len(aw) >= 1]
    t0 = min([m.created_at for m in memories] + [now] + aw_times) if memories else now
    if t1 <= t0:
        t0 = now - 60

    plot_w, plot_h = W - ML - MR, H - MT - MB

    def px(t: float) -> float:
        return ML + (t - t0) / (t1 - t0) * plot_w

    def py(s: float) -> float:
        return MT + (1.0 - s) * plot_h

    static: list[str] = []
    plot: list[str] = []
    static.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">')
    static.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # 标题
    counts = {t: len(agent.store.by_tier(t)) for t in Tier}
    static.append(
        f'<text x="{ML}" y="30" font-size="20" font-weight="bold" fill="#222">'
        f'记忆强度变化曲线 — 预测至 {fmt_delta(t1, now)}'
        f'（{tau_summary(agent.cfg)} · Hot {counts[Tier.HOT]} / Warm {counts[Tier.WARM]} / Cold {counts[Tier.COLD]}）</text>'
    )
    static.append(f'<text x="{ML}" y="50" font-size="12" fill="#888">实线=遗忘曲线预测，圆点=观测采样（每轮对话自动记录），灰色虚线=实际观测轨迹，黑色虚线=当前时刻，类型色虚线=配置τ的典型遗忘参考（绿=技能/紫=语义/橙=情景）；菱形=唤醒事件，红条=实测跳升 dev，青条=类型预期 expected，徽章=比值 dev/expected</text>')

    # 贴合度摘要（第二行）：实测τ vs 配置τ，偏离 100% 处提示该类型 τ 配置失准
    fit = fit_report(agent, now)
    if fit["overall"]["obs"] > 0:
        bits = []
        for t in MemType:
            d = fit["by_type"][t.value]
            pct = f"{d['fit'] * 100:.0f}%" if d["fit"] is not None else "—"
            bits.append(f"{t.value} {pct}")
        static.append(
            f'<text x="{ML}" y="68" font-size="12" fill="#666">'
            f'贴合度（实测τ vs 配置τ）: {" ｜ ".join(bits)} — 偏离 100% 处提示该类型 τ 配置失准</text>'
        )

    # 图例（右上）：层级 + 唤醒信号徽章（learn_tau 校准过程中比值趋 1）
    lx, ly = W - MR - 210, MT + 16
    for i, t in enumerate(Tier):
        y = ly + i * 22
        static.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 26}" y2="{y}" stroke="{TIER_COLORS[t]}" stroke-width="3"/>')
        static.append(f'<text x="{lx + 34}" y="{y + 4}" font-size="13" fill="#444">{TIER_LABELS[t]} {counts[t]}条</text>')
    y_aw = ly + 3 * 22
    static.append(f'<path d="M{lx} {y_aw - 5} l4 5 l-4 5 l-4 -5 z" fill="#8a8f98" stroke="#fff" stroke-width="1"/>')
    static.append(f'<text x="{lx + 34}" y="{y_aw + 4}" font-size="12" fill="#444">唤醒事件（信号徽章=比值）</text>')
    y_aw2 = y_aw + 18
    static.append(f'<rect x="{lx}" y="{y_aw2 - 9}" width="30" height="13" rx="2.5" fill="#e34a2f"/>')
    static.append(f'<text x="{lx + 15}" y="{y_aw2 + 1}" font-size="8.5" fill="#fff" text-anchor="middle">1.4×</text>')
    static.append(f'<text x="{lx + 34}" y="{y_aw2}" font-size="11" fill="#888">>1 唤醒比预期剧烈（τ 应下调）</text>')

    # 网格与 Y 轴
    for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = py(s)
        plot.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + plot_w}" y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        plot.append(f'<text x="{ML - 10}" y="{y + 4:.1f}" font-size="12" fill="#888" text-anchor="end">{s:.1f}</text>')
    # 强度下限 0.2 虚线标注
    yf = py(0.2)
    plot.append(f'<line x1="{ML}" y1="{yf:.1f}" x2="{ML + plot_w}" y2="{yf:.1f}" stroke="#f0b840" stroke-width="1.2" stroke-dasharray="6 4"/>')
    plot.append(f'<text x="{ML + plot_w}" y="{yf - 6:.1f}" font-size="11" fill="#c08a20" text-anchor="end">强度下限 0.2</text>')
    plot.append(f'<text x="{ML - 10}" y="{MT - 10}" font-size="12" fill="#888" text-anchor="end">强度</text>')

    # X 轴刻度
    step = _nice_step(t1 - t0)
    x = math.ceil(t0 / step) * step
    while x <= t1:
        if x >= t0:
            plot.append(f'<line x1="{px(x):.1f}" y1="{MT}" x2="{px(x):.1f}" y2="{MT + plot_h}" stroke="#f0f0f0" stroke-width="1"/>')
            plot.append(f'<text x="{px(x):.1f}" y="{MT + plot_h + 20}" font-size="12" fill="#888" text-anchor="middle">{fmt_delta(x, now)}</text>')
        x += step
    plot.append(f'<text x="{ML + plot_w / 2}" y="{H - 18}" font-size="12" fill="#888" text-anchor="middle">相对现在的时间（0=现在）</text>')

    # 当前时刻竖线
    plot.append(f'<line x1="{px(now):.1f}" y1="{MT}" x2="{px(now):.1f}" y2="{MT + plot_h}" stroke="#333" stroke-width="1.2" stroke-dasharray="4 4"/>')
    plot.append(f'<text x="{px(now):.1f}" y="{MT - 8}" font-size="12" fill="#333" text-anchor="middle">现在</text>')

    return static, plot, _ChartCtx(t0, t1, px, py, memories, counts, fit)


def render_svg(
    agent,
    path: str = "memories_curves.svg",
    horizon_seconds: float | None = None,
    now: float | None = None,
    samples: int = 200,
) -> str:
    """把全部记忆的强度曲线渲染成一张 SVG，返回文件路径。"""
    now = now if now is not None else time.time()
    horizon = horizon_seconds or default_horizon(agent)
    static, plot, ctx = _scaffold(agent, now, horizon, samples)
    parts = static + plot
    px, py, memories = ctx.px, ctx.py, ctx.memories
    t0, t1 = ctx.t0, ctx.t1

    # 类型参考曲线（底层虚线）：配置 τ 的"典型遗忘"预期斜率，与全部记忆曲线同图对照
    # （类型色：技能绿/语义紫/情景橙；一条从 t0 创建、重要0.1、零检索记忆的衰减）
    for mt in MemType:
        ref_pts = " ".join(
            f"{px(t):.1f},{py(s):.1f}"
            for t, s in _reference_series(agent, mt, t0, t1, samples)
        )
        tip = f"参考：{mt.value} 典型遗忘（τ={fmt_duration(agent.cfg.tau_for(mt))}）"
        parts.append(
            f'<polyline points="{ref_pts}" fill="none" stroke="{MTYPE_COLORS[mt]}" '
            f'stroke-width="1.6" stroke-dasharray="6 4" stroke-opacity="0.75">'
            f'<title>{_esc(tip)}</title></polyline>'
        )

    # 曲线：预测折线（实线，按层级配色，线宽 ∝ 重要性）
    for mem in memories:
        pts = " ".join(f"{px(t):.1f},{py(s):.1f}" for t, s in strength_series(agent, mem, now, horizon, samples))
        label = (mem.summary or mem.content)[:60]
        tip = f"{mem.id} [{mem.tier.value}] 强度历史 {len(mem.history)} 次 · {label}"
        width = 1.2 + mem.importance * 2.6  # 重要性越高线越粗
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{TIER_COLORS[mem.tier]}" '
            f'stroke-width="{width:.2f}" stroke-opacity="0.85"><title>{_esc(tip)}</title></polyline>'
        )
        # 检索事件环标（access_count 增大的采样时刻）
        for ts in access_events(mem):
            if t0 <= ts <= t1:
                s = strength_at(mem.history, ts)
                parts.append(
                    f'<circle cx="{px(ts):.1f}" cy="{py(s):.1f}" r="4.6" fill="none" '
                    f'stroke="{TIER_COLORS[mem.tier]}" stroke-width="1.5">'
                    f'<title>检索命中 {mem.id}</title></circle>'
                )

    # 实际观测轨迹（灰色虚线，穿过所有采样点）：偏离预测实线处 = 检索/再巩固干扰或 τ 失准
    for mem in memories:
        pts = " ".join(
            f"{px(row[0]):.1f},{py(row[1]):.1f}" for row in mem.history if t0 <= row[0] <= t1
        )
        if len(mem.history) >= 2:
            tip = f"实际观测轨迹 {mem.id}（偏离预测实线处即发生干扰或 τ 失准）"
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="#555" stroke-width="1.3" '
                f'stroke-dasharray="3 3" stroke-opacity="0.65"><title>{_esc(tip)}</title></polyline>'
            )

    # 实际采样点（圆点）
    for mem in memories:
        color = TIER_COLORS[mem.tier]
        for row in mem.history:
            ts, s = row[0], row[1]
            if t0 <= ts <= t1:
                tip = f"{mem.id} 采样 {fmt_delta(ts, now)}：强度 {s:.2f}（{mem.summary or mem.content[:30]}）"
                parts.append(
                    f'<circle cx="{px(ts):.1f}" cy="{py(s):.1f}" r="3.2" fill="{color}" '
                    f'stroke="#fff" stroke-width="1"><title>{_esc(tip)}</title></circle>'
                )

    # 唤醒事件标注（全部历史）：菱形 = 唤醒点（唤醒后实测强度）；红条 = 实测
    # 跳升 dev（真实 τ 延续预测 → 实测），青条 = 类型预期 expected（模型 τ 延续
    # 预测 → 实测）——两条都结束于实测点高度，红条长于青条即"唤醒比类型预期
    # 剧烈"（比值>1，τ 应下调）；信号徽章 = 比值 dev/expected，颜色随校准状态
    # 变化（红 >1 → 信号活跃；青 ≤1 → 已校准）——learn_tau 收敛时徽章红色渐
    # 渐转青、比值趋 1，直观展示信号随学习轮次衰减收敛。
    for mem in memories:
        for i, ev in enumerate(_awakening_events(mem), 1):
            ts, dev, exp = ev["ts"], ev["dev"], ev["expected"]
            if not (t0 <= ts <= t1):
                continue
            s_act = strength_at(mem.history, ts)
            px_ts = px(ts)
            py_act = py(s_act)
            y_dev = py(max(0.2, s_act - dev))
            y_exp = py(max(0.2, s_act - exp))
            parts.append(
                f'<path d="M{px_ts:.1f} {py_act:.1f} l4 -4 l4 4 l-4 4 z" '
                f'fill="{TIER_COLORS[mem.tier]}" stroke="#fff" stroke-width="1">'
                f'<title>唤醒 {mem.id} 第{i}次 · dev {dev} vs 预期 {exp}</title></path>'
            )
            parts.append(
                f'<line x1="{px_ts:.1f}" y1="{y_dev:.1f}" x2="{px_ts + 9:.1f}" y2="{y_dev:.1f}" '
                f'stroke="#e34a2f" stroke-width="2.5">'
                f'<title>实测跳升 dev {dev}（真实 τ 延续预测 → 实测）</title></line>'
            )
            parts.append(
                f'<line x1="{px_ts + 2:.1f}" y1="{y_exp:.1f}" x2="{px_ts + 11:.1f}" y2="{y_exp:.1f}" '
                f'stroke="#2a9d8f" stroke-width="2.5">'
                f'<title>类型预期 expected {exp}（模型 τ 延续预测 → 实测）</title></line>'
            )
            # 信号徽章：比值 dev/expected，颜色 = 校准状态（红 >1 → τ 应下调；
            # 青 ≤1 → 已校准）。比值趋 1 即 learn_tau 收敛（信号衰减）。
            if ev["ratio"] is not None:
                ratio = ev["ratio"]
                badge = "1.0×" if abs(ratio - 1.0) < 0.05 else f"{ratio:.1f}×"
                bcolor = "#e34a2f" if ratio > 1.05 else ("#2a9d8f" if ratio < 0.95 else "#95a5a6")
                bw = 26 + len(badge) * 4
                bx = px_ts - bw / 2 + 8
                by = py_act - 20
                parts.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="13" rx="2.5" fill="{bcolor}" opacity="0.92">'
                    f'<title>信号徽章：比值 {ratio}（{"唤醒比类型预期剧烈 → τ 应下调" if ratio > 1 else "比预期温和/已校准"}）'
                    f'—— learn_tau 校准过程中该值趋 1</title></rect>'
                )
                parts.append(
                    f'<text x="{bx + bw / 2:.1f}" y="{by + 10:.1f}" font-size="8.5" fill="#fff" '
                    f'text-anchor="middle">{badge}</text>'
                )

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# ---------- 按类型分面板（对比遗忘斜率） ----------

TYPE_PANEL_H = 200   # 子图绘图区高度
TYPE_LABEL_H = 30    # 子图标题行高度
TYPE_GAP = 26        # 子图间距
TYPE_HEADER_H = 80   # 顶部标题区高度


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def _reference_series(agent, mtype: MemType, t0: float, t1: float, samples: int = 200) -> list[list]:
    """某类型的"典型遗忘参考曲线"：一条新记忆（重要0.1、零检索）按该类型 τ 的衰减。"""
    return [
        [t, agent.strength_at_state(mtype, last_access=t0, access_count=0, importance=0.1, t=t)]
        for t in _linspace(t0, t1, samples)
    ]


STRENGTH_FLOOR = 0.2  # 与 agent.py 的检索强度下限一致（触底后不再衰减）


def _floor_time(
    agent, mtype: MemType, last_access: float, access_count: int,
    importance: float, now: float, tau: float,
) -> float | None:
    """预测该状态从 now 起多久衰减到强度下限 0.2（30τ 内未触底返回 None）。"""
    if agent.strength_at_state(mtype, last_access, access_count, importance, now) <= STRENGTH_FLOOR + 1e-6:
        return 0.0
    step = max(1e-6, tau / 10.0)
    t = now
    for _ in range(301):  # 最多推 30τ
        t += step
        if agent.strength_at_state(mtype, last_access, access_count, importance, t) <= STRENGTH_FLOOR + 1e-6:
            return t - now
    return None


def forgetting_slope(agent, mem, now: float | None = None) -> dict:
    """某记忆的遗忘斜率：每 τ 强度下降量 + 触底时间，并与同类型典型遗忘对比。

    对比用**触底时间**而非斜率比——斜率归一化到每 τ 后纯 recency 衰减的绝对
    下降量跨类型相同（都是 e⁻¹ 比例），且触底钳制会扭曲斜率比（同 learn_tau 的
    "触底段不参与反推"教训）；触底时间区分度大且直观。返回：
    - slope_per_tau：每 τ 的强度变化（≤0，已触底≈0）；
    - time_to_floor：按模型预测多久衰减到 0.2 下限（None = 30τ 内不触底，
      被检索/重要性抬高到永不触底）；
    - ref_time_to_floor：同类型典型（重要0.1、零检索）的触底时间；
    - ratio = 记忆触底时间 / 参考触底时间：>1 更持久（忘得慢），<1 更快触底；
    - label："持久 N 倍" / "快 N% 触底" / "≈ 典型"。
    """
    now = now if now is not None else time.time()
    tau = agent._tau_for(mem)
    slope = agent._strength_at(mem, now + tau) - agent._strength_at(mem, now)
    ttf = _floor_time(agent, mem.mtype, mem.last_access, mem.access_count, mem.importance, now, tau)
    ref_ttf = _floor_time(agent, mem.mtype, now, 0, 0.1, now, tau)
    ratio = (ttf / ref_ttf) if (ttf is not None and ref_ttf and ref_ttf > 1e-9) else None
    if ratio is None:
        label = "持久（不触底）" if ttf is None else "≈ 典型"
    elif abs(ratio - 1.0) < 0.15:
        label = "≈ 典型"
    elif ratio > 1.0:
        label = f"持久 {ratio:.1f} 倍"
    else:
        label = f"快 {(1.0 - ratio) * 100:.0f}% 触底"
    return {
        "slope_per_tau": round(slope, 4),
        "time_to_floor": round(ttf, 1) if ttf is not None else None,
        "ref_time_to_floor": round(ref_ttf, 1) if ref_ttf is not None else None,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "label": label,
    }


def floor_verification(agent, mem, now: float | None = None) -> dict:
    """实测触底验证：用 _observe 的观测采样跟踪实际触底时刻，与预测对比。

    观测采样（_record_sample）按"真实"τ（true_tau_by_type，未配置时同模型 τ）
    记录强度并钳到下限 0.2，所以历史里第一个强度 ≤ 0.2 的采样点 = 实测触底
    时刻；对比 forgetting_slope 从同一状态（该衰减段起始的最后访问）按模型 τ
    预测的触底时长——把"遗忘斜率"从纯预测升级为预测 vs 实际。返回：
    - floored：是否已观测到触底采样；
    - actual_ts / actual_dt：首次触底采样时刻 / 距该段起始最后访问的时长（秒）；
    - predicted_dt：从同一状态按模型 τ 预测的触底时长（None = 模型预测不触底）；
    - ratio = actual_dt / predicted_dt：<1 实际更快触底（衰减快于预期），>1 更慢；
    - status："not_floored" / "verified"；
    - label：人类可读结论（实测 X vs 预测 Y → 快/慢/贴合）。
    """
    now = now if now is not None else time.time()
    tau = agent._tau_for(mem)
    floor_row = next((r for r in mem.history if r[1] <= STRENGTH_FLOOR + 1e-9), None)
    if floor_row is None:
        return {
            "floored": False,
            "actual_ts": None,
            "actual_dt": None,
            "predicted_dt": None,
            "ratio": None,
            "status": "not_floored",
            "label": "尚未实测触底",
        }
    ts, _s, la, acc, imp = floor_row
    actual_dt = ts - la
    predicted_dt = _floor_time(agent, mem.mtype, la, acc, imp, la, tau)
    if predicted_dt is None:
        label = f"实测 {fmt_duration(actual_dt)} 触底（模型预测不触底）"
        ratio = None
    else:
        ratio = actual_dt / predicted_dt
        if abs(ratio - 1.0) < 0.15:
            label = f"实测 {fmt_duration(actual_dt)} vs 预测 {fmt_duration(predicted_dt)}（≈贴合）"
        elif ratio < 1.0:
            label = f"实测 {fmt_duration(actual_dt)}，比预测快 {(1.0 - ratio) * 100:.0f}%（衰减快于预期）"
        else:
            label = f"实测 {fmt_duration(actual_dt)}，比预测慢 {(ratio - 1.0) * 100:.0f}%（衰减慢于预期）"
    return {
        "floored": True,
        "actual_ts": round(ts, 3),
        "actual_dt": round(actual_dt, 3),
        "predicted_dt": round(predicted_dt, 3) if predicted_dt is not None else None,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "status": "verified",
        "label": label,
    }


def render_svg_by_type(
    agent,
    path: str = "memories_by_type.svg",
    horizon_seconds: float | None = None,
    now: float | None = None,
    samples: int = 200,
) -> str:
    """按类型分面板渲染曲线：技能/语义/情景各一张子图，共享横轴对比遗忘斜率。

    每张子图：该类型记忆的强度曲线（层级配色、线宽∝重要性）+ 灰色虚线参考曲线
    （按该类型 τ 的典型遗忘），并标注配置τ / 实测τ / 贴合度。
    """
    now = now if now is not None else time.time()
    horizon = horizon_seconds or default_horizon(agent)
    memories = sorted(agent.store.all(), key=lambda m: (-m.importance, m.created_at))
    t1 = now + horizon
    t0 = min([m.created_at for m in memories] + [now]) if memories else now
    if t1 <= t0:
        t0 = now - 60

    plot_w = W - ML - MR
    H = TYPE_HEADER_H + 3 * (TYPE_LABEL_H + TYPE_PANEL_H) + 2 * TYPE_GAP + 44

    def px(t: float) -> float:
        return ML + (t - t0) / (t1 - t0) * plot_w

    fit = fit_report(agent, now)
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

    # 顶部标题 + 层级图例
    parts.append(
        f'<text x="{ML}" y="30" font-size="20" font-weight="bold" fill="#222">'
        f'按类型分面板：遗忘斜率对比（横轴共享，可直接对比衰减快慢）</text>'
    )
    parts.append(f'<text x="{ML}" y="52" font-size="12" fill="#888">{_esc(tau_summary(agent.cfg))} · 灰色虚线=该类型典型遗忘参考曲线</text>')
    counts = {t: len(agent.store.by_tier(t)) for t in Tier}
    lx = W - MR - 230
    for i, t in enumerate(Tier):
        y = 24 + i * 18
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 22}" y2="{y}" stroke="{TIER_COLORS[t]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 28}" y="{y + 4}" font-size="12" fill="#444">{TIER_LABELS[t]} {counts[t]}条</text>')

    for i, mt in enumerate(MemType):
        top = TYPE_HEADER_H + i * (TYPE_LABEL_H + TYPE_PANEL_H + TYPE_GAP)
        members = [m for m in memories if m.mtype is mt]
        d = fit["by_type"][mt.value]
        bits = [f"{mt.value}（{len(members)} 条）", f"τ={fmt_duration(agent.cfg.tau_for(mt))}"]
        if d["tau_est"] is not None:
            bits.append(f"实测τ≈{fmt_duration(d['tau_est'])} 贴合度{d['fit'] * 100:.0f}%")
        parts.append(
            f'<text x="{ML}" y="{top + 19}" font-size="14" font-weight="bold" fill="#333">'
            f'{_esc(" · ".join(bits))}</text>'
        )

        y0 = top + TYPE_LABEL_H

        def py(s: float) -> float:
            return y0 + (1.0 - s) * TYPE_PANEL_H

        # 网格
        for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            y = py(s)
            parts.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + plot_w}" y2="{y:.1f}" stroke="#ececec" stroke-width="1"/>')
        # 强度下限
        yf = py(0.2)
        parts.append(f'<line x1="{ML}" y1="{yf:.1f}" x2="{ML + plot_w}" y2="{yf:.1f}" stroke="#f0b840" stroke-width="1" stroke-dasharray="5 4"/>')
        parts.append(f'<text x="{ML + plot_w}" y="{yf - 5:.1f}" font-size="10" fill="#c08a20" text-anchor="end">强度下限 0.2</text>')

        # 参考曲线（该类型典型遗忘斜率）
        ref_pts = " ".join(
            f"{px(t):.1f},{py(s):.1f}" for t, s in _reference_series(agent, mt, t0, t1, samples)
        )
        parts.append(
            f'<polyline points="{ref_pts}" fill="none" stroke="#999" stroke-width="1.4" '
            f'stroke-dasharray="5 4"><title>参考：{mt.value} 典型遗忘（τ={fmt_duration(agent.cfg.tau_for(mt))}）</title></polyline>'
        )

        # 该类型记忆曲线（层级配色、线宽∝重要性）
        for mem in members:
            width = 1.2 + mem.importance * 2.6
            pts = " ".join(
                f"{px(t):.1f},{py(s):.1f}"
                for t, s in strength_series(agent, mem, now, horizon, samples)
            )
            sl = forgetting_slope(agent, mem, now)
            fc = floor_verification(agent, mem, now)
            tip = f"{mem.id} [{mem.tier.value}] {mem.summary or mem.content[:50]} · 遗忘斜率 每τ {sl['slope_per_tau']:.2f}（{sl['label']}）"
            if fc["floored"]:
                tip += f" · 触底验证 {fc['label']}"
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{TIER_COLORS[mem.tier]}" '
                f'stroke-width="{width:.2f}" stroke-opacity="0.85"><title>{_esc(tip)}</title></polyline>'
            )
            for row in mem.history:
                ts, s = row[0], row[1]
                if t0 <= ts <= t1:
                    parts.append(
                        f'<circle cx="{px(ts):.1f}" cy="{py(s):.1f}" r="2.4" '
                        f'fill="{TIER_COLORS[mem.tier]}" stroke="#fff" stroke-width="0.8"/>'
                    )

        if not members:
            parts.append(
                f'<text x="{ML + plot_w / 2}" y="{y0 + TYPE_PANEL_H / 2}" font-size="13" '
                f'fill="#bbb" text-anchor="middle">无 {mt.value} 类记忆</text>'
            )

        # 底部面板才画 X 轴刻度（共享横轴）
        if i == len(list(MemType)) - 1:
            step = _nice_step(t1 - t0)
            x = math.ceil(t0 / step) * step
            while x <= t1:
                if x >= t0:
                    parts.append(f'<line x1="{px(x):.1f}" y1="{y0 + TYPE_PANEL_H}" x2="{px(x):.1f}" y2="{y0 + TYPE_PANEL_H + 5}" stroke="#999"/>')
                    parts.append(f'<text x="{px(x):.1f}" y="{y0 + TYPE_PANEL_H + 18}" font-size="11" fill="#888" text-anchor="middle">{fmt_delta(x, now)}</text>')
                x += step
            parts.append(
                f'<text x="{ML + plot_w / 2}" y="{H - 16}" font-size="12" fill="#888" text-anchor="middle">相对现在的时间（0=现在）</text>'
            )

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# ---------- CSV / JSON 导出 ----------

def _awakening_events(mem) -> list[dict]:
    """从记忆的 awakenings 日志推导唤醒事件明细（供 JSON/CSV 导出）。

    每个事件 {ts, dev, expected, ratio, mtype}（六元组唤醒日志还带 dt / n_cold）：
    dev = 实测跳升强度，expected = 同事件的类型预期偏差（模型信念 τ 的延续预测），
    ratio = dev/expected（>1 = 唤醒比类型预期剧烈 → 可塑性更活跃；
    <1 = 比预期温和）。ratio 仅在 dev、expected 都 > 0 时给出（与学习器
    `_tau_awakening_estimate` / `_awakening_drift_estimate` 的门控一致），
    否则 None。旧格式三元组（无类型预期偏差，早期版本遗留）跳过——比值
    无定义，原样保留在 mem.awakenings 里供完整追溯。

    六元组日志（[ts, dev, expected, mtype, 埋藏时长, 检索次数]）额外带
    dt / n_cold 键（四元组时无此键），供事件级导出按衰减公式精确重算。
    """
    events: list[dict] = []
    for aw in mem.awakenings:
        if len(aw) < 4:
            continue
        ts, dev, expected, mtype = aw[0], float(aw[1]), float(aw[2]), aw[3]
        ratio = None
        if dev > 0 and expected > 0:
            ratio = round(dev / expected, 4)
        ev = {
            "ts": ts,
            "dev": round(dev, 4),
            "expected": round(expected, 4),
            "ratio": ratio,
            "mtype": mtype,
        }
        if len(aw) >= 6:
            ev["dt"] = round(float(aw[4]), 1)
            ev["n_cold"] = int(aw[5])
        events.append(ev)
    return events


def export_csv(
    agent,
    path: str = "memories_curves.csv",
    horizon_seconds: float | None = None,
    now: float | None = None,
) -> str:
    """长格式 CSV：每行一条记忆在某一时刻的强度（row_type="sample"），
    并追加该记忆的**唤醒事件行**（row_type="awakening"）——实测偏差 dev 放
    strength 列、类型预期与比值放 content 列（"expected=… ratio=…"），
    mtype 取**唤醒时刻**的类型。外部工具按 row_type 过滤即可在同一张表里
    同时拿到曲线采样与唤醒明细。"""
    now = now if now is not None else time.time()
    horizon = horizon_seconds or default_horizon(agent)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "memory_id", "tier", "mtype", "kind", "importance",
                    "access_count", "t_relative_seconds", "strength", "content"])
        for mem in sorted(agent.store.all(), key=lambda m: m.id):
            for t, s in strength_series(agent, mem, now, horizon):
                w.writerow(["sample", mem.id, mem.tier.value, mem.mtype.value, mem.kind,
                            mem.importance, mem.access_count, round(t - now, 1), round(s, 4),
                            mem.summary or mem.content])
            for ev in _awakening_events(mem):
                ratio = ev["ratio"] if ev["ratio"] is not None else ""
                w.writerow(["awakening", mem.id, mem.tier.value, ev["mtype"], mem.kind,
                            mem.importance, mem.access_count, round(ev["ts"] - now, 1),
                            ev["dev"], f"expected={ev['expected']} ratio={ratio}"])
    return path


def export_json(
    agent,
    path: str = "memories_curves.json",
    horizon_seconds: float | None = None,
    now: float | None = None,
) -> str:
    """结构化 JSON：预测序列 + 实际采样 + 元数据。

    顶层三表与交互仪表盘一致：profiles（类型画像）+ signal_drift（信号漂移对比）
    + health（τ 两路信号健康检查合表）——静态导出同屏可查。
    """
    from .agent import (  # 函数级导入：避免模块级环
        awakening_signal_periods,
        awakening_signal_stats,
        tau_learner_health,
    )
    from .profiles import type_profiles  # 函数级导入：profiles 依赖本模块的 fmt_duration

    now = now if now is not None else time.time()
    horizon = horizon_seconds or default_horizon(agent)
    data = []
    for mem in sorted(agent.store.all(), key=lambda m: m.id):
        data.append(
            {
                "id": mem.id,
                "tier": mem.tier.value,
                "mtype": mem.mtype.value,
                "mtype_confidence": mem.mtype_confidence,
                "kind": mem.kind,
                "importance": mem.importance,
                "access_count": mem.access_count,
                "content": mem.content,
                "summary": mem.summary,
                "revision_count": mem.revision_count,
                "labile_until": mem.labile_until,
                "revisions": mem.revisions,
                "semanticization_score": round(agent._semanticization_score(mem), 4),
                "migrations": mem.migrations,
                "checks": mem.checks,
                # 唤醒事件明细：awakenings = 原始四元组（完整追溯）；
                # awakening_events = 推导明细（实测/预期/比值，旧格式三元组跳过），
                # 供外部工具按 mtype 分组直接分析类型可塑性。
                "awakenings": mem.awakenings,
                "awakening_events": _awakening_events(mem),
                "series": strength_series(agent, mem, now, horizon),
                "recorded": [[row[0], row[1]] for row in mem.history],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"now": now, "horizon_seconds": horizon,
             "fit": fit_report(agent, now), "memories": data,
             "reconsolidation_factors": {
                 t.value: agent.cfg.reconsolidation_by_type.get(t, {})
                 for t in MemType
             },
             "learned_plasticity": agent.store.meta.get("learned_plasticity"),
             "profiles": [p.to_dict() for p in type_profiles(agent.cfg, awakening_signal_stats(agent))],
             # 信号漂移：最近 30 天 vs 更早的方向一致性对比（方向翻转 = 类型行为
             # 随时间变化，需重新审视配置）。
             "signal_drift": awakening_signal_periods(agent),
             # τ 两路信号健康检查合表（与仪表盘一致）：干净段 vs 唤醒的方向一致性
             # + 行动建议 + 置信度（单一事实源 agent.tau_learner_health）。
             "health": tau_learner_health(agent)},
            f, ensure_ascii=False, indent=2,
        )
    return path


# ---------- τ 学习收敛轨迹 ----------

def tau_rounds(agent) -> list[dict]:
    """learn_tau 的学习轮次：合并估计 + 两路信号的独立估计 + 唤醒信号原始值。

    每行 {mtype, old_tau, tau_est, new_tau, confidence, clean_est, aw_est,
    ratio, dev, expected}——dev/expected 为本次更新实际使用的唤醒中位值
    （11 列学习历史，dev > expected = 该类型埋得比信念深 → 下调 τ 的方向
    依据）。clean_est/aw_est/ratio/dev/expected 缺源的轮次是 None（旧 6/9
    列历史同样 None——向后兼容）。
    """
    out: list[dict] = []
    for row in agent._learn_history:
        out.append({
            "mtype": row[1], "old_tau": row[2], "tau_est": row[3],
            "new_tau": row[4], "confidence": row[5],
            "clean_est": row[6] if len(row) > 6 else None,
            "aw_est": row[7] if len(row) > 7 else None,
            "ratio": row[8] if len(row) > 8 else None,
            "dev": row[9] if len(row) > 9 else None,
            "expected": row[10] if len(row) > 10 else None,
        })
    return out


def export_tau_trajectory(agent, path: str = "tau_convergence.csv") -> str:
    """学习轨迹 CSV：每行一轮学习（合并估计 + 两路独立估计 + 唤醒中位比值
    + 唤醒信号原始值 dev/expected——方向可复盘）。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "mtype", "old_tau_seconds", "tau_est_seconds",
                    "new_tau_seconds", "confidence", "clean_est_seconds",
                    "awakening_est_seconds", "awakening_ratio",
                    "awakening_dev", "awakening_expected"])
        for i, r in enumerate(tau_rounds(agent), 1):
            w.writerow([
                i, r["mtype"], r["old_tau"], r["tau_est"], r["new_tau"],
                r["confidence"],
                r["clean_est"] if r["clean_est"] is not None else "",
                r["aw_est"] if r["aw_est"] is not None else "",
                r["ratio"] if r["ratio"] is not None else "",
                r["dev"] if r["dev"] is not None else "",
                r["expected"] if r["expected"] is not None else "",
            ])
    return path


def render_tau_convergence(agent, path: str = "tau_convergence.svg") -> str:
    """learn_tau 两路信号的收敛轨迹图（按学习轮次）。

    每个有学习记录的类型一张面板，上下两个子图：
    - **τ（对数轴）**：配置 τ 的 EMA 轨迹（实线，轮次 0 = 初始信念）、
      干净段反推 τ_est（紫虚线）、唤醒偏差 τ_est（橙虚线）、真实 τ（灰
      虚线，配置了 true_tau_by_type 时）；
    - **唤醒比值 dev/expected**：随轮次逼近 1（灰虚线 = 与真实一致）。

    两条 τ_est 线都向真实 τ 收敛 = 两路信号互相印证；比值趋 1 = 唤醒不再
    比类型预期剧烈（τ 已校准）。
    """
    rounds = tau_rounds(agent)
    by_type: dict[str, list[dict]] = {}
    for r in rounds:
        by_type.setdefault(r["mtype"], []).append(r)
    true_tau = agent.cfg.true_tau_by_type

    W2, ML2, MR2 = 1100, 96, 30
    title_h, sub_h, gap = 34, 190, 22
    H2 = max(240, 56 + len(by_type) * (title_h + 2 * sub_h + gap) + 40)
    pw = W2 - ML2 - MR2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W2}" height="{H2}" '
        f'font-family="Microsoft YaHei, sans-serif">',
        f'<text x="{W2 / 2}" y="30" font-size="16" font-weight="bold" fill="#222" '
        f'text-anchor="middle">learn_tau 两路信号收敛轨迹（按学习轮次）</text>',
    ]

    def px(rnd: float, n: int) -> float:
        return ML2 + rnd / max(n, 1) * pw

    if not rounds:
        parts.append(
            f'<text x="{W2 / 2}" y="120" font-size="13" fill="#999" text-anchor="middle">'
            f'（尚无学习轮次——运行 learn_tau 且产生参数更新后才会出现轨迹）</text>'
        )
        parts.append("</svg>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return path

    y = 56
    for mtype_val, rs in by_type.items():
        n = len(rs)
        mtype = MemType(mtype_val)
        color = MTYPE_COLORS.get(mtype, "#666")
        real = true_tau.get(mtype) if mtype in true_tau else None
        final = rs[-1]["new_tau"]
        # 面板标题 + 图例
        title = (f'{mtype.value} · τ: {fmt_duration(rs[0]["old_tau"])} → '
                 f'{fmt_duration(final)}' + (f'（真实 {fmt_duration(real)}）' if real else ''))
        parts.append(f'<text x="{ML2}" y="{y + 14}" font-size="14" font-weight="bold" '
                     f'fill="{color}">{title}</text>')
        legends = [("#222", "配置τ(EMA)"), ("#7048e8", "干净段τ_est"),
                   ("#f08c00", "唤醒τ_est"), ("#8a8f98", "真实τ/比值1")]
        lx = ML2 + pw
        for lc, ll in reversed(legends):  # 从右往左排
            lx -= 14 + 8 * len(ll) + 10
            dash = ' stroke-dasharray="5,3"' if lc != "#222" else ""
            parts.append(
                f'<line x1="{lx}" y1="{y + 9}" x2="{lx + 12}" y2="{y + 9}" '
                f'stroke="{lc}" stroke-width="2"{dash}/>'
            )
            parts.append(f'<text x="{lx + 14}" y="{y + 13}" font-size="10" fill="#666">{ll}</text>')
        y += title_h

        # 上子图：τ（对数轴）
        ys, ye = y, y + sub_h
        vals = [r["old_tau"], r["new_tau"]]
        vals += [r["clean_est"] for r in rs if r["clean_est"]]
        vals += [r["aw_est"] for r in rs if r["aw_est"]]
        if real:
            vals.append(real)
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            vmin, vmax = vmin * 0.5, vmax * 2
        lmin, lmax = math.log10(vmin), math.log10(vmax)

        def py(v: float) -> float:
            return ye - (math.log10(max(v, 1e-9)) - lmin) / (lmax - lmin) * (ye - ys)

        k = math.floor(lmin)
        while k <= math.ceil(lmax):
            v = 10 ** k
            if vmin <= v <= vmax:
                parts.append(f'<line x1="{ML2}" y1="{py(v):.1f}" x2="{W2 - MR2}" y2="{py(v):.1f}" stroke="#eee"/>')
                parts.append(f'<text x="{ML2 - 6}" y="{py(v) + 4:.1f}" font-size="10" fill="#999" '
                             f'text-anchor="end">{fmt_duration(v)}</text>')
            k += 1
        if real:
            parts.append(f'<line x1="{ML2}" y1="{py(real):.1f}" x2="{W2 - MR2}" y2="{py(real):.1f}" '
                         f'stroke="#8a8f98" stroke-dasharray="5,4"/>')
            parts.append(f'<text x="{W2 - MR2 - 4}" y="{py(real) - 5:.1f}" font-size="10" fill="#8a8f98" '
                         f'text-anchor="end">真实 τ={fmt_duration(real)}</text>')
        for label, key, lc in [("干净段", "clean_est", "#7048e8"), ("唤醒", "aw_est", "#f08c00")]:
            pts = [(i, r[key]) for i, r in enumerate(rs, 1) if r[key] is not None]
            if pts:
                d = "M" + " L".join(f"{px(i, n):.1f},{py(v):.1f}" for i, v in pts)
                parts.append(f'<path d="{d}" fill="none" stroke="{lc}" stroke-width="2" stroke-dasharray="6,3"/>')
                for i, v in pts:
                    parts.append(f'<circle cx="{px(i, n):.1f}" cy="{py(v):.1f}" r="3" fill="{lc}"/>')
        # EMA 轨迹：轮次 0 = 初始信念，之后每轮 = 更新后的 τ
        ema = [(0, rs[0]["old_tau"])] + [(i, r["new_tau"]) for i, r in enumerate(rs, 1)]
        d = "M" + " L".join(f"{px(i, n):.1f},{py(v):.1f}" for i, v in ema)
        parts.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{px(0, n):.1f}" cy="{py(rs[0]["old_tau"]):.1f}" r="3.5" fill="#fff" stroke="#222"/>')
        parts.append(f'<text x="14" y="{ys + 12}" font-size="10" fill="#666">τ（对数）</text>')
        y = ye + gap

        # 下子图：唤醒比值 dev/expected
        ys2, ye2 = y, y + sub_h
        ratio_pts = [(i, r["ratio"]) for i, r in enumerate(rs, 1) if r["ratio"] is not None]
        if ratio_pts:
            rv = [v for _, v in ratio_pts]
            rmin = min(0.85, min(rv) - 0.1)
            rmax = max(1.15, max(rv) + 0.1)
        else:
            rmin, rmax = 0.85, 1.15

        def pry(v: float) -> float:
            return ye2 - (v - rmin) / (rmax - rmin) * (ye2 - ys2)

        for v in (0.5, 0.75, 1.0, 1.25, 1.5):
            if rmin <= v <= rmax:
                parts.append(f'<line x1="{ML2}" y1="{pry(v):.1f}" x2="{W2 - MR2}" y2="{pry(v):.1f}" stroke="#eee"/>')
                parts.append(f'<text x="{ML2 - 6}" y="{pry(v) + 4:.1f}" font-size="10" fill="#999" '
                             f'text-anchor="end">{v}</text>')
        parts.append(f'<line x1="{ML2}" y1="{pry(1.0):.1f}" x2="{W2 - MR2}" y2="{pry(1.0):.1f}" '
                     f'stroke="#8a8f98" stroke-dasharray="5,4"/>')
        parts.append(f'<text x="{ML2 + 4}" y="{pry(1.0) - 5:.1f}" font-size="10" fill="#8a8f98">比值=1（与真实一致）</text>')
        if ratio_pts:
            d = "M" + " L".join(f"{px(i, n):.1f},{pry(v):.1f}" for i, v in ratio_pts)
            parts.append(f'<path d="{d}" fill="none" stroke="#f08c00" stroke-width="2"/>')
            for i, v in ratio_pts:
                parts.append(f'<circle cx="{px(i, n):.1f}" cy="{pry(v):.1f}" r="3" fill="#f08c00"/>')
        else:
            parts.append(
                f'<text x="{ML2 + pw / 2}" y="{(ys2 + ye2) / 2 + 4}" font-size="11" fill="#bbb" '
                f'text-anchor="middle">（该类型无唤醒观测——仅干净段源）</text>'
            )
        parts.append(f'<text x="14" y="{ys2 + 12}" font-size="10" fill="#666">唤醒比值</text>')
        # x 轴刻度 + 轮次标签
        for i in range(0, n + 1):
            parts.append(f'<line x1="{px(i, n):.1f}" y1="{ye2}" x2="{px(i, n):.1f}" y2="{ye2 + 4}" stroke="#999"/>')
            if i == 0 or i == n:
                parts.append(f'<text x="{px(i, n):.1f}" y="{ye2 + 16}" font-size="10" fill="#888" '
                             f'text-anchor="middle">{"初始" if i == 0 else i}</text>')
        parts.append(f'<text x="{W2 - MR2}" y="{ye2 + 30}" font-size="10" fill="#888" text-anchor="end">学习轮次</text>')
        y = ye2 + gap

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def export_awakenings_csv(
    agent,
    path: str = "memories_awakenings.csv",
    now: float | None = None,
) -> str:
    """唤醒事件明细 CSV：每行一次唤醒（实测/预期/比值）。

    列：memory_id（可与曲线 CSV 连接）、mtype（**唤醒时刻**的类型——记忆
    之后可能迁移类型）、ts_relative_seconds（相对现在，负值=过去）、dev
    （实测跳升强度）、expected（同事件类型预期偏差）、ratio（dev/expected，
    >1 = 唤醒比类型预期剧烈 → 可塑性更活跃）。

    供外部工具按类型直接分析可塑性：`df[df.mtype=="episodic"]` 的 ratio
    分布 > 1 即该类型被唤醒得比模型预期剧烈（τ 配置偏大 / 可塑性配置偏小）。
    旧格式唤醒日志（无类型预期）不导出（比值无定义）。空记忆库 → 仅表头。
    """
    now = now if now is not None else time.time()
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["memory_id", "mtype", "ts_relative_seconds", "dev", "expected", "ratio"])
        for mem in sorted(agent.store.all(), key=lambda m: m.id):
            for ev in _awakening_events(mem):
                w.writerow([
                    mem.id, ev["mtype"], round(ev["ts"] - now, 1),
                    ev["dev"], ev["expected"],
                    ev["ratio"] if ev["ratio"] is not None else "",
                ])
    return path
