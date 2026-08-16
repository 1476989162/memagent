"""唤醒链路曲线连续性验证：导出唤醒前后的强度曲线（recorded 采样）。

背景：recall() 是 move 语义 + 全量继承——唤醒记忆带着 Cold 阶段的观测轨迹
（history）重生。本脚本把**实际观测采样**（recorded，非模型预测线）画成
SVG：唤醒前的 Cold 衰减段 vs 唤醒后的同前缀 + 唤醒跳升 + 重建段继续衰减，
并用数据断言"无缝衔接"。

用法：
    python recall_curve_check.py              合成判别场景（默认）
    python recall_curve_check.py --real [路径]  追加真实持久化场景：从
        memories_session.json 加载真实决策记忆 → 老化 → sleep 压缩成 Cold →
        唤醒 → 临时持久化往返 → 验证真实数据的曲线连续性（只读真实文件）
    python recall_curve_check.py --awakened [路径]  多次唤醒记忆检查：从真实
        记忆库挑一条 awakenings > 1 的记忆，逐次标注全部唤醒事件（dev vs
        类型预期双条 + 信号方向徽章）——真实库无可选对象时合成一条；
        --awakened-store 可单独指定库路径（默认与 --real 共用）；路径若是
        --export-signals 导出的 JSON（顶层 events）则直接从导出文件挑多次
        唤醒记忆做逐次标注（导出 → 验证闭环）
输出（当前工作目录）：
    recall_curve_before.svg   唤醒前：Cold 衰减段（合成场景）
    recall_curve_after.svg    唤醒后：继承前缀 + 跳升 + 重建衰减（合成场景）
    recall_curve_overlay.svg  两者叠加，标注唤醒点（合成场景）
    recall_curve_fit.svg      预测线 vs recorded 叠加（蓝=模型延续预测，
                              灰/橙=实际采样；唤醒点处双条并排：红=实测偏差、
                              青=类型预期偏差，基线连线=信号幅度，附信号方向）
    recall_curve_real.svg     真实数据叠加图（--real 时额外生成）
    recall_curve_real_fit.svg 真实数据预测线 vs recorded（--real 时额外生成）
    recall_curve_awakened.svg 多次唤醒记忆轨迹 + 逐次唤醒标注（--awakened 时生成）
    recall_curve_awakened_export.svg 导出 JSON 事件时间线 + 双条标注
                          （--awakened 指向 --export-signals 的 JSON 时生成）
并打印连续性验证结论（前缀一致 / 跳升点 / 尾部衰减 / 唤醒点偏差）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType, Tier
from memagent.visualize import strength_at

DEFAULT_REAL_STORE = "memories_session.json"

# ---------- 验证核心（可被测试导入） ----------


def build_scenario() -> tuple[MemoryAgent, list, list, list]:
    """构建判别场景：Cold 纯衰减段 → 唤醒 → 重建段继续衰减。

    返回 (agent, before, after, cold)：
    before = 唤醒前 Cold 的 recorded（[[ts, strength], ...]）
    after  = 唤醒后记忆的 recorded（继承前缀 + 唤醒采样 + 尾部采样）
    cold   = Cold 记忆对象（供测试取 id 等）
    """
    clock = [1000.0]
    agent = MemoryAgent(
        cfg=AgentConfig(
            reconsolidate=False,
            true_tau_by_type={MemType.EPISODIC: 20000.0},  # 真实衰减可见
        ),
        now_fn=lambda: clock[0],
    )
    m = agent.remember("我昨天去吃了火锅", importance=0.3)
    for _ in range(4):                      # 纯衰减采样（无检索/重要性变化）
        clock[0] += 5000
        agent._record_sample(m)
    # 保留完整历史行 [时间戳, 强度, 最后访问, 检索次数, 重要性]——状态列
    # 是各采样时刻模型使用的真实状态（预测线/偏差计算直接取用，无需重建）。
    before = [list(r) for r in m.history]
    m.demote_to_cold("我昨天去吃了火锅")

    clock[0] += 5000                        # 唤醒前流逝 → 唤醒点与末采样错开
    revived = agent.recall(m.id[:6])
    clock[0] += 5000                        # 重建段继续衰减（Warm 尾部）
    agent._record_sample(revived)
    clock[0] += 5000
    agent._record_sample(revived)
    after = [list(r) for r in revived.history]
    return agent, before, after, m


def verify_continuity(before: list, after: list) -> dict:
    """数据级连续性验证，返回结论 dict。"""
    prefix_ok = after[: len(before)] == before          # 继承前缀逐位一致
    jump = after[len(before)] if len(after) > len(before) else None   # 唤醒采样
    tail_decays = (
        len(after) >= 3 and after[-2][1] > after[-1][1]  # 重建段继续衰减
    )
    return {
        "prefix_ok": bool(prefix_ok),
        "jump": jump,
        "tail_decays": bool(tail_decays),
        "n_before": len(before),
        "n_after": len(after),
    }


def predict_line(agent, mtype, state, t0, t1, tau_override: float | None = None,
                 n: int = 60) -> list[list[float]]:
    """模型预测线：从给定状态 (last_access, access_count, importance) 按遗忘公式
    外推 [t0, t1] 的强度，返回 [[ts, strength], ...]。

    tau_override 必须与采样时一致（合成场景用真实 τ）：纯衰减段上状态不变，
    预测线与 recorded 逐点重合——偏差只出现在状态变化点（如唤醒）。
    """
    la, count, imp = state
    pts = []
    for i in range(n):
        t = t0 + (t1 - t0) * i / (n - 1)
        pts.append([
            t,
            agent.strength_at_state(mtype, la, count, imp, t, tau_override=tau_override),
        ])
    return pts


_AWAKENING_EPS = 1e-6  # 方向判别容差：dev 是两强度之差、expected 是独立取整值


def awakening_direction(dev: float, expected: float) -> str:
    """唤醒信号方向：实测偏差 vs 类型预期偏差。

    up   = 实测 > 预期 → 唤醒比类型预期剧烈 → learn_tau 下调 τ、learn_plasticity 上调可塑性
    down = 实测 < 预期 → 温和 → 反向校准
    flat = 持平（自洽环境，无学习信号）。容差 _AWAKENING_EPS 防浮点尾差误判。
    """
    if dev > expected + _AWAKENING_EPS:
        return "up"
    if dev < expected - _AWAKENING_EPS:
        return "down"
    return "flat"


def _format_deviation_line(dev_info: dict) -> str:
    """报告行 ⑥：唤醒点偏差 + 类型预期参考 + 信号方向（与 SVG 标注一致）。"""
    line = (f"⑥ 唤醒点偏差：实测 {dev_info['actual']:.3f} vs 模型延续预测 "
            f"{dev_info['predicted']:.3f} → 偏差 {dev_info['deviation']:+.3f}")
    if "expected" in dev_info:
        cmp = {"up": ">", "down": "<", "flat": "="}[dev_info["signal"]]
        sig = {"up": "上调", "down": "下调", "flat": "持平"}[dev_info["signal"]]
        line += (f"；类型预期 {dev_info['expected']:+.3f} → 信号: {sig}"
                 f"（实测 {cmp} 类型预期）")
    else:
        line += "（模型只按旧状态延续，预测不到唤醒刷新）"
    return line


def awakening_deviation(verdict: dict, pred_pre: list,
                        awakenings: list | None = None) -> dict | None:
    """唤醒点偏差：实测跳升强度 vs 模型延续预测（未唤醒假想）的差值。

    verdict["jump"] = [唤醒采样时间戳, 实测强度]；pred_pre 取唤醒时刻最近的
    预测点。正偏差 = 测试效应（模型只按旧状态延续，预测不到唤醒刷新）。

    awakenings: 该记忆的唤醒观测行（[时间戳, 实测偏差, 类型预期偏差, 类型]，
    来自 _observe_awakening）。提供时按时间戳匹配，附带类型预期参考：
    - expected       = 类型预期偏差（同一事件、同一状态，只把 τ 换成模型信念）
    - expected_point = 模型 τ 延续预测的唤醒前强度（= actual − expected）
    - signal         = dev vs expected 的方向（"up" 实测更剧烈 / "down" 更温和 /
      "flat" 持平）——learn_tau 与 learn_plasticity 据此产生学习信号。
    """
    jump = verdict.get("jump")
    if not jump:
        return None
    tj, sj = jump[0], jump[1]
    pv = min(pred_pre, key=lambda p: abs(p[0] - tj))[1]
    info = {"ts": tj, "actual": sj, "predicted": pv, "deviation": sj - pv}
    if awakenings:
        aw = next(
            (a for a in awakenings
             if len(a) >= 4 and abs(float(a[0]) - tj) < 1.0),
            None,
        )
        if aw is not None:
            expected = float(aw[2])
            # 有事件记录时用事件记录的偏差（aw[1]）——它与类型预期同尺度
            # （唤醒跳升按该类型实测可塑性因子调制），曲线反推的 sj−pv 不含
            # 可塑性刻度，跨尺度比较会误判方向（自洽环境伪「上调」）。
            info["deviation"] = float(aw[1])
            info["expected"] = expected
            info["expected_point"] = sj - expected   # 模型延续预测的唤醒前强度
            info["signal"] = awakening_direction(info["deviation"], expected)
    return info


# ---------- SVG 渲染（自包含，无外部依赖） ----------

_SVG_W, _SVG_H = 900, 400
_ML, _MR, _MT, _MB = 64, 20, 46, 46


def _render_overlay_svg(before: list, after: list, path: str,
                        verdict: dict) -> str:
    """叠加图：Cold 衰减段（灰虚线）+ 唤醒后（实线，同前缀）+ 唤醒点标注。"""
    pts = before + after
    t0, t1 = min(p[0] for p in pts), max(p[0] for p in pts)
    smax = max(p[1] for p in pts)
    plot_w, plot_h = _SVG_W - _ML - _MR, _SVG_H - _MT - _MB
    span = (t1 - t0) or 1.0

    def px(t): return _ML + (t - t0) / span * plot_w
    def py(s): return _MT + plot_h - (s / (smax * 1.15)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'font-family="system-ui, sans-serif">',
        f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>',
        f'<text x="{_ML}" y="26" font-size="17" font-weight="bold" fill="#222">'
        f'唤醒链路曲线连续性：Cold 衰减段 → 唤醒跳升 → Warm 重建衰减</text>',
    ]
    # 坐标轴
    parts.append(f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + plot_h}" '
                 f'stroke="#999" stroke-width="1"/>')
    parts.append(f'<line x1="{_ML}" y1="{_MT + plot_h}" x2="{_ML + plot_w}" y2="{_MT + plot_h}" '
                 f'stroke="#999" stroke-width="1"/>')
    parts.append(f'<text x="{_ML}" y="{_MT - 6}" font-size="11" fill="#777">强度</text>')
    parts.append(f'<text x="{_ML + plot_w - 40}" y="{_MT + plot_h + 20}" font-size="11" fill="#777">时间</text>')

    # 唤醒前：Cold 衰减段（灰虚线）
    pre_pts = " ".join(f"{px(p[0]):.1f},{py(p[1]):.1f}" for p in before)
    parts.append(f'<polyline points="{pre_pts}" fill="none" stroke="#999" '
                 f'stroke-width="2.4" stroke-dasharray="7 5" stroke-opacity="0.9"/>')

    # 唤醒后：继承前缀 + 跳升 + 重建衰减（橙色实线）
    post_pts = " ".join(f"{px(p[0]):.1f},{py(p[1]):.1f}" for p in after)
    parts.append(f'<polyline points="{post_pts}" fill="none" stroke="#e07b39" '
                 f'stroke-width="2.4" stroke-linejoin="round"/>')

    # 唤醒点标注
    if verdict["jump"]:
        tj, sj = verdict["jump"][0], verdict["jump"][1]
        t_prev, s_prev = before[-1][0], before[-1][1]
        parts.append(f'<line x1="{px(tj):.1f}" y1="{py(min(s_prev, sj) - 20):.1f}" '
                     f'x2="{px(tj):.1f}" y2="{py(max(s_prev, sj) + 20):.1f}" '
                     f'stroke="#c0392b" stroke-width="1.6" stroke-dasharray="3 3"/>')
        parts.append(f'<circle cx="{px(tj):.1f}" cy="{py(sj):.1f}" r="5" fill="#c0392b"/>')
        parts.append(f'<circle cx="{px(t_prev):.1f}" cy="{py(s_prev):.1f}" r="4" fill="#555"/>')
        parts.append(f'<text x="{px(tj) + 8:.1f}" y="{py(sj) - 10:.1f}" font-size="11" fill="#c0392b">'
                     f'唤醒点（测试效应跳升 {s_prev:.2f}→{sj:.2f}）</text>')

    # 图例 + 结论
    lx = _ML + 8
    ly = _MT + plot_h - 96
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#999" '
                 f'stroke-width="2.4" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 4}" font-size="12" fill="#555">'
                 f'唤醒前：Cold 衰减段（{verdict["n_before"]} 个采样）</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 24}" x2="{lx + 26}" y2="{ly + 24}" '
                 f'stroke="#e07b39" stroke-width="2.4"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 28}" font-size="12" fill="#555">'
                 f'唤醒后：同前缀 + 跳升 + 重建衰减（{verdict["n_after"]} 个采样）</text>')
    status = "✔ 无缝衔接" if (verdict["prefix_ok"] and verdict["tail_decays"]) else "✘ 存在断层"
    color = "#1a8f4c" if verdict["prefix_ok"] else "#c0392b"
    parts.append(f'<text x="{lx}" y="{_SVG_H - 24}" font-size="13" font-weight="bold" fill="{color}">'
                 f'{status}：前缀逐位一致={verdict["prefix_ok"]} · 尾部继续衰减={verdict["tail_decays"]}</text>')
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def _render_single_svg(points: list, path: str, title: str, color: str) -> str:
    """单段曲线图（唤醒前 / 唤醒后各一张）。"""
    t0, t1 = min(p[0] for p in points), max(p[0] for p in points)
    smax = max(p[1] for p in points)
    plot_w, plot_h = _SVG_W - _ML - _MR, _SVG_H - _MT - _MB
    span = (t1 - t0) or 1.0

    def px(t): return _ML + (t - t0) / span * plot_w
    def py(s): return _MT + plot_h - (s / (smax * 1.15)) * plot_h

    pts = " ".join(f"{px(p[0]):.1f},{py(p[1]):.1f}" for p in points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'font-family="system-ui, sans-serif">',
        f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>',
        f'<text x="{_ML}" y="26" font-size="17" font-weight="bold" fill="#222">{title}</text>',
        f'<text x="{_ML}" y="44" font-size="12" fill="#888">'
        f'{len(points)} 个观测采样（recorded，非预测线）</text>',
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.6" '
        f'stroke-linejoin="round"/>',
        *[f'<circle cx="{px(p[0]):.1f}" cy="{py(p[1]):.1f}" r="3.6" fill="{color}"/>'
          for p in points],
        "</svg>",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def _render_fit_svg(before: list, after: list, verdict: dict,
                    pred_pre: list, pred_post: list, path: str,
                    dev_info: dict | None = None) -> str:
    """预测线 vs recorded 叠加图：同一张 SVG 直观看模型预测与实际观测。

    蓝实线 = 模型延续预测（未唤醒假想：从最后 Cold 采样状态外推，穿过衰减段
    后继续平滑衰减）；灰虚线 = recorded Cold 衰减段；橙虚 = 唤醒后新状态的模型
    预测；橙实线 = recorded 唤醒后重建段。唤醒时刻红标垂直间隙 = 唤醒点偏差
    （实测跳升 − 模型延续预测，正 = 测试效应）。
    """
    all_pts = before + after + pred_pre + pred_post
    t0, t1 = min(p[0] for p in all_pts), max(p[0] for p in all_pts)
    smax = max(p[1] for p in all_pts)
    plot_w, plot_h = _SVG_W - _ML - _MR, _SVG_H - _MT - _MB
    span = (t1 - t0) or 1.0

    def px(t): return _ML + (t - t0) / span * plot_w
    def py(s): return _MT + plot_h - (s / (smax * 1.15)) * plot_h

    def poly(pts, stroke, width, dash=None, opacity=1.0):
        p = " ".join(f"{px(x[0]):.1f},{py(x[1]):.1f}" for x in pts)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{p}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width}"{d} stroke-opacity="{opacity}"/>')

    if dev_info is None:
        dev_info = awakening_deviation(verdict, pred_pre)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'font-family="system-ui, sans-serif">',
        f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>',
        f'<text x="{_ML}" y="26" font-size="17" font-weight="bold" fill="#222">'
        f'模型预测 vs recorded：唤醒点偏差（recall 前后同一遗忘公式）</text>',
        f'<text x="{_ML}" y="44" font-size="12" fill="#888">'
        f'蓝线 = 模型延续预测（未唤醒假想）；灰/橙点 = 实际观测采样；垂直间隙 = 唤醒点偏差</text>',
        # 模型延续预测：从最后 Cold 采样状态外推（穿过衰减段，唤醒点后继续平滑衰减）
        poly(pred_pre, "#3a7bd5", 2.2, opacity=0.9),
        # 唤醒前 recorded：Cold 衰减段（灰虚线）
        poly(before, "#999", 2.4, dash="7 5", opacity=0.9),
        # 唤醒后模型预测：新状态（橙虚）——与重建段 recorded 同公式，重合
        poly(pred_post, "#e07b39", 1.8, dash="4 4", opacity=0.7),
        # 唤醒后 recorded：重建段（橙实线）
        poly(after, "#e07b39", 2.4),
        *[f'<circle cx="{px(p[0]):.1f}" cy="{py(p[1]):.1f}" r="3.4" fill="#777"/>'
          for p in before],
        *[f'<circle cx="{px(p[0]):.1f}" cy="{py(p[1]):.1f}" r="3.4" fill="#e07b39"/>'
          for p in after],
    ]
    # 唤醒点偏差标注：实测跳升 vs 模型延续预测的垂直间隙 + 类型预期参考
    if dev_info:
        tj, sj = dev_info["ts"], dev_info["actual"]
        pv = dev_info["predicted"]
        dev = dev_info["deviation"]
        x0 = px(tj)
        parts.append(f'<circle cx="{x0:.1f}" cy="{py(sj):.1f}" r="5" fill="#c0392b"/>')
        parts.append(f'<circle cx="{x0:.1f}" cy="{py(pv):.1f}" r="4" fill="#3a7bd5"/>')
        if "expected" in dev_info:
            # 两列垂直条并排：左红 = 实测偏差（真实 τ），右青 = 类型预期（模型 τ）
            exp = dev_info["expected"]
            pe = dev_info["expected_point"]
            xr, xt = x0 - 8, x0 + 8
            parts.append(f'<line x1="{xr:.1f}" y1="{py(sj):.1f}" x2="{xr:.1f}" '
                         f'y2="{py(pv):.1f}" stroke="#c0392b" stroke-width="2.6"/>')
            parts.append(f'<line x1="{xt:.1f}" y1="{py(sj):.1f}" x2="{xt:.1f}" '
                         f'y2="{py(pe):.1f}" stroke="#2a9d8f" stroke-width="2.6" '
                         f'stroke-dasharray="2 3"/>')
            # 底部基线：两个延续预测的连线（长度 = 信号幅度 dev − expected）
            parts.append(f'<line x1="{xr:.1f}" y1="{py(pv):.1f}" x2="{xt:.1f}" '
                         f'y2="{py(pe):.1f}" stroke="#999" stroke-width="1" '
                         f'stroke-dasharray="2 2" stroke-opacity="0.6"/>')
            # 底部端点：真实 τ 延续（蓝圆）与模型 τ 延续（青菱形）
            parts.append(f'<circle cx="{xr:.1f}" cy="{py(pv):.1f}" r="3.6" fill="#3a7bd5"/>')
            parts.append(f'<path d="M {xt:.1f} {py(pe) - 4:.1f} L {xt + 4:.1f} {py(pe):.1f} '
                         f'L {xt:.1f} {py(pe) + 4:.1f} L {xt - 4:.1f} {py(pe):.1f} Z" '
                         f'fill="none" stroke="#2a9d8f" stroke-width="1.8"/>')
            sig_text, sig_color = {
                "up":   ("信号: 唤醒比类型预期剧烈（τ↓ · 可塑性↑）", "#c0392b"),
                "down": ("信号: 唤醒比类型预期温和（τ↑ · 可塑性↓）", "#1a8f4c"),
                "flat": ("信号: 实测≈类型预期（持平，无学习信号）", "#888"),
            }[dev_info["signal"]]
            ytxt = py(min(pv, pe)) + 20
            parts.append(f'<text x="{xr:.1f}" y="{ytxt:.1f}" font-size="12" '
                         f'font-weight="bold" fill="#c0392b">实测 {dev:+.3f}</text>')
            parts.append(f'<text x="{xt + 10:.1f}" y="{ytxt:.1f}" font-size="12" '
                         f'fill="#2a9d8f">类型预期 {exp:+.3f}</text>')
            parts.append(f'<text x="{x0:.1f}" y="{ytxt + 18:.1f}" font-size="12" '
                         f'font-weight="bold" fill="{sig_color}" text-anchor="middle">'
                         f'{sig_text}</text>')
        else:
            lo, hi = min(sj, pv), max(sj, pv)
            parts.append(f'<line x1="{x0:.1f}" y1="{py(lo - 20):.1f}" '
                         f'x2="{x0:.1f}" y2="{py(hi + 20):.1f}" '
                         f'stroke="#c0392b" stroke-width="1.6" stroke-dasharray="3 3"/>')
            parts.append(f'<text x="{x0 + 8:.1f}" y="{py((sj + pv) / 2):.1f}" font-size="13" '
                         f'font-weight="bold" fill="#c0392b">偏差 {dev:+.3f}</text>')
            parts.append(f'<text x="{x0 + 8:.1f}" y="{py((sj + pv) / 2) + 16:.1f}" '
                         f'font-size="11" fill="#777">实测 {sj:.3f} vs 预测 {pv:.3f}</text>')
    # 图例 + 结论
    has_exp = bool(dev_info and "expected" in dev_info)
    lx, ly = _ML + 8, _MT + plot_h - (176 if has_exp else 128)
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" '
                 f'stroke="#3a7bd5" stroke-width="2.2"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 4}" font-size="12" fill="#555">'
                 f'模型预测（未唤醒延续）</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 24}" x2="{lx + 26}" y2="{ly + 24}" '
                 f'stroke="#999" stroke-width="2.4" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 28}" font-size="12" fill="#555">'
                 f'recorded：Cold 衰减段（{len(before)} 个采样）</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 48}" x2="{lx + 26}" y2="{ly + 48}" '
                 f'stroke="#e07b39" stroke-width="2.4"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 52}" font-size="12" fill="#555">'
                 f'recorded：唤醒后重建段（{len(after)} 个采样，含跳升）</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 72}" x2="{lx + 26}" y2="{ly + 72}" '
                 f'stroke="#e07b39" stroke-width="1.8" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 76}" font-size="12" fill="#555">'
                 f'模型预测（唤醒后新状态，与重建段同公式）</text>')
    if has_exp:
        parts.append(f'<line x1="{lx}" y1="{ly + 96}" x2="{lx + 26}" y2="{ly + 96}" '
                     f'stroke="#2a9d8f" stroke-width="2.6" stroke-dasharray="2 3"/>')
        parts.append(f'<text x="{lx + 32}" y="{ly + 100}" font-size="12" fill="#555">'
                     f'类型预期偏差（同一事件、同一状态，只把 τ 换成模型信念）</text>')
        parts.append(f'<path d="M {lx + 13} {ly + 116} L {lx + 17} {ly + 120} '
                     f'L {lx + 13} {ly + 124} L {lx + 9} {ly + 120} Z" '
                     f'fill="none" stroke="#2a9d8f" stroke-width="1.8"/>')
        parts.append(f'<text x="{lx + 32}" y="{ly + 124}" font-size="12" fill="#555">'
                     f'模型 τ 延续预测点（基线连线 = 信号幅度）</text>')
    status = "✔ 预测线穿过衰减段，偏差仅在唤醒点" if dev_info else "✔ 无缝衔接"
    color = "#1a8f4c" if (verdict["prefix_ok"] and verdict["tail_decays"]) else "#c0392b"
    if dev_info and has_exp:
        sig_label = {"up": "上调", "down": "下调", "flat": "持平"}[dev_info["signal"]]
        parts.append(f'<text x="{lx}" y="{_SVG_H - 24}" font-size="13" font-weight="bold" fill="{color}">'
                     f'{status} · 实测 {dev_info["deviation"]:+.3f} vs 类型预期 '
                     f'{dev_info["expected"]:+.3f} · 信号: {sig_label}</text>')
    elif dev_info:
        parts.append(f'<text x="{lx}" y="{_SVG_H - 24}" font-size="13" font-weight="bold" fill="{color}">'
                     f'{status} · 唤醒点偏差 {dev_info["deviation"]:+.3f}</text>')
    else:
        parts.append(f'<text x="{lx}" y="{_SVG_H - 24}" font-size="13" font-weight="bold" fill="{color}">'
                     f'{status}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def build_real_scenario(store_path: str = DEFAULT_REAL_STORE, age_days: float = 180.0):
    """真实持久化场景：加载真实决策记忆 → 驱动完整生命周期 → 持久化往返。

    真实持久化数据通常只有 1 条创建采样、无唤醒标记——直接扫描没有相邻采样对
    可验证。因此把真实记忆**驱动过完整生命周期**再验证：老化（注入时钟）→
    逐段观测采样 → sleep 压缩成 Cold（真实 merge_similar / extractive_summary）→
    记录 Cold 衰减段 → 唤醒一条 Cold（move 语义 + 全量继承）→ 记录重建段 →
    保存到**临时文件**并重新加载（真实 JSON 往返）→ 返回
    (reloaded, before, after, revived, stats)。

    before = 唤醒前 Cold 的 recorded（继承进唤醒记忆的前缀）
    after  = 往返加载后唤醒记忆的 recorded（继承前缀 + 唤醒跳升 + 尾部衰减）
    验证与合成场景同一套 verify_continuity；前缀逐位一致还顺带断言持久化往返
    无损（浮点经 JSON 往返精确保持）。**只读 store_path**：唯一写盘是临时文件。
    """
    clock = [0.0]
    agent = MemoryAgent(persist_path=store_path, now_fn=lambda: clock[0])
    total = len(agent.store.all())
    if total == 0:
        raise RuntimeError(f"{store_path} 里没有记忆")
    # 时钟起点在真实数据最后采样之后（保证单调，不倒退）
    clock[0] = max(
        (r[0] for m in agent.store.all() for r in m.history), default=time.time()
    ) + 1.0
    step = age_days * 24 * 3600 / 3
    for _ in range(3):                       # 老化 3 段 → 真实衰减段采样
        clock[0] += step
        for m in agent.store.all():
            agent._record_sample(m)
    sleep_report = agent.sleep()             # 久未访问的真实记忆聚类压缩成 Cold
    colds = agent.store.by_tier(Tier.COLD)
    if not colds:
        raise RuntimeError(
            f"老化 {age_days:.0f} 天后 sleep 未压缩出 Cold"
            f"（候选需闲置 > 2×τ 且检索 ≤ {agent.cfg.cold_max_access} 次）"
        )
    cold = colds[0]
    clock[0] += step
    agent._record_sample(cold)               # Cold 衰减段尾部采样
    before = [list(r) for r in cold.history]
    clock[0] += step / 3
    revived = agent.recall(cold.id[:6])      # move 语义：原 Cold 移除，唤醒即取代
    if revived is None:
        raise RuntimeError("唤醒失败（recall 返回 None）")
    for _ in range(2):                       # 重建段继续衰减（Warm 尾部）
        clock[0] += step / 3
        agent._record_sample(revived)

    # 持久化往返：重定向保存到临时文件（绝不触碰真实文件）→ 重新加载
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        agent.store.path = tmp
        agent.store.save()
        reloaded = MemoryAgent(persist_path=tmp, now_fn=lambda: clock[0])
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    revived2 = next((m for m in reloaded.store.all() if m.id == revived.id), None)
    if revived2 is None:
        raise RuntimeError("持久化往返后找不到唤醒记忆（save/load 异常）")
    after = [list(r) for r in revived2.history]
    stats = {
        "total": total,
        "age_days": age_days,
        "cold_compressed": sleep_report["cold_compressed"],
        "clusters": sleep_report["clusters"],
        "revived_preview": revived.content[:30],
    }
    return reloaded, before, after, revived2, stats


def _synthesize_multi_awakening() -> tuple[MemoryAgent, "object", list]:
    """合成一条多次唤醒记忆（真实库无可选对象时的回退）：模拟时钟驱动
    Cold↔Warm 往返多次，每条往返产生一条唤醒观测（滚 12 条），返回
    (agent, memory, 唤醒事件列表)。
    """
    clock = [0.0]
    agent = MemoryAgent(
        cfg=AgentConfig(
            tau_by_type={MemType.EPISODIC: 3 * 86400.0},
            true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
            tau_learning_rate=0.3,
            joint_awakening=True,
        ),
        now_fn=lambda: clock[0],
    )
    m = agent.store.add("我昨天去吃了火锅（合成：多次唤醒）", importance=0.3,
                        mtype=MemType.EPISODIC)
    m.access_count = 2
    m.last_access = clock[0]
    for _ in range(3):                       # 3 轮 × 2 次往返 = 6 条唤醒观测
        for _ in range(2):
            clock[0] += 1.1 * 3 * 86400
            m.demote_to_cold("火锅聚餐（已归档）")
            m = agent.recall(m.id[:6])
            agent._record_sample(m)
        agent.learn_tau(force=True)
    return agent, m, _awakening_events(m)


def _awakening_events(mem) -> list[dict]:
    """记忆的唤醒事件明细：{ts, dev, expected, ratio, mtype, dt, n_cold}。
    与 visualize._awakening_events 同语义（旧格式 4 元组无 expected 时 ratio
    为 None），供 --awakened 模式标注全部唤醒事件。
    """
    events: list[dict] = []
    for aw in mem.awakenings:
        if not aw or len(aw) < 3:
            continue
        ts, dev = float(aw[0]), float(aw[1])
        expected = float(aw[2]) if len(aw) >= 4 else None
        ratio = None
        if expected is not None and dev > 0 and expected > 0:
            ratio = round(dev / expected, 4)
        events.append({
            "ts": ts,
            "dev": round(dev, 4),
            "expected": expected,
            "ratio": ratio,
            "mtype": aw[3] if len(aw) >= 4 else mem.mtype.value,
            "dt": float(aw[4]) if len(aw) > 4 and aw[4] is not None else None,
            "n_cold": int(aw[5]) if len(aw) > 5 and aw[5] is not None else None,
        })
    return events


def build_awakened_scenario(store_path: str = DEFAULT_REAL_STORE) -> tuple:
    """--awakened 场景：从真实记忆库挑一条**多次唤醒**记忆（awakenings > 1），
    逐次标注全部唤醒事件。

    返回 (agent, mem, events, source)：
    - 真实库存在 awakenings 数 > 1 的记忆 → 直接用（source="real"）；
    - 否则合成一条（source="synthetic"，标注"回退"）。mem 的 history 提供
      强度轨迹，events 是唤醒事件明细（每条含实测 dev / 类型预期 expected /
      比值 / 埋藏时长 / 检索次数）。只读 store_path。
    """
    if os.path.exists(store_path):
        try:
            agent = MemoryAgent(persist_path=store_path)
            cands = [m for m in agent.store.all() if len(m.awakenings) > 1]
            if cands:
                # 选唤醒次数最多的一条（多事件最丰富，标注最有意义）
                mem = max(cands, key=lambda m: len(m.awakenings))
                return agent, mem, _awakening_events(mem), "real"
        except Exception as e:               # 文件损坏/格式异常 → 回退合成
            print(f"  (提示: {store_path} 读取失败（{e}），回退合成场景)")
    return _synthesize_multi_awakening() + ("synthetic",)


def _load_exported_events(path: str) -> list | None:
    """识别 --export-signals 导出的 JSON（顶层 `events` 列表）并返回事件明细。

    导出文件每事件带 memory_id / mtype / ts / dev / expected / ratio / dt /
    n_cold——与记忆库文件（顶层 meta + memories）结构不同，据此判别。非导出
    文件（记忆库 / 不存在 / 损坏 / 空 events）返回 None，调用方走记忆库分支。
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    evs = data.get("events")
    if not isinstance(evs, list) or not evs:
        return None
    if not all(isinstance(e, dict) and "memory_id" in e for e in evs):
        return None
    return evs


def build_awakened_scenario_from_export(path: str, events: list) -> tuple | None:
    """从导出 JSON 的 events 挑一条**多次唤醒**记忆做逐次标注。

    按 memory_id 分组 → 选唤醒事件最多的一条（须 > 1，多次唤醒才有逐次标注
    意义），事件按时刻排序。返回 (mem_id, mtype, events, source="export")；
    无可选对象（全部 ≤ 1 次）返回 None。只读导出文件。
    """
    grouped: dict[str, list] = {}
    for ev in events:
        grouped.setdefault(ev["memory_id"], []).append(ev)
    cands = {k: v for k, v in grouped.items() if len(v) > 1}
    if not cands:
        return None
    mem_id = max(cands, key=lambda k: len(cands[k]))
    evs = sorted(cands[mem_id], key=lambda e: e["ts"])
    return mem_id, evs[0]["mtype"], evs, "export"


def _render_awakened_svg(agent, mem, events: list, path: str) -> str:
    """多次唤醒记忆曲线：完整 history 轨迹 + **逐次标注全部唤醒事件**。

    每个唤醒事件一个标注组：◇ 菱形 = 唤醒点（该时刻实测强度），红条 = 实测
    跳升 dev，青条 = 类型预期 expected，右端信号徽章 = 比值 + 方向（红 ↓ =
    唤醒比类型预期剧烈 → τ 应下调；青 ↑ = 温和 → τ 应上调；灰 ✓ = 已校准）。
    事件从左到右 = 唤醒先后顺序（最后一次在最右）。
    """
    history = [list(r) for r in mem.history]
    t0, t1 = min(p[0] for p in history), max(p[0] for p in history)
    if not history:
        t0, t1 = 0.0, 1.0
    elif t1 <= t0:
        t1 = t0 + 1.0
    smax = max(p[1] for p in history)
    plot_w, plot_h = _SVG_W - _ML - _MR, _SVG_H - _MT - _MB
    span = (t1 - t0) or 1.0

    def px(t): return _ML + (t - t0) / span * plot_w
    def py(s): return _MT + plot_h - (s / (smax * 1.15)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'font-family="system-ui, sans-serif">',
        f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>',
        f'<text x="{_ML}" y="26" font-size="16" font-weight="bold" fill="#222">'
        f'多次唤醒记忆曲线 — {mem.id} [{mem.tier.value}] [{mem.mtype.value}] · '
        f'{len(events)} 次唤醒事件逐次标注</text>',
        f'<text x="{_ML}" y="44" font-size="12" fill="#888">'
        f'◇ = 唤醒点（实测强度）· 红条 = 实测跳升 dev · 青条 = 类型预期 expected · '
        f'徽章 = 比值 dev/expected + 信号方向（红↓τ应下调 / 青↑应上调 / 灰✓已校准）</text>',
    ]
    # 坐标轴
    parts.append(f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + plot_h}" stroke="#999"/>')
    parts.append(f'<line x1="{_ML}" y1="{_MT + plot_h}" x2="{_ML + plot_w}" y2="{_MT + plot_h}" stroke="#999"/>')
    parts.append(f'<text x="{_ML}" y="{_MT - 6}" font-size="11" fill="#777">强度</text>')
    parts.append(f'<text x="{_ML + plot_w - 40}" y="{_MT + plot_h + 20}" font-size="11" fill="#777">时间（相对）</text>')

    # 强度轨迹（recorded 采样折线）
    pts = " ".join(f"{px(p[0]):.1f},{py(p[1]):.1f}" for p in history)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#3a7bd5" '
                 f'stroke-width="2.2" stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{px(history[0][0]):.1f}" cy="{py(history[0][1]):.1f}" '
                 f'r="3.6" fill="#3a7bd5"/>')
    parts.append(f'<circle cx="{px(history[-1][0]):.1f}" cy="{py(history[-1][1]):.1f}" '
                 f'r="3.6" fill="#3a7bd5"/>')

    # 逐次标注全部唤醒事件
    for i, ev in enumerate(events, 1):
        ts = ev["ts"]
        if not (t0 <= ts <= t1):
            continue
        s_act = strength_at(mem.history, ts)
        x0 = px(ts)
        y_act = py(s_act)
        # 唤醒点竖参考线 + 菱形
        parts.append(f'<line x1="{x0:.1f}" y1="{_MT}" x2="{x0:.1f}" y2="{_MT + plot_h}" '
                     f'stroke="#c0392b" stroke-width="1.2" stroke-dasharray="3 3" '
                     f'stroke-opacity="0.5"/>')
        parts.append(f'<path d="M{x0:.1f} {y_act:.1f} l5 -5 l5 5 l-5 5 z" fill="#c0392b" '
                     f'stroke="#fff" stroke-width="1"><title>唤醒 #{i} t={ts:.0f} '
                     f'dev {ev["dev"]} vs 预期 {ev["expected"]}</title></path>')
        # 红条 = 实测跳升 dev，青条 = 类型预期 expected（都结束于实测点高度）
        if ev["expected"] is not None:
            y_dev = py(max(0.2, s_act - ev["dev"]))
            y_exp = py(max(0.2, s_act - ev["expected"]))
            xr, xt = x0 - 9, x0 + 9
            parts.append(f'<line x1="{xr:.1f}" y1="{y_act:.1f}" x2="{xr:.1f}" y2="{y_dev:.1f}" '
                         f'stroke="#e34a2f" stroke-width="2.6"/>')
            parts.append(f'<line x1="{xt:.1f}" y1="{y_act:.1f}" x2="{xt:.1f}" y2="{y_exp:.1f}" '
                         f'stroke="#2a9d8f" stroke-width="2.6"/>')
            # 信号徽章：比值 + 方向
            ratio = ev["ratio"]
            if ratio is not None:
                if ratio > 1.05:
                    bcolor, mark, hint = "#e34a2f", "↓", "唤醒比类型预期剧烈 → τ 应下调"
                elif ratio < 0.95:
                    bcolor, mark, hint = "#2a9d8f", "↑", "唤醒比类型预期温和 → τ 应上调"
                else:
                    bcolor, mark, hint = "#95a5a6", "✓", "已校准（信号衰减收敛）"
            else:
                bcolor, mark, hint = "#95a5a6", "—", "旧格式事件（无类型预期）"
            label = f"#{i} {mark} {ratio if ratio is not None else '—'}"
            bw = 34 + len(str(label)) * 5.4
            bx = min(x0 + 24, _SVG_W - _MR - bw)
            by = max(_MT + 6, y_act - 34)
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="14" '
                         f'rx="3" fill="{bcolor}" opacity="0.92">'
                         f'<title>#{i} 比值 {ratio} — {hint}</title></rect>')
            parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by + 11:.1f}" font-size="9" '
                         f'fill="#fff" text-anchor="middle">{label}</text>')
        else:
            # 旧格式 4 元组：只标实测跳升
            parts.append(f'<text x="{x0 + 8:.1f}" y="{y_act - 8:.1f}" font-size="10" '
                         f'fill="#c0392b">#{i} dev {ev["dev"]:+.3f}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def _render_awakened_events_svg(mem_id: str, mtype: str, events: list,
                                path: str) -> str:
    """导出 JSON 事件标注图：无强度轨迹，改为**事件时间线**。

    x 轴 = 距首个唤醒事件的相对时间（天），y 轴 = 跳升强度幅度；每个事件：
    红色虚线竖参考线 + 红条（实测跳升 dev）/ 青条（类型预期 expected）从基线
    升起（高度 ∝ 幅度，端点带数值标签）+ 顶部菱形 + 信号徽章（比值 + 方向，
    红↓τ应下调 / 青↑应上调 / 灰✓已校准）。事件从左到右 = 唤醒先后。
    """
    t0, t1 = events[0]["ts"], events[-1]["ts"]
    span = (t1 - t0) or 1.0
    ymax = max(max(ev["dev"], ev["expected"] or 0.0) for ev in events) * 1.15 or 1.0
    plot_w, plot_h = _SVG_W - _ML - _MR, _SVG_H - _MT - _MB

    def px(t): return _ML + (t - t0) / span * plot_w
    def py(v): return _MT + plot_h - (v / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'font-family="system-ui, sans-serif">',
        f'<rect x="0" y="0" width="{_SVG_W}" height="{_SVG_H}" fill="#ffffff"/>',
        f'<text x="{_ML}" y="26" font-size="16" font-weight="bold" fill="#222">'
        f'导出 JSON 唤醒事件时间线 — {mem_id} [{mtype}] · '
        f'{len(events)} 次唤醒事件逐次标注</text>',
        f'<text x="{_ML}" y="44" font-size="12" fill="#888">'
        f'红条 = 实测跳升 dev · 青条 = 类型预期 expected（高度 ∝ 幅度）· '
        f'徽章 = 比值 dev/expected + 信号方向（红↓τ应下调 / 青↑应上调 / 灰✓已校准）</text>',
    ]
    # 坐标轴 + 基线
    parts.append(f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + plot_h}" stroke="#999"/>')
    parts.append(f'<line x1="{_ML}" y1="{_MT + plot_h}" x2="{_ML + plot_w}" y2="{_MT + plot_h}" stroke="#999"/>')
    parts.append(f'<text x="{_ML}" y="{_MT - 6}" font-size="11" fill="#777">跳升强度</text>')
    parts.append(f'<text x="{_ML + plot_w - 40}" y="{_MT + plot_h + 20}" font-size="11" fill="#777">距首事件时间（天）</text>')

    for i, ev in enumerate(events, 1):
        x0 = px(ev["ts"])
        y_base = py(0)
        y_dev = py(ev["dev"])
        y_exp = py(ev["expected"] or 0.0)
        y_top = min(y_dev, y_exp)
        # 竖参考线 + 顶部菱形（两条中较高者）
        parts.append(f'<line x1="{x0:.1f}" y1="{_MT}" x2="{x0:.1f}" y2="{_MT + plot_h}" '
                     f'stroke="#c0392b" stroke-width="1.2" stroke-dasharray="3 3" '
                     f'stroke-opacity="0.5"/>')
        parts.append(f'<path d="M{x0:.1f} {y_top:.1f} l5 -5 l5 5 l-5 5 z" fill="#c0392b" '
                     f'stroke="#fff" stroke-width="1"><title>唤醒 #{i} 相对 {fmt_ts(ev["ts"] - t0)} '
                     f'dev {ev["dev"]} vs 预期 {ev["expected"]}</title></path>')
        # 红条 = 实测 dev（左）、青条 = 类型预期 expected（右），端点带数值
        parts.append(f'<line x1="{x0 - 9:.1f}" y1="{y_base:.1f}" x2="{x0 - 9:.1f}" y2="{y_dev:.1f}" '
                     f'stroke="#e34a2f" stroke-width="2.6"/>')
        parts.append(f'<line x1="{x0 + 9:.1f}" y1="{y_base:.1f}" x2="{x0 + 9:.1f}" y2="{y_exp:.1f}" '
                     f'stroke="#2a9d8f" stroke-width="2.6"/>')
        parts.append(f'<text x="{x0 - 9:.1f}" y="{y_dev - 4:.1f}" font-size="8.5" '
                     f'fill="#e34a2f" text-anchor="middle">{ev["dev"]:.2f}</text>')
        parts.append(f'<text x="{x0 + 9:.1f}" y="{y_exp - 4:.1f}" font-size="8.5" '
                     f'fill="#2a9d8f" text-anchor="middle">{ev["expected"]:.2f}</text>')
        # 信号徽章：比值 + 方向
        ratio = ev["ratio"]
        if ratio is not None:
            if ratio > 1.05:
                bcolor, mark, hint = "#e34a2f", "↓", "唤醒比类型预期剧烈 → τ 应下调"
            elif ratio < 0.95:
                bcolor, mark, hint = "#2a9d8f", "↑", "唤醒比类型预期温和 → τ 应上调"
            else:
                bcolor, mark, hint = "#95a5a6", "✓", "已校准（信号衰减收敛）"
        else:
            bcolor, mark, hint = "#95a5a6", "—", "旧格式事件（无类型预期）"
        label = f"#{i} {mark} {ratio if ratio is not None else '—'}"
        bw = 34 + len(str(label)) * 5.4
        bx = min(x0 + 24, _SVG_W - _MR - bw)
        by = max(_MT + 6, y_top - 34)
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="14" '
                     f'rx="3" fill="{bcolor}" opacity="0.92">'
                     f'<title>#{i} 比值 {ratio} — {hint}</title></rect>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by + 11:.1f}" font-size="9" '
                     f'fill="#fff" text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def run_awakened_check_from_export(path: str, events: list) -> bool:
    """--awakened 指向 --export-signals 导出的 JSON：直接从导出文件挑一条
    多次唤醒记忆做逐次标注（导出 → 验证闭环，无强度轨迹，标注事件时间线）。
    """
    print(f"== 导出 JSON 唤醒检查（--awakened {path}）==")
    sc = build_awakened_scenario_from_export(path, events)
    if sc is None:
        print("✘ 导出 events 中无多次唤醒记忆（全部 ≤ 1 次）→ 跳过"
              "（可直接用记忆库路径跑 --awakened）")
        return False
    mem_id, mtype, evs, _source = sc
    print(f"① 从导出 JSON 选中记忆 {mem_id} [{mtype}]，唤醒事件 {len(evs)} 条 > 1，逐次标注：")
    for i, ev in enumerate(evs, 1):
        line = f"   #{i} t≈{fmt_ts(ev['ts'])}: dev {ev['dev']:+.3f}"
        if ev["expected"] is not None:
            cmp = {1: ">", -1: "<", 0: "="}[_sign(ev["dev"] - ev["expected"])]
            sig = _signal_text(ev)
            line += f" vs 类型预期 {ev['expected']:+.3f}（{cmp}）比值 {ev['ratio']} → {sig}"
        if ev["dt"] is not None:
            line += f"（埋藏 {ev['dt'] / 86400:.1f} 天 / 检索 {ev['n_cold']} 次）"
        print(line)
    print("结论: ✔ 导出事件的 dev vs expected 双条 + 信号方向已逐次标注")
    f = _render_awakened_events_svg(mem_id, mtype, evs,
                                    "recall_curve_awakened_export.svg")
    print(f"产物: {f}（事件时间线：红条 dev / 青条 expected / 顶部菱形 + 信号徽章）")
    return True


def run_awakened_check(store_path: str = DEFAULT_REAL_STORE) -> bool:
    """--awakened 模式：真实记忆库挑多次唤醒记忆，逐次标注全部唤醒事件。

    路径若是 --export-signals 导出的 JSON（顶层 events）→ 直接走导出模式
    （从 JSON 挑多次唤醒记忆，事件时间线标注）；否则走记忆库模式。"""
    exported = _load_exported_events(store_path)
    if exported is not None:
        return run_awakened_check_from_export(store_path, exported)
    agent, mem, events, source = build_awakened_scenario(store_path)
    print(f"== 多次唤醒记忆检查（--awakened）==")
    if source == "real":
        print(f"① 从 {store_path} 选中「{(mem.summary or mem.content)[:40]}」"
              f"（id={mem.id} [{mem.tier.value}] [{mem.mtype.value}]，"
              f"awakenings={len(mem.awakenings)} 条 > 1）")
    else:
        print("① 真实库无可选多次唤醒记忆 → 合成一条（3 轮 × 2 次 Cold↔Warm 往返）")
    print(f"② 强度轨迹采样 {len(mem.history)} 条，唤醒事件 {len(events)} 条，逐次标注：")
    for i, ev in enumerate(events, 1):
        line = f"   #{i} t≈{fmt_ts(ev['ts'])}: dev {ev['dev']:+.3f}"
        if ev["expected"] is not None:
            cmp = {1: ">", -1: "<", 0: "="}[_sign(ev["dev"] - ev["expected"])]
            sig = _signal_text(ev)
            line += f" vs 类型预期 {ev['expected']:+.3f}（{cmp}）比值 {ev['ratio']} → {sig}"
        if ev["dt"] is not None:
            line += f"（埋藏 {ev['dt'] / 86400:.1f} 天 / 检索 {ev['n_cold']} 次）"
        print(line)
    print("结论: " + ("✔ 全部唤醒事件已逐次标注（含 dev vs expected 双条 + 信号方向）"
                      if events else "✘ 无唤醒事件可标注"))
    f = _render_awakened_svg(agent, mem, events, "recall_curve_awakened.svg")
    print(f"产物: {f}（完整轨迹 + 逐次唤醒标注）")
    return bool(events)


def fmt_ts(ts: float) -> str:
    """时间戳 → 人类可读（相对天数）。"""
    return f"{ts / 86400:.1f}天" if abs(ts) > 86400 else f"{ts:.0f}s"


def _sign(x: float) -> int:
    if x > _AWAKENING_EPS:
        return 1
    if x < -_AWAKENING_EPS:
        return -1
    return 0


def _signal_text(ev: dict) -> str:
    if ev["ratio"] is None:
        return "无信号（旧格式）"
    if ev["ratio"] > 1.05:
        return "唤醒比类型预期剧烈 → τ 应下调"
    if ev["ratio"] < 0.95:
        return "唤醒比类型预期温和 → τ 应上调"
    return "已校准（信号衰减收敛）"


def run_real_check(store_path: str = DEFAULT_REAL_STORE) -> bool:
    """真实持久化场景：加载真实记忆 → 生命周期 → 持久化往返 → 连续性验证。

    真实文件缺失时打印提示并返回 True（不阻断合成场景检查）。
    """
    if not os.path.exists(store_path):
        print(f"== 真实持久化场景（{store_path}）==")
        print("跳过：文件不存在（--real [路径] 可指定其他记忆文件）")
        return True
    try:
        reloaded, before, after, revived, stats = build_real_scenario(store_path)
    except RuntimeError as e:
        print(f"== 真实持久化场景（{store_path}）==")
        print(f"失败：{e}")
        return False
    verdict = verify_continuity(before, after)
    ok = verdict["prefix_ok"] and verdict["tail_decays"]
    # 模型预测线（真实场景无 true_tau → 与采样同用配置 τ，纯衰减段预测=recorded）
    n = len(before)
    tau = reloaded._true_tau_for(revived)
    pred_pre = predict_line(reloaded, revived.mtype, tuple(before[-1][2:5]),
                            before[0][0], after[-1][0], tau_override=tau)
    pred_post = predict_line(reloaded, revived.mtype, tuple(after[n][2:5]),
                             after[n][0], after[-1][0], tau_override=tau)
    dev_info = awakening_deviation(verdict, pred_pre, revived.awakenings)
    print(f"== 真实持久化场景（{store_path}，真实决策记忆 {stats['total']} 条）==")
    print(f"① 老化 {stats['age_days']:.0f} 天 + sleep 压缩："
          f"{stats['cold_compressed']} 条 → Cold（{stats['clusters']} 簇）")
    print(f"② 唤醒「{stats['revived_preview']}」：继承 Cold 采样 {verdict['n_before']} 条，"
          f"重建段 {verdict['n_after'] - verdict['n_before']} 条")
    print(f"③ 前缀逐位一致（真实数据经 JSON 持久化往返无损）: {verdict['prefix_ok']}")
    if verdict["jump"]:
        _tp, s_prev = before[-1][0], before[-1][1]
        tj, sj = verdict["jump"][0], verdict["jump"][1]
        print(f"④ 唤醒点 t={tj:.0f}: 强度 {s_prev:.3f} → {sj:.3f}（测试效应跳升）")
    print(f"⑤ 重建段尾部继续衰减: {verdict['tail_decays']}")
    if dev_info:
        print(_format_deviation_line(dev_info))
    print("结论: " + ("✔ 无缝衔接——真实记忆在持久化往返 + 唤醒链路下曲线连续"
                      if ok else "✘ 存在断层"))
    f = _render_overlay_svg(before, after, "recall_curve_real.svg", verdict)
    f_fit = _render_fit_svg(before, after, verdict, pred_pre, pred_post,
                            "recall_curve_real_fit.svg", dev_info)
    print(f"产物: {f}（叠加）/ {f_fit}（预测线 vs recorded，唤醒点偏差）")
    return ok


def main(argv: list[str] | None = None) -> int:
    from memagent.cli import enable_utf8

    enable_utf8()  # 子进程/管道下强制 UTF-8 输出，避免 Windows GBK 解码崩溃
    parser = argparse.ArgumentParser(
        description="唤醒链路曲线连续性验证（合成判别场景 + 可选真实持久化场景）",
    )
    parser.add_argument(
        "--real", nargs="?", const=DEFAULT_REAL_STORE, metavar="路径",
        help="额外运行真实持久化场景：从 memories_session.json 加载真实决策记忆，"
             "驱动完整生命周期（老化→sleep 压缩→唤醒）→ 临时持久化往返 → 验证曲线连续性",
    )
    parser.add_argument(
        "--awakened", nargs="?", const=DEFAULT_REAL_STORE, metavar="路径",
        help="多次唤醒记忆检查：从真实记忆库挑一条 awakenings > 1 的记忆，逐次标注"
             "全部唤醒事件（dev vs 类型预期双条 + 信号方向）——真实库无可选对象时合成一条；"
             "路径若是 --export-signals 导出的 JSON（顶层 events）则直接从导出文件挑多次唤醒记忆标注",
    )
    parser.add_argument(
        "--awakened-store", default=DEFAULT_REAL_STORE, metavar="路径",
        help="--awakened 的记忆库路径（默认 memories_session.json；--real 指定了路径则共用）",
    )
    args = parser.parse_args(argv)
    awakened_store = args.awakened_store
    if args.awakened is not None and args.awakened != DEFAULT_REAL_STORE:
        awakened_store = args.awakened            # --awakened 显式给了路径优先
    elif args.real is not None:
        awakened_store = args.real                # --real 指定了路径则共用同一库

    agent, before, after, cold = build_scenario()
    verdict = verify_continuity(before, after)

    f_before = _render_single_svg(before, "recall_curve_before.svg",
                                  "唤醒前：Cold 衰减段", "#999")
    f_after = _render_single_svg(after, "recall_curve_after.svg",
                                 "唤醒后：继承前缀 + 跳升 + 重建衰减", "#e07b39")
    f_overlay = _render_overlay_svg(before, after, "recall_curve_overlay.svg", verdict)

    # 模型预测线（与采样同 τ）：未唤醒延续（全时间轴）+ 唤醒后新状态（重建段）
    n = len(before)
    tau = agent._true_tau_for(cold)
    pred_pre = predict_line(agent, cold.mtype, tuple(before[-1][2:5]),
                            before[0][0], after[-1][0], tau_override=tau)
    pred_post = predict_line(agent, cold.mtype, tuple(after[n][2:5]),
                             after[n][0], after[-1][0], tau_override=tau)
    revived = next((m for m in agent.store.all() if m.awakenings), None)
    dev_info = awakening_deviation(verdict, pred_pre,
                                   revived.awakenings if revived else None)
    f_fit = _render_fit_svg(before, after, verdict, pred_pre, pred_post,
                            "recall_curve_fit.svg", dev_info)

    print("== 唤醒链路曲线连续性验证（合成判别场景）==")
    print(f"① 唤醒前 Cold 采样 {verdict['n_before']} 条（衰减段）")
    print(f"② 唤醒后采样 {verdict['n_after']} 条（继承 {verdict['n_before']} + 唤醒 1 + 尾部 2）")
    print(f"③ 前缀逐位一致（Cold 衰减段完整继承）: {verdict['prefix_ok']}")
    if verdict["jump"]:
        t_prev, s_prev = before[-1][0], before[-1][1]
        tj, sj = verdict["jump"][0], verdict["jump"][1]
        print(f"④ 唤醒点 t={tj:.0f}: 强度 {s_prev:.3f} → {sj:.3f}（测试效应跳升）")
    print(f"⑤ 重建段尾部继续衰减: {verdict['tail_decays']}")
    if dev_info:
        print(_format_deviation_line(dev_info))
    verdict_ok = verdict["prefix_ok"] and verdict["tail_decays"]
    conclusion = "✔ 无缝衔接——Cold 衰减段与 Warm 重建段是同一记忆的连续轨迹" if verdict_ok else "✘ 存在断层"
    print(f"结论: {conclusion}")
    print(f"产物: {f_before} / {f_after} / {f_overlay} / {f_fit}")

    ok = verdict["prefix_ok"] and verdict["tail_decays"]
    if args.real is not None:
        ok_real = run_real_check(args.real)
        ok = ok and ok_real
    if args.awakened is not None:
        ok_aw = run_awakened_check(awakened_store)
        ok = ok and ok_aw
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
