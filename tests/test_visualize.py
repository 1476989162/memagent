"""可视化测试：预测曲线单调衰减、SVG/CSV/JSON 导出。"""

import csv
import json
import time

from memagent import MemoryAgent
from memagent.visualize import export_csv, export_json, fmt_delta, render_svg, strength_series


def _agent() -> MemoryAgent:
    from memagent.agent import AgentConfig

    return MemoryAgent(cfg=AgentConfig(tau_seconds=30.0))


def test_series_monotonic_decay_after_last_access():
    agent = _agent()
    mem = agent.remember("一条低频记忆")
    now = time.time()
    series = strength_series(agent, mem, now, horizon=300.0)
    strengths = [s for _, s in series]
    # 从最后一次访问之后，强度应单调不增（指数衰减 + 下限钳制）
    assert all(strengths[i] >= strengths[i + 1] - 1e-9 for i in range(len(strengths) - 1))
    assert strengths[0] >= strengths[-1]


def test_render_svg_creates_file(tmp_path):
    agent = _agent()
    agent.remember("我叫小林，喜欢爬山")
    out = render_svg(agent, str(tmp_path / "curves.svg"), horizon_seconds=120.0)
    svg = open(out, encoding="utf-8").read()
    assert "<svg" in svg and "</svg>" in svg
    assert "polyline" in svg  # 预测曲线
    assert "circle" in svg    # 实际采样点


def test_export_csv_rows(tmp_path):
    agent = _agent()
    agent.remember("我叫小林")
    out = export_csv(agent, str(tmp_path / "curves.csv"), horizon_seconds=120.0)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200  # 默认 samples=200 个时间点
    assert rows[0]["content"] == "我叫小林"


def test_export_json_roundtrip(tmp_path):
    agent = _agent()
    mem = agent.remember("我叫小林")
    out = export_json(agent, str(tmp_path / "curves.json"), horizon_seconds=120.0)
    data = json.load(open(out, encoding="utf-8"))
    assert len(data["memories"]) == 1
    assert data["memories"][0]["id"] == mem.id
    assert len(data["memories"][0]["series"]) == 200


def test_fmt_delta():
    now = time.time()
    assert fmt_delta(now - 90, now) == "-2分钟"
    assert fmt_delta(now + 2 * 86400, now) == "+2.0天"
    assert fmt_delta(now - 30, now) == "-30秒"


def test_main_plot_overlays_type_reference_curves(tmp_path):
    """主图叠加 3 条类型参考曲线（配置τ的典型遗忘，类型色虚线）。"""
    agent = _agent()
    agent.remember("我叫小林，喜欢爬山")
    agent.remember("我昨天去吃了火锅")
    out = render_svg(agent, str(tmp_path / "curves.svg"), horizon_seconds=120.0)
    svg = open(out, encoding="utf-8").read()
    # 三种类型颜色各出现一次作为参考曲线（虚线 6 4）
    for color in ("#2f9e44", "#7048e8", "#f08c00"):
        assert f'stroke="{color}"' in svg
    assert svg.count('stroke-dasharray="6 4"') >= 3
    # 副标题提示参考曲线含义
    assert "典型遗忘参考" in svg


def test_reference_curve_follows_type_tau(tmp_path):
    """参考曲线坡度随类型 τ：技能参考曲线衰减比情景慢。"""
    from memagent.agent import AgentConfig
    from memagent.memory import MemType
    from memagent.visualize import _reference_series

    agent = MemoryAgent(cfg=AgentConfig(
        tau_seconds=30.0,
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
    ))
    now = time.time()
    t0, t1 = now - 20, now + 100
    sk = _reference_series(agent, MemType.SKILL, t0, t1, 50)
    ep = _reference_series(agent, MemType.EPISODIC, t0, t1, 50)
    # 起点相同（同一新记忆基线），但同样时间后技能衰减明显慢于情景
    assert abs(sk[0][1] - ep[0][1]) < 0.05
    mid_t = t0 + 10
    sk_mid = min(sk, key=lambda p: abs(p[0] - mid_t))[1]
    ep_mid = min(ep, key=lambda p: abs(p[0] - mid_t))[1]
    assert sk_mid > ep_mid + 0.1  # τ=90 的技能衰减显著慢于 τ=8 的情景
