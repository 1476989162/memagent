"""遗忘斜率测试：每τ强度变化率 + 触底时间对比、面板交互数据。"""

import json
import re
import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.interactive import render_interactive_html
from memagent.memory import MemType
from memagent.visualize import floor_verification, forgetting_slope, render_svg_by_type


def _agent() -> MemoryAgent:
    return MemoryAgent(cfg=AgentConfig(
        tau_seconds=30.0,
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
    ))


def test_slope_negative_and_floor_time_tracks_tau():
    agent = _agent()
    sk = agent.remember("我会弹钢琴", importance=0.1, mtype=MemType.SKILL)
    ep = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    now = time.time()
    s_sk = forgetting_slope(agent, sk, now)
    s_ep = forgetting_slope(agent, ep, now)
    # 每 τ 强度下降（≤0）
    assert s_sk["slope_per_tau"] <= 0
    assert s_ep["slope_per_tau"] <= 0
    # 触底时间区分度：情景（τ=8s）远快于技能（τ=90s）
    assert s_ep["time_to_floor"] is not None and s_sk["time_to_floor"] is not None
    assert s_ep["time_to_floor"] < s_sk["time_to_floor"] * 0.5
    # 刚写入未强化 = 典型 → ratio ≈ 1
    assert s_sk["ratio"] is not None and abs(s_sk["ratio"] - 1.0) < 0.2
    assert s_ep["ratio"] is not None and abs(s_ep["ratio"] - 1.0) < 0.2


def test_slope_compares_to_reference():
    agent = _agent()
    # 检索加固后的记忆基线更高 → 触底时间更长 → 比典型持久
    mem = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    agent.retrieve("昨天去吃了火锅", k=1)
    agent.retrieve("昨天去吃了火锅", k=1)
    s = forgetting_slope(agent, mem)
    # 检索加固抬高了 freq/importance 基线 → 比典型持久
    # （基线可能高到 30τ 内不触底 → ratio=None，label 为"持久（不触底）"）
    assert s["label"].startswith("持久")
    if s["ratio"] is not None:
        assert s["ratio"] > 1.0
    # 早已触底的记忆：触底时间 0、斜率≈0
    mem2 = agent.remember("一条零访问记忆", importance=0.05, mtype=MemType.EPISODIC)
    mem2.last_access = time.time() - 1000
    s2 = forgetting_slope(agent, mem2)
    assert s2["time_to_floor"] == 0.0
    assert abs(s2["slope_per_tau"]) < 0.05


def test_dashboard_embeds_slope_and_renders_row(tmp_path):
    agent = _agent()
    agent.remember("我昨天去吃了火锅", importance=0.1)
    out = render_interactive_html(agent, str(tmp_path / "dash.html"), horizon_seconds=120.0)
    html = open(out, encoding="utf-8").read()
    assert "遗忘斜率" in html                    # 详情面板行
    assert "fmtFloor" in html and "m.slope.time_to_floor" in html
    blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
    data = json.loads(blob)
    for m in data["memories"]:
        assert "slope" in m
        assert {"slope_per_tau", "time_to_floor", "ref_time_to_floor", "ratio", "label"} <= set(m["slope"])


def test_static_by_type_title_has_slope(tmp_path):
    agent = _agent()
    agent.remember("我昨天去吃了火锅", importance=0.1)
    out = render_svg_by_type(agent, str(tmp_path / "by_type.svg"), horizon_seconds=120.0)
    svg = open(out, encoding="utf-8").read()
    assert "遗忘斜率" in svg  # 曲线 title 带斜率（悬停可见）


def test_sub_curve_titles_have_slope(tmp_path):
    agent = _agent()
    agent.remember("我昨天去吃了火锅", importance=0.1)
    out = render_interactive_html(agent, str(tmp_path / "dash.html"), horizon_seconds=120.0)
    html = open(out, encoding="utf-8").read()
    # JS 渲染模板：子图曲线 title 带斜率（悬停可见）
    assert "遗忘斜率 每τ " in html
    assert "m.slope.slope_per_tau.toFixed(2)" in html


# ---------- 触底验证：实测触底时刻 vs 预测 ----------

def _clock_agent(true_episodic=None):
    clock = {"t": 1000.0}
    cfg = AgentConfig(
        tau_seconds=30.0,
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 30.0},
    )
    if true_episodic is not None:
        cfg.true_tau_by_type = {MemType.EPISODIC: true_episodic}
    return MemoryAgent(cfg=cfg, now_fn=lambda: clock["t"]), clock


def _observe_until_floor(agent, mem, clock, max_steps=300):
    for _ in range(max_steps):
        clock["t"] += 1.0
        agent._observe()
        if mem.history[-1][1] <= 0.2 + 1e-6:
            return True
    return False


def test_floor_verification_matches_prediction_without_true_tau():
    """无真实 τ 时观测=模型 → 实测触底与预测贴合（ratio≈1）。"""
    agent, clock = _clock_agent()
    mem = agent.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC)
    assert _observe_until_floor(agent, mem, clock)
    fc = floor_verification(agent, mem)
    assert fc["floored"] is True and fc["status"] == "verified"
    assert fc["actual_dt"] > 0 and fc["predicted_dt"] > 0
    assert abs(fc["ratio"] - 1.0) < 0.2
    assert "贴合" in fc["label"]


def test_floor_verification_true_tau_faster_than_prediction():
    """真实 τ=4s vs 模型 30s：实测触底远早于预测（ratio<0.6）。"""
    agent, clock = _clock_agent(true_episodic=4.0)
    mem = agent.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC)
    assert _observe_until_floor(agent, mem, clock)
    fc = floor_verification(agent, mem)
    assert fc["floored"] is True
    assert fc["predicted_dt"] > fc["actual_dt"]
    assert fc["ratio"] < 0.6
    assert "快" in fc["label"] and "预测" in fc["label"]


def test_floor_verification_not_floored_yet():
    """刚写入、远未触底 → 未实测（status=not_floored）。"""
    agent, clock = _clock_agent()
    mem = agent.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC)
    agent._observe()
    fc = floor_verification(agent, mem)
    assert fc["floored"] is False and fc["status"] == "not_floored"
    assert fc["actual_dt"] is None


def test_floor_verification_embedded_in_dashboard(tmp_path):
    """仪表盘数据嵌入 floor_check，详情面板有触底验证行。"""
    agent, clock = _clock_agent(true_episodic=4.0)
    mem = agent.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC)
    _observe_until_floor(agent, mem, clock)
    out = render_interactive_html(agent, str(tmp_path / "dash.html"), horizon_seconds=120.0)
    html = open(out, encoding="utf-8").read()
    assert "触底验证" in html and "fmtFloorCheck" in html
    blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
    data = json.loads(blob)
    fm = next(m for m in data["memories"] if m["id"] == mem.id)
    assert fm["floor_check"]["floored"] is True
    assert fm["floor_check"]["predicted_dt"] > fm["floor_check"]["actual_dt"]


def test_by_type_svg_title_has_floor_check(tmp_path):
    """静态 by_type SVG 的曲线 title 带触底验证结论。"""
    agent, clock = _clock_agent(true_episodic=4.0)
    mem = agent.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC)
    _observe_until_floor(agent, mem, clock)
    out = render_svg_by_type(agent, str(tmp_path / "by_type.svg"), horizon_seconds=120.0)
    svg = open(out, encoding="utf-8").read()
    assert "触底验证" in svg and "比预测快" in svg
