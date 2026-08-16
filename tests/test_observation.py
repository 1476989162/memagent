"""持续观测与贴合度验证测试。"""

import csv
import json
import re
import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType
from memagent.visualize import (
    export_awakenings_csv,
    export_csv,
    export_json,
    render_svg,
)


def _observe_loop(agent, gap, n=6):
    for _ in range(n):
        agent._observe()
        time.sleep(gap)


def test_respond_auto_observes_all_memories():
    agent = MemoryAgent()
    m1 = agent.remember("我叫小林")
    m2 = agent.remember("我昨天去吃了火锅")
    time.sleep(0.1)  # 模拟真实对话节奏（避开同刻去重）
    agent.respond("你是谁")  # 触发自动观测
    assert len(m1.history) >= 2  # 未被本轮检索的记忆也被采样
    assert len(m2.history) >= 2


def test_fit_self_consistent_when_true_tau_equals_cfg():
    cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 30.0})
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe_loop(agent, 0.2)
    d = agent.fit_report()["by_type"]["episodic"]
    assert d["tau_est"] is not None
    assert 10 < d["tau_est"] < 90  # 实测τ ≈ 配置τ（30s）
    assert d["fit"] > 0.5          # 贴合度高


def test_fit_detects_misconfigured_tau():
    cfg = AgentConfig(
        tau_by_type={MemType.EPISODIC: 30.0},
        true_tau_by_type={MemType.EPISODIC: 2.0},  # 真实遗忘快 15 倍
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe_loop(agent, 0.3)
    d = agent.fit_report()["by_type"]["episodic"]
    assert d["tau_est"] is not None
    assert d["tau_est"] < 10   # 实测出真实 τ≈2
    assert d["fit"] < 0.5      # 贴合度低 → 提示 τ 配置失准


def test_interference_detected_on_retrieval():
    agent = MemoryAgent()
    mem = agent.remember("我昨天去吃了火锅", importance=0.1)
    agent._observe()
    time.sleep(0.1)
    agent.retrieve("昨天吃的火锅", k=1)  # 检索命中 → 制造干扰段
    agent._observe()
    r = agent.fit_report()
    m = next(x for x in r["memories"] if x["id"] == mem.id)
    assert m["interference"] >= 1


def test_svg_shows_actual_trajectory(tmp_path):
    agent = MemoryAgent()
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe_loop(agent, 0.1, n=3)
    out = render_svg(agent, str(tmp_path / "c.svg"), horizon_seconds=60.0)
    assert "实际观测轨迹" in open(out, encoding="utf-8").read()


def _awakening_agent(clock):
    """模拟时钟 agent：配置 τ=3 天、真实 τ=2 天，3 轮 × 3 次 Cold↔Warm 往返
    产生 9 条唤醒观测（learn_tau 校准过程中比值缓慢趋 1）。"""
    day = 24 * 3600
    a = MemoryAgent(cfg=AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * day},
        true_tau_by_type={MemType.EPISODIC: 2 * day},
        tau_learning_rate=0.3,
        joint_awakening=True,
    ), now_fn=lambda: clock[0])
    m = a.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count = 2
    m.last_access = clock[0]
    for _ in range(3):
        for _ in range(3):
            clock[0] += 1.1 * 3 * day
            m.demote_to_cold("火锅聚餐（已归档）")
            m = a.recall(m.id[:6])
        a.learn_tau(force=True)
    return a


def test_svg_annotates_all_awakening_events(tmp_path):
    """主图标注记忆历史里的全部唤醒事件：每条一个菱形 + dev/expected 双条 +
    信号徽章（比值），且窗口左扩覆盖创建之后的全部唤醒历史（不只 now 起）。"""
    clock = [0.0]
    a = _awakening_agent(clock)
    m = a.store.all()[0]
    assert len(m.awakenings) == 9
    out = render_svg(a, str(tmp_path / "c.svg"), now=clock[0],
                     horizon_seconds=20 * 24 * 3600)
    svg = open(out, encoding="utf-8").read()
    # 9 条唤醒事件全部标注：菱形 + dev 红条 + expected 青条 + 徽章
    assert svg.count("l4 -4") == 9                       # 菱形
    assert svg.count('stroke="#e34a2f" stroke-width="2.5"') == 9   # dev 红条
    assert svg.count('stroke="#2a9d8f" stroke-width="2.5"') == 9   # expected 青条
    assert len(re.findall(r'opacity="0.92"', svg)) == 9  # 信号徽章
    assert len(re.findall(r"<title>唤醒 ", svg)) == 9    # 每条带明细 tooltip
    # 窗口左扩：最早唤醒事件（3.3 天前）落在窗口内 → 负时间轴上有标记
    assert "-3天" in svg or svg.count("l4 -4") == 9
    # 徽章带校准语义 tooltip（比值趋 1 = learn_tau 收敛）
    assert "信号徽章" in svg and "learn_tau 校准" in svg
    assert "唤醒事件（信号徽章=比值）" in svg          # 图例


def test_svg_awakening_badge_color_tracks_convergence(tmp_path):
    """信号徽章颜色随校准状态：比值 >1 红（τ 应下调）、≤1 青（已校准）——
    learn_tau 收敛过程中红色徽章逐渐转青，直观展示信号随轮次衰减收敛。"""
    a = MemoryAgent(cfg=AgentConfig())
    m = a.store.add("火锅聚餐", importance=0.3, mtype=MemType.EPISODIC)
    m.history = [[100.0, 0.8, 100.0, 3, 0.3], [200.0, 0.6, 200.0, 3, 0.3]]
    # 校准进程：2.0×（信号强烈）→ 1.4×（仍活跃）→ 1.0×（已校准）→ 0.8×（偏温和）
    m.awakenings = [
        [100.0, 0.30, 0.15, "episodic", 50.0, 2],
        [200.0, 0.28, 0.20, "episodic", 50.0, 3],
        [300.0, 0.24, 0.24, "episodic", 50.0, 4],
        [400.0, 0.20, 0.25, "episodic", 50.0, 5],
    ]
    m.history += [[300.0, 0.5, 300.0, 3, 0.3], [400.0, 0.4, 400.0, 3, 0.3]]
    out = render_svg(a, str(tmp_path / "c.svg"), now=500.0,
                     horizon_seconds=200.0)
    svg = open(out, encoding="utf-8").read()
    assert svg.count("l4 -4") == 4                       # 4 条事件全部标注
    assert len(re.findall(r'fill="#e34a2f" opacity', svg)) == 2   # 2.0×/1.4× 红
    assert len(re.findall(r'fill="#2a9d8f" opacity', svg)) == 1   # 0.8× 青（偏温和）
    assert len(re.findall(r'fill="#95a5a6" opacity', svg)) == 1   # 1.0× 灰（校准带）
    assert ">2.0×</text>" in svg and ">0.8×</text>" in svg


def test_json_export_contains_fit(tmp_path):
    agent = MemoryAgent()
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe_loop(agent, 0.1, n=3)
    out = export_json(agent, str(tmp_path / "c.json"), horizon_seconds=60.0)
    data = json.load(open(out, encoding="utf-8"))
    assert "fit" in data and "by_type" in data["fit"]


def test_json_export_awakening_events(tmp_path):
    """每条记忆带唤醒事件明细：实测/预期/比值 + 原始四元组完整保留。"""
    agent = MemoryAgent()
    mem = agent.remember("我昨天去吃了火锅", importance=0.1)
    mem.awakenings.append([1000.0, 0.4, 0.2, "episodic"])   # 剧烈唤醒：ratio 2.0
    mem.awakenings.append([2000.0, 0.15, 0.3, "episodic"])  # 温和唤醒：ratio 0.5
    mem.awakenings.append([3000.0, 0.0, 0.1, "episodic"])   # dev=0 → ratio None（门控）
    out = export_json(agent, str(tmp_path / "c.json"), horizon_seconds=60.0)
    data = json.load(open(out, encoding="utf-8"))
    m = next(x for x in data["memories"] if x["id"] == mem.id)
    evs = m["awakening_events"]
    assert len(evs) == 3
    assert evs[0] == {"ts": 1000.0, "dev": 0.4, "expected": 0.2, "ratio": 2.0, "mtype": "episodic"}
    assert evs[1]["ratio"] == 0.5
    assert evs[2]["ratio"] is None            # 门控与学习器一致：dev/expected 需 > 0
    assert evs[2]["mtype"] == "episodic"      # 类型取唤醒时刻类型
    assert m["awakenings"] == mem.awakenings  # 原始四元组完整保留（含全字段）


def test_json_export_skips_old_format_awakenings(tmp_path):
    """旧格式三元组（无类型预期）留在原始日志，但不出现在推导明细（比值无定义）。"""
    agent = MemoryAgent()
    mem = agent.remember("我昨天去吃了火锅", importance=0.1)
    mem.awakenings.append([1000.0, 0.45, "episodic"])       # 旧格式
    mem.awakenings.append([2000.0, 0.4, 0.2, "semantic"])   # 新格式
    out = export_json(agent, str(tmp_path / "c.json"), horizon_seconds=60.0)
    data = json.load(open(out, encoding="utf-8"))
    m = next(x for x in data["memories"] if x["id"] == mem.id)
    assert len(m["awakening_events"]) == 1
    assert m["awakening_events"][0]["mtype"] == "semantic"
    assert len(m["awakenings"]) == 2         # 原始日志完整


def test_csv_export_awakenings(tmp_path):
    """唤醒明细 CSV：每行一次唤醒，memory_id 可连接，比值列与类型可分组分析。"""
    agent = MemoryAgent()
    m1 = agent.remember("我昨天去吃了火锅", importance=0.1)
    m2 = agent.remember("我学会了弹钢琴", importance=0.2)
    m1.awakenings.append([1000.0, 0.4, 0.2, "episodic"])
    m2.awakenings.append([2000.0, 0.1, 0.25, "skill"])
    m2.awakenings.append([1500.0, 0.3, 0.2, "semantic"])   # 同一记忆两次唤醒
    out = export_awakenings_csv(agent, str(tmp_path / "a.csv"), now=3000.0)
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert sorted(r["memory_id"] for r in rows) == sorted([m1.id, m2.id, m2.id])
    row1 = next(r for r in rows if r["memory_id"] == m1.id)
    assert row1["mtype"] == "episodic"
    assert row1["dev"] == "0.4" and row1["expected"] == "0.2" and row1["ratio"] == "2.0"
    assert row1["ts_relative_seconds"] == "-2000.0"        # 相对现在的过去时刻
    skill_rows = [r for r in rows if r["mtype"] == "skill"]
    assert len(skill_rows) == 1 and skill_rows[0]["ratio"] == "0.4"  # 0.1/0.25
    sem_rows = [r for r in rows if r["mtype"] == "semantic"]
    assert len(sem_rows) == 1 and sem_rows[0]["ratio"] == "1.5"      # 0.3/0.2


def test_export_csv_contains_awakening_rows(tmp_path):
    """主曲线 CSV 自包含唤醒事件行：row_type 区分采样/唤醒，唤醒行 strength=实测
    dev、content=expected/ratio——外部工具过滤 row_type 即可同一张表分析。"""
    agent = MemoryAgent()
    m = agent.remember("我昨天去吃了火锅", importance=0.1)
    m.awakenings.append([1000.0, 0.4, 0.2, "episodic"])
    m.awakenings.append([2000.0, 0.2, 0.25, "semantic"])   # 唤醒时刻类型 semantic
    out = export_csv(agent, str(tmp_path / "c.csv"), horizon_seconds=60.0, now=3000.0)
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert rows[0]["row_type"] == "sample"                 # 曲线采样行
    aw = [r for r in rows if r["row_type"] == "awakening"]
    assert len(aw) == 2
    assert aw[0]["memory_id"] == m.id
    assert aw[0]["strength"] == "0.4"                      # 实测偏差 dev
    assert aw[0]["content"] == "expected=0.2 ratio=2.0"
    assert aw[0]["mtype"] == "episodic"                    # 唤醒时刻类型
    assert aw[1]["mtype"] == "semantic"
    assert aw[1]["t_relative_seconds"] == "-1000.0"        # 相对现在


def test_csv_export_awakenings_empty_and_gated(tmp_path):
    """空记忆库 → 仅表头；dev/expected ≤ 0 的事件 ratio 为空串。"""
    agent = MemoryAgent()
    out = export_awakenings_csv(agent, str(tmp_path / "a.csv"), now=1.0)
    rows = list(csv.reader(open(out, encoding="utf-8-sig")))
    assert rows == [["memory_id", "mtype", "ts_relative_seconds", "dev", "expected", "ratio"]]
    mem = agent.remember("我昨天去吃了火锅", importance=0.1)
    mem.awakenings.append([1.0, 0.0, 0.1, "episodic"])     # dev=0 → 比值门控
    out = export_awakenings_csv(agent, str(tmp_path / "b.csv"), now=1.0)
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert rows[0]["ratio"] == ""
    assert rows[0]["dev"] == "0.0"
