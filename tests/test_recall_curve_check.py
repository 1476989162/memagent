"""唤醒链路曲线连续性验证测试：recall_curve_check.py 的数据断言与 SVG 产物。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from memagent import MemoryAgent

import recall_curve_check
from recall_curve_check import build_scenario, verify_continuity

ROOT = Path(__file__).resolve().parents[1]
REAL_STORE = ROOT / "memories_session.json"


def test_continuity_seamless_for_real_recall():
    """真实唤醒链路：Cold 衰减段 → 唤醒跳升 → 重建衰减，前缀逐位一致（无缝）。"""
    _agent, before, after, _cold = build_scenario()
    v = verify_continuity(before, after)
    assert v["prefix_ok"] is True          # 继承前缀逐位一致
    assert v["tail_decays"] is True        # 重建段继续衰减
    assert v["n_before"] == 5 and v["n_after"] == 8
    # 唤醒点：末采样 0.260 → 唤醒采样 0.525（测试效应跳升），时间戳在后
    assert v["jump"] is not None
    assert v["jump"][0] > before[-1][0]
    assert v["jump"][1] > before[-1][1]


def test_continuity_detects_gap():
    """检查器有判别力：after 缺少继承前缀（模拟唤醒不继承历史）→ 报断层。"""
    _agent, before, _after, _cold = build_scenario()
    broken = [[p[0] + 1000, p[1]] for p in before] + [[99999.0, 0.5]]  # 前缀错位
    v = verify_continuity(before, broken)
    assert v["prefix_ok"] is False
    # 完全空历史（修复前行为：唤醒记忆只有 1 条采样）→ 无前缀可言
    v2 = verify_continuity(before, [[99999.0, 0.5]])
    assert v2["prefix_ok"] is False


def test_render_svgs_created(tmp_path):
    """三张 SVG 产物生成且含折线元素。"""
    _agent, before, after, _cold = build_scenario()
    v = verify_continuity(before, after)
    f1 = recall_curve_check._render_single_svg(before, str(tmp_path / "b.svg"), "t", "#999")
    f2 = recall_curve_check._render_single_svg(after, str(tmp_path / "a.svg"), "t", "#e07b39")
    f3 = recall_curve_check._render_overlay_svg(before, after, str(tmp_path / "o.svg"), v)
    for f in (f1, f2, f3):
        assert os.path.exists(f)
        assert "<polyline" in open(f, encoding="utf-8").read()
    assert "无缝衔接" in open(f3, encoding="utf-8").read()


def test_prediction_line_and_awakening_deviation():
    """模型预测 vs recorded：纯衰减段预测线穿过实测采样（同公式同状态 → 逐位一致），
    唤醒点偏差为正（测试效应：模型按旧状态延续，预测不到唤醒刷新）。"""
    agent, before, after, cold = build_scenario()
    verdict = verify_continuity(before, after)
    n = len(before)
    tau = agent._true_tau_for(cold)
    pred_pre = recall_curve_check.predict_line(
        agent, cold.mtype, tuple(before[-1][2:5]), before[0][0], after[-1][0],
        tau_override=tau,
    )
    # 纯衰减段：在最后 Cold 采样时刻的预测强度 == 实测强度
    # （recorded 把强度 round 到 4 位，预测是未取整值——按 4 位对齐后逐位一致）
    p_at_last = recall_curve_check.predict_line(
        agent, cold.mtype, tuple(before[-1][2:5]),
        before[-1][0], before[-1][0], tau_override=tau)[0]
    assert round(p_at_last[1], 4) == before[-1][1]
    # 唤醒点偏差：实测跳升 > 模型延续预测（正 = 测试效应）
    dev = recall_curve_check.awakening_deviation(verdict, pred_pre)
    assert dev is not None
    assert dev["actual"] == verdict["jump"][1]
    assert dev["deviation"] > 0
    # 唤醒后新状态预测与重建段 recorded 同公式 → 尾部末采样 4 位对齐一致
    p_tail = recall_curve_check.predict_line(
        agent, cold.mtype, tuple(after[n][2:5]),
        after[-1][0], after[-1][0], tau_override=tau)[0]
    assert round(p_tail[1], 4) == after[-1][1]


def test_render_fit_svg_created(tmp_path):
    """预测线 vs recorded 叠加图：预测线（蓝）+ 唤醒点双条偏差标注——
    红=实测偏差、青=类型预期偏差（基线连线=信号幅度）+ 信号方向。"""
    agent, before, after, cold = build_scenario()
    verdict = verify_continuity(before, after)
    n = len(before)
    tau = agent._true_tau_for(cold)
    pred_pre = recall_curve_check.predict_line(
        agent, cold.mtype, tuple(before[-1][2:5]), before[0][0], after[-1][0],
        tau_override=tau,
    )
    pred_post = recall_curve_check.predict_line(
        agent, cold.mtype, tuple(after[n][2:5]), after[n][0], after[-1][0],
        tau_override=tau,
    )
    revived = next((m for m in agent.store.all() if m.awakenings), None)
    assert revived is not None
    dev = recall_curve_check.awakening_deviation(
        verdict, pred_pre, revived.awakenings,
    )
    assert "expected" in dev and dev["expected"] > 0
    assert dev["expected_point"] == dev["actual"] - dev["expected"]
    assert dev["signal"] == "up"   # 真实 τ(20000s) << 模型 τ(3 天) → 实测 >> 类型预期
    f = recall_curve_check._render_fit_svg(
        before, after, verdict, pred_pre, pred_post, str(tmp_path / "fit.svg"), dev,
    )
    text = open(f, encoding="utf-8").read()
    assert "<polyline" in text
    assert "#3a7bd5" in text      # 模型延续预测（蓝实线）
    assert "#2a9d8f" in text      # 类型预期条（青虚线）与菱形端点
    assert "实测 +" in text and "类型预期 +" in text
    assert "基线连线 = 信号幅度" in text
    assert "信号: 唤醒比类型预期剧烈（τ↓ · 可塑性↑）" in text


def test_awakening_deviation_signal_direction():
    """信号方向判别：dev > expected → up；dev < expected → down；相等 → flat。"""
    verdict = {"jump": [100.0, 0.5]}
    pred_pre = [[99.0, 0.375], [100.0, 0.375], [101.0, 0.375]]  # dev = 0.125 精确
    base = recall_curve_check.awakening_deviation(verdict, pred_pre)
    assert "expected" not in base          # 无唤醒观测 → 只有基本偏差
    up = recall_curve_check.awakening_deviation(
        verdict, pred_pre, [[100.0, 0.08, 0.04, "episodic"]])
    assert up["signal"] == "up" and up["expected"] == 0.04
    down = recall_curve_check.awakening_deviation(
        verdict, pred_pre, [[100.0, 0.03, 0.15, "episodic"]])
    assert down["signal"] == "down"
    flat = recall_curve_check.awakening_deviation(
        verdict, pred_pre, [[100.0, 0.125, 0.125, "episodic"]])
    assert flat["signal"] == "flat"
    # 浮点尾差不误判：dev 与 expected 差 1e-7（容差 1e-6 内）→ 持平
    near = recall_curve_check.awakening_deviation(
        verdict, pred_pre, [[100.0, 0.125, 0.1249999, "episodic"]])
    assert near["signal"] == "flat"
    # 时间戳不匹配（唤醒行缺失）→ 回退基本偏差
    miss = recall_curve_check.awakening_deviation(
        verdict, pred_pre, [[50.0, 0.08, 0.04, "episodic"]])
    assert "expected" not in miss


def test_cli_exits_zero_and_writes_artifacts(tmp_path):
    """CLI 可运行：退出码 0、结论打印、产物写在工作目录（不污染项目根）。"""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "recall_curve_check.py")],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(tmp_path), timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "无缝衔接" in proc.stdout and "前缀逐位一致" in proc.stdout
    assert "唤醒点偏差" in proc.stdout and "+0." in proc.stdout
    assert "类型预期 +" in proc.stdout and "信号: 上调" in proc.stdout
    for name in ("recall_curve_before.svg", "recall_curve_after.svg",
                 "recall_curve_overlay.svg", "recall_curve_fit.svg"):
        assert os.path.exists(tmp_path / name)
    assert os.path.exists(tmp_path / "recall_curve_fit.svg")
    fit = open(tmp_path / "recall_curve_fit.svg", encoding="utf-8").read()
    assert "类型预期" in fit and "#2a9d8f" in fit and "信号: 上调" in fit
    assert not os.path.exists(ROOT / "recall_curve_before.svg")  # 未污染项目根


def test_real_scenario_continuity():
    """真实持久化场景：真实决策记忆 → 老化 → sleep 压缩 → 唤醒 → 临时持久化往返，
    曲线无缝衔接（前缀逐位一致 + 唤醒跳升 + 尾部衰减），唤醒标记与继承轨迹在往返后保留。"""
    if not REAL_STORE.exists():
        pytest.skip("缺少 memories_session.json（真实记忆库）")
    reloaded, before, after, revived, stats = recall_curve_check.build_real_scenario(
        str(REAL_STORE)
    )
    v = verify_continuity(before, after)
    assert v["prefix_ok"] is True
    assert v["tail_decays"] is True
    assert v["n_after"] == v["n_before"] + 3      # 继承 N + 唤醒 1 + 尾部 2
    assert v["jump"] is not None
    assert v["jump"][1] > before[-1][1]            # 唤醒点确有测试效应跳升
    assert stats["total"] >= 1 and stats["cold_compressed"] >= 1
    assert revived.awakened_at is not None          # 往返后复活标记保留
    assert revived.tier.value == "warm"


def test_real_scenario_reads_store_but_never_mutates():
    """只读保证：真实持久化文件在场景运行前后逐字节一致（写盘只走临时文件）。"""
    if not REAL_STORE.exists():
        pytest.skip("缺少 memories_session.json（真实记忆库）")
    before_bytes = REAL_STORE.read_bytes()
    recall_curve_check.build_real_scenario(str(REAL_STORE))
    assert REAL_STORE.read_bytes() == before_bytes


# ---------- --awakened 多次唤醒记忆检查 ----------


def _multi_awakening_store(tmp_path, n_rounds: int = 3) -> str:
    """建一条多次唤醒记忆的临时持久化库（3 轮 × 2 次 Cold↔Warm 往返 = 6 条唤醒）。"""
    from memagent.agent import AgentConfig
    from memagent.memory import MemType

    day = 86400.0
    store = str(tmp_path / "multi_awakenings.json")
    clock = [0.0]
    a = MemoryAgent(cfg=AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * day},
        true_tau_by_type={MemType.EPISODIC: 2 * day},
        tau_learning_rate=0.3,
        joint_awakening=True,
    ), now_fn=lambda: clock[0])
    m = a.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count = 2
    m.last_access = clock[0]
    for _ in range(n_rounds):
        for _ in range(2):
            clock[0] += 1.1 * 3 * day
            m.demote_to_cold("火锅聚餐（已归档）")
            m = a.recall(m.id[:6])
            a._record_sample(m)
        a.learn_tau(force=True)
    a.store.path = store
    a.store.save()
    return store


def test_awakened_scenario_real_store(tmp_path):
    """真实库存在多次唤醒记忆 → 直接选中（source=real），事件明细完整
    （dev/expected/比值/埋藏时长/检索次数）。"""
    store = _multi_awakening_store(tmp_path)
    agent, mem, events, source = recall_curve_check.build_awakened_scenario(store)
    assert source == "real"
    assert len(mem.awakenings) == 6 and len(events) == 6
    ev = events[0]
    assert {"ts", "dev", "expected", "ratio", "mtype", "dt", "n_cold"} <= set(ev)
    assert ev["ratio"] > 1.0          # 真实 τ(2天) < 模型 τ(3天) → 埋得更深
    assert ev["dt"] > 0 and ev["n_cold"] >= 2
    assert ev["mtype"] == "episodic"


def test_awakened_scenario_synthetic_fallback(tmp_path):
    """真实库无可选对象（文件不存在）→ 合成一条多次唤醒记忆（source=synthetic）。"""
    agent, mem, events, source = recall_curve_check.build_awakened_scenario(
        str(tmp_path / "nonexistent.json")
    )
    assert source == "synthetic"
    assert len(mem.awakenings) > 1 and len(events) > 1
    assert all(e["ratio"] is not None for e in events)   # 6 元组观测都有类型预期


def test_awakened_svg_annotates_all_events(tmp_path):
    """SVG 逐次标注全部唤醒事件：每个事件一个菱形 + dev 红条 + expected 青条 +
    信号徽章（比值+方向），且徽章数 == 事件数。"""
    agent, mem, events, _ = recall_curve_check.build_awakened_scenario(
        str(tmp_path / "nonexistent.json")
    )
    f = recall_curve_check._render_awakened_svg(
        agent, mem, events, str(tmp_path / "awakened.svg")
    )
    text = open(f, encoding="utf-8").read()
    assert text.count("l5 -5 l5 5") == len(events)               # 菱形
    assert text.count('stroke="#e34a2f" stroke-width="2.6"') == len(events)  # dev 红条
    assert text.count('stroke="#2a9d8f" stroke-width="2.6"') == len(events)  # expected 青条
    assert text.count('rx="3" fill=') == len(events)             # 信号徽章
    assert "多次唤醒记忆曲线" in text
    assert "逐次标注" in text
    assert "#1" in text and f"#{len(events)}" in text            # 事件序号
    assert "τ 应下调" in text                                     # 方向语义（比值>1）


def test_awakened_svg_old_format_fallback(tmp_path):
    """旧格式 4 元组唤醒（无类型预期）→ 不崩溃，只标实测跳升（无徽章）。"""
    agent, mem, events, _ = recall_curve_check.build_awakened_scenario(
        str(tmp_path / "nonexistent.json")
    )
    events2 = [dict(e, expected=None, ratio=None) for e in events]  # 模拟旧格式
    f = recall_curve_check._render_awakened_svg(
        agent, mem, events2, str(tmp_path / "old.svg")
    )
    text = open(f, encoding="utf-8").read()
    assert text.count("l5 -5 l5 5") == len(events2)              # 菱形仍在
    assert text.count('rx="3" fill=') == 0                        # 无信号徽章
    assert "旧格式事件（无类型预期）" in text or "dev +" in text


# ---------- --awakened 指向 --export-signals 导出的 JSON ----------


def _export_json(path, events: list) -> None:
    """写一份 --export-signals 形状的导出 JSON（顶层 now/recent_seconds/events）。"""
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"now": 40 * 86400, "recent_seconds": 30 * 86400,
                   "stats": {}, "periods": {}, "events": events}, f,
                  ensure_ascii=False, indent=2)


def test_load_exported_events_detection(tmp_path):
    """判别：导出 JSON（顶层 events 列表）→ 返回事件；记忆库形状 / 不存在 → None。"""
    ev = [{"memory_id": "m1", "mtype": "episodic", "ts": 1.0,
           "dev": 0.4, "expected": 0.3, "ratio": 1.33, "dt": None,
           "n_cold": None, "ts_relative_seconds": -1.0}]
    _export_json(str(tmp_path / "sig.json"), ev)
    got = recall_curve_check._load_exported_events(str(tmp_path / "sig.json"))
    assert got == ev
    # 记忆库形状（meta + memories）→ None，不误判
    store = str(tmp_path / "store.json")
    with open(store, "w", encoding="utf-8") as f:
        import json
        json.dump({"meta": {}, "memories": [{"id": "x1", "awakenings": []}]}, f)
    assert recall_curve_check._load_exported_events(store) is None
    # 损坏 / 不存在 / 空 events → None
    bad = str(tmp_path / "bad.json")
    open(bad, "w", encoding="utf-8").write("{ not json")
    assert recall_curve_check._load_exported_events(bad) is None
    assert recall_curve_check._load_exported_events(
        str(tmp_path / "missing.json")) is None


def test_build_awakened_scenario_from_export(tmp_path):
    """从导出 JSON 挑**多次唤醒**记忆：按 memory_id 分组、选事件最多的一条、
    按时刻排序；无多次唤醒（全 ≤ 1）→ None。"""
    evs = [
        {"memory_id": "A", "mtype": "episodic", "ts": 35 * 86400,
         "dev": 0.42, "expected": 0.40, "ratio": 1.05, "dt": None, "n_cold": None},
        {"memory_id": "A", "mtype": "episodic", "ts": 5 * 86400,
         "dev": 0.45, "expected": 0.38, "ratio": 1.18, "dt": None, "n_cold": None},
        {"memory_id": "A", "mtype": "episodic", "ts": 20 * 86400,
         "dev": 0.43, "expected": 0.39, "ratio": 1.10, "dt": None, "n_cold": None},
        {"memory_id": "B", "mtype": "semantic", "ts": 8 * 86400,
         "dev": 0.20, "expected": 0.25, "ratio": 0.80, "dt": None, "n_cold": None},
    ]
    path = str(tmp_path / "sig.json")
    _export_json(path, evs)
    sc = recall_curve_check.build_awakened_scenario_from_export(
        path, recall_curve_check._load_exported_events(path))
    assert sc is not None
    mem_id, mtype, picked, source = sc
    assert mem_id == "A" and mtype == "episodic" and source == "export"
    assert [e["ts"] for e in picked] == sorted(e["ts"] for e in picked)  # 按时刻排序
    assert [e["ts"] for e in picked] == [5 * 86400, 20 * 86400, 35 * 86400]
    # 全部 ≤ 1 次 → None
    _export_json(str(tmp_path / "single.json"), evs[:1] + evs[3:])
    p2 = str(tmp_path / "single.json")
    assert recall_curve_check.build_awakened_scenario_from_export(
        p2, recall_curve_check._load_exported_events(p2)) is None


def test_awakened_from_export_json(tmp_path, capsys, monkeypatch):
    """导出 → 验证闭环：run_awakened_check 识别导出 JSON、选中多次唤醒记忆、
    逐次打印（含比值收敛方向）、渲染事件时间线 SVG。"""
    monkeypatch.chdir(tmp_path)   # 产物写当前工作目录
    evs = [
        {"memory_id": "A", "mtype": "episodic", "ts": 5 * 86400,
         "dev": 0.45, "expected": 0.38, "ratio": 1.1842, "dt": 2.5 * 86400,
         "n_cold": 3, "ts_relative_seconds": -30 * 86400},
        {"memory_id": "A", "mtype": "episodic", "ts": 20 * 86400,
         "dev": 0.43, "expected": 0.39, "ratio": 1.1026, "dt": 2.5 * 86400,
         "n_cold": 3, "ts_relative_seconds": -15 * 86400},
        {"memory_id": "A", "mtype": "episodic", "ts": 35 * 86400,
         "dev": 0.42, "expected": 0.40, "ratio": 1.05, "dt": 2.5 * 86400,
         "n_cold": 3, "ts_relative_seconds": -5 * 86400},
        {"memory_id": "B", "mtype": "semantic", "ts": 8 * 86400,
         "dev": 0.20, "expected": 0.25, "ratio": 0.80, "dt": None,
         "n_cold": None, "ts_relative_seconds": -2 * 86400},
    ]
    p = str(tmp_path / "sig.json")
    _export_json(p, evs)
    ok = recall_curve_check.run_awakened_check(p)
    assert ok is True
    out = capsys.readouterr().out
    assert "导出 JSON 唤醒检查" in out
    assert "选中记忆 A [episodic]，唤醒事件 3 条 > 1" in out
    assert "#1" in out and "#3" in out and "已校准（信号衰减收敛）" in out  # 收敛方向
    assert "埋藏 2.5 天 / 检索 3 次" in out
    f = tmp_path / "recall_curve_awakened_export.svg"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    assert text.count("l5 -5 l5 5") == 3                              # 菱形
    assert text.count('stroke="#e34a2f" stroke-width="2.6"') == 3    # dev 红条
    assert text.count('stroke="#2a9d8f" stroke-width="2.6"') == 3    # expected 青条
    assert text.count('rx="3" fill=') == 3                            # 信号徽章
    assert "导出 JSON 唤醒事件时间线" in text and "事件时间线" in text


def test_awakened_from_export_no_multi(tmp_path, capsys):
    """导出 JSON 无多次唤醒记忆 → 明确跳过（不合成，避免误导），返回 False。"""
    evs = [{"memory_id": "m1", "mtype": "episodic", "ts": 1.0,
            "dev": 0.4, "expected": 0.3, "ratio": 1.33, "dt": None,
            "n_cold": None, "ts_relative_seconds": -1.0}]
    p = str(tmp_path / "sig.json")
    _export_json(p, evs)
    ok = recall_curve_check.run_awakened_check(p)
    assert ok is False
    out = capsys.readouterr().out
    assert "无多次唤醒记忆（全部 ≤ 1 次）→ 跳过" in out
    assert not (tmp_path / "recall_curve_awakened_export.svg").exists()


def test_cli_awakened_mode(tmp_path):
    """CLI --awakened：从真实库选中多次唤醒记忆（awakenings>1）、逐次打印全部
    唤醒事件（dev vs 预期 + 比值 + 方向）、产物写工作目录。"""
    store = _multi_awakening_store(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "recall_curve_check.py"), "--awakened", store],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(tmp_path), timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "多次唤醒记忆检查" in proc.stdout
    assert "awakenings=6 条 > 1" in proc.stdout                   # 真实库选中
    assert "#1" in proc.stdout and "#6" in proc.stdout            # 逐次标注
    assert "dev +0.4" in proc.stdout and "vs 类型预期" in proc.stdout
    assert "比值 1.19" in proc.stdout and "τ 应下调" in proc.stdout
    assert "埋藏 3.3 天 / 检索" in proc.stdout                     # 6 元组明细
    assert os.path.exists(tmp_path / "recall_curve_awakened.svg")
    assert not os.path.exists(ROOT / "recall_curve_awakened.svg")  # 未污染项目根


def test_cli_real_scenario(tmp_path):
    """CLI --real：额外运行真实场景、退出码 0、产物写工作目录（不污染项目根）。"""
    if not REAL_STORE.exists():
        pytest.skip("缺少 memories_session.json（真实记忆库）")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "recall_curve_check.py"), "--real", str(REAL_STORE)],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(tmp_path), timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "真实持久化场景" in proc.stdout
    assert "无缝衔接" in proc.stdout
    assert "唤醒点偏差" in proc.stdout
    assert "类型预期 +" in proc.stdout and "信号: 持平" in proc.stdout
    assert os.path.exists(tmp_path / "recall_curve_real.svg")
    assert os.path.exists(tmp_path / "recall_curve_real_fit.svg")
    fit = open(tmp_path / "recall_curve_real_fit.svg", encoding="utf-8").read()
    assert "类型预期" in fit and "#2a9d8f" in fit
    assert not os.path.exists(ROOT / "recall_curve_real.svg")  # 未污染项目根
    assert not os.path.exists(ROOT / "recall_curve_real_fit.svg")
