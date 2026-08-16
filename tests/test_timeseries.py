"""activity_timeseries 时间切片的单元测试。

背景（2026-08-15）：需要观察「活性×Δ 相关」随回放扩窗（轮 ~214 上线\nreplay-rounds 10）的时间变化，从单点相关变成时间序列。本测试固化：\n  - 窗口切分与领域聚合；\n  - r_mean（窗口内均分×活性）与 r_delta（窗口内 Δ×活性）计算；\n  - 样本不足窗口跳过；\n  - 截断/失败/抖动轮排除；\n  - 扩窗前 vs 后对比渲染。\n"""

import sys
import io
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import track_coder_trend as tct


def _mem(mid, dom, acc):
    return {"id": mid, "kind": "skill", "content": f"[{dom}/坑] 规则 {mid}",
            "access_count": acc}


def _log(rounds):
    """rounds: [(no, domain, score), ...] → 日志文本。"""
    lines = []
    for no, dom, score in rounds:
        lines.append(f"[t] === 第 {no} 轮 ===")
        lines.append(f"[t] 抽题: 领域「{dom}」题目：...")
        lines.append(f"[t] 生成代码: 1000 字符（完整）")
        lines.append(f"[t] 自检验: 五维 {{语法正确性={score}, API 规范性={score}, "
                     f"铁律遵守={score}, 实战可用性={score}, 最佳实践={score}}}")
    return "\n".join(lines)


def test_timeseries_windows_basic():
    """窗口切分：每 20 轮一个窗口，含领域聚合。"""
    rounds = []
    for i in range(1, 25):            # 轮 1-24 → 窗口 1-20, 21-40
        rounds.append((i, ["A", "B", "C"][i % 3], 5.0))
    log = _log(rounds)
    mems = [_mem("a", "A", 5), _mem("b", "B", 3), _mem("c", "C", 2)]
    series = tct.activity_timeseries(log, mems, window=20)
    assert len(series) >= 2
    assert series[0]["lo"] == 1 and series[0]["hi"] == 20
    assert series[0]["n_dom"] == 3
    assert "r_mean" in series[0] and "r_delta" in series[0]


def test_timeseries_includes_both_metrics():
    """r_mean 与 r_delta 都在输出中，且 Δ 只在 ≥2 轮领域上算。"""
    rounds = [
        (1, "A", 4.0), (2, "A", 6.0), (3, "A", 8.0),   # A: Δ+4.0, 3 轮
        (4, "B", 5.0), (5, "B", 5.0),                  # B: Δ 0, 2 轮
        (6, "C", 7.0),                                  # C: 1 轮, Δ 不计
    ]
    log = _log(rounds)
    mems = [_mem("a", "A", 6), _mem("b", "B", 3), _mem("c", "C", 1)]
    series = tct.activity_timeseries(log, mems, window=20)
    assert len(series) == 1
    s = series[0]
    assert s["n_dom"] == 3 and s["n_delta"] == 2   # C 只有 1 轮不进 Δ 样本
    assert s["r_delta"] == 0.0                        # Δ 样本仅 2 个，spearman 降级 0
    # r_mean: A=(4+6+8)/3=6, B=5, C=7; 活性规则数(≥3): A=1,B=1,C=0
    assert abs(s["r_mean"] - tct.spearman([1, 1, 0], [6.0, 5.0, 7.0])) < 1e-9


def test_timeseries_skips_small_windows():
    """领域 <3 的窗口跳过。"""
    rounds = [(1, "A", 5.0), (2, "B", 5.0)]   # 只有 2 领域
    log = _log(rounds)
    mems = [_mem("a", "A", 5), _mem("b", "B", 3)]
    assert tct.activity_timeseries(log, mems, window=20) == []


def test_timeseries_excludes_trunc_fail():
    """截断轮不计入窗口聚合。"""
    rounds = [(1, "A", 5.0), (2, "A", 6.0), (3, "B", 4.0), (4, "B", 4.0), (5, "C", 6.0)]
    log = _log(rounds)
    # 把 B 的轮标为截断
    log = log.replace("[t] 生成代码: 1000 字符（完整）\n[t] 自检验: 五维 {语法正确性=4.0,",
                      "[t] 生成代码: 100 字符（截断（代码块未闭合））\n[t] 自检验: 五维 {语法正确性=4.0,")
    mems = [_mem("a", "A", 5), _mem("b", "B", 4), _mem("c", "C", 3)]
    series = tct.activity_timeseries(log, mems, window=20)
    # B 被排除（截断）→ 只剩 A, C 2 领域 → 窗口跳过
    assert series == []


def test_timeseries_multi_window_sequence():
    """多窗口序列按轮号升序返回，窗口边界正确。"""
    rounds = []
    for i in range(1, 61):
        dom = ["A", "B", "C", "D"][i % 4]
        rounds.append((i, dom, 5.0))
    log = _log(rounds)
    mems = [_mem("a", "A", 5), _mem("b", "B", 3), _mem("c", "C", 2), _mem("d", "D", 1)]
    series = tct.activity_timeseries(log, mems, window=20)
    assert len(series) == 3
    assert [s["lo"] for s in series] == [1, 21, 41]
    assert all(s["n_dom"] == 4 for s in series)


def test_timeseries_block_renders():
    """渲染含扩窗前/后对比。"""
    series = [
        {"lo": 1, "hi": 20, "n_dom": 10, "n_delta": 0, "r_mean": -0.5, "r_delta": 0.0},
        {"lo": 201, "hi": 220, "n_dom": 10, "n_delta": 0, "r_mean": 0.4, "r_delta": 0.0},
        {"lo": 221, "hi": 240, "n_dom": 10, "n_delta": 0, "r_mean": 0.6, "r_delta": 0.0},
    ]
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        tct.print_timeseries_block(series, window=20)
    finally:
        sys.stdout = old
    text = out.getvalue()
    assert "时间切片" in text
    assert "扩窗(轮214)前" in text and "后" in text
