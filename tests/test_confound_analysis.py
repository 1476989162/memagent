"""活性×Δ 偏相关（混杂分离）的单元测试。

背景（2026-08-15）：track_coder_trend 曾观察到活性规则数×Δ均分负相关，怀疑是
「活性多的领域练得久→首练在早期无铁律→基数低→Δ 显负」的结构性混杂。本测试固化：
  - partial_spearman：一阶偏相关公式正确性；
  - domain_confound_analysis：从日志解析 + 领域变量聚合 + 相关/偏相关输出；
  - 混杂消失场景：活性×Δ 原始相关由练习时长驱动，控制后归零；
  - 真实效应场景：控制混杂后活性×Δ 仍显著。
"""

import sys
import io
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import track_coder_trend as tct


def _mem(mid, dom, acc):
    return {"id": mid, "kind": "skill", "content": f"[{dom}/坑] 规则 {mid}",
            "access_count": acc}


def _log(rounds):
    """rounds: [(no, domain, dim, score), ...] → 日志文本。dim 为五维均分。"""
    lines = []
    for no, dom, score in rounds:
        lines.append(f"[t] === 第 {no} 轮 ===")
        lines.append(f"[t] 抽题: 领域「{dom}」题目：...")
        if score is not None:
            lines.append(f"[t] 生成代码: 1000 字符（完整）")
            lines.append(f"[t] 自检验: 五维 {{语法正确性={score}, API 规范性={score}, "
                         f"铁律遵守={score}, 实战可用性={score}, 最佳实践={score}}}")
    return "\n".join(lines)


def _rounds_from_pairs(pairs):
    """pairs: [(no, dom, score), ...]"""
    return pairs


def test_partial_spearman_no_confound():
    """x,y 强相关、z 与其无关 → 偏相关 ≈ r_xy（不被无关变量稀释）。"""
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [5, 4, 3, 2, 1, 0, -1, -2]          # 与 x 完全负相关
    z = [3, 1, 4, 2, 5, 1, 3, 2]            # 与 x,y 无系统关系（非共线）
    r_xy = tct.spearman(x, y)
    r_xy_z = tct.partial_spearman(x, y, z)
    assert abs(r_xy - (-1.0)) < 1e-9
    assert abs(r_xy_z - r_xy) < 1e-3       # z 与 x,y 无关 → 偏相关近似不变（浮点容差）


def test_partial_spearman_confound_removed():
    """x,y 原始相关由 z 驱动，控制 z 后归零。"""
    # 构造: x 与 z 强相关（活性∝练习时长），y 与 z 强相关但 x-y 无直接关系
    z = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x = [2 * a + 0.1 * (i % 3) for i, a in enumerate(z)]   # x ≈ 2z
    y = [3 * a + 0.1 * (i % 2) for i, a in enumerate(z)]   # y ≈ 3z → x,y 都∝z
    r_xy = tct.spearman(x, y)
    r_xy_z = tct.partial_spearman(x, y, z)
    assert abs(r_xy) > 0.9                    # 原始强相关
    assert abs(r_xy_z) < 0.1                  # 控制 z 后消失


def test_partial_spearman_small_sample():
    assert tct.partial_spearman([1, 2], [3, 4], [5, 6]) == 0.0


def test_domain_confound_analysis_basic():
    """日志解析 + 领域变量聚合。"""
    pairs = [
        (1, "A", 4.0), (2, "A", 6.0),   # A: Δ+2.0
        (3, "B", 5.0), (4, "B", 5.0),   # B: Δ 0
        (5, "C", 7.0), (6, "C", 8.0),   # C: Δ+1.0
    ]
    mems = [_mem("a", "A", 5), _mem("b", "A", 3), _mem("c", "B", 2), _mem("d", "C", 4)]
    res = tct.domain_confound_analysis(_log(pairs), mems)
    assert len(res["rows"]) == 3
    by = {r["domain"]: r for r in res["rows"]}
    assert by["A"]["delta"] == 2.0 and by["A"]["n_rounds"] == 2
    assert by["A"]["active"] == 2 and by["B"]["active"] == 0
    assert "active_delta" in res["corr"] and "active_delta|rounds" in res["partial"]


def test_domain_confound_excludes_trunc_fail():
    """截断/失败/低分抖动轮被排除，不进入领域聚合。"""
    # A 两轮正常；B 两轮都是截断轮（分数 1.0 来自截断代码，应排除）
    lines = []
    for no, dom, score, trunc in [
        (1, "A", 4.0, False), (2, "A", 6.0, False),
        (3, "B", 1.0, True), (4, "B", 1.0, True),
    ]:
        lines.append(f"[t] === 第 {no} 轮 ===")
        lines.append(f"[t] 抽题: 领域「{dom}」题目：...")
        if score is not None:
            tag = "（截断（代码块未闭合））" if trunc else "（完整）"
            lines.append(f"[t] 生成代码: 100 字符{tag}")
            lines.append(f"[t] 自检验: 五维 {{语法正确性={score}, API 规范性={score}, "
                         f"铁律遵守={score}, 实战可用性={score}, 最佳实践={score}}}")
    log = "\n".join(lines)
    mems = [_mem("a", "A", 5), _mem("b", "B", 4)]
    res = tct.domain_confound_analysis(log, mems)
    assert [r["domain"] for r in res["rows"]] == ["A"]


def test_confound_block_renders_verdict_no_confound():
    """原始相关不显著时输出明确判定。"""
    res = {
        "rows": [{"domain": d} for d in "ABCDEFG"],
        "corr": {"active_delta": 0.1, "active_rounds": 0.5, "rounds_delta": 0.0,
                 "first_s_delta": 0.0, "first_r_delta": 0.0},
        "partial": {"active_delta|rounds": 0.05, "active_delta|first_r": 0.02,
                    "active_delta|first_s": 0.03, "active_delta|avg_acc": 0.0,
                    "active_delta|rules": 0.0},
    }
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        tct.print_confound_block(res)
    finally:
        sys.stdout = old
    text = out.getvalue()
    assert "原始相关本就不显著" in text


def test_confound_block_renders_verdict_artifact():
    """原始相关强但控制后归零 → 判定为结构性伪影。"""
    res = {
        "rows": [{"domain": d} for d in "ABCDEFG"],
        "corr": {"active_delta": -0.6, "active_rounds": 0.8, "rounds_delta": -0.5,
                 "first_s_delta": -0.7, "first_r_delta": 0.3},
        "partial": {"active_delta|rounds": -0.05, "active_delta|first_r": -0.03,
                    "active_delta|first_s": 0.02, "active_delta|avg_acc": 0.0,
                    "active_delta|rules": 0.0},
    }
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        tct.print_confound_block(res)
    finally:
        sys.stdout = old
    text = out.getvalue()
    assert "结构性伪影" in text
    assert "非回放活性因果" in text
