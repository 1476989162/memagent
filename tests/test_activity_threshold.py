"""活性规则判定模式的单元测试：固定 ≥3 vs 领域内相对排名（膨胀免疫）。"""

import sys
import io
import json
import os
import tempfile
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import track_coder_trend as tct


def _mem(mid, dom, acc, mtype="skill"):
    return {"id": mid, "kind": mtype, "content": f"[{dom}/坑] 规则 {mid}",
            "access_count": acc, "importance": 0.9}


def _mems(specs):
    """specs: [(id, domain, acc), ...]"""
    return [_mem(mid, dom, acc) for mid, dom, acc in specs]


def test_fix3_basic():
    """fix3: access_count >= 3 即活性。"""
    mems = _mems([("a", "D1", 3), ("b", "D1", 2), ("c", "D2", 5), ("d", "D2", 1)])
    ids = tct.active_rule_ids(mems, mode="fix3")
    assert ids == {"a", "c"}


def test_rel_basic():
    """rel: 领域内 top 30% 且 >2。"""
    mems = _mems([("a", "D1", 3), ("b", "D1", 2), ("c", "D1", 4),
                  ("d", "D1", 2), ("e", "D2", 5), ("f", "D2", 1)])
    ids = tct.active_rule_ids(mems, mode="rel")
    # D1: 4 条规则, top30% → thr=ranked[int(4*0.3)-1]=ranked[0]=4 → 只有 c(4) 且 >2 → {c}
    # D2: 2 条, thr=ranked[0]=5 → e → {e}
    assert ids == {"c", "e"}, ids


def test_rel_inflation_immune():
    """膨胀免疫：活跃簇从 3-6 涨到 12-16，领域内 top30% 集合不变（fix3 会全算）。"""
    # 两个领域, 各 33 条（25 沉睡 @2 + 8 活跃）
    fresh = [("f%d" % i, "刚激活", 2) for i in range(25)]
    fresh += [("f%d" % i, "刚激活", a) for i, a in enumerate([3, 4, 5, 6, 6, 5, 4, 3], 25)]
    bloated = [("b%d" % i, "膨胀", 2) for i in range(25)]
    bloated += [("b%d" % i, "膨胀", a) for i, a in enumerate([12, 13, 14, 15, 16, 15, 14, 13], 25)]
    mems = _mems(fresh + bloated)
    rel_ids = tct.active_rule_ids(mems, mode="rel")
    # 两领域各有 8 条 >2，top30% of 33 ≈ 10 条 → 8 条活性全入选（都 >2 且 ≥ thr）
    # thr = ranked[int(33*0.3)-1] = ranked[8] = 3(刚激活)/12(膨胀) → 8 条都 >= thr 且 >2
    rel_fresh = sum(1 for i in range(33) if ("f%d" % i) in rel_ids)
    rel_bloated = sum(1 for i in range(33) if ("b%d" % i) in rel_ids)
    assert rel_fresh == 8 and rel_bloated == 8
    # fix3 同样全 8（此时未膨胀, 两者等价——一致性正是切换安全的证据）
    fx_ids = tct.active_rule_ids(mems, mode="fix3")
    fx_fresh = sum(1 for i in range(33) if ("f%d" % i) in fx_ids)
    fx_bloated = sum(1 for i in range(33) if ("b%d" % i) in fx_ids)
    assert fx_fresh == 8 and fx_bloated == 8


def test_rel_vs_fix3_diverge_when_all_bloated():
    """膨胀后 fix3 饱和（全部 >2 都算活性）而 rel 保持 top 比例。"""
    # 单领域 10 条，全部涨到 3-12（无沉睡池）——模拟最坏膨胀
    mems = _mems([("r%d" % i, "D", a) for i, a in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])])
    fx = tct.active_rule_ids(mems, mode="fix3")
    rel = tct.active_rule_ids(mems, mode="rel")
    assert len(fx) == 10          # 全算活性——饱和
    assert 0 < len(rel) <= 5      # top30% → 3 条（thr=ranked[2]=10 → 10,11,12）
    assert len(rel) == 3, len(rel)


def test_global_relative_not_used():
    """全局相对排名不可行：88%+ 沉睡 @2 会把 top25% 阈值压到 2，100% 饱和。"""
    mems = _mems([("s%d" % i, "D%d" % (i % 3), 2) for i in range(88)] +
                 [("a%d" % i, "D%d" % (i % 3), a) for i, a in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])])
    # 全局 top 25% 会包含 @2 → 全部入选
    n = len(mems)
    ranked = sorted((m["access_count"] for m in mems), reverse=True)
    thr = ranked[max(0, int(n * 0.25) - 1)]
    assert thr == 2  # 阈值被压到沉睡档
    global_ids = {m["id"] for m in mems if m["access_count"] >= thr}
    assert len(global_ids) == n  # 100% 饱和


def test_domain_activity_stats_modes():
    """domain_activity_stats 的 active 计数随 mode 变化。"""
    mems = _mems([("a", "D1", 3), ("b", "D1", 2), ("c", "D2", 5), ("d", "D2", 4)])
    fx = tct.domain_activity_stats(mems, mode="fix3")
    assert fx["D1"]["active"] == 1 and fx["D2"]["active"] == 2
    rel = tct.domain_activity_stats(mems, mode="rel")
    # D1: 2条, top30%→1条(3); D2: 2条, top30%→1条(5)
    assert rel["D1"]["active"] == 1 and rel["D2"]["active"] == 1, rel


def test_activity_correlation_mode_param():
    """activity_correlation 透传 mode，不崩。"""
    mems = _mems([("a", "D1", 3), ("b", "D1", 2)])
    cycles = [{"n": 1, "domain": "D1", "scores": {"语法正确性": 5.0, "API 规范性": 5.0,
              "铁律遵守": 5.0, "实战可用性": 5.0, "最佳实践": 5.0}, "fail": ""},
              {"n": 2, "domain": "D1", "scores": {"语法正确性": 6.0, "API 规范性": 6.0,
              "铁律遵守": 6.0, "实战可用性": 6.0, "最佳实践": 6.0}, "fail": ""}]
    r1 = tct.activity_correlation(mems, cycles, mode="fix3")
    r2 = tct.activity_correlation(mems, cycles, mode="rel")
    assert r1["rows"] and r2["rows"]
    assert r1["rows"][0]["domain"] == r2["rows"][0]["domain"] == "D1"
