"""全领域「注入→产出」对照的判定逻辑单元测试（纯 analyze 函数，不读真实日志）。"""
import domain_gap_report as dgr

FIX = dgr.FIX_ROUND


def _r(no, domain, avg, tag="轮"):
    return {"no": no, "tag": tag, "domain": domain, "avg": avg}


def _rules(**kw):
    out = {}
    for d, n in kw.items():
        out[d] = {"rules": n, "pitfalls": 0, "improve": n}
    return out


def test_lifted_domain_not_in_gap():
    rounds = [_r(7, "分级数据", 1.2), _r(112, "分级数据", 1.2),
              _r(FIX + 10, "分级数据", 6.6)]
    rows, gaps, pending, first, _ = dgr.analyze(rounds, _rules(分级数据=5))
    row = next(r for r in rows if r["domain"] == "分级数据")
    assert row["verdict"] == "拉起"
    assert row["before"] == (1.2, 2) and row["after"] == (6.6, 1)
    assert gaps == [] and pending == [] and first == []


def test_gap_rule_but_score_not_lifted():
    rounds = [_r(50, "JSON相关", 7.6), _r(106, "JSON相关", 6.6),
              _r(FIX + 9, "JSON相关", 4.6)]
    rows, gaps, pending, first, _ = dgr.analyze(rounds, _rules(**{"JSON相关": 20}))
    row = next(r for r in rows if r["domain"] == "JSON相关")
    assert row["verdict"] == "未拉起"
    assert len(gaps) == 1 and gaps[0]["domain"] == "JSON相关"
    assert pending == [] and first == []


def test_flat_high_score_not_gap():
    rounds = [_r(1, "Table", 9.6), _r(FIX + 5, "Table", 7.8)]
    rows, gaps, pending, first, _ = dgr.analyze(rounds, _rules(Table=14))
    row = next(r for r in rows if r["domain"] == "Table")
    assert row["verdict"] == "持平(高分)"
    assert gaps == []


def test_pending_rules_but_no_after_sample():
    rounds = [_r(3, "TreeView", 7.1)]
    rows, gaps, pending, first, _ = dgr.analyze(rounds, _rules(TreeView=22))
    row = next(r for r in rows if r["domain"] == "TreeView")
    assert row["verdict"] == "待验证"
    assert len(pending) == 1 and pending[0]["domain"] == "TreeView"
    assert gaps == [] and first == []


def test_first_after_no_baseline():
    rounds = [_r(FIX + 20, "本地WEB", 5.0)]
    rows, gaps, pending, first, _ = dgr.analyze(rounds, _rules(**{"本地WEB": 11}))
    row = next(r for r in rows if r["domain"] == "本地WEB")
    assert row["verdict"] == "首练于修复后"
    assert len(first) == 1 and first[0]["domain"] == "本地WEB"
    assert gaps == [] and pending == []


def test_no_rules_never_gap():
    rounds = [_r(2, "新领域", 3.0), _r(FIX + 1, "新领域", 3.5)]
    rows, gaps, pending, first, _ = dgr.analyze(rounds, _rules())
    row = next(r for r in rows if r["domain"] == "新领域")
    assert row["verdict"] == "无规则"
    assert gaps == [] and pending == [] and first == []


def test_round_boundary_and_special_excluded():
    # 轮 113（修复前最后一轮）归 before；轮 114 起归 after；专项/验证轮不计入对照
    rounds = [_r(FIX - 1, "A", 5.0), _r(FIX, "A", 8.0),
              _r(1, "A", 6.0, tag="分级数据专项"), _r(1, "A", 6.5, tag="分级数据验证")]
    rows, gaps, pending, first, special = dgr.analyze(rounds, _rules(A=3))
    row = next(r for r in rows if r["domain"] == "A")
    assert row["before"] == (5.0, 1) and row["after"] == (8.0, 1)
    assert row["verdict"] == "拉起"
    assert len(special["A"]) == 2  # 专项/验证轮单独归入 special，不影响均分
