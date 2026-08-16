"""track_coder_trend 规则活跃度 × 领域走势关联分析的单元测试。

背景（2026-08-15）：sleep 桥 + 回放扩窗后，规则 access_count 分层积累。
本测试固化：
  - domain_activity_stats：按领域聚合活性规则（≥3 次回放）与平均回放次数；
  - domain_score_trend：领域首练→最近均分走势；
  - spearman：秩相关（含并列排名、样本不足降级）；
  - activity_correlation：关联表 + 分组对比。
"""
import json

import track_coder_trend as tct


def _mems():
    return [
        {"kind": "skill", "content": "[Table/坑] 铁律A", "access_count": 5},
        {"kind": "skill", "content": "[Table/改进] 教训B", "access_count": 3},
        {"kind": "skill", "content": "[Table/代码] 模板C", "access_count": 2},
        {"kind": "skill", "content": "[DataTable/改进] 教训D", "access_count": 2},
        {"kind": "skill", "content": "[DataTable/坑] 铁律E", "access_count": 2},
        {"kind": "router", "content": "[Table/路由] 路由"},
    ]


def _cycles():
    def c(n, d, scores):
        return {"n": n, "domain": d, "code": 10, "scores": scores,
                "distilled": 0, "fail": "", "src": "log"}
    return [
        c(1, "Table", {"语法正确性": 4, "API 规范性": 4, "铁律遵守": 4,
                       "实战可用性": 4, "最佳实践": 4}),       # 4.0
        c(2, "DataTable", {"语法正确性": 2, "API 规范性": 2, "铁律遵守": 2,
                           "实战可用性": 2, "最佳实践": 2}),   # 2.0
        c(3, "Table", {"语法正确性": 8, "API 规范性": 8, "铁律遵守": 8,
                       "实战可用性": 8, "最佳实践": 8}),       # 8.0
        c(4, "DataTable", {"语法正确性": 2, "API 规范性": 2, "铁律遵守": 2,
                           "实战可用性": 2, "最佳实践": 2}),   # 2.0
        c(5, "JSON相关", {"语法正确性": 6, "API 规范性": 6, "铁律遵守": 6,
                          "实战可用性": 6, "最佳实践": 6}),    # 6.0
    ]


def test_domain_activity_stats():
    act = tct.domain_activity_stats(_mems())
    # Table：3 条规则，2 条活性（5/3），平均回放 (5+3+2)/3 = 3.33
    assert act["Table"]["active"] == 2
    assert abs(act["Table"]["avg_acc"] - 10 / 3) < 1e-6
    assert act["Table"]["rules"] == 3
    # DataTable：2 条规则，0 活性
    assert act["DataTable"]["active"] == 0
    assert act["DataTable"]["avg_acc"] == 2.0
    # JSON相关：无规则（不出现）
    assert "JSON相关" not in act


def test_domain_score_trend():
    trend = tct.domain_score_trend(_cycles())
    assert trend["Table"] == {"first": 4.0, "last": 8.0, "delta": 4.0}
    assert trend["DataTable"] == {"first": 2.0, "last": 2.0, "delta": 0.0}
    assert "JSON相关" not in trend  # 只有一轮，样本不足


def test_spearman_rank_correlation():
    # 完全正相关 → 1.0
    assert abs(tct.spearman([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9
    # 完全负相关 → -1.0
    assert abs(tct.spearman([1, 2, 3, 4], [8, 6, 4, 2]) - -1.0) < 1e-9
    # 样本不足 → 0.0
    assert tct.spearman([1, 2], [3, 4]) == 0.0


def test_spearman_ties():
    # 并列数据不崩，秩相关在 [-1,1] 内
    r = tct.spearman([2, 2, 3], [1, 2, 3])
    assert -1.0 <= r <= 1.0


def test_activity_correlation_positive_delta():
    corr = tct.activity_correlation(_mems(), _cycles())
    # Table 活性 2、Δ+4.0；DataTable 活性 0、Δ0.0
    rows = {r["domain"]: r for r in corr["rows"]}
    assert rows["Table"]["active"] == 2 and rows["Table"]["delta"] == 4.0
    assert rows["DataTable"]["active"] == 0 and rows["DataTable"]["delta"] == 0.0
    # 活性领域 Δ 均值 > 无活性领域 Δ 均值
    assert corr["active_n"] == 1 and corr["inactive_n"] == 1
    assert corr["active_delta_avg"] == 4.0
    assert corr["inactive_delta_avg"] == 0.0
    assert corr["spearman"] == 0.0  # 样本 2 < 3，按设计降级为 0


def test_activity_correlation_insufficient():
    corr = tct.activity_correlation(_mems(), _cycles()[:1])  # 只有一轮
    assert corr["rows"] == []
    assert corr["spearman"] == 0.0


def test_activity_block_renders(monkeypatch, tmp_path):
    mem_path = tmp_path / "mem.json"
    mem_path.write_text(json.dumps({"memories": _mems()}), encoding="utf-8")
    monkeypatch.setattr(tct, "FT_MEM", mem_path)
    out = []
    import io
    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        tct.main.__wrapped__ if hasattr(tct.main, "__wrapped__") else None
    # 直接测 print_activity_block 更稳定
    corr = tct.activity_correlation(_mems(), _cycles())
    with contextlib.redirect_stdout(buf):
        tct.print_activity_block(corr)
    text = buf.getvalue()
    assert "规则活跃度 × 领域分数走势" in text
    assert "Table" in text and "DataTable" in text
    assert "Spearman" in text
    assert "Δ均分均值" in text
