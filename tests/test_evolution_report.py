"""evolution_report 睡眠回放再激活统计的单元测试。

背景（2026-08-15）：coder 记忆接入 MemoryAgent store 后，agent.sleep() 的回放
真正生效（每轮新沉淀规则被再激活，access_count 2→3+）。本测试固化：
  - 每轮「睡眠: 回放 N」日志解析；
  - access_count 分布 + 活性规则（≥3 次）按领域聚合；
  - 活性领域 vs 无活性领域的「铁律遵守」维度对比（遵守率代理）；
  - 样本不足时降级提示。
"""
import evolution_report as er


def _log() -> str:
    # 轮 1/2：有回放（接入后）；轮 3：无回放（旧段/无沉淀轮）；轮 4：无回放
    return """[t] === 第 1 轮 ===
[t] 抽题: 领域「Table」题目：...
[t] 自检验: 五维 {语法正确性=8.0, API 规范性=7.0, 铁律遵守=9.0, 实战可用性=6.0, 最佳实践=5.0}
[t] 睡眠: 回放 2 · 冷压缩 0 · 演化入库 0
[t] === 第 2 轮 ===
[t] 抽题: 领域「Table」题目：...
[t] 自检验: 五维 {语法正确性=7.0, API 规范性=6.0, 铁律遵守=8.0, 实战可用性=7.0, 最佳实践=6.0}
[t] 睡眠: 回放 3 · 冷压缩 0 · 演化入库 0
[t] === 第 3 轮 ===
[t] 抽题: 领域「DataTable」题目：...
[t] 自检验: 五维 {语法正确性=5.0, API 规范性=5.0, 铁律遵守=4.0, 实战可用性=5.0, 最佳实践=5.0}
[t] 睡眠: 回放 0 · 冷压缩 0 · 演化入库 0
[t] === 第 4 轮 ===
[t] 抽题: 领域「JSON相关」题目：...
[t] 自检验: 五维 {语法正确性=6.0, API 规范性=6.0, 铁律遵守=5.0, 实战可用性=6.0, 最佳实践=5.0}
[t] 睡眠: 回放 0 · 冷压缩 0 · 演化入库 0
"""


def _mems():
    # Table 有活性规则（access_count 3+），DataTable/JSON相关 无
    return [
        {"kind": "skill", "content": "[Table/坑] 铁律A", "access_count": 5},
        {"kind": "skill", "content": "[Table/改进] 教训B", "access_count": 3},
        {"kind": "skill", "content": "[DataTable/改进] 教训C", "access_count": 2},
        {"kind": "skill", "content": "[JSON相关/坑] 铁律D", "access_count": 2},
        {"kind": "router", "content": "[Table/路由] 路由"},
    ]


def test_replay_stats_parses_rounds():
    st = er.replay_stats(_log(), _mems())
    assert st["rounds_total"] == 4
    assert st["replay_rounds"] == 2       # 轮1/2 有回放
    assert st["replay_total"] == 5         # 2 + 3
    assert st["timeline"] == [(1, 2), (2, 3)]


def test_replay_stats_access_distribution():
    st = er.replay_stats(_log(), _mems())
    assert st["acc_dist"] == {2: 2, 3: 1, 5: 1}   # 三条 skill + 一条 skill
    assert st["acc3_plus"] == 2            # Table 两条 ≥3
    assert st["skills_total"] == 4
    assert st["active_domains_n"] == 1     # 只有 Table 有活性规则


def test_replay_stats_rule_dimension_comparison():
    st = er.replay_stats(_log(), _mems())
    # Table（活性）铁律遵守 9.0/8.0 → 8.5；DataTable 4.0；JSON相关 5.0 → 4.5
    assert st["active_dim_avg"] == 8.5
    assert st["inactive_dim_avg"] == 4.5
    assert st["active_dim_n"] == 1
    assert st["inactive_dim_n"] == 2


def test_replay_block_renders_verdict():
    md = er.replay_block(_log(), _mems())
    assert "睡眠回放再激活" in md
    assert "回放发生 2/4 轮 · 累计回放 5 条规则" in md
    assert "轮1:2 · 轮2:3" in md
    assert "铁律遵守维度: 活性领域 8.5（n=1） vs 无活性领域 4.5（n=2） → Δ+4.0，活性领域铁律遵守更高" in md


def test_replay_stats_insufficient_data_degrades():
    # 只有无回放轮（轮3/4），且记忆无活性规则 → 对比样本不足
    log3 = _log()[_log().index("=== 第 3 轮 ==="):]  # 只留轮3 起
    st = er.replay_stats(log3, _mems())
    assert st["replay_total"] == 0
    # 轮3/4 领域是 DataTable/JSON相关，均无活性规则 → inactive 有样本、active 无
    assert st["active_dim_avg"] is None
    assert st["inactive_dim_avg"] == 4.5
    md = er.replay_block(log3, _mems())
    assert "铁律遵守对比: 样本不足" in md
