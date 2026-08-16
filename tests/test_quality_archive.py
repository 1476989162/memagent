"""休息间隙质量快照归档（snapshot_quality_archive）的单元测试。

背景（2026-08-15）：休息机制此前是纯计时器，治理已把部分休息变成产出；
本功能让每次休息都留下一份可回溯质量档案（均分曲线/领域覆盖/注入核对）。
"""
import autonomous_coder as ac


def _fake_log() -> str:
    return """[2026-08-15 17:00:00] === 第 1 轮 ===
[2026-08-15 17:00:00] 抽题: 领域「Table」题目：...
[2026-08-15 17:00:30] 注入覆盖: 坑规则 3/3 已注入
[2026-08-15 17:01:00] 自检验: 五维 {语法正确性=8.0, API 规范性=7.0, 铁律遵守=6.0, 实战可用性=9.0, 最佳实践=5.0}
[2026-08-15 17:01:10] 休息 95s ...
[2026-08-15 17:02:00] === 第 2 轮 ===
[2026-08-15 17:02:00] 抽题: 领域「DataTable」题目：...
[2026-08-15 17:02:30] 注入覆盖: 坑规则 0/0 已注入
[2026-08-15 17:03:00] 自检验: 五维 {语法正确性=2.0, API 规范性=2.0, 铁律遵守=2.0, 实战可用性=2.0, 最佳实践=2.0}
[2026-08-15 17:03:10] 休息 88s ...
"""


def _fake_mems():
    return [
        {"kind": "skill", "content": "[Table/坑] 铁律1", "access_count": 5},
        {"kind": "skill", "content": "[Table/改进] 改进1", "access_count": 2},
        {"kind": "skill", "content": "[DataTable/坑] 铁律2", "access_count": 3},
        {"kind": "router", "content": "[Table/路由] 路由1"},
    ]


def test_snapshot_data_parses_scores(monkeypatch, tmp_path):
    log_path = tmp_path / "fake.log"
    log_path.write_text(_fake_log(), encoding="utf-8")
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", _fake_mems)
    d = ac._snapshot_data(round_no=210, govern_rep={"total": 2})
    # 均分：轮1 (8+7+6+9+5)/5=7.0；轮2 全 2.0
    assert [n for n, _ in d["recent"]] == [1, 2]
    avgs = [a for _, a in d["recent"]]
    assert avgs == [7.0, 2.0]
    assert d["all_avg"] == 4.5
    assert d["all_n"] == 2
    assert d["inject_ok"] == 2
    assert d["round"] == 210
    # 记忆规模：skill 3 条，坑 2 条，被回放≥3 次 2 条
    assert d["mems"] == 4
    assert d["skills"] == 3
    assert d["pitfalls"] == 2
    assert d["access3"] == 2


def test_snapshot_md_contains_all_sections(monkeypatch, tmp_path):
    log_path = tmp_path / "fake.log"
    log_path.write_text(_fake_log(), encoding="utf-8")
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", _fake_mems)
    d = ac._snapshot_data(round_no=210, govern_rep={"total": 2})
    md = ac._snapshot_md(d)
    assert "## 快照 @ 轮 210" in md
    assert "均分曲线" in md and "轮1:7.0" in md
    assert "累计均分**: 4.50（2 轮）" in md
    assert "领域覆盖" in md
    assert "注入核对" in md and "注入截断告警 0" in md
    assert "规则治理" in md and "共清理 2 条" in md
    assert "记忆规模" in md and "被回放≥3次 2" in md


def test_snapshot_archive_appends_and_writes_latest(monkeypatch, tmp_path):
    log_path = tmp_path / "fake.log"
    log_path.write_text(_fake_log(), encoding="utf-8")
    archive = tmp_path / "quality_archive.md"
    latest = tmp_path / "quality_latest.md"
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", _fake_mems)
    monkeypatch.setattr(ac, "ARCHIVE_DOC", archive)
    monkeypatch.setattr(ac, "ARCHIVE_LATEST", latest)
    # 第一次归档
    p1 = ac.snapshot_quality_archive(210, {"total": 0})
    assert p1 == latest
    txt1 = archive.read_text(encoding="utf-8")
    assert txt1.startswith("# 质量快照归档")
    assert "轮 210" in txt1
    # 第二次归档 → 追加（历史累积）
    ac.snapshot_quality_archive(211)
    txt2 = archive.read_text(encoding="utf-8")
    assert txt2.count("## 快照 @ 轮") == 2
    assert "轮 211" in txt2
    # latest 只保留最新
    assert latest.read_text(encoding="utf-8").count("## 快照 @ 轮") == 1


def test_snapshot_no_scores_tolerated(monkeypatch, tmp_path):
    log_path = tmp_path / "empty.log"
    log_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", lambda: [])
    d = ac._snapshot_data(210)
    assert d["recent"] == []
    assert d["all_avg"] is None
    md = ac._snapshot_md(d)
    assert "累计均分**: 暂无" in md
    assert "领域覆盖**: 0/44" in md


def test_snapshot_includes_replay_activity(monkeypatch, tmp_path):
    """快照数据含回放活性：活性规则数/覆盖领域/Δ铁律遵守。"""
    log_path = tmp_path / "fake.log"
    log_path.write_text(_fake_log(), encoding="utf-8")
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", _fake_mems)
    d = ac._snapshot_data(round_no=210)
    # Table 活性(5), DataTable 活性(3) → 2 条活性规则, 2 领域
    assert d["active_rules"] == 2
    assert d["active_domains"] == 2
    # Δ铁律遵守：Table 遵守 (6+2)/2=4.0(活性) vs DataTable 2.0(无活性?)——DataTable 有活性(3)→两者都活性
    assert d["replay_delta"] is None  # 两领域均有活性 → 无无活性对照 → None
    md = ac._snapshot_md(d)
    assert "**回放活性**" in md
    assert "规则 2 · 覆盖 2 领域" in md


def test_replay_activity_stats_pure():
    """_replay_activity_stats 纯函数：正确聚合活性与 Δ。"""
    log = """[t] === 第 1 轮 ===
[t] 抽题: 领域「Table」题目：...
[t] 自检验: 五维 {语法正确性=8.0, API 规范性=7.0, 铁律遵守=9.0, 实战可用性=6.0, 最佳实践=5.0}
[t] === 第 2 轮 ===
[t] 抽题: 领域「DataTable」题目：...
[t] 自检验: 五维 {语法正确性=5.0, API 规范性=5.0, 铁律遵守=4.0, 实战可用性=5.0, 最佳实践=5.0}
[t] === 第 3 轮 ===
[t] 抽题: 领域「JSON相关」题目：...
[t] 自检验: 五维 {语法正确性=6.0, API 规范性=6.0, 铁律遵守=5.0, 实战可用性=6.0, 最佳实践=5.0}
"""
    mems = [
        {"kind": "skill", "content": "[Table/坑] 铁律A", "access_count": 5},
        {"kind": "skill", "content": "[DataTable/坑] 铁律B", "access_count": 2},
        {"kind": "skill", "content": "[JSON相关/坑] 铁律C", "access_count": 2},
    ]
    st = ac._replay_activity_stats(log, mems)
    assert st["active_rules"] == 1          # 只有 Table 5>=3
    assert st["active_domains"] == 1
    # 活性领域 Table 遵守 9.0; 无活性 DataTable 4.0 / JSON相关 5.0 → Δ = 9 - 4.5 = 4.5
    assert st["replay_delta"] == 4.5
    assert st["act_n"] == 1 and st["inact_n"] == 2


def test_append_replay_trend_every_20_rounds(monkeypatch, tmp_path):
    """每 20 轮追加一行；非 20 倍轮跳过；同轮幂等。"""
    trend = tmp_path / "replay_activity_trend.md"
    monkeypatch.setattr(ac, "REPLAY_TREND_DOC", trend)
    d20 = {"round": 220, "time": "2026-08-15 20:00:00", "active_rules": 100,
           "active_domains": 20, "replay_delta": 1.5, "act_n": 30, "inact_n": 5}
    p = ac.append_replay_trend(d20)
    assert p == trend
    txt = trend.read_text(encoding="utf-8")
    assert "| 220 |" in txt and "100" in txt and "+1.50" in txt
    # 非 20 倍轮 → None
    assert ac.append_replay_trend({**d20, "round": 221}) is None
    # 同轮幂等：不重复追加
    ac.append_replay_trend(d20)
    assert trend.read_text(encoding="utf-8").count("| 220 |") == 1
    # 下一 20 倍轮追加新行
    ac.append_replay_trend({**d20, "round": 240, "active_rules": 130})
    txt = trend.read_text(encoding="utf-8")
    assert "| 240 |" in txt and txt.count("| 240 |") == 1


def test_snapshot_archive_triggers_trend(monkeypatch, tmp_path):
    """snapshot_quality_archive 在 20 倍轮自动追加时间序列。"""
    log_path = tmp_path / "fake.log"
    log_path.write_text(_fake_log(), encoding="utf-8")
    archive = tmp_path / "quality_archive.md"
    latest = tmp_path / "quality_latest.md"
    trend = tmp_path / "replay_activity_trend.md"
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", _fake_mems)
    monkeypatch.setattr(ac, "ARCHIVE_DOC", archive)
    monkeypatch.setattr(ac, "ARCHIVE_LATEST", latest)
    monkeypatch.setattr(ac, "REPLAY_TREND_DOC", trend)
    # 轮 220（20 倍）→ 追加；轮 221 → 不追加
    ac.snapshot_quality_archive(220)
    assert trend.exists() and "| 220 |" in trend.read_text(encoding="utf-8")
    ac.snapshot_quality_archive(221)
    assert "| 221 |" not in trend.read_text(encoding="utf-8")


def _fake_log_with_trend() -> str:
    """20 轮日志：均分先升后降再升（制造 MA5 拐点）+ 治理/截断行。"""
    lines = []
    for i in range(1, 21):
        lines.append(f"[2026-08-15 17:{i:02d}:00] === 第 {i} 轮 ===")
        lines.append(f"[2026-08-15 17:{i:02d}:00] 抽题: 领域「Table」题目：...")
        # 分数：轮1-6 升（4→9），轮7-12 降（9→4），轮13-20 升（4→8）
        if i <= 6:
            sc = 4 + i
        elif i <= 12:
            sc = 15 - i
        else:
            sc = 4 + (i - 12)
        lines.append(f"[2026-08-15 17:{i:02d}:30] 自检验: 五维 {{语法正确性={sc}.0, API 规范性={sc}.0, "
                     f"铁律遵守={sc}.0, 实战可用性={sc}.0, 最佳实践={sc}.0}}")
    lines.append("[2026-08-15 17:21:00] === 第 21 轮 ===")
    lines.append("[2026-08-15 17:21:00] 规则治理: 去重 2 · 毒规则 1 · 残缺 1（共清理 4 条）")
    lines.append("[2026-08-15 17:22:00] === 第 22 轮 ===")
    lines.append("[2026-08-15 17:22:00] 抽题: 领域「Table」题目：...")
    lines.append("[2026-08-15 17:22:30] 生成代码: 500 字符（截断（代码块未闭合））")
    lines.append("[2026-08-15 17:22:30] ⚠ 高优先级截断告警: 「Table」代码截断——强制下轮重练")
    return "\n".join(lines)


def test_snapshot_ma5_trend_and_pivots(monkeypatch, tmp_path):
    """MA5 趋势线 + 拐点标注：方向反转且幅度≥0.5 处识别。"""
    log_path = tmp_path / "fake.log"
    log_path.write_text(_fake_log_with_trend(), encoding="utf-8")
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", _fake_mems)
    d = ac._snapshot_data(round_no=22)
    # MA5 有 20 个点（最近 20 轮）
    assert len(d["ma5"]) == 20
    assert d["ma5"][0][0] == 1 and d["ma5"][-1][0] == 20
    # 拐点：先升后降处（轮 ~8）与降后升处（轮 ~13）应被识别
    tags = [t for _, _, t in d["pivots"]]
    assert "顶" in tags and "底" in tags
    # 治理/截断轮次标注
    assert d["last_govern_round"] == 21
    assert d["last_trunc_round"] == 22
    md = ac._snapshot_md(d)
    assert "**MA5 趋势线**" in md
    assert "MA5 拐点" in md
    assert "最近规则治理 轮21" in md
    assert "最近截断告警 轮22" in md


def test_snapshot_ma5_flat_no_pivots(monkeypatch, tmp_path):
    """均分平坦无拐点：MA5 存在但 pivots 空。"""
    lines = []
    for i in range(1, 11):
        lines.append(f"[t] === 第 {i} 轮 ===")
        lines.append(f"[t] 抽题: 领域「Table」题目：...")
        lines.append(f"[t] 自检验: 五维 {{语法正确性=6.0, API 规范性=6.0, 铁律遵守=6.0, "
                     f"实战可用性=6.0, 最佳实践=6.0}}")
    log_path = tmp_path / "flat.log"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(ac, "LOG_PATH", log_path)
    monkeypatch.setattr(ac, "load_ft_memory", lambda: [])
    d = ac._snapshot_data(10)
    assert len(d["ma5"]) == 10
    assert d["pivots"] == []
