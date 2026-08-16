"""记忆类型画像测试：τ / 再巩固因子 / 压缩阈值统一成一张配置表并导出。"""

import json
import os
import tempfile

from memagent.agent import AgentConfig
from memagent.memory import MemType
from memagent.profiles import TYPE_LABELS, format_profiles, type_profiles


def test_profiles_cover_all_types_in_order():
    cfg = AgentConfig()
    ps = type_profiles(cfg)
    assert [p.mtype for p in ps] == ["skill", "semantic", "episodic"]
    assert [p.label for p in ps] == ["技能", "语义", "情景"]
    assert ps[0].label == TYPE_LABELS[MemType.SKILL]


def test_profiles_reflect_custom_config():
    cfg = AgentConfig(
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
        reconsolidation_by_type={
            MemType.SKILL: {"drift": 0.15, "importance": 0.2},
            MemType.EPISODIC: {"drift": 2.5, "importance": 1.5},
        },
        cold_after_seconds=3.0,
    )
    ps = {p.mtype: p for p in type_profiles(cfg)}
    assert ps["skill"].tau_seconds == 90.0
    assert ps["episodic"].tau_seconds == 8.0
    assert ps["skill"].drift_factor == 0.15
    assert ps["episodic"].drift_factor == 2.5
    assert ps["semantic"].importance_factor == 1.0  # 未配置 → 默认 1.0


def test_cold_after_absolute_vs_derived():
    # 绝对模式：所有类型同一阈值
    abs_cfg = AgentConfig(cold_after_seconds=3.0, tau_by_type={MemType.SKILL: 90.0, MemType.EPISODIC: 8.0})
    ps = {p.mtype: p for p in type_profiles(abs_cfg)}
    assert ps["skill"].cold_after_seconds == 3.0 == ps["episodic"].cold_after_seconds
    # 按类型推导：cold_after = cold_after_tau × τ
    der_cfg = AgentConfig(cold_after_seconds=None, cold_after_tau=2.0,
                          tau_by_type={MemType.SKILL: 90.0, MemType.EPISODIC: 8.0})
    ps = {p.mtype: p for p in type_profiles(der_cfg)}
    assert ps["skill"].cold_after_seconds == 180.0
    assert ps["skill"].cold_after_tau == 2.0
    assert ps["episodic"].cold_after_seconds == 16.0


def test_to_dict_human_readable_text():
    cfg = AgentConfig(tau_by_type={MemType.SKILL: 45.0}, cold_after_seconds=3.0)
    d = type_profiles(cfg)[0].to_dict()
    assert d["tau_text"] == "45秒"
    assert d["cold_after_text"] == "3秒"
    assert set(d) >= {"mtype", "label", "tau_seconds", "drift_factor",
                      "importance_factor", "cold_after_seconds", "cold_after_tau",
                      "awakening_signal", "signal_text"}
    assert d["awakening_signal"] is None and d["signal_text"] == "无观测"


def test_profiles_with_awakening_signal():
    """传入唤醒信号统计：每类型带 direction + 一致性列文本。"""
    sig = {
        "skill": {"events": 0},
        "semantic": {"events": 2, "dominant": "down", "consistency": 1.0},
        "episodic": {"events": 3, "dominant": "up", "consistency": 0.67,
                      "dev": [0.30, 0.32, 0.33], "expected": [0.27, 0.28, 0.29],
                      "ratio_med": 1.138, "up": 2, "down": 1, "flat": 0},
    }
    ps = {p.mtype: p for p in type_profiles(AgentConfig(), sig)}
    assert ps["skill"].awakening_signal == {"events": 0}
    assert ps["skill"].to_dict()["signal_text"] == "无观测"
    assert ps["semantic"].to_dict()["signal_text"] == "↓下调·100%（2条）"
    assert ps["episodic"].to_dict()["signal_text"] == "↑上调·67%（3条）"
    s = format_profiles(AgentConfig(), sig)
    assert "唤醒信号（实测）" in s
    assert "↓下调·100%（2条）" in s and "↑上调·67%（3条）" in s and "无观测" in s


def test_format_profiles_table():
    cfg = AgentConfig(
        tau_by_type={MemType.SKILL: 45.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
        cold_after_seconds=3.0,
    )
    s = format_profiles(cfg)
    for key in ("技能", "语义", "情景", "τ", "drift", "压缩阈值", "45秒", "8秒"):
        assert key in s


def test_format_profiles_with_health():
    """传入 health 合表：追加 干净段/唤醒 方向与 一致性 三列——与仪表盘、
    CSV 合表同源（终端三处输出一致）。"""
    health = {
        "by_type": {
            "skill": {"clean": {"direction": None}, "awakening": {"direction": None},
                      "consistency": "no_data"},
            "semantic": {"clean": {"direction": "up"}, "awakening": {"direction": None},
                          "consistency": "one_sided"},
            "episodic": {"clean": {"direction": "down"}, "awakening": {"direction": "up"},
                          "consistency": "conflict"},
        },
        "summary": {},
    }
    s = format_profiles(AgentConfig(), health=health)
    assert "τ 两路一致性" in s                    # 标题标注新增列
    assert "干净段" in s and "一致性" in s
    # 情景：干净段 ↓ + 唤醒 ↑ → ✘冲突（与仪表盘/CSV 同语义）
    assert "✘冲突" in s
    # 语义：单源 → △单源；技能：无信号 → —无信号
    assert "△单源" in s and "—无信号" in s
    # 行方向箭头（情景行含 ↓ 与 ↑）
    ep_line = [ln for ln in s.splitlines() if "情景" in ln][0]
    assert "↓" in ep_line and "↑" in ep_line
    # 不传 health → 无三列（向后兼容）
    assert "干净段" not in format_profiles(AgentConfig())


def test_dashboard_embeds_profiles():
    from memagent import MemoryAgent
    from memagent.memory import MemType as _MT

    agent = MemoryAgent(cfg=AgentConfig(
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
        cold_after_seconds=3.0,
    ))
    agent.remember("我叫小林")
    for dev, exp in [(0.32, 0.28), (0.30, 0.27)]:  # episodic 上调×2
        m = agent.store.add("我昨天去吃了火锅", importance=0.1, mtype=_MT.EPISODIC)
        m.awakenings.append([1.0, dev, exp, "episodic"])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dash.html")
        agent.plot_interactive(p)
        html = open(p, encoding="utf-8").read()
        assert "记忆类型画像" in html
        assert "renderProfiles" in html
        assert "sig-badge" in html and "唤醒信号" in html
        blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
        data = json.loads(blob)
        assert len(data["profiles"]) == 3
        assert data["profiles"][0]["mtype"] == "skill"
        assert data["profiles"][2]["tau_text"] == "8秒"
        # 画像列带实测唤醒信号：episodic ↑上调、其余无观测
        assert data["profiles"][2]["signal_text"] == "↑上调·100%（2条）"
        assert data["profiles"][2]["awakening_signal"]["events"] == 2
        assert data["profiles"][0]["signal_text"] == "无观测"


def test_json_export_includes_profiles():
    from memagent import MemoryAgent

    agent = MemoryAgent(cfg=AgentConfig(tau_by_type={MemType.EPISODIC: 8.0}))
    agent.remember("我昨天去吃了火锅")
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "m")
        agent.plot_curves(base)
        with open(base + ".json", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["profiles"]) == 3
        assert data["profiles"][2]["mtype"] == "episodic"
        # 信号漂移对比一并导出（近 30 天 vs 更早，无观测时 verdict=无观测）
        assert "signal_drift" in data
        assert data["signal_drift"]["by_type"]["episodic"]["verdict"] == "无观测"
        # health 合表一并导出（与仪表盘一致：by_type + summary + warnings，
        # 含 suggest/confidence；冲突告警非空即 CI 红灯）
        assert "health" in data
        h = data["health"]
        assert set(h) == {"by_type", "summary", "warnings", "actions"}
        assert h["warnings"] == []   # 空库无冲突 → warnings 空
        assert h["actions"] == []    # 空库无行动项 → actions 空（CI 非空才需处理）
        assert h["summary"] == {"agree": 0, "conflict": 0, "one_sided": 0,
                                "no_data": 3}
        ep = h["by_type"]["episodic"]
        assert set(ep) == {"clean", "awakening", "consistency", "suggest",
                           "confidence"}
        assert ep["consistency"] == "no_data" and ep["suggest"] == "无信号"
        assert ep["confidence"] == "—"


def test_dashboard_embeds_signal_drift():
    """仪表盘数据带 signal_drift（时间窗对比），JS 渲染漂移提示行。"""
    from memagent import MemoryAgent
    from memagent.memory import MemType as _MT

    agent = MemoryAgent(now_fn=lambda: 40 * 86400)
    agent.remember("我叫小林")
    for ts, dev, exp in [(5, 0.20, 0.25), (10, 0.21, 0.26),
                         (32, 0.32, 0.28), (35, 0.30, 0.27)]:
        m = agent.store.add("我昨天去吃了火锅", importance=0.1, mtype=_MT.EPISODIC)
        m.awakenings.append([ts * 86400, dev, exp, "episodic"])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dash.html")
        agent.plot_interactive(p)
        html = open(p, encoding="utf-8").read()
        blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
        data = json.loads(blob)
        ep = data["signal_drift"]["by_type"]["episodic"]
        assert ep["verdict"] == "方向翻转"
        assert data["signal_drift"]["recent_seconds"] == 30 * 86400
        assert "sig-drift" in html
        assert "天 vs 更早" in html  # JS 运行时由 recent_seconds 拼出「近30天 vs 更早」


def test_dashboard_embeds_tau_health():
    """仪表盘数据带 τ 两路信号健康检查（health），画像面板渲染干净段/唤醒方向
    与一致性徽章（单一事实源 agent.tau_learner_health）。"""
    from memagent import MemoryAgent
    from memagent.memory import MemType as _MT

    agent = MemoryAgent(cfg=AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * 86400.0},
        true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
    ), now_fn=lambda: 0.0)
    m = agent.store.add("两路同向的决策记忆", importance=0.3, mtype=_MT.EPISODIC)
    m.access_count = 0
    m.last_access = 0.0
    t = 0.0
    for _ in range(4):   # 3 个干净段 → 反推 τ≈2 天 < 配置 3 天 → down
        s = agent.strength_at_state(_MT.EPISODIC, 0.0, 0, 0.1, t,
                                    tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.5, 0.4, "episodic"])    # ratio > 1 → down
    m.awakenings.append([2.0, 0.45, 0.38, "episodic"])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dash.html")
        agent.plot_interactive(p)
        html = open(p, encoding="utf-8").read()
        blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
        data = json.loads(blob)
        h = data["health"]
        assert set(h) == {"by_type", "summary", "warnings", "actions"}
        assert h["warnings"] == []   # 无冲突类型 → 空（CI 按非空判定红灯）
        # actions = 行动清单（与终端/CSV suggest_adjust 同源）：episodic τ↓ 在列
        assert h["actions"] == [{"mtype": "episodic", "suggest": "τ↓",
                                  "confidence": "弱",
                                  "reason": "配置偏大 · 忘得比信念快"}]
        ep = h["by_type"]["episodic"]
        assert ep["consistency"] == "agree" and ep["suggest"] == "τ↓"
        assert ep["confidence"] == "弱"       # 唤醒 2 条 < 门控 → 弱
        assert ep["clean"]["direction"] == "down" and ep["clean"]["n"] == 3
        assert ep["awakening"]["direction"] == "down"
        assert h["by_type"]["skill"]["consistency"] == "no_data"
        assert h["by_type"]["skill"]["confidence"] == "—"
        # JS 渲染：三列头 + 徽章映射 + 建议 tooltip（运行时由 by_type 拼出）
        assert "干净段" in html and "唤醒" in html and "一致性" in html
        assert "✔ 一致" in html and "✘ 冲突" in html and "建议: " in html
        assert "MEM.health.by_type" in html
        # 行动徽章列（suggest_adjust 接进画像面板）：τ↓红 / τ↑青 / 需检查橙
        assert "行动" in html and "act-badge" in html
        assert "#e34a2f" in html and "#2a9d8f" in html and "#e8590c" in html
        assert "τ 应下调（配置偏大）" in html and "两路信号冲突，先排查再调参" in html
        # 与信号漂移行联动：徽章/漂移项都带 data-mt，JS 有 toggleDriftLink
        assert 'data-mt="' in html and "toggleDriftLink" in html
        assert "drift-item" in html and "linked" in html
        # 一致性徽章 → 主图类型唤醒联动（✘ 冲突 = 定位哪几起事件造成两路冲突）
        assert 'cs-badge" data-mt=' in html       # 一致性徽章带类型
        assert "linkTypeAwakenings" in html and "showTypeCallout" in html
        assert "type-linked" in html and "type-linked-mode" in html
        assert "与干净段相反" in html and ".act-row" in html


def test_dashboard_warn_row_evidence():
    """冲突类型行 ⚠ 高亮 + 点击展开两路证据行（health.warnings 同源）：
    警告行/证据行/⚠ 标记模板、setWarnEv 切换函数、与一致性徽章联动接线。"""
    from memagent import MemoryAgent
    from memagent.memory import MemType as _MT

    agent = MemoryAgent(cfg=AgentConfig(
        tau_by_type={_MT.EPISODIC: 3 * 86400.0},
        true_tau_by_type={_MT.EPISODIC: 2 * 86400.0},
    ), now_fn=lambda: 0.0)
    m = agent.store.add("冲突决策记忆", importance=0.3, mtype=_MT.EPISODIC)
    m.access_count = 0
    m.last_access = 0.0
    t = 0.0
    for _ in range(4):   # 3 个干净段 → 反推 τ≈2 天 < 配置 3 天 → down
        s = agent.strength_at_state(_MT.EPISODIC, 0.0, 0, 0.1, t,
                                    tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.50, 0.62, "episodic"])   # ratio < 1 → up（冲突）
    m.awakenings.append([2.0, 0.45, 0.58, "episodic"])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dash.html")
        agent.plot_interactive(p)
        html = open(p, encoding="utf-8").read()
        blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
        data = json.loads(blob)
        # 数据带 warnings（episodic 冲突：两路证据 + 建议 + 冲突成因事件）
        w = data["health"]["warnings"]
        assert len(w) == 1 and w[0]["mtype"] == "episodic"
        assert "干净段 3 条" in w[0]["clean_evidence"]
        assert "中位比值" in w[0]["awakening_evidence"]
        assert "事件注入" in w[0]["suggestion"]
        # 冲突成因事件：index（主图 data-evi 同语义）+ CSV 行号（与导出同算法）
        evs = w[0]["events"]
        assert len(evs) == 2
        assert [e["row"] for e in evs] == [2, 3]
        assert [e["index"] for e in evs] == [0, 1]
        assert all(e["direction"] == "up" for e in evs)
        assert set(evs[0]) >= {"memory_id", "index", "ts", "dev",
                               "expected", "ratio", "direction", "row"}
        # 模板：警告行（橙边+⚠ 标记）+ 隐藏证据行（colspan + 两路证据 + 建议）
        assert "warn-row" in html and "warn-ev" in html and "warn-mark" in html
        assert 'tr.warn-row' in html and 'tr.warn-ev' in html
        assert "两路信号冲突" in html
        assert 'class="warn-row"' in html
        # JS：setWarnEv 切换函数 + 警告行点击绑定 + 一致性徽章联动展开（linkTypeAwakenings 内 setWarnEv）
        assert "function setWarnEv" in html
        assert "tr.warn-row').forEach" in html
        assert "setWarnEv(mt, true)" in html and "setWarnEv(mt, false)" in html
        assert "setWarnEv(tr.dataset.mt)" in html
        # Esc/deselect 收起所有证据行
        assert "tr.warn-ev').forEach" in html
        # 证据行内冲突事件：wev-evts 模板 + warn-ev-row（行号 + 记忆 id + data-evi）
        # + 点击绑定 showAwakening 定位主图对应唤醒点
        assert ".wev-evts" in html and ".warn-ev-row" in html
        assert "warn-ev-row" in html
        assert "[行 " in html and "wev-rowno" in html
        assert "data-evi=\"" in html
        assert "showAwakening(row.dataset.mem" in html
        assert "点击定位主图对应唤醒点" in html
        # Shift 多选聚合：wev-aggr 占位 + 提示 + 聚合函数/判定文案
        assert 'class="wev-aggr" data-mt=' in html
        assert "Shift 点击多选聚合" in html and "Shift 点击多选" in html
        assert "function toggleConflictSel" in html
        assert "function renderConflictAggregate" in html
        assert "function conflictEvKey" in html and "function _ratioMed" in html
        assert "✔ 移除后两路一致——冲突消除" in html
        assert "✘ 移除后仍冲突" in html and "剩余观测不足，无法判定" in html
        assert "renderConflictAggregate(box.dataset.mt)" in html
        # 聚合面板快捷操作：全选/清空按钮 + 全体方向占比
        assert "aggr-btns" in html and "aggr-btn" in html
        assert "aggr-all" in html and "aggr-clear" in html
        assert "全选" in html and "清空" in html
        assert "function selectAllConflict" in html and "function clearConflictSel" in html
        assert "function syncSelVisual" in html
        # 方向占比可视化条：三段色条（↑青/↓红/＝灰）等比例显示取代纯文本百分比，
        # 图例保留计数、title 悬浮保留精确 n/N (pct%)
        assert "方向分布（全体" in html
        assert "function _distBar" in html
        assert "aggr-distbar" in html and "aggr-distlegend" in html
        assert "aggr-seg-up" in html and "aggr-seg-down" in html and "aggr-seg-flat" in html
        assert "Math.round(n / total * 100)" in html
        assert "_distBar(distAll, total.length" in html
        assert "style=\"width:' + pct(n) + '%\"" in html
        # 第二条色条（选中集分布）+ 全体条选中段描边高亮（选反向/全选/清空同步）
        assert "选中集分布（已选 " in html
        assert "_distBar(dist, nSel, '选中集分布" in html
        assert "aggr-sel" in html and "inset 0 0 0 2px #1f6feb" in html
        assert "selDist ? (selDist[s[0]] || 0) : 0" in html
        assert "' · 选中 ' + selN + ' 起'" in html
        # 色段可点击：data-dir + selectDirConflict（只圈该方向事件，与选反向互补）
        assert "aggr-seg-btn" in html and "data-dir=\"" in html
        assert "function selectDirConflict" in html
        assert "selectDirConflict(seg.dataset.mt, seg.dataset.dir)" in html
        assert "if (_evDir(ev) === dir)" in html
        assert "点击色段只圈出该方向事件" in html
        assert "aggr-seg-btn:hover" in html
        assert "selectAllConflict(mt)" in html and "clearConflictSel(mt)" in html
        # 「选反向」：一键圈定与干净段相反方向的事件（dir ≠ clean）——
        # 与 clashes 标橙共用同一判定（_evDir 抽取后两处同源）
        assert "aggr-clash" in html
        assert "选反向" in html
        assert "function selectClashConflict" in html
        assert "selectClashConflict(mt)" in html
        assert "与 clashes 标橙同一判定" in html
        assert "function _evDir" in html
        assert "const dir = _evDir(ev)" in html
        assert "clean && dir && dir !== clean" in html
        # 全选/清空/选反向/Shift 多选状态互通：统一刷新路径（refreshConflictSel = 高亮
        # + 聚合面板重渲染 + callout CSV 预览徽章），面板内嵌选中集 CSV 行预览
        assert "function refreshConflictSel" in html
        assert "function syncCsvPreview" in html and "function _setCsvSelBadge" in html
        assert "function _selCsvBlock" in html
        assert "aggr-csvsel" in html and "aggr-csvrow" in html
        assert "选中集 CSV 行预览" in html
        assert "已选 ✓" in html and "'未选'" in html
        assert "refreshConflictSel(mt)" in html
        assert "syncSelVisual(mt);\n  renderConflictAggregate(mt);\n  syncCsvPreview(mt);" in html
        # deselect/Esc 清空多选状态后重渲染聚合面板（面板回到 0 选中基线）
        assert "delete selConflictEvts[k]; });" in html
        assert "Object.keys(selConflictEvts).forEach(function (k) { delete selConflictEvts[k]; });\n  document.querySelectorAll('.wev-aggr')" in html
        # CSV 行预览：点击事件 → callout 展示原始 CSV 行（行号 + 全字段）
        assert "function csvRowPreviewHtml" in html
        assert 'data-row="' in html and "(ev.row || '')" in html
        assert "csvRowPreviewHtml(mem, ev, csvRow)" in html
        assert "原始 CSV 行预览" in html and "csv-row-box" in html
        assert "memory_id, mtype, ts, ts_relative_seconds" in html
        assert "dt_seconds, retrievals_before" in html
        assert "csv-line" in html and "csv-cell" in html
        assert "展示 CSV 行" in html
        # 非冲突类型无警告行（skill 行不带 warn-row）
        assert data["health"]["by_type"]["skill"]["consistency"] == "no_data"
        assert data["health"]["warnings"] and data["health"]["warnings"][0]["mtype"] == "episodic"


def test_dashboard_aggregations_replay():
    """health.aggregations 接进仪表盘：plot_interactive(aggregations=...) 回放
    历史聚合结论（verdict 徽章 + resolved 自动附带剔除后证据包），点击主图高亮
    事件子集（linkAggregation / showAggCallout / clearAggLink / deselect 接线）。"""
    from memagent import MemoryAgent
    from memagent.memory import MemType as _MT

    agent = MemoryAgent(cfg=AgentConfig(
        tau_by_type={_MT.EPISODIC: 3 * 86400.0},
        true_tau_by_type={_MT.EPISODIC: 2 * 86400.0},
    ), now_fn=lambda: 0.0)
    m = agent.store.add("冲突决策记忆", importance=0.3, mtype=_MT.EPISODIC)
    m.access_count = 0
    m.last_access = 0.0
    t = 0.0
    for _ in range(4):
        s = agent.strength_at_state(_MT.EPISODIC, 0.0, 0, 0.1, t,
                                    tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.50, 0.62, "episodic"])   # 0.806 up
    m.awakenings.append([2.0, 0.45, 0.58, "episodic"])   # 0.776 up
    m2 = agent.store.add("正常情景记忆", importance=0.2, mtype=_MT.EPISODIC)
    m2.awakenings.append([1.5, 0.4, 0.32, "episodic"])   # 1.25 down
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "dash.html")
        agent.plot_interactive(p, aggregations=[
            {"mtype": "episodic", "events": [f"{m.id}:0", f"{m.id}:1"]},
            {"mtype": "episodic", "events": [f"{m2.id}:0"]},
        ])
        html = open(p, encoding="utf-8").read()
        blob = html.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
        data = json.loads(blob)
        aggs = data["health"]["aggregations"]
        assert len(aggs) == 2
        r, s_ = aggs[0], aggs[1]
        assert r["verdict"] == "resolved" and r["events"] == [f"{m.id}:0", f"{m.id}:1"]
        # 方向占比分布（复刻仪表盘色条）随回放嵌入
        assert r["all_dist"] == {"up": 2, "down": 1, "flat": 0} and r["all_n"] == 3
        assert r["all_dist_pct"] == {"up": 67, "down": 33, "flat": 0}
        assert r["selected_dist_pct"] == {"up": 100, "down": 0, "flat": 0}
        rc = r["recomputed"]
        assert rc["excluded"] == [f"{m.id}:0", f"{m.id}:1"]
        assert rc["health"]["by_type"]["episodic"]["consistency"] == "agree"
        assert rc["health"]["warnings"] == []
        assert s_["verdict"] == "still_conflict" and "recomputed" not in s_
        # 面板模板：verdict 徽章 + 事件子集摘要 + 剔除后证据包行
        assert "agg-hist" in html and "agg-hist-row" in html
        assert "历史聚合结论（health.aggregations" in html
        assert "✔ 冲突消除" in html and "✘ 仍冲突" in html
        assert "已选 ' + a.selected_n + ' 起" in html and "agg-rec" in html and "剔除后" in html
        # JS：linkAggregation 高亮事件子集（data-mem + data-evi 精确匹配）
        assert "function linkAggregation" in html
        assert "function showAggCallout" in html and "function clearAggLink" in html
        assert "linkAggregation(parseInt(row.dataset.idx, 10))" in html
        assert ".awake-mark[data-mem=\"" in html and "data-evi=\"' + i" in html
        assert "aggLinked === idx" in html
        # 一键导出选择集：aggr-export 按钮 + aggregationsExportPayload / downloadAggregations
        # （--aggregations-file 直接读取，免手写 memory_id 列表）
        assert "aggr-export" in html and "导出聚合" in html
        assert "function aggregationsExportPayload" in html
        assert "function downloadAggregations" in html
        assert "downloadAggregations()" in html
        assert "--aggregations-file 直接读取" in html
        assert "{ mtype: mt, events: Array.from(s).sort() }" in html
        assert "a.download = 'aggregations.json'" in html
        # 复制到剪贴板选项（覆盖无下载权限环境）：clipboard API + execCommand 回退
        assert "aggr-copy" in html and "复制" in html
        assert "function copyAggregations" in html and "function legacyCopy" in html
        assert "copyAggregations(bCpy)" in html
        assert "navigator.clipboard && navigator.clipboard.writeText" in html
        assert "document.execCommand('copy')" in html
        assert "已复制 ✓" in html
        # 选反向选择集一键生成 --exclude-events 参数串 + 证据行顶部复制按钮
        assert "wev-copy-excl" in html and "复制 --exclude-events" in html
        assert "function excludeEventsParam" in html
        assert "function copyExcludeEventsParam" in html and "function _flashBtn" in html
        assert "copyExcludeEventsParam(b)" in html
        assert "keys.sort().join(',')" in html
        assert "--exclude-events 参数: <code>' +" in html
        # deselect/Esc 清空聚合联动
        assert "aggLinked = null;" in html
        assert "classList.remove('linked')" in html
        # 不带 aggregations → health 无 aggregations 键（向后兼容）
        p2 = os.path.join(td, "dash2.html")
        agent.plot_interactive(p2)
        html2 = open(p2, encoding="utf-8").read()
        blob2 = html2.split('<script id="memdata" type="application/json">')[1].split("</script>")[0]
        assert "aggregations" not in json.loads(blob2)["health"]
