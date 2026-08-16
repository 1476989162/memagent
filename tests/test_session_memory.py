"""会话记忆测试：提交提炼、去重强化、注入排序、主题检索、非 git 降级、CLI。"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import session_memory
from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType

SCRIPT = Path(__file__).resolve().parents[1] / "session_memory.py"

COMMITS = [
    ("a1b2c3", "再巩固因子学习器改用中位数抗离群事件"),
    ("d4e5f6", "语义化迁移增加双阈值滞回"),
]


def test_record_from_commits():
    agent = MemoryAgent()
    n = session_memory.record(agent, COMMITS, notes=[])
    assert n == 2
    contents = [m.content for m in agent.store.all()]
    assert "开发决策：再巩固因子学习器改用中位数抗离群事件" in contents
    assert "开发决策：语义化迁移增加双阈值滞回" in contents


def test_record_notes_and_dedup_strengthens():
    """同一决策重复沉淀 → 去重合并（条数不变）+ 测试效应强化（次数增加）。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=["重要：实验脚本用可注入时钟"])
    session_memory.record(agent, COMMITS, notes=["重要：实验脚本用可注入时钟"])
    assert len(agent.store.all()) == 3  # 2 提交 + 1 note，重复记录不新增
    note = next(m for m in agent.store.all() if "可注入时钟" in m.content)
    assert note.access_count >= 1  # 去重命中 = 强化（测试效应）


def test_inject_block_shows_summary_for_cold(capsys):
    """Cold 记忆在注入块里展示摘要而非深藏 content（与核心 /memories 一致）。"""
    agent = MemoryAgent()
    m = agent.store.add("开发决策：旧版对比实验的原始内容", importance=0.5)
    m.demote_to_cold("开发决策：遗忘斜率对比用触底时间")
    session_memory.inject_block(agent, topic=None, k=5)
    out = capsys.readouterr().out
    assert "遗忘斜率对比用触底时间" in out
    assert "旧版对比实验的原始内容" not in out


def test_short_topic_hint_counts_summary_matches(capsys):
    """短主题提示的含词计数覆盖摘要（与核心 content+summary 检查一致）——
    「触底」只在摘要里，计数仍为 1/1。"""
    agent = MemoryAgent()
    m = agent.store.add("开发决策：对照实验", importance=0.2)
    m.demote_to_cold("开发决策：触底时间")
    session_memory.pick_decisions(agent, topic="触底", k=5)
    out = capsys.readouterr().out
    assert "[1/1 条含主题词]" in out


def test_short_topic_substring_rerank(capsys):
    """短主题（<3 字）子串优先重排：含主题词的决策排在不含的前面（消除哈希碰撞干扰）。"""
    agent = MemoryAgent()
    session_memory.record(agent, [], [
        "遗忘斜率对比用触底时间而非每 τ 斜率比",      # 含「触底」
        "对照实验靠可注入时钟 now_fn 确定性快进",      # 不含但可能与「触底」哈希碰撞
        "贴合度 fit = 1 − |实测τ − 配置τ| / 配置τ",
    ])
    picked = session_memory.pick_decisions(agent, topic="触底", k=5)
    # 含主题词的（触底时间）必须排在最前
    assert "触底时间" in picked[0][0].content, [m.content for m, _ in picked]
    # 顺序：所有含词的在所有不含词的前面（无交错）
    flags = ["触底" in m.content for m, _ in picked]
    assert flags == sorted(flags, reverse=True)
    out = capsys.readouterr().out
    assert "子串优先重排" in out and "建议加长" in out


def test_short_topic_rerank_only_orders_retrieved(capsys):
    """短主题重排只调顺序：返回的都是检索命中的（rel 过滤不变），含词的排最前。"""
    agent = MemoryAgent()
    session_memory.record(agent, [], ["遗忘斜率对比用触底时间", "贴合度 fit 公式"])
    picked = session_memory.pick_decisions(agent, topic="触底", k=2)
    assert picked, "应至少命中含触底的决策"
    assert "触底时间" in picked[0][0].content
    assert all(m.content in {"开发决策：遗忘斜率对比用触底时间", "开发决策：贴合度 fit 公式"}
               for m, _ in picked)
    capsys.readouterr()


def test_normal_topic_no_rerank(capsys):
    """长主题（≥3 字）不重排、不打印提示，仍按 rel 排序（top1 精准命中）。"""
    agent = MemoryAgent()
    session_memory.record(agent, [], ["语义化双阈值滞回避免振荡", "循环导入用函数级导入"])
    picked = session_memory.pick_decisions(agent, topic="语义化", k=2)
    assert "语义化" in picked[0][0].content
    out = capsys.readouterr().out
    assert "子串优先重排" not in out


def test_start_injection_ranked_by_strength():
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    picked = session_memory.inject_block(agent, topic=None, k=2)
    assert len(picked) == 2
    assert picked[0][1] >= picked[1][1]  # 强度降序


def test_start_injection_by_topic():
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    picked = session_memory.inject_block(agent, topic="再巩固因子", k=5)
    contents = [m.content for m, _s in picked]
    assert any("再巩固因子" in c for c in contents)  # 主题检索命中


def test_get_recent_commits_not_a_repo(tmp_path):
    assert session_memory.get_recent_commits(cwd=str(tmp_path)) == []  # 无 git 仓库 → 降级


def test_write_context_file(tmp_path):
    """生成独立注入 prompt 文件：含注入决策与使用说明。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    p = session_memory.write_context_file(agent, str(tmp_path / "ctx.md"), topic="再巩固", k=2)
    text = open(p, encoding="utf-8").read()
    assert "# memagent 会话上下文" in text
    assert "再巩固因子学习器改用中位数抗离群事件" in text
    assert "--inject-agents-md" in text  # 提示如何维护 AGENTS.md


def test_inject_into_agents_md_creates(tmp_path):
    """AGENTS.md 不存在 → 创建，含 marker 区块。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    p = session_memory.inject_into_agents_md(agent, str(tmp_path / "AGENTS.md"), k=2)
    text = open(p, encoding="utf-8").read()
    assert session_memory.INJECT_MARKER_START in text
    assert session_memory.INJECT_MARKER_END in text
    assert "再巩固因子学习器改用中位数抗离群事件" in text


def test_inject_into_agents_md_replaces_not_duplicates(tmp_path):
    """二次注入替换旧区块（marker 单一），不重复堆积。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    p = str(tmp_path / "AGENTS.md")
    session_memory.inject_into_agents_md(agent, p, k=1)
    session_memory.inject_into_agents_md(agent, p, k=2)
    text = open(p, encoding="utf-8").read()
    assert text.count(session_memory.INJECT_MARKER_START) == 1
    assert text.count("开发决策：") == 2  # 只有最新一次的注入条数


def test_inject_into_agents_md_inserts_at_top_of_existing(tmp_path):
    """已有 AGENTS.md（无 marker）→ 区块插入顶部，原内容保留。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    p = str(tmp_path / "AGENTS.md")
    open(p, "w", encoding="utf-8").write("# 原有项目指令\n- 已有规则\n")
    session_memory.inject_into_agents_md(agent, p, k=1)
    text = open(p, encoding="utf-8").read()
    assert text.index(session_memory.INJECT_MARKER_START) < text.index("原有项目指令")
    assert "已有规则" in text  # 原内容保留


def test_export_agents_md_full_document(tmp_path):
    """全量导出：按类型分组、带标注、含全部决策（非 top-k 注入）。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=["可塑性学习器用中位数抗离群"])
    p = session_memory.export_agents_md(agent, str(tmp_path / "AGENTS.md"))
    text = open(p, encoding="utf-8").read()
    assert "# 项目决策记忆（由 memagent 自动生成）" in text
    assert "再巩固因子学习器改用中位数抗离群事件" in text
    assert "可塑性学习器用中位数抗离群" in text
    assert "## semantic" in text or "## skill" in text
    assert "检索 0 次" in text  # 标注
    assert text.count("开发决策：") == 3  # 全部导出，非 top-k


def test_export_agents_md_dual_writes_both_in_sync(tmp_path, monkeypatch):
    """双格式：同一份内容同时写入 AGENTS.md + CLAUDE.md（逐字节一致，保持同步）。"""
    monkeypatch.chdir(tmp_path)
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=["可塑性学习器用中位数抗离群"])
    exports = session_memory.export_agents_md(agent, dual=True)
    assert set(exports) == {"AGENTS.md", "CLAUDE.md"}
    a = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    c = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert a == c
    assert "可塑性学习器用中位数抗离群" in a
    assert "同时刷新" in a and "CLAUDE.md" in a


def test_export_agents_md_single_path_only(tmp_path, monkeypatch):
    """显式文件名 → 只写该文件，不生成 AGENTS.md/CLAUDE.md（向后兼容）。"""
    monkeypatch.chdir(tmp_path)
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=[])
    p = session_memory.export_agents_md(agent, str(tmp_path / "custom.md"))
    assert p == str(tmp_path / "custom.md")
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()


def test_cli_export_agents_md_dual_no_arg(tmp_path):
    """CLI 不带文件名 → 同时生成 AGENTS.md + CLAUDE.md（内容同步），不污染项目根。"""
    persist = str(tmp_path / "mem.json")
    subprocess.run(
        [sys.executable, "session_memory.py", "--record", "--note", "再巩固因子学习器用中位数抗离群事件",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    e = subprocess.run(
        [sys.executable, str(SCRIPT), "--export-agents-md", "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=tmp_path,
    )
    assert e.returncode == 0, e.stdout + e.stderr
    assert "AGENTS.md + CLAUDE.md" in e.stdout
    a = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    c = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    assert a == c
    assert "再巩固因子学习器用中位数抗离群事件" in a


def test_eval_agents_md_coverage(tmp_path):
    """评估：文档含全部答案关键词 → 100% 可及；缺关键词 → 标出缺失。"""
    agent = MemoryAgent()
    session_memory.record(agent, [("a1", "再巩固可塑性 = 1 − importance，importance ≥ 0.8 完全冻结")], notes=[])
    # 补齐其余问题的答案关键词
    extra = [
        "开发决策：可塑性学习器用中位数聚合实测因子抗离群事件",
        "开发决策：语义化双阈值滞回（3.0 / 0.8）避免来回振荡",
        "开发决策：遗忘斜率对比用触底时间而非每 τ 斜率比",
        "开发决策：技能类一致性校验钩子：技能回忆是验证不是吸收情境",
        "开发决策：查询同义扩展：人称互换 + 同义词替换",
        "开发决策：对照实验靠可注入时钟 now_fn 确定性快进",
        "开发决策：贴合度 fit = 1 − |实测τ − 配置τ| / 配置τ",
    ]
    for n in extra:
        agent.remember(n)
    p = session_memory.export_agents_md(agent, str(tmp_path / "AGENTS.md"))
    report = session_memory.eval_agents_md(agent, path=p)
    assert all(r["ok"] for r in report["coverage"]), report["coverage"]


def test_eval_recall_chain_verdict(capsys):
    """唤醒链路检查：判别场景全过（无缝衔接），结论输出完整。"""
    v = session_memory.eval_recall_chain()
    assert v["prefix_ok"] and v["tail_decays"]
    assert v["n_before"] == 5 and v["n_after"] == 8  # 创建采样 1 + 手动 4；继承 5 + 唤醒 1 + 尾部 2
    out = capsys.readouterr().out
    assert "== 唤醒链路连续性检查 ==" in out
    assert "无缝衔接" in out
    assert "唤醒点" in out
    assert "⑤ 重建段尾部继续衰减: True" in out


def test_eval_recall_chain_reports_store_awakened(capsys):
    """真实记忆库统计：唤醒过的长生命周期记忆计入报告，唤醒信号节照常输出。"""
    a = MemoryAgent()
    m = a.store.add("我昨天去吃了火锅", importance=0.1)
    m.demote_to_cold("我昨天去吃了火锅")
    a.recall(m.id[:6])
    v = session_memory.eval_recall_chain(a)
    assert v["awakened_in_store"] == 1
    out = capsys.readouterr().out
    assert "长生命周期记忆 1 条" in out
    assert "⑦ 唤醒偏差信号" in out
    assert "episodic" in out and "1 条" in out     # recall 产生 1 条自洽观测


def _signal_clock_agent(now_days: float) -> "MemoryAgent":
    """构造带时间轴的唤醒信号 agent：episodic 早期下调×2、近期上调×3。"""
    clock = [now_days * 86400]
    a = MemoryAgent(now_fn=lambda: clock[0])
    for ts, dev, exp in [(5, 0.20, 0.25), (10, 0.21, 0.26),
                         (32, 0.32, 0.28), (35, 0.30, 0.27), (38, 0.33, 0.29)]:
        m = a.store.add("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
        m.awakenings.append([ts * 86400, dev, exp, "episodic"])
    return a


def test_awakening_signal_stats_window_seconds():
    """时间窗参数：只看最近 N 天内的唤醒事件（相对 now），绝对方可回退。"""
    a = _signal_clock_agent(40)
    all_s = session_memory.awakening_signal_stats(a)
    assert all_s["episodic"]["events"] == 5
    # 近 30 天 → since=10 天，边界含 ts=10（第 2 条下调）→ 4 条（1 下 3 上）
    w30 = session_memory.awakening_signal_stats(a, window_seconds=30 * 86400)
    assert w30["episodic"]["events"] == 4 and w30["episodic"]["dominant"] == "up"
    # 近 10 天 → 3 条全上调
    w10 = session_memory.awakening_signal_stats(a, window_seconds=10 * 86400)
    assert w10["episodic"]["events"] == 3 and w10["episodic"]["up"] == 3
    assert w10["episodic"]["dev"] == [0.30, 0.32, 0.33]
    # 绝对窗口：since/until 组合出任意时段（第 1 条）
    seg = session_memory.awakening_signal_stats(a, since=0.0, until=6 * 86400)
    assert seg["episodic"]["events"] == 1 and seg["episodic"]["dominant"] == "down"
    # 无观测类型 window 后仍 {"events": 0}
    assert w10["skill"] == {"events": 0}


def test_awakening_signal_periods_verdicts():
    """时段对比判定：方向翻转（早期下调 → 近期上调）、无观测、仅早期、稳定。"""
    from memagent.agent import awakening_signal_periods

    a = _signal_clock_agent(40)
    p = awakening_signal_periods(a, recent_seconds=30 * 86400)
    ep = p["by_type"]["episodic"]
    assert ep["verdict"] == "方向翻转" and ep["direction_changed"]
    assert ep["recent"]["events"] == 4 and ep["earlier"]["events"] == 1
    assert ep["recent"]["dominant"] == "up" and ep["earlier"]["dominant"] == "down"
    assert p["by_type"]["skill"]["verdict"] == "无观测"
    assert p["by_type"]["semantic"]["verdict"] == "无观测"
    # 稳定：两段同向（近 60 天窗口 → 早期为空？不——用 15 天：近 15 天 3 上调、
    # 更早 2 下调仍翻转；改用全部上调的窗口验证稳定分支）
    b = MemoryAgent(now_fn=lambda: 40 * 86400)
    for ts, dev, exp in [(5, 0.32, 0.28), (10, 0.30, 0.27), (35, 0.33, 0.29)]:
        m = b.store.add("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
        m.awakenings.append([ts * 86400, dev, exp, "episodic"])
    pb = awakening_signal_periods(b, recent_seconds=15 * 86400)
    e2 = pb["by_type"]["episodic"]
    assert e2["verdict"] == "稳定" and not e2["direction_changed"]
    assert e2["consistency_delta"] == 0.0
    # 仅早期有观测
    c = MemoryAgent(now_fn=lambda: 40 * 86400)
    m = c.store.add("北京是中国的首都", importance=0.1, mtype=MemType.SEMANTIC)
    m.awakenings.append([8 * 86400, 0.20, 0.25, "semantic"])
    pc = awakening_signal_periods(c, recent_seconds=30 * 86400)
    assert pc["by_type"]["semantic"]["verdict"] == "仅早期有观测"


def test_signal_drift_table_cli():
    """CLI /signal 表格：两段方向、判定标记与判读文本齐全。"""
    a = _signal_clock_agent(40)
    s = a.signal_drift_table(30)
    assert "唤醒信号漂移（近 30 天 vs 更早" in s
    assert "↑上调·75%（4条）" in s and "↓下调·100%（1条）" in s
    assert "⚠ 方向翻转" in s
    assert "✔ 无观测" in s


def test_awakening_signal_stats_per_type():
    """唤醒信号统计：按类型聚合 dev/expected 分布与方向一致性，旧格式三元组跳过。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    for dev, exp in [(0.32, 0.28), (0.30, 0.27), (0.33, 0.29)]:  # episodic 全上调
        m = a.store.add("我昨天去吃了火锅", importance=0.1)
        m.awakenings.append([1.0, dev, exp, "episodic"])
    m = a.store.add("北京是中国的首都", importance=0.1)
    m.awakenings.append([1.0, 0.20, 0.25, "semantic"])          # semantic 下调
    m.awakenings.append([1.0, 0.21, 0.26, "semantic"])
    m.awakenings.append([1.0, 0.30, "semantic"])                # 旧格式 → 跳过
    s = session_memory.awakening_signal_stats(a)
    ep = s["episodic"]
    assert ep["events"] == 3 and ep["up"] == 3 and ep["down"] == 0
    assert ep["dominant"] == "up" and ep["consistency"] == 1.0
    assert ep["dev"] == [0.30, 0.32, 0.33]       # 中位 0.32
    assert ep["expected"] == [0.27, 0.28, 0.29]  # 中位 0.28
    # ratio 中位数 = 各事件 dev/expected 的中位数（非中位之比）：
    # 1.1111 / 1.1379 / 1.1428 → 1.138
    assert ep["ratio_med"] == 1.138
    se = s["semantic"]
    assert se["events"] == 2 and se["down"] == 2 and se["dominant"] == "down"
    assert s["skill"]["events"] == 0


def _two_signal_agent(clean_dir=None, aw_dir=None, n_clean=3):
    """构造带两路信号的 agent：干净段（真实 τ=2天 < 配置 3天 → down）+
    唤醒观测（可配方向）。clean_dir=None 则不造干净段；aw_dir 控制唤醒方向。"""
    cfg = AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * 86400.0},
        true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
    )
    a = MemoryAgent(cfg=cfg, now_fn=lambda: 0.0)
    mem = a.store.add("我昨天去看了场电影", importance=0.1, mtype=MemType.EPISODIC)
    mem.access_count = 0
    mem.last_access = 0.0
    if clean_dir is not None:
        t = 0.0
        for _ in range(n_clean + 1):  # n_clean+1 个采样 → n_clean 个干净段
            s = a.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                    tau_override=2 * 86400.0)
            mem.history.append([t, round(s, 4), 0.0, 0, 0.1])
            t += 0.5 * 86400.0
    if aw_dir == "down":
        mem.awakenings.append([1.0, 0.5, 0.4, "episodic"])
        mem.awakenings.append([2.0, 0.45, 0.38, "episodic"])
    elif aw_dir == "up":
        mem.awakenings.append([1.0, 0.2, 0.25, "episodic"])
        mem.awakenings.append([2.0, 0.21, 0.26, "episodic"])
    elif aw_dir == "flat":
        mem.awakenings.append([1.0, 0.4, 0.4, "episodic"])
    return a


def test_tau_learner_health_agree():
    """两路同向（实测 τ < 配置 → 应下调；唤醒 ratio > 1 → 应下调）→ 一致。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")
    h = session_memory.tau_learner_health(a)
    ep = h["by_type"]["episodic"]
    assert ep["clean"]["direction"] == "down" and ep["clean"]["n"] == 3
    assert ep["awakening"]["direction"] == "down"
    assert ep["consistency"] == "agree"
    assert h["summary"]["agree"] == 1


def test_tau_learner_health_conflict_and_flat():
    """两路反向 → 冲突；两路都无偏差（ratio=1、实测τ≈配置）→ 一致（已校准）。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="up")
    h = session_memory.tau_learner_health(a)
    assert h["by_type"]["episodic"]["consistency"] == "conflict"
    assert h["summary"]["conflict"] == 1
    # 校准场景：配置 τ=3 天、真实 2 天但采样间隔极短 → 实测τ≈3 天？不——用干净段
    # 直接构造 flat 更可控：让干净段真实 τ == 配置 τ
    cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 2 * 86400.0})
    a2 = MemoryAgent(cfg=cfg, now_fn=lambda: 0.0)
    m = a2.store.add("x", importance=0.1, mtype=MemType.EPISODIC)
    m.access_count = 0
    m.last_access = 0.0
    t = 0.0
    for _ in range(4):
        s = a2.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                 tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.4, 0.4, "episodic"])   # ratio=1 → flat
    h2 = session_memory.tau_learner_health(a2)
    ep2 = h2["by_type"]["episodic"]
    assert ep2["clean"]["direction"] == "flat"
    assert ep2["awakening"]["direction"] == "flat"
    assert ep2["consistency"] == "agree"


def test_tau_learner_health_one_sided_and_no_data():
    """单源 → 无法交叉印证；无数据 → 无信号。"""
    a = _two_signal_agent(clean_dir=None, aw_dir="up")
    h = session_memory.tau_learner_health(a)
    assert h["by_type"]["episodic"]["consistency"] == "one_sided"
    assert h["summary"]["one_sided"] == 1
    b = MemoryAgent()
    h2 = session_memory.tau_learner_health(b)
    assert h2["summary"]["no_data"] == 3
    assert all(v["consistency"] == "no_data" for v in h2["by_type"].values())


def test_eval_tau_learner_prints(capsys):
    """第 ⑧ 节输出：学习器状态、两路方向、一致性判定、摘要。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")
    h = session_memory.eval_tau_learner(a)
    out = capsys.readouterr().out
    assert "== τ 学习器健康检查" in out
    assert "学习器状态: tau_learning=开" in out
    assert "episodic" in out and "实测τ" in out
    assert "应下调（配置偏大）" in out
    assert "✔ 两路一致" in out and "收敛方向明确" in out
    assert "摘要: 一致 1" in out
    assert h["by_type"]["episodic"]["consistency"] == "agree"


def test_eval_recall_chain_reports_awakening_signal(capsys):
    """第 ⑦ 节从真实记忆库扫描唤醒信号：分布、主导方向、一致性 ≥60% 的提示。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    for dev, exp in [(0.32, 0.28), (0.30, 0.27), (0.33, 0.29)]:
        m = a.store.add("我昨天去吃了火锅", importance=0.1)
        m.awakenings.append([1.0, dev, exp, "episodic"])
    v = session_memory.eval_recall_chain(a)
    s = v["awakening_signal"]
    assert s["episodic"]["events"] == 3 and s["episodic"]["dominant"] == "up"
    out = capsys.readouterr().out
    assert "⑦ 唤醒偏差信号" in out
    assert "episodic" in out and "3 条" in out
    assert "dev 中位 0.3200" in out
    assert "主导 上调（一致性 100%）" in out
    assert "τ 配置偏大、可塑性配置偏小" in out


def test_eval_learner_response_synthetic_fallback(capsys):
    """无高一致性信号 → 回退合成演示 agent：单次 sleep 校准 τ 与 drift，
    都朝已知真实值方向移动（τ 3→逼近 2 天、drift 1→逼近 3.5）。"""
    r = session_memory.eval_learner_response(MemoryAgent())
    out = capsys.readouterr().out
    assert r["source"] == "synthetic" and r["signals"] == []
    ep = r["deltas"]["episodic"]
    assert ep["tau_d"] < 0            # τ 下调（3.0 → 2.7x，朝真实 2 天）
    assert ep["drift"] > 0            # drift 上调（1.0 → 1.6x，朝真实 3.5）
    assert "回退受控合成演示 agent" in out
    assert "真实 τ=2.0 天 vs 配置信念 3 天" in out
    assert "τ 3.000 → 2.749 天（Δ-0.251 → 逼近真实 2.0 天）" in out
    assert "校准完成: episodic 的参数已按信号方向调整" in out
    assert r["before"]["episodic"]["tau_d"] == 3.0
    assert r["after"]["episodic"]["drift"] > 1.0


def test_eval_learner_response_real_path(capsys):
    """真实库扫到高一致性信号（episodic 两路一致都应下调）→ 用真实 agent 跑
    sleep() 校准：τ 3.0 → 2.85（朝真实 2 天），drift 无修订样本不动。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")
    r = session_memory.eval_learner_response(a)
    out = capsys.readouterr().out
    assert r["source"] == "real" and r["signals"] == ["episodic"]
    d = r["deltas"]["episodic"]
    assert d["tau_d"] < 0 and d["drift"] == 0.0
    assert "episodic 两路方向一致且非持平 → 高一致性信号" in out
    assert "用真实记忆库演示" in out
    assert "τ 3.000 → 2.850 天（Δ-0.150）" in out
    assert "sleep 内自动触发 learn_tau + learn_plasticity" in out


def test_eval_learner_response_none(capsys):
    """未提供 agent → source=none，跳过。"""
    r = session_memory.eval_learner_response(None)
    assert r == {"source": "none"}
    assert "跳过" in capsys.readouterr().out


def test_eval_sync_returns_both_reports(tmp_path):
    """收工一步验证：返回合并报告（加载评估 + 唤醒链路 + τ 学习器健康
    + 学习器响应演示）。"""
    agent = MemoryAgent()
    session_memory.record(agent, COMMITS, notes=["可塑性学习器用中位数抗离群"])
    p = session_memory.export_agents_md(agent, str(tmp_path / "AGENTS.md"))
    rep = session_memory.eval_sync(agent, p)
    assert set(rep) == {"agents_md", "recall_chain", "tau_learner",
                        "learner_response"}
    assert rep["recall_chain"]["prefix_ok"]
    assert rep["agents_md"]["coverage"]
    assert "summary" in rep["tau_learner"]
    assert rep["learner_response"]["source"] in ("real", "synthetic")


def test_eval_agents_md_missing_keywords(tmp_path):
    """空文档 → 全部不可及，missing 列表完整。"""
    p = str(tmp_path / "AGENTS.md")
    open(p, "w", encoding="utf-8").write("# 空文件")
    report = session_memory.eval_agents_md(MemoryAgent(), path=p)
    assert all(not r["ok"] for r in report["coverage"])
    assert report["coverage"][0]["missing"]


def test_sync_three_steps(tmp_path, monkeypatch):
    """一键闭环：沉淀提交决策 + 刷新 AGENTS.md/CLAUDE.md + 统计正确。

    chdir 到 tmp 避免覆盖项目根的真实 AGENTS.md（相对路径导出）。
    """
    monkeypatch.chdir(tmp_path)
    agent = MemoryAgent()
    r = session_memory.sync(agent, commits=COMMITS, notes=["补充决策：学习器用中位数"])
    assert r["commits"] == 2 and r["recorded"] == 3
    assert r["total"] == 3
    assert len(r["exports"]) == 2 and r["export"] == r["exports"][0] == "AGENTS.md"
    text = open(r["export"], encoding="utf-8").read()
    assert "再巩固因子学习器改用中位数抗离群事件" in text
    assert "补充决策：学习器用中位数" in text
    # 双格式同步：CLAUDE.md 与 AGENTS.md 逐字节一致
    assert open("CLAUDE.md", encoding="utf-8").read() == text


def test_sync_repeat_does_not_duplicate(tmp_path, monkeypatch):
    """重复 sync 同一批提交 → 决策去重合并，AGENTS.md 条数不增。"""
    monkeypatch.chdir(tmp_path)
    agent = MemoryAgent()
    session_memory.sync(agent, commits=COMMITS, notes=[])
    r2 = session_memory.sync(agent, commits=COMMITS, notes=[])
    assert r2["recorded"] == 2  # 仍是写入 2 条（去重命中，非新增）
    assert r2["total"] == 2


def test_cli_sync_degrades_without_git(tmp_path):
    """非 git 仓库：--sync 降级（0 提交）+ 导出 AGENTS.md，--eval 联动可跑。"""
    persist = str(tmp_path / "mem.json")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--sync", "--note", "关键决策：学习器用中位数",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=tmp_path,
    )
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--sync", "--eval", "--note", "补充：语义化滞回",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=tmp_path,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "① 从 git log 提炼 0 条提交" in r.stdout
    assert "③ 刷新全量导出" in r.stdout
    assert "加载效果评估" in r.stdout       # --eval 联动：AGENTS.md 加载评估
    assert "唤醒链路连续性检查" in r.stdout  # --eval 联动：唤醒链路连续性检查
    # 双格式：--sync 在 tmp 里生成 AGENTS.md + CLAUDE.md（不污染项目根）
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")


def test_export_signals_json_and_csv(tmp_path):
    """第 ⑦ 节数据源导出：JSON（stats + periods 漂移对比）+ CSV（每类型全字段行）。"""
    a = _signal_clock_agent(40)
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    j = json.load(open(ex["json"], encoding="utf-8"))
    assert j["stats"]["episodic"]["events"] == 5
    assert j["periods"]["by_type"]["episodic"]["verdict"] == "方向翻转"
    rows = list(csv.reader(open(ex["csv"], encoding="utf-8")))
    assert rows[0][0] == "mtype" and "drift_verdict" in rows[0]
    ep = dict(zip(rows[0], rows[3]))  # 第 4 行 = episodic
    assert ep["events"] == "5" and ep["dominant"] == "up"
    assert ep["dev_med"] == "0.3" and ep["ratio_med"] == "1.111"
    assert ep["recent_events"] == "4" and ep["earlier_dominant"] == "down"
    assert ep["drift_verdict"] == "方向翻转" and ep["direction_changed"] == "True"
    # 无观测类型行：计数 0、判定字段保留
    sk = dict(zip(rows[0], rows[1]))
    assert sk["events"] == "0" and sk["drift_verdict"] == "无观测"


def test_export_signals_exclude_events(tmp_path):
    """--exclude-events：排除事件后重算唤醒统计/漂移/health——冲突库排除
    冲突侧事件后 health 从 conflict 变为 agree（冲突剔除假设检验）。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="up")   # episodic 冲突
    base = session_memory.export_signals(a, str(tmp_path / "base"))
    assert base["health"]["summary"]["conflict"] == 1
    assert base["health"]["warnings"]
    m = a.store.all()[0]
    # 排除两条唤醒 up 事件（index 0/1）→ 剩余无唤醒 → 单源/无信号，冲突消除
    ex = session_memory.export_signals(
        a, str(tmp_path / "ex"), exclude_events=[f"{m.id}:0", f"{m.id}:1"])
    h = ex["health"]["by_type"]["episodic"]
    assert h["consistency"] != "conflict"        # 冲突消除（此处唤醒清零 → no_data）
    assert ex["health"]["warnings"] == []
    assert ex["excluded"] == [f"{m.id}:0", f"{m.id}:1"]
    assert ex["stats"]["episodic"]["events"] == 0
    assert ex["events"] == []                    # 事件明细同步排除
    # 剔除前后对比块：before conflict → after 冲突消除（CI 直接读结论）
    ec = ex["health"]["exclude_compare"]
    assert ec["before"]["by_type"]["episodic"]["consistency"] == "conflict"
    assert ec["after"]["by_type"]["episodic"]["consistency"] != "conflict"
    assert ec["before"]["warnings_n"] >= 1 and ec["after"]["warnings_n"] == 0
    j = json.load(open(ex["json"], encoding="utf-8"))
    assert j["health"]["summary"]["conflict"] == 0
    assert j["health"]["exclude_compare"]["after"]["summary"]["conflict"] == 0
    assert j["health"]["exclude_compare"]["before"]["summary"]["conflict"] == 1
    # 无排除的基线导出不带对比块
    assert "exclude_compare" not in base["health"]
    # CSV 事件明细也为空（只余表头）
    rows = list(csv.reader(open(ex["events_csv"], encoding="utf-8")))
    assert len(rows) == 1


def test_export_signals_exclude_resolves_conflict(tmp_path):
    """排除冲突侧事件后两路同向 → health 从 conflict 变为 agree（χ验证
    “去掉这批事件后两路一致”落盘）。"""
    from memagent import MemoryAgent

    a = MemoryAgent(cfg=AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * 86400.0},
        true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
    ), now_fn=lambda: 0.0)
    m = a.store.add("冲突记忆", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count, m.last_access = 0, 0.0
    t = 0.0
    for _ in range(4):
        s = a.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.50, 0.62, "episodic"])   # ratio<1 → up（冲突侧）
    m.awakenings.append([2.0, 0.45, 0.58, "episodic"])   # ratio<1 → up
    m2 = a.store.add("正常情景", importance=0.2, mtype=MemType.EPISODIC)
    m2.awakenings.append([1.5, 0.4, 0.32, "episodic"])   # ratio>1 → down
    base = session_memory.export_signals(a, str(tmp_path / "b"))
    assert base["health"]["by_type"]["episodic"]["consistency"] == "conflict"
    ex = session_memory.export_signals(
        a, str(tmp_path / "r"), exclude_events=[f"{m.id}:0", f"{m.id}:1"])
    ep = ex["health"]["by_type"]["episodic"]
    assert ep["consistency"] == "agree" and ep["suggest"] == "τ↓"
    assert ex["health"]["warnings"] == []
    acts = ex["health"]["actions"]
    assert acts == [{"mtype": "episodic", "suggest": "τ↓",
                     "confidence": "弱",
                     "reason": "配置偏大 · 忘得比信念快"}]


def test_export_signals_aggregations(tmp_path):
    """仪表盘 Shift 多选聚合结论回放 health.aggregations：事件子集 + 全体/
    移除后中位与方向 + 判定——与仪表盘 JS 判定同闸门，CI 可回放人工结论。"""
    from memagent import MemoryAgent

    a = MemoryAgent(cfg=AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * 86400.0},
        true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
    ), now_fn=lambda: 0.0)
    m = a.store.add("冲突记忆", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count, m.last_access = 0, 0.0
    t = 0.0
    for _ in range(4):
        s = a.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.50, 0.62, "episodic"])   # 0.806 up（冲突侧）
    m.awakenings.append([2.0, 0.45, 0.58, "episodic"])   # 0.776 up
    m2 = a.store.add("正常情景", importance=0.2, mtype=MemType.EPISODIC)
    m2.awakenings.append([1.5, 0.4, 0.32, "episodic"])   # 1.25 down
    ex = session_memory.export_signals(a, str(tmp_path / "sig"), aggregations=[
        {"mtype": "episodic", "events": [f"{m.id}:0", f"{m.id}:1"]},
        {"mtype": "episodic", "events": [f"{m2.id}:0"]},
    ])
    aggs = ex["health"]["aggregations"]
    assert len(aggs) == 2
    r = aggs[0]   # 选两条 up → 剩余 [1.25] → down == clean → resolved
    assert r["events"] == [f"{m.id}:0", f"{m.id}:1"]
    assert r["selected_n"] == 2 and r["selected_dist"] == {"up": 2, "down": 0, "flat": 0}
    assert r["remaining_n"] == 1 and r["remaining_median_ratio"] == 1.25
    assert r["remaining_direction"] == "down" and r["clean_direction"] == "down"
    assert r["verdict"] == "resolved" and "冲突消除" in r["verdict_text"]
    assert r["all_median_ratio"] == 0.8065 and r["all_direction"] == "up"
    # 方向占比分布（外部工具复刻仪表盘色条）：全体/选中集各方向计数 + 占比 + 中位
    assert r["all_dist"] == {"up": 2, "down": 1, "flat": 0} and r["all_n"] == 3
    assert r["all_dist_pct"] == {"up": 67, "down": 33, "flat": 0}
    assert r["selected_n_ratio"] == 2
    assert r["selected_dist_pct"] == {"up": 100, "down": 0, "flat": 0}
    s = aggs[1]
    assert s["selected_dist"] == {"up": 0, "down": 1, "flat": 0}
    assert s["selected_dist_pct"] == {"up": 0, "down": 100, "flat": 0}
    assert s["all_dist_pct"] == {"up": 67, "down": 33, "flat": 0}
    s = aggs[1]   # 只选 down → 剩余两条 up → still_conflict
    assert s["verdict"] == "still_conflict" and "仍冲突" in s["verdict_text"]
    # resolved → 自动附带排除后重算证据包（--exclude-events 同链路）
    rc = r.get("recomputed")
    assert rc and rc["excluded"] == [f"{m.id}:0", f"{m.id}:1"]
    rh = rc["health"]
    assert rh["by_type"]["episodic"]["consistency"] == "agree"
    assert rh["by_type"]["episodic"]["suggest"] == "τ↓"
    assert rh["warnings"] == [] and rh["summary"]["conflict"] == 0
    assert rc["stats"]["episodic"]["events"] == 1   # 剩余 1 起（1.25 down）
    assert set(rc) >= {"excluded", "stats", "periods", "health"}
    # 未解决（still_conflict / insufficient）不附带重算
    assert "recomputed" not in s
    # JSON 同步携带
    j = json.load(open(ex["json"], encoding="utf-8"))
    assert len(j["health"]["aggregations"]) == 2
    assert j["health"]["aggregations"][0]["verdict"] == "resolved"
    jr = j["health"]["aggregations"][0]["recomputed"]
    assert jr["health"]["summary"]["conflict"] == 0
    assert jr["excluded"] == [f"{m.id}:0", f"{m.id}:1"]
    assert "recomputed" not in j["health"]["aggregations"][1]


def test_cli_export_signals_aggregations(tmp_path):
    """CLI --aggregations TYPE:key,key → health.aggregations 回放 + resolved
    自动附带剔除后重算证据包（--exclude-events 联动）。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="up")
    m = a.store.all()[0]
    m2 = a.store.add("正常情景", importance=0.2, mtype=MemType.EPISODIC)
    m2.awakenings.append([1.5, 0.4, 0.32, "episodic"])   # 1.25 down
    a.store.path = persist
    a.store.save()
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "sig"), "--persist", persist,
         "--aggregations", f"episodic:{m.id}:0,{m.id}:1"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "聚合结论回放（health.aggregations）" in r.stdout
    assert "已自动附带剔除后重算证据包（exclude 2 起）" in r.stdout
    j = json.load(open(str(tmp_path / "sig.json"), encoding="utf-8"))
    agg = j["health"]["aggregations"][0]
    assert agg["mtype"] == "episodic" and agg["verdict"] == "resolved"
    rc = agg["recomputed"]
    assert rc["excluded"] == [f"{m.id}:0", f"{m.id}:1"]
    assert rc["health"]["by_type"]["episodic"]["consistency"] == "agree"
    assert rc["health"]["warnings"] == []


def _clash_library(now_fn=lambda: 0.0):
    """冲突库：干净段 down + 唤醒 up/up/down——clashes = 两条 up。"""
    cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 3 * 86400.0},
                      true_tau_by_type={MemType.EPISODIC: 2 * 86400.0})
    a = MemoryAgent(cfg=cfg, now_fn=now_fn)
    m = a.store.add("冲突记忆", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count, m.last_access = 0, 0.0
    t = 0.0
    for _ in range(4):
        s = a.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                tau_override=2 * 86400.0)
        m.history.append([t, round(s, 4), 0.0, 0, 0.1])
        t += 0.5 * 86400.0
    m.awakenings.append([1.0, 0.50, 0.62, "episodic"])   # 0.806 up（clash）
    m.awakenings.append([2.0, 0.45, 0.58, "episodic"])   # 0.776 up（clash）
    m2 = a.store.add("正常情景", importance=0.2, mtype=MemType.EPISODIC)
    m2.awakenings.append([1.5, 0.4, 0.32, "episodic"])   # 1.25 down（一致，不排除）
    return a, m, m2


def test_clash_event_keys():
    """clash_event_keys：按 dir ≠ clean 圈定冲突成因（与仪表盘「选反向」同判定）
    ——up/up/down + 干净段 down → 恰两条 up；无比值/旧格式事件不圈。"""
    from memagent.agent import clash_event_keys, tau_learner_health

    a, m, m2 = _clash_library()
    h = tau_learner_health(a)
    assert h["by_type"]["episodic"]["consistency"] == "conflict"
    keys = clash_event_keys(a, h)
    assert keys == {(m.id, 0), (m.id, 1)}
    # 无干净段方向（one_sided 库）→ 无 clash（与选反向 clean && 同闸门）
    cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 3 * 86400.0})
    b = MemoryAgent(cfg=cfg, now_fn=lambda: 0.0)
    mm = b.store.add("x", importance=0.1, mtype=MemType.EPISODIC)
    mm.awakenings.append([1.0, 0.2, 0.25, "episodic"])
    assert clash_event_keys(b, tau_learner_health(b)) == set()


def test_export_signals_exclude_clashes(tmp_path):
    """--exclude-clashes：按 dir ≠ clean 自动排除冲突成因后重判 health——
    conflict → agree（τ↓）、warnings 清零、excluded_clashes 落盘。"""
    a, m, _ = _clash_library()
    ex = session_memory.export_signals(a, str(tmp_path / "sig"),
                                       exclude_clashes=True)
    assert ex["excluded_clashes"] == [f"{m.id}:0", f"{m.id}:1"]
    ep = ex["health"]["by_type"]["episodic"]
    assert ep["consistency"] == "agree" and ep["suggest"] == "τ↓"
    assert ex["health"]["warnings"] == []
    assert ex["health"]["summary"]["conflict"] == 0
    assert ex["stats"]["episodic"]["events"] == 1   # 剩余 1.25 down
    # 剔除前后对比块：每类型 before/after 的一致性 + 两路方向并列
    ec = ex["health"]["exclude_compare"]
    b, af = ec["before"]["by_type"]["episodic"], ec["after"]["by_type"]["episodic"]
    assert b["consistency"] == "conflict" and b["clean_direction"] == "down"
    assert b["awakening_direction"] == "up" and b["suggest"] == "需检查"
    assert af["consistency"] == "agree" and af["awakening_direction"] == "down"
    assert af["suggest"] == "τ↓"
    assert ec["before"]["warnings_n"] >= 1 and ec["after"]["warnings_n"] == 0
    assert ec["before"]["summary"]["conflict"] == 1
    assert ec["after"]["summary"]["conflict"] == 0
    # JSON 同步
    j = json.load(open(ex["json"], encoding="utf-8"))
    assert j["excluded_clashes"] == [f"{m.id}:0", f"{m.id}:1"]
    assert j["health"]["summary"]["conflict"] == 0
    assert j["health"]["exclude_compare"]["after"]["by_type"]["episodic"]["consistency"] == "agree"
    # 无冲突库 → 空（不误排除，无对比块）
    b = MemoryAgent()
    ex2 = session_memory.export_signals(b, str(tmp_path / "sig2"),
                                        exclude_clashes=True)
    assert ex2["excluded_clashes"] == []
    assert "exclude_compare" not in ex2["health"]


def test_cli_export_signals_exclude_clashes(tmp_path):
    """CLI --export-signals --exclude-clashes → 自动剔除冲突成因 + 打印重判说明
    + JSON 落盘（与仪表盘「选反向」同判定）。"""
    persist = str(tmp_path / "mem.json")
    a, m, _ = _clash_library()
    a.store.path = persist
    a.store.save()
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "sig"), "--persist", persist, "--exclude-clashes"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已按 dir ≠ clean 自动排除 2 起冲突成因事件" in r.stdout
    assert f"{m.id}:0, {m.id}:1" in r.stdout
    assert "重判后: 一致 1 · 冲突 0" in r.stdout
    assert "剔除前后对比（health.exclude_compare）" in r.stdout
    assert "conflict → agree" in r.stdout
    assert "建议 需检查 → τ↓" in r.stdout
    j = json.load(open(str(tmp_path / "sig.json"), encoding="utf-8"))
    assert j["excluded_clashes"] == [f"{m.id}:0", f"{m.id}:1"]
    assert j["health"]["by_type"]["episodic"]["consistency"] == "agree"
    assert j["health"]["warnings"] == []
    assert j["health"]["exclude_compare"]["before"]["summary"]["conflict"] == 1
    assert j["health"]["exclude_compare"]["after"]["summary"]["conflict"] == 0


def test_load_aggregations_file(tmp_path):
    """--aggregations-file 从 JSON 读取聚合选择集（仪表盘「导出聚合」格式），
    免手写 memory_id 列表。"""
    # 列表格式（仪表盘导出）
    p1 = str(tmp_path / "a.json")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump([{"mtype": "episodic", "events": ["m1:0", "m1:1"]}], f)
    assert session_memory._load_aggregations_file(p1) == [
        {"mtype": "episodic", "events": ["m1:0", "m1:1"]}]
    # 包裹对象格式 {aggregations: [...]}
    p2 = str(tmp_path / "b.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump({"aggregations": [{"mtype": "episodic", "events": ["m2:0"]}]}, f)
    assert session_memory._load_aggregations_file(p2) == [
        {"mtype": "episodic", "events": ["m2:0"]}]
    # 非法条目跳过 + 缺文件/坏 JSON → 空
    p3 = str(tmp_path / "c.json")
    with open(p3, "w", encoding="utf-8") as f:
        json.dump([{"mtype": "episodic", "events": ["m3:0"]},
                   {"mtype": "episodic"},  # 无 events → 跳过
                   "bad"], f)              # 非 dict → 跳过
    assert session_memory._load_aggregations_file(p3) == [
        {"mtype": "episodic", "events": ["m3:0"]}]
    assert session_memory._load_aggregations_file(str(tmp_path / "nope.json")) == []
    p4 = str(tmp_path / "bad.json")
    with open(p4, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert session_memory._load_aggregations_file(p4) == []


def test_cli_export_signals_aggregations_file(tmp_path):
    """CLI --aggregations-file（仪表盘导出的选择集 JSON）→ 聚合回放 + resolved
    证据包，与手写 --aggregations 结果一致。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="up")
    m = a.store.all()[0]
    m2 = a.store.add("正常情景", importance=0.2, mtype=MemType.EPISODIC)
    m2.awakenings.append([1.5, 0.4, 0.32, "episodic"])
    a.store.path = persist
    a.store.save()
    agg_file = str(tmp_path / "aggregations.json")
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump([{"mtype": "episodic",
                    "events": [f"{m.id}:0", f"{m.id}:1"]}], f)
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "sig"), "--persist", persist,
         "--aggregations-file", agg_file],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "读取 1 条聚合规格（--aggregations-file）" in r.stdout
    assert "✔ 移除后两路一致——冲突消除（resolved）" in r.stdout
    j = json.load(open(str(tmp_path / "sig.json"), encoding="utf-8"))
    agg = j["health"]["aggregations"][0]
    assert agg["events"] == [f"{m.id}:0", f"{m.id}:1"]
    assert agg["verdict"] == "resolved"
    assert agg["recomputed"]["health"]["by_type"]["episodic"]["consistency"] == "agree"


def test_cli_export_signals_exclude(tmp_path):
    """CLI --export-signals --exclude-events memory_id:序号,... → 排除后重算
    health 并打印说明。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="up")
    a.store.path = persist
    a.store.save()
    m = a.store.all()[0]
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "sig"), "--persist", persist,
         "--exclude-events", f"{m.id}:0,{m.id}:1"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已排除 2 起事件" in r.stdout
    j = json.load(open(str(tmp_path / "sig.json"), encoding="utf-8"))
    assert j["health"]["summary"]["conflict"] == 0
    assert j["stats"]["episodic"]["events"] == 0


def test_export_signals_events_csv(tmp_path):
    """事件级导出：每条唤醒事件一行（含来源记忆 id / 绝对 ts / 比值），
    旧格式跳过、6 元组透传埋藏时长与检索次数、按事件时刻排序。"""
    a = _signal_clock_agent(40)
    # 追加一条 6 元组事件（透传 dt / n_cold）与一条旧格式（无 expected → 跳过）
    m6 = a.store.add("六元组唤醒", importance=0.1, mtype=MemType.EPISODIC)
    m6.awakenings.append([30 * 86400, 0.31, 0.27, "episodic", 3.2 * 86400, 4])
    m6.awakenings.append([31 * 86400, 0.45, "episodic"])   # 旧格式 → 跳过
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    assert ex["events_csv"] == str(tmp_path / "sig_events.csv")
    rows = list(csv.reader(open(ex["events_csv"], encoding="utf-8")))
    assert rows[0] == ["memory_id", "mtype", "ts", "ts_relative_seconds",
                       "dev", "expected", "ratio", "dt_seconds",
                       "retrievals_before"]
    assert len(rows) == 7          # 表头 + 5 条（_signal_clock_agent）+ 1 条 6 元组
    body = rows[1:]
    ids = {m.id for m in a.store.all()}
    assert all(r[0] in ids for r in body)   # 来源记忆 id 齐全且可追溯
    ts = [float(r[2]) for r in body]
    assert ts == sorted(ts)        # 按事件时刻排序（任意时间窗重切片）
    # 6 元组行：dt / n_cold 透传
    six = [r for r in body if r[7]]
    assert len(six) == 1 and six[0][7] == "276480.0" and six[0][8] == "4"
    assert six[0][1] == "episodic" and six[0][5] == "0.27" and six[0][6] == "1.1481"
    # 4 元组行：dt / n_cold 空、比值照常
    four = [r for r in body if not r[7]]
    assert len(four) == 5 and all(r[8] == "" for r in four)
    # 比值 / 相对时间：最近事件 ts_relative 为负（过去）
    assert body[-1][6] == "1.1379" and float(body[-1][3]) < 0
    # JSON 也带事件明细（含 memory_id 与原始字段）
    j = json.load(open(ex["json"], encoding="utf-8"))
    evs = j["events"]
    assert len(evs) == 6
    assert set(evs[0]) == {"memory_id", "mtype", "ts", "ts_relative_seconds",
                           "dev", "expected", "ratio", "dt", "n_cold"}
    assert evs[-1]["memory_id"] in {m.id for m in a.store.all()}


def test_export_signals_includes_tau_health(tmp_path):
    """CSV 每类型追加 τ 两路信号方向一致性列（干净段 vs 唤醒），JSON 带 health
    结构——信号导出与健康检查合表。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")   # episodic 两路同向
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    rows = list(csv.reader(open(ex["csv"], encoding="utf-8")))
    hdr = rows[0]
    for col in ("clean_n", "clean_tau_est_s", "clean_cfg_tau_s",
                "clean_direction", "awakening_direction", "tau_consistency",
                "suggest_adjust", "suggest_confidence"):
        assert col in hdr
    ep = dict(zip(hdr, rows[3]))   # 第 4 行 = episodic
    assert ep["clean_n"] == "3" and ep["clean_direction"] == "down"
    assert ep["awakening_direction"] == "down"
    assert ep["tau_consistency"] == "agree"
    assert ep["suggest_adjust"] == "τ↓"     # agree 同向 down → 直接给调整方向
    assert ep["suggest_confidence"] == "弱"  # 唤醒 2 条 < tau_min_awakenings → 弱
    # 实测 τ < 配置 τ（忘得比信念快）→ clean_tau_est_s 数值小于 cfg
    assert float(ep["clean_tau_est_s"]) < float(ep["clean_cfg_tau_s"])
    # 无信号类型行：clean 空 / direction 空 / no_data / 无动作 / 无置信度
    sk = dict(zip(hdr, rows[1]))
    assert sk["clean_direction"] == "" and sk["awakening_direction"] == ""
    assert sk["tau_consistency"] == "no_data"
    assert sk["suggest_adjust"] == "无信号"
    assert sk["suggest_confidence"] == "—"
    # JSON health：by_type 结构与 summary（by_type 含 suggest 行动建议）
    j = json.load(open(ex["json"], encoding="utf-8"))
    h = j["health"]
    assert set(h) == {"by_type", "summary", "warnings", "actions"}
    assert h["warnings"] == []   # 无冲突类型 → warnings 空（CI 按非空判定红灯）
    # actions = 行动清单（τ↓/τ↑/需检查）——agree 的 episodic 在列，无信号类型不在
    acts = h["actions"]
    assert acts == [{"mtype": "episodic", "suggest": "τ↓",
                     "confidence": "弱",
                     "reason": "配置偏大 · 忘得比信念快"}]
    assert h["summary"]["agree"] == 1 and h["summary"]["no_data"] == 2
    epj = h["by_type"]["episodic"]
    assert set(epj) == {"clean", "awakening", "consistency", "suggest",
                        "confidence"}
    assert epj["consistency"] == "agree"
    assert epj["suggest"] == "τ↓"
    assert epj["confidence"] == "弱"
    assert epj["clean"]["direction"] == "down"
    assert epj["awakening"]["direction"] == "down"


def test_export_signals_tau_health_conflict(tmp_path):
    """两路反向 → tau_consistency=conflict（合表里直接暴露信号冲突，与健康检查一致）。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="up")    # 干净段下调 vs 唤醒上调
    # 行号直接进 health（agent 层单一实现，无需导出）
    raw = session_memory.tau_learner_health(a)
    raw_evs = raw["warnings"][0]["events"]
    assert [e["row"] for e in raw_evs] == [2, 3]
    assert "ts_relative_seconds" in raw_evs[0]
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    rows = list(csv.reader(open(ex["csv"], encoding="utf-8")))
    ep = dict(zip(rows[0], rows[3]))
    assert ep["clean_direction"] == "down"
    assert ep["awakening_direction"] == "up"
    assert ep["tau_consistency"] == "conflict"
    assert ep["suggest_adjust"] == "需检查"   # 冲突 → 先排查再调参
    assert ep["suggest_confidence"] == "—"    # 冲突不是调参建议，无置信度
    j = json.load(open(ex["json"], encoding="utf-8"))
    assert j["health"]["summary"]["conflict"] == 1
    assert j["health"]["by_type"]["episodic"]["consistency"] == "conflict"
    assert j["health"]["by_type"]["episodic"]["suggest"] == "需检查"
    assert j["health"]["by_type"]["episodic"]["confidence"] == "—"
    # actions 同步：conflict → 需检查 在行动清单里（CI 非空即需处理）
    acts = j["health"]["actions"]
    assert len(acts) == 1
    base = {k: acts[0][k] for k in ("mtype", "suggest", "confidence", "reason")}
    assert base == {"mtype": "episodic", "suggest": "需检查",
                    "confidence": "—",
                    "reason": "两路信号冲突，先排查再调参"}
    # 需检查条目带冲突事件明细（行号/记忆id/比值/方向）——CI 直接读冲突成因
    evs = acts[0]["events"]
    assert [e["row"] for e in evs] == [2, 3]          # 对应 events CSV 行号
    assert len({e["memory_id"] for e in evs}) == 1
    assert [e["ratio"] for e in evs] == [0.8, 0.8077]
    assert all(e["direction"] == "up" for e in evs)   # ratio<1 → 与干净段相反
    assert evs[0]["csv"].endswith("_events.csv")
    assert "dev" in evs[0] and "expected" in evs[0]
    assert "ts_relative_seconds" in evs[0]
    # 冲突告警写进导出 JSON（health.warnings：类型 + 两路证据 + 建议）
    warns = j["health"]["warnings"]
    assert len(warns) == 1 and warns[0]["mtype"] == "episodic"
    w = warns[0]
    assert w["consistency"] == "conflict"
    assert "干净段 3 条: 实测τ≈" in w["clean_evidence"]
    assert "应下调（配置偏大）" in w["clean_evidence"]
    assert "唤醒 2 条: 中位比值" in w["awakening_evidence"]
    assert "应上调（配置偏小）" in w["awakening_evidence"]
    assert "观测" in w["suggestion"] and "事件注入" in w["suggestion"]


def test_suggest_adjust_all_states():
    """建议列五态全覆盖：agree 同向给方向（τ↓/τ↑）、agree 双 flat → 已校准、
    conflict → 需检查、one_sided → 需补观测、no_data → 无信号。"""
    def _tau_agent(true_days: float, cfg_days: float, ratio: float) -> MemoryAgent:
        """可控 τ 方向 agent：true τ 决定干净段反推方向（>cfg → up、<cfg → down、
        相等 → flat），ratio 决定唤醒方向（<1 → up、>1 → down、=1 → flat）。"""
        cfg = AgentConfig(tau_by_type={MemType.EPISODIC: cfg_days * 86400.0})
        a = MemoryAgent(cfg=cfg, now_fn=lambda: 0.0)
        m = a.store.add("x", importance=0.1, mtype=MemType.EPISODIC)
        m.access_count = 0
        m.last_access = 0.0
        t = 0.0
        for _ in range(4):
            s = a.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                    tau_override=true_days * 86400.0)
            m.history.append([t, round(s, 4), 0.0, 0, 0.1])
            t += 0.5 * 86400.0
        dev, exp = 0.2, 0.2 / ratio      # ratio = dev/expected
        m.awakenings.append([1.0, dev, exp, "episodic"])
        m.awakenings.append([2.0, dev, exp, "episodic"])
        return a

    cases = [
        (_tau_agent(true_days=2, cfg_days=3, ratio=1.25), "τ↓"),   # 都 down
        (_tau_agent(true_days=4, cfg_days=2, ratio=0.8), "τ↑"),    # 都 up
        (_tau_agent(true_days=2, cfg_days=2, ratio=1.0), "已校准"),  # 都 flat
        (_two_signal_agent(clean_dir="down", aw_dir="up"), "需检查"),
        (_two_signal_agent(clean_dir=None, aw_dir="up"), "需补观测"),
        (MemoryAgent(), "无信号"),
    ]
    for a, want in cases:
        h = session_memory.tau_learner_health(a)
        assert h["by_type"]["episodic"]["suggest"] == want, (h, want)


def test_warn_conflict_types_evidence(tmp_path, capsys):
    """冲突类型告警：附两路原始证据行（干净段 n/实测τ/配置τ/方向 + 唤醒
    n/中位比值/方向）+ 排查提示；无冲突类型时静默。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="up")   # episodic 冲突
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    session_memory._warn_conflict_types(ex)
    out = capsys.readouterr().out
    assert "⚠ 需排查类型（两路信号冲突）" in out
    assert "✘ episodic:" in out
    assert "干净段 3 条: 实测τ≈" in out and "应下调（配置偏大）" in out
    assert "唤醒 2 条: 中位比值" in out and "应上调（配置偏小）" in out
    assert "观测" in out and "事件注入" in out
    # 一致 → 静默
    b = _two_signal_agent(clean_dir="down", aw_dir="down")
    ex2 = session_memory.export_signals(b, str(tmp_path / "agree"))
    session_memory._warn_conflict_types(ex2)
    assert capsys.readouterr().out == ""


def test_cli_export_signals_conflict_warning(tmp_path):
    """CLI --export-signals 检测到冲突类型 → 打印告警 + 两路证据行。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="up")   # episodic 冲突
    a.store.path = persist
    a.store.save()
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "sig"), "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已导出唤醒信号" in r.stdout
    assert "⚠ 需排查类型（两路信号冲突）" in r.stdout
    assert "干净段 3 条" in r.stdout and "唤醒 2 条" in r.stdout
    assert "观测" in r.stdout


def test_strict_exit_code_states(tmp_path):
    """--strict 双档退出码：冲突（warnings）→ 1 需排查、无冲突但 τ↓/τ↑ → 2
    需校准、皆无 → 0；未开 --strict 恒 0（向后兼容）。"""
    conflict = _two_signal_agent(clean_dir="down", aw_dir="up")
    ex = session_memory.export_signals(conflict, str(tmp_path / "c"))
    assert session_memory._strict_exit_code(ex, True) == 1    # 需排查优先
    assert session_memory._strict_exit_code(ex, False) == 0   # 默认不阻塞
    agree = _two_signal_agent(clean_dir="down", aw_dir="down")   # τ↓ 行动项
    ex2 = session_memory.export_signals(agree, str(tmp_path / "a"))
    assert session_memory._strict_exit_code(ex2, True) == 2   # 无冲突但需校准
    assert session_memory._strict_exit_code(ex2, False) == 0
    none = MemoryAgent()
    ex3 = session_memory.export_signals(none, str(tmp_path / "n"))
    assert session_memory._strict_exit_code(ex3, True) == 0   # 皆无 → 0


def test_apply_suggestions_calibrates(tmp_path, capsys):
    """--apply-suggestions：τ↓ 行动项确认后跑 sleep() 让学习器实际校准——
    校准前后 τ/drift 对比 + learned_tau 持久化（重启即加载）。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")   # episodic agree down → τ↓
    persist = str(tmp_path / "mem.json")
    a.store.path = persist
    a.store.save()
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    assert ex["health"]["by_type"]["episodic"]["suggest"] == "τ↓"
    before = a.cfg.tau_for(MemType.EPISODIC)
    r = session_memory.apply_suggestions(a, ex, yes=True)
    assert r["cancelled"] is False and "episodic" in r["deltas"]
    assert r["deltas"]["episodic"]["tau_d"] < 0        # τ 朝真实值下调
    assert a.cfg.tau_for(MemType.EPISODIC) < before       # 校准后 τ 变小
    out = capsys.readouterr().out
    assert "== 行动执行（--apply-suggestions）==" in out
    assert "episodic（τ↓" in out and "置信度" in out
    assert "τ 3.000 →" in out and "drift" in out and "校准完成" in out
    # 持久化：重启后 learned_tau 加载（τ 仍 < 3 天）
    reloaded = MemoryAgent(persist_path=persist)
    assert reloaded.cfg.tau_for(MemType.EPISODIC) < 3 * 86400.0
    # 元数据落盘
    meta = json.load(open(persist, encoding="utf-8"))["meta"]
    assert meta["learned_tau"]["episodic"] < 3 * 86400.0


def test_apply_suggestions_cancel_and_conflict(tmp_path, monkeypatch, capsys):
    """确认拒绝（n）/EOF → 取消不校准；冲突类型（需检查）不是自动执行项。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")
    ex = session_memory.export_signals(a, str(tmp_path / "s"))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    r = session_memory.apply_suggestions(a, ex, yes=False)
    assert r["cancelled"] is True and not r["deltas"]
    assert "已取消——未执行校准" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    r2 = session_memory.apply_suggestions(a, ex, yes=False)
    assert r2["cancelled"] is True
    # 冲突：无可自动执行行动项，需检查明确跳过
    c = _two_signal_agent(clean_dir="down", aw_dir="up")
    exc = session_memory.export_signals(c, str(tmp_path / "c"))
    rc = session_memory.apply_suggestions(c, exc, yes=True)
    assert rc["actionable"] == [] and rc["skipped"] == {"episodic": "需检查"}
    assert not rc["deltas"]
    assert "冲突需先排查" in capsys.readouterr().out


def test_cli_export_signals_apply_suggestions(tmp_path):
    """CLI --export-signals --apply-suggestions --yes：端到端校准——stdout 含
    校准对比，持久化文件 learned_tau 更新（episodic < 3 天）。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="down")
    a.store.path = persist
    a.store.save()
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "sig"), "--persist", persist,
         "--apply-suggestions", "--yes"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "== 行动执行（--apply-suggestions）==" in r.stdout
    assert "τ 3.000 →" in r.stdout and "校准完成" in r.stdout
    meta = json.load(open(persist, encoding="utf-8"))["meta"]
    assert meta["learned_tau"]["episodic"] < 3 * 86400.0


def test_cli_export_signals_strict(tmp_path):
    """CLI --export-signals --strict：冲突库 → returncode 1（需排查）、τ↓ 库
    → returncode 2（需校准）、不带 --strict → 0。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="up")   # episodic 冲突
    a.store.path = persist
    a.store.save()
    # 冲突 + --strict → 1（需排查）
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "s1"), "--persist", persist, "--strict"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "⚠ --strict: 1 个冲突类型 → 退出码 1" in r.stdout
    # 不带 --strict → 照常 0
    r2 = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "s2"), "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    # τ↓ 库（agree 无冲突）+ --strict → 2（需校准）
    persist2 = str(tmp_path / "mem2.json")
    b = _two_signal_agent(clean_dir="down", aw_dir="down")
    b.store.path = persist2
    b.store.save()
    r3 = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals",
         str(tmp_path / "s3"), "--persist", persist2, "--strict"],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r3.returncode == 2, r3.stdout + r3.stderr
    assert "退出码 2（需校准" in r3.stdout


def test_read_suggest_adjust_from_csv(tmp_path):
    """读回导出 CSV 的 suggest_adjust 列（导出 → 读回 → 行动）：按类型返回建议；
    文件缺失/损坏回退空表。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="up")   # episodic 冲突
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    got = session_memory._read_suggest_adjust(ex["csv"])
    assert ("episodic", "需检查") in got
    assert ("skill", "无信号") in got
    assert session_memory._read_suggest_adjust(
        str(tmp_path / "missing.csv")) == []


def test_print_adjust_actions(capsys, tmp_path):
    """行动清单：检测到 τ↓/τ↑/需检查 时打印逐类型行动项（含置信度）+ sleep()
    校准提示；无行动项时静默。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="down")  # episodic τ↓（弱）
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    session_memory._print_adjust_actions(ex)
    out = capsys.readouterr().out
    assert "行动清单（suggest_adjust）" in out
    assert "episodic: τ↓（弱 · 配置偏大 · 忘得比信念快）" in out
    assert "跑 sleep() 让学习器实际校准" in out
    assert "learn_tau + learn_plasticity" in out
    assert "置信度弱 = 观测不足" in out
    # 无行动项（全 no_data）→ 静默
    b = MemoryAgent()
    ex2 = session_memory.export_signals(b, str(tmp_path / "none"))
    session_memory._print_adjust_actions(ex2)
    assert capsys.readouterr().out == ""


def test_print_conflict_events_detail(tmp_path, capsys):
    """需检查 类型打印冲突唤醒事件明细 + events CSV 对应行号——排障不用翻文件。"""
    a = _two_signal_agent(clean_dir="down", aw_dir="up")   # episodic 冲突
    ex = session_memory.export_signals(a, str(tmp_path / "sig"))
    session_memory._print_adjust_actions(ex)
    out = capsys.readouterr().out
    assert "⚠ episodic: 两路信号冲突" in out
    assert "唤醒事件明细（2 条 →" in out
    assert "_events.csv 第 2-3 行" in out          # 引用 events CSV 对应行
    assert "[行 2]" in out and "[行 3]" in out
    assert "记忆 " in out and "dev " in out and "vs 预期" in out
    assert "比值 0.800 ↑ 应上调" in out            # ratio<1 → 应上调（冲突侧）
    assert "比值 0.808 ↑ 应上调" in out
    # 无事件级明细的类型 → 提示空，不报错（事件现在直接来自 health.warnings）
    b = _two_signal_agent(clean_dir="down", aw_dir="up")
    exb = session_memory.export_signals(b, str(tmp_path / "sig2"))
    exb["health"]["warnings"][0]["events"] = []   # 模拟事件级明细缺失
    session_memory._print_adjust_actions(exb)
    out2 = capsys.readouterr().out
    assert "（episodic 无事件级唤醒明细）" in out2


def test_suggest_confidence_rubric():
    """置信度标尺：agree 按两路观测条数给 强/中/弱（避免单条观测就建议调参），
    冲突/无信号 → —、单源 → 弱。"""
    def conf_agent(clean_n: int, aw_n: int) -> "MemoryAgent":
        cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 3 * 86400.0},
                          true_tau_by_type={MemType.EPISODIC: 2 * 86400.0})
        a = MemoryAgent(cfg=cfg, now_fn=lambda: 0.0)
        m = a.store.add("x", importance=0.1, mtype=MemType.EPISODIC)
        m.access_count = 0
        m.last_access = 0.0
        t = 0.0
        for _ in range(clean_n + 1):   # clean_n 个干净段
            s = a.strength_at_state(MemType.EPISODIC, 0.0, 0, 0.1, t,
                                    tau_override=2 * 86400.0)
            m.history.append([t, round(s, 4), 0.0, 0, 0.1])
            t += 0.5 * 86400.0
        for i in range(aw_n):
            m.awakenings.append([i + 1.0, 0.5, 0.4, "episodic"])  # ratio>1 → down
        return a

    cases = [
        (conf_agent(6, 6), "强"),   # 两路都 ≥ 2×门控
        (conf_agent(3, 3), "中"),   # 两路都过门控
        (conf_agent(3, 1), "弱"),   # 单条唤醒观测 → 弱（避免过早调参）
        (conf_agent(4, 2), "弱"),   # 唤醒不足门控 → 弱
        (_two_signal_agent(clean_dir="down", aw_dir="up"), "—"),   # 冲突
        (_two_signal_agent(clean_dir=None, aw_dir="up"), "弱"),      # 单源
        (MemoryAgent(), "—"),                                         # 无信号
    ]
    for a, want in cases:
        h = session_memory.tau_learner_health(a)["by_type"]["episodic"]
        assert h["confidence"] == want, (h, want)


def test_cli_sync_eval_export_signals_actions(tmp_path):
    """组合 --sync --eval --export-signals：收工验证读取导出 CSV 的 suggest_adjust，
    检测到 τ↓ → 打印行动清单 + sleep() 校准提示。"""
    persist = str(tmp_path / "mem.json")
    a = _two_signal_agent(clean_dir="down", aw_dir="down")   # episodic τ↓
    a.store.path = persist
    a.store.save()
    r = subprocess.run(
        [sys.executable, os.path.abspath("session_memory.py"), "--sync", "--eval",
         "--export-signals", str(tmp_path / "sig"), "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已导出唤醒信号" in r.stdout
    assert "行动清单（suggest_adjust）" in r.stdout
    assert "episodic: τ↓" in r.stdout
    assert "跑 sleep() 让学习器实际校准" in r.stdout


def test_cli_export_signals_standalone(tmp_path):
    """单独运行 --export-signals：从当前记忆库导出 JSON + CSV。"""
    persist = str(tmp_path / "mem.json")
    subprocess.run(
        [sys.executable, "session_memory.py", "--record", "--note", "开发决策：学习器用中位数",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    )
    base = str(tmp_path / "sig")
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--export-signals", base, "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0 and "已导出唤醒信号" in r.stdout
    assert os.path.exists(base + ".json") and os.path.exists(base + ".csv")
    assert os.path.exists(base + "_events.csv")            # 事件级明细
    with open(base + ".json", encoding="utf-8") as f:
        j = json.load(f)
    assert set(j) == {"now", "recent_seconds", "stats", "periods", "events",
                      "health", "excluded", "excluded_clashes"}


def test_cli_sync_eval_export_signals(tmp_path):
    """组合 --sync --eval --export-signals：收工验证后追加导出信号文件。"""
    script = os.path.abspath("session_memory.py")
    r = subprocess.run(
        [sys.executable, script, "--sync", "--eval", "--export-signals",
         str(tmp_path / "sig"), "--persist", str(tmp_path / "mem.json")],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
        cwd=str(tmp_path),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已导出唤醒信号" in r.stdout
    assert os.path.exists(str(tmp_path / "sig.json"))
    assert os.path.exists(str(tmp_path / "sig.csv"))
    assert os.path.exists(str(tmp_path / "sig_events.csv"))


def test_cli_export_and_eval(tmp_path):
    persist = str(tmp_path / "mem.json")
    subprocess.run(
        [sys.executable, "session_memory.py", "--record", "--note", "再巩固因子学习器用中位数抗离群事件",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    )
    out = str(tmp_path / "AGENTS.md")
    e = subprocess.run(
        [sys.executable, "session_memory.py", "--export-agents-md", out, "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert e.returncode == 0 and "已全量导出" in e.stdout
    v = subprocess.run(
        [sys.executable, "session_memory.py", "--eval-agents-md", out, "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert v.returncode == 0 and "加载效果评估" in v.stdout


def test_cli_record_and_start(tmp_path):
    persist = str(tmp_path / "mem.json")
    # 收工：无 git 仓库时只沉淀 --note（降级）
    r = subprocess.run(
        [sys.executable, "session_memory.py", "--record", "--note", "关键决策：学习器用中位数",
         "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "已沉淀" in r.stdout
    # 开工：注入该决策
    s = subprocess.run(
        [sys.executable, "session_memory.py", "--start", "--persist", persist],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert s.returncode == 0, s.stdout + s.stderr
    assert "开发决策：关键决策：学习器用中位数" in s.stdout
    assert "==== memagent 决策记忆注入" in s.stdout
