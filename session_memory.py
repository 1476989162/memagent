"""session_memory.py — 把 memagent 用作项目的"外置会话记忆"。

- **收工沉淀**（--record）：从 git log 提炼本次提交（关键决策线索）+ --note
  手动补充，全部写入记忆层（memories_session.json）。同一决策重复记录会被
  去重合并、测试效应强化——被反复确认的决策越来越强（memagent 机制）；
- **开工注入**（--start）：检索相关决策并输出可直接粘贴给 agent 的上下文块。
  --topic 按主题检索；无 topic 取记忆层当前强度最高的决策（重要决策长期
  保留、被遗忘的自动沉底）。

用法：
    python session_memory.py --start [--topic 关键词] [--k 5]       # 开工注入
    python session_memory.py --record [--since "2 hours ago"] [--note "补充"]  # 收工沉淀
    python session_memory.py --show [--topic 关键词]                # 查看记忆层
    python session_memory.py --reset                                # 清空

与真实 agent 的关系：主流 agent 的长期记忆是静态文件（CLAUDE.md/AGENTS.md），
本脚本演示"动态决策记忆"——决策自动分类、按重要性分层、重复确认强化、
长期不用衰减，且注入块可直接粘贴进任何 agent 的上下文。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.instructions import (
    MD_GROUP_TITLES as _MD_GROUP_TITLES,
    build_injection_md as _build_injection_md_impl,
    export_agents_md as _export_agents_md_impl,
    pick_decisions as _pick_decisions_impl,
    ranked_decisions,
)
from memagent.memory import MemType
from memagent.synonyms import is_short_query

DEFAULT_PERSIST = "memories_session.json"


def get_recent_commits(since: str | None = None, limit: int = 10, cwd: str | None = None) -> list[tuple[str, str]]:
    """返回最近提交 [(hash, subject), ...]。非 git 仓库 / 无提交 → []。"""
    cmd = ["git", "log", "--pretty=format:%h%x09%s"]
    if since:
        cmd += ["--since", since]
    cmd += ["-n", str(limit)]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10, check=True, cwd=cwd,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, ValueError):
        return []
    return [
        (h, s) for h, s in (line.split("\t", 1) for line in out.stdout.splitlines() if "\t" in line)
    ]


def sync(agent: MemoryAgent, since: str | None = None, limit: int = 10,
         notes: list[str] | None = None, commits: list[tuple[str, str]] | None = None) -> dict:
    """一键闭环：git log 提炼新提交 → 沉淀决策 → 刷新 AGENTS.md 全量导出。

    commits 显式传入时（测试/外部管道）跳过 git 拉取。返回统计。
    """
    if commits is None:
        commits = get_recent_commits(since=since, limit=limit)
    recorded = record(agent, commits, notes or [])
    exports = export_agents_md(agent, "AGENTS.md", dual=True)  # 双格式，内容同步
    total = sum(1 for m in agent.store.all() if m.kind != "turn")
    return {"commits": len(commits), "recorded": recorded,
            "export": exports[0], "exports": exports, "total": total}


def record(agent: MemoryAgent, commits: list[tuple[str, str]], notes: list[str]) -> int:
    """把提交信息与手动补充沉淀进记忆层，返回写入条数（去重合并的算命中强化）。"""
    n = 0
    for _hash, subject in commits:
        if subject.strip():
            agent.remember(f"开发决策：{subject.strip()}")
            n += 1
    for note in notes:
        if note.strip():
            agent.remember(f"开发决策：{note.strip()}")
            n += 1
    agent.store.save()
    return n


def _ranked_decisions(agent: MemoryAgent, k: int) -> list[tuple]:
    """按当前强度取记忆层最强的决策（排除对话流水）。

    实现已上移到 memagent.instructions（与 MCP 工具共用），此处保留
    兼容别名。
    """
    return ranked_decisions(agent, k)


def pick_decisions(agent: MemoryAgent, topic: str | None = None, k: int = 5) -> list[tuple]:
    """选择要注入的决策：主题检索（rel 排序）或按强度 top-k。返回 [(记忆, 强度)]。

    核心实现在 memagent.instructions（与 MCP 工具 memagent_start 共用）；
    此处保留 CLI 专属的短主题加长提示。

    短主题（< SHORT_QUERY_LEN 字）由 memagent.synonyms 的通用函数做
    子串优先重排（与 remember_agent 入口行为一致），并提示建议加长。
    """
    picked = _pick_decisions_impl(agent, topic, k)
    if topic and agent.cfg.rerank_short_query and is_short_query(topic, agent.cfg.rerank_short_len):
        # 含词计数与核心一致：content + 摘要（Cold 命中词可能只在摘要里）
        n = sum(1 for m, _s in picked if topic.lower() in (m.content + (m.summary or "")).lower())
        print(f"（提示：主题「{topic}」仅 {len(topic.strip())} 字，已做子串优先重排"
              f"[{n}/{len(picked)} 条含主题词]；建议加长以获得更精确检索）")
    return picked


def inject_block(agent: MemoryAgent, topic: str | None = None, k: int = 5) -> list[tuple]:
    """打印可直接粘贴的上下文块（开工默认模式）。"""
    picked = pick_decisions(agent, topic, k)
    print(f"==== memagent 决策记忆注入（{len(picked)} 条"
          f"{' · 主题「' + topic + '」' if topic else ''}）====")
    for i, (m, s) in enumerate(picked, 1):
        # Cold 记忆展示摘要而非深藏 content（与核心 /memories 展示一致）
        print(f"{i}. [{m.mtype.value}] {m.summary or m.content}（强度 {s:.2f} · 检索 {m.access_count} 次）")
    print("==== 注入结束（以上为跨会话沉淀的开发决策，可粘贴给 agent）====")
    return picked


INJECT_MARKER_START = "<!-- memagent-injection:start -->"
INJECT_MARKER_END = "<!-- memagent-injection:end -->"


def build_injection_md(agent: MemoryAgent, topic: str | None = None, k: int = 5) -> str:
    """生成注入块的 markdown 文本（供写文件 / 维护 AGENTS.md 区块）。

    核心实现在 memagent.instructions（与 MCP 工具 memagent_start 共用）。
    """
    return _build_injection_md_impl(
        agent, topic, k,
        refresh_hint="重新运行 `python session_memory.py --inject-agents-md` 更新本区块。",
    )


def write_context_file(agent: MemoryAgent, path: str = "session_context.md",
                       topic: str | None = None, k: int = 5) -> str:
    """生成独立注入 prompt 文件（开工时让 agent 读取/粘贴）。"""
    block = build_injection_md(agent, topic, k)
    text = (
        "# memagent 会话上下文\n\n"
        f"生成时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}。\n"
        "用法：开工时把本文件内容作为上下文提供给 agent，或\n"
        "运行 `python session_memory.py --inject-agents-md` 直接维护 AGENTS.md 顶部。\n\n"
        f"{block}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# 全量导出与加载评估：模拟 Claude Code/Codex 等"全量加载指令文件"的 agent
# （分组标题与导出实现见 memagent.instructions，导入别名为 _MD_GROUP_TITLES）


def export_agents_md(agent: MemoryAgent, path: str = "AGENTS.md", max_items: int = 50,
                     dual: bool = False) -> str | tuple[str, str]:
    """把记忆层全部决策导出成完整 AGENTS.md 风格文档（按类型分组、带标注）。

    与 --inject-agents-md 的区别：注入是顶部 top-k 动态区块，这里导出
    完整决策库（全部条目），供全量加载指令文件的 agent 在会话开始时读取。

    核心实现在 memagent.instructions（与 MCP 工具 memagent_export 共用）。

    dual=True 时用同一份内容同时写入 AGENTS.md 与 CLAUDE.md（逐字节一致，
    保持同步——Codex 加载前者、Claude Code 加载后者），返回两个路径；
    否则只写 path 并返回该路径。
    """
    return _export_agents_md_impl(
        agent, path, max_items=max_items, dual=dual,
        refresh_hint=(
            "重新运行 `python session_memory.py --export-agents-md` 可同时刷新\n"
            "AGENTS.md 与 CLAUDE.md（内容同步）。"
        ),
    )


# 主题问题集：每个问题附带"文件中必须出现才能回答"的关键词
EVAL_QUESTIONS = [
    ("可塑性学习器为什么用中位数聚合？", ["中位数", "抗离群"]),
    ("语义化迁移怎么避免来回振荡？", ["滞回", "3.0", "0.8"]),
    ("遗忘斜率对比用什么指标而非斜率比？", ["触底时间"]),
    ("技能类记忆回忆时的行为是什么？", ["校验", "验证"]),
    ("查询同义扩展包含哪两个机制？", ["人称互换", "同义词"]),
    ("对照实验为什么需要可注入时钟？", ["now_fn", "确定性"]),
    ("贴合度怎么计算？", ["fit", "实测τ", "配置τ"]),
    ("再巩固可塑性与重要性的关系？", ["1 − importance", "冻结"]),
]


def eval_agents_md(agent: MemoryAgent, path: str = "AGENTS.md") -> dict:
    """实测"全量加载 agent"的加载效果：

    1) 内容覆盖检查（离线）：对每个主题问题，检查 AGENTS.md 全文是否包含
       回答所需关键词——模拟全量加载后信息是否可及；
    2) LLM 问答（可选）：配置了 OPENAI_API_KEY 时，把文档作为上下文让 LLM
       回答同一组问题，检查答案是否含期望关键词（真实的提取质量验证）。
    """
    with open(path, encoding="utf-8") as f:
        doc = f.read()
    report = {"coverage": [], "llm": []}
    for q, kws in EVAL_QUESTIONS:
        ok = all(kw in doc for kw in kws)
        report["coverage"].append({"question": q, "ok": ok, "missing": [k for k in kws if k not in doc]})
    n_cov = sum(1 for r in report["coverage"] if r["ok"])
    # LLM 深度验证（可选）：把整个文档作为上下文回答同一组问题
    llm_on = bool(os.environ.get("OPENAI_API_KEY"))
    if llm_on:
        for q, kws in EVAL_QUESTIONS:
            answer = _ask_llm(doc, q)
            hit = any(k in (answer or "") for k in kws)
            report["llm"].append({"question": q, "answer": answer, "ok": hit})
    print(f"== 加载效果评估（{path}，共 {len(doc)} 字符）==")
    print(f"① 内容覆盖：{n_cov}/{len(EVAL_QUESTIONS)} 个问题的答案关键词在文件中可及")
    for r in report["coverage"]:
        tag = "✓" if r["ok"] else f"✗ 缺 {r['missing']}"
        print(f"   {tag} {r['question']}")
    if llm_on:
        n_llm = sum(1 for r in report["llm"] if r["ok"])
        print(f"② LLM 问答（基于文档上下文）：{n_llm}/{len(EVAL_QUESTIONS)} 回答含期望要点")
        for r in report["llm"]:
            tag = "✓" if r["ok"] else "✗"
            print(f"   {tag} {r['question']} → {(r['answer'] or '')[:60]}")
    else:
        print("② LLM 问答：跳过（未设置 OPENAI_API_KEY；设置后可验证真实提取质量）")
    print("结论：全量加载 agent 开工读取本文件后，回答上述问题所需信息"
          + ("全部可及" if n_cov == len(EVAL_QUESTIONS) else f"仅有 {n_cov}/{len(EVAL_QUESTIONS)} 可及"))
    return report


def eval_recall_chain(agent: MemoryAgent | None = None) -> dict:
    """唤醒链路连续性检查（收工验证的一部分，随 --sync --eval 联动）。

    跑 recall_curve_check 的判别场景（Cold 纯衰减段 → 唤醒 → 重建段），用数据
    断言唤醒不产生曲线断层——recall 的 move 语义 + 全量继承（history 轨迹）
    必须在机制层面持续成立；并统计当前记忆库中唤醒过的长生命周期记忆
    （修订/观测轨迹可追溯）与唤醒偏差信号（awakenings 四元组按类型聚合的
    dev/expected 分布 + 信号方向一致性，见 awakening_signal_stats）。
    返回结论 dict（不改变退出码，与加载评估一致）。
    """
    from recall_curve_check import build_scenario, verify_continuity

    _a, before, after, _cold = build_scenario()
    verdict = verify_continuity(before, after)
    n = len(before)
    print("== 唤醒链路连续性检查 ==")
    print(f"① 唤醒前 Cold 采样 {n} 条（纯衰减段）")
    print(f"② 唤醒后采样 {len(after)} 条（继承 {n} + 唤醒 1 + 尾部 {len(after) - n - 1}）")
    print(f"③ 前缀逐位一致（Cold 衰减段完整继承）: {verdict['prefix_ok']}")
    if verdict["jump"]:
        t_prev, s_prev = before[-1][0], before[-1][1]
        tj, sj = verdict["jump"][0], verdict["jump"][1]
        print(f"④ 唤醒点 t={tj:.0f}: 强度 {s_prev:.3f} → {sj:.3f}（测试效应跳升）")
    print(f"⑤ 重建段尾部继续衰减: {verdict['tail_decays']}")
    ok = bool(verdict["prefix_ok"] and verdict["tail_decays"])
    print("结论: " + ("✔ 无缝衔接——Cold 衰减段与 Warm 重建段是同一记忆的连续轨迹"
                      if ok else "✘ 存在断层"))
    if agent is not None:
        awakened = [m for m in agent.store.all() if m.awakened_at is not None]
        n_rev = sum(m.revision_count for m in awakened)
        n_hist = sum(len(m.history) for m in awakened)
        print(f"⑥ 当前记忆库长生命周期记忆 {len(awakened)} 条"
              f"（累计修订 {n_rev} 次 / 观测轨迹 {n_hist} 条，可追溯）")
        stats = awakening_signal_stats(agent)
        total_events = sum(v.get("events", 0) for v in stats.values())
        print(f"⑦ 唤醒偏差信号（真实记忆库 awakenings 扫描，共 {total_events} 条观测）:")
        if total_events == 0:
            print("   无唤醒观测——本库尚未产生（recall 唤醒时自动记录 [ts, dev, expected, 类型]）")
        for t in ("skill", "semantic", "episodic"):
            s = stats[t]
            if s["events"] == 0:
                print(f"   {t:<8} 无观测")
                continue
            dmin, dmed, dmax = s["dev"]
            emin, emed, emax = s["expected"]
            ratio = f" ratio {s['ratio_med']}" if s["ratio_med"] is not None else ""
            print(f"   {t:<8} {s['events']} 条  dev 中位 {dmed:.4f}"
                  f"（{dmin:.4f}~{dmax:.4f}）  expected 中位 {emed:.4f}"
                  f"（{emin:.4f}~{emax:.4f}）{ratio}")
            print(f"           方向: 上调 {s['up']} · 下调 {s['down']} · 持平 {s['flat']}  "
                  f"主导 {_DIR_LABELS[s['dominant']]}（一致性 {s['consistency']:.0%}）")
            if s["consistency"] >= 0.6 and s["dominant"] != "flat":
                hint = ("该类型唤醒比类型预期剧烈：τ 配置偏大、可塑性配置偏小"
                        if s["dominant"] == "up"
                        else "该类型唤醒比类型预期温和：τ 配置偏小、可塑性配置偏大")
                print(f"           一致性 ≥ 60% → 提示: {hint}（学习器开启时自动校准）")
        verdict = {**verdict, "awakened_in_store": len(awakened),
                   "store_revisions": n_rev, "store_history": n_hist,
                   "awakening_signal": stats}
    return verdict


_DIR_LABELS = {"up": "上调", "down": "下调", "flat": "持平"}


def awakening_signal_stats(agent: MemoryAgent, *args, **kwargs) -> dict:
    """从真实记忆库扫描 awakenings 四元组，统计各类型实测/预期偏差分布与信号方向一致性。

    委托 memagent.agent.awakening_signal_stats（单一事实源，CLI /types、
    仪表盘画像列与收工验证共用），支持其时间窗参数（window_seconds / since /
    until / now）。每类型返回 {"events", "dev": [min, 中位, max], "expected":
    [min, 中位, max], "ratio_med": 中位 dev/expected, "up"/"down"/"flat":
    方向计数, "dominant": 主导方向, "consistency": 主导方向占比}；无观测的
    类型返回 {"events": 0}。
    """
    from memagent.agent import awakening_signal_stats as _core

    return _core(agent, *args, **kwargs)


def _signals_print_suffix(ex: dict) -> str:
    """导出完成提示的括号内容：观测/事件数 + 漂移窗口 + τ 一致性 + 行动建议。"""
    s = ex["health"]["summary"]
    sug: dict[str, int] = {}
    for h in ex["health"]["by_type"].values():
        sug[h["suggest"]] = sug.get(h["suggest"], 0) + 1
    action = " · ".join(f"{k}{v}" for k, v in sug.items()
                        if k in ("τ↓", "τ↑", "需检查", "需补观测"))
    return (f"{sum(v.get('events', 0) for v in ex['stats'].values())} 条观测、"
            f"{len(ex['events'])} 条事件明细，含近 "
            f"{ex['periods']['recent_seconds'] / 86400:g} 天漂移对比；"
            f"τ 两路一致性 一致 {s['agree']} · 冲突 {s['conflict']} · "
            f"单源 {s['one_sided']} · 无信号 {s['no_data']}"
            f"；建议 {action or '无行动项'}")


def _warn_conflict_types(ex: dict) -> None:
    """导出后检测两路信号冲突的类型：打印告警 + 两路原始证据行——收工验证直接
    指出需排查的类型（学习器仍会按置信度加权折中，但冲突意味着观测可能被污染）。
    无冲突类型时静默。
    """
    warnings = (ex.get("health") or {}).get("warnings") or []
    if not warnings:
        return
    print("⚠ 需排查类型（两路信号冲突）:")
    for w in warnings:
        print(f"  ✘ {w['mtype']}: {w['clean_evidence']}；{w['awakening_evidence']}")
        print(f"    → {w['suggestion']}")


_ADJUST_ACTIONS = ("τ↓", "τ↑", "需检查")
from memagent.agent import _ADJUST_CN  # 单一事实源：行动项中文语义（agent.py）


def _strict_exit_code(ex: dict, strict: bool) -> int:
    """--strict 双档退出码：CI 区分"需排查"与"需校准"两种红灯。

    - warnings 非空（冲突类型）→ **1 需排查**（先检查观测污染/事件注入，
      不自动调参——最高优先级，同时存在时覆盖）；
    - 无冲突但存在 τ↓/τ↑ 行动项 → **2 需校准**（非阻塞红灯，可直接
      --apply-suggestions 执行）；
    - 两者皆无 → 0（无信号/仅需补观测，不阻塞）。
    未开 --strict → 0（向后兼容，默认行为不变）。
    """
    if not strict:
        return 0
    h = ex.get("health") or {}
    n_conflict = len(h.get("warnings") or [])
    n_calib = sum(1 for a in (h.get("actions") or [])
                  if a.get("suggest") in ("τ↓", "τ↑"))
    if n_conflict:
        print(f"⚠ --strict: {n_conflict} 个冲突类型 → 退出码 1（需排查，CI 阻塞）")
        return 1
    if n_calib:
        print(f"⚠ --strict: {n_calib} 个类型需校准（τ↓/τ↑）→ 退出码 2（需校准，"
              f"可 --apply-suggestions 执行）")
        return 2
    return 0


def _read_suggest_adjust(csv_path: str) -> list[tuple[str, str]]:
    """读导出 CSV 的 `suggest_adjust` 列：返回 [(类型, 建议), ...]。

    行动清单以**写出的 CSV 文件**为数据源（导出 → 读回 → 行动），顺带验证
    落盘产物与 health 一致；文件缺失/损坏 → 回退空表。
    """
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            return [(r["mtype"], r["suggest_adjust"])
                    for r in csv.DictReader(f) if r.get("suggest_adjust")]
    except (OSError, ValueError):
        return []


def _attach_action_events(health: dict, events_csv: str) -> dict:
    """给 agent 层已附的冲突成因事件（warnings/actions 同源，行号/相对时间已由
    agent.py 统一算出）补来源文件（csv 文件名）——CI 读导出 JSON 按行号可翻
    `{base}_events.csv` 核对。无事件的类型不附加。"""
    for holder in [*health.get("actions", []), *health.get("warnings", [])]:
        for ev in holder.get("events") or []:
            ev["csv"] = events_csv
    return health


def _print_adjust_actions(ex: dict) -> None:
    """收工验证行动清单：读导出 CSV 的 suggest_adjust 列，检测到 τ↓ / τ↑ /
    需检查 时打印行动清单并提示跑 sleep() 让学习器实际校准。检测到需检查时
    额外打印该冲突类型的唤醒事件明细（引用 events CSV 对应行）——排障不用
    再翻文件。无行动项时静默。
    """
    actions = [p for p in _read_suggest_adjust(ex["csv"])
               if p[1] in _ADJUST_ACTIONS]
    if not actions:
        return
    print("== 行动清单（suggest_adjust）==")
    conf = {t: h.get("confidence", "—")
            for t, h in ex["health"]["by_type"].items()}
    for t, sug in actions:
        if sug == "需检查":
            print(f"  ⚠ {t}: {_ADJUST_CN[sug]}（证据见上方告警）")
            warn = next((w for w in (ex["health"].get("warnings") or [])
                         if w["mtype"] == t), None)
            evs = (warn or {}).get("events") or []   # 行号直接来自 health
            if evs:
                first, last = evs[0]["row"], evs[-1]["row"]
                rng = f"{first}" if first == last else f"{first}-{last}"
                print(f"      唤醒事件明细（{len(evs)} 条 → "
                      f"{ex['events_csv']} 第 {rng} 行）:")
                d_cn = {"down": "↓ 应下调", "up": "↑ 应上调",
                        "flat": "= 已校准", "legacy": "— 旧格式"}
                for n, e in enumerate(evs, 1):
                    ratio = e.get("ratio")
                    ratio_txt = f"{ratio:.3f}" if ratio is not None else "—"
                    rel = e.get("ts_relative_seconds")
                    rel_txt = (f"{rel / 3600:+.1f}h" if rel is not None else "—")
                    print(f"        #{n} [行 {e['row']}] 记忆 {e['memory_id']} · "
                          f"相对 {rel_txt} · dev {e['dev']} vs 预期 "
                          f"{e['expected']} · 比值 {ratio_txt} "
                          f"{d_cn.get(e['direction'], '— 未判定')}")
            else:
                print(f"      （{t} 无事件级唤醒明细）")
        else:
            print(f"  → {t}: {sug}（{conf[t]} · {_ADJUST_CN[sug]}）"
                  f"→ 跑 sleep() 让学习器实际校准")
    print("  提示: sleep() 末尾自动触发 learn_tau + learn_plasticity，按当前信号"
          "校准 τ 与 drift——CLI 里用 python -m memagent 后 /sleep，或调 "
          "MemoryAgent.sleep() 后查看 cfg.tau_by_type；置信度弱 = 观测不足，"
          "建议先积累唤醒观测再校准")


def apply_suggestions(agent: MemoryAgent, ex: dict, yes: bool = False,
                      duration: float = 6 * 3600) -> dict:
    """--apply-suggestions：按 suggest_adjust 行动项批量执行校准。

    行动项 = suggest ∈ {τ↓, τ↑} 的类型（agree 且方向明确——可信可执行）；
    冲突（需检查）需先排查、单源（需补观测）观测不足、无信号无可调——均不
    自动执行。确认后跑一次 sleep()——末尾自动触发 learn_tau + learn_plasticity，
    按当前观测信号实际校准（学习器门控不足的类型自然跳过，与建议方向无关）。
    校准结果写入 store.meta（learned_tau / learned_plasticity）并 save() 落盘
    ——真正的 apply：后续会话重启即加载。打印校准前后各类型 τ/drift 对比。
    返回 {"actionable", "skipped", "cancelled", "before", "after",
    "deltas", "sleep_report"}。
    """
    h = ex["health"]["by_type"]
    actionable = [t for t in MemType if h[t.value]["suggest"] in ("τ↓", "τ↑")]
    skipped = {t.value: h[t.value]["suggest"] for t in MemType
               if h[t.value]["suggest"] not in ("τ↓", "τ↑", "无信号")}
    print("== 行动执行（--apply-suggestions）==")
    if not actionable:
        print("  无 τ↓/τ↑ 行动项——冲突需先排查、单源需补观测、无信号无可调，"
              "均不自动执行")
        if skipped:
            print("  跳过: " + "、".join(
                f"{t}（{_ADJUST_CN.get(s, s)}）" for t, s in skipped.items()))
        return {"actionable": [], "skipped": skipped, "cancelled": False,
                "before": {}, "after": {}, "deltas": {}, "sleep_report": None}
    names = "、".join(f"{t.value}（{h[t.value]['suggest']} · 置信度 "
                       f"{h[t.value].get('confidence', '—')}）" for t in actionable)
    print(f"  行动项: {names}")
    if skipped:
        print("  跳过: " + "、".join(
            f"{t}（{_ADJUST_CN.get(s, s)}）" for t, s in skipped.items()))
    if not yes:
        try:
            ans = input(f"  确认对 {len(actionable)} 个类型批量执行 sleep() 校准？[y/N] ")
            ans = ans.strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("  已取消——未执行校准（加 --yes 跳过确认）")
            return {"actionable": actionable, "skipped": skipped, "cancelled": True,
                    "before": {}, "after": {}, "deltas": {}, "sleep_report": None}
    before = _snapshot_calibration(agent)
    sleep_report = agent.sleep(duration=duration)
    after = _snapshot_calibration(agent)
    deltas = {t: {k: round(after[t][k] - before[t][k], 3) for k in before[t]}
              for t in before if before[t] != after[t]}
    for t in MemType:
        b, a = before[t.value], after[t.value]
        d = deltas.get(t.value)
        if not d:
            continue
        print(f"   {t.value:<9} τ {b['tau_d']:.3f} → {a['tau_d']:.3f} 天"
              f"（Δ{d['tau_d']:+.3f}）  ·  drift {b['drift']:.3f} → "
              f"{a['drift']:.3f}（Δ{d['drift']:+.3f}）")
    if deltas:
        print(f"  校准完成: {'、'.join(deltas)} 参数已按信号调整并持久化到记忆库")
    else:
        print("  本次 sleep 无参数变化——信号未达学习器门控，继续积累观测")
    agent.store.save()   # 落盘：learned_tau / learned_plasticity 下次重启即加载
    return {"actionable": actionable, "skipped": skipped, "cancelled": False,
            "before": before, "after": after, "deltas": deltas,
            "sleep_report": sleep_report}


def _split_exclude(specs: str | None) -> list[str]:
    """CLI 逗号分隔的排除列表 → 条目列表（None/空 → []）。"""
    if not specs:
        return []
    return [s.strip() for s in specs.split(",") if s.strip()]


def _parse_aggregations(specs: list[str]) -> list[dict]:
    """解析 --aggregations 'TYPE:memid:idx,memid:idx'（可多次）→
    [{"mtype", "events": [key, ...]}]（事件 key = memory_id:序号，仪表盘
    Shift 多选即此 key）。非法条目跳过。"""
    out: list[dict] = []
    for spec in specs:
        mtype, _, keys_spec = spec.partition(":")
        keys = [k.strip() for k in keys_spec.split(",") if k.strip()]
        if mtype and keys:
            out.append({"mtype": mtype.strip(), "events": keys})
        else:
            print(f"  ⚠ 忽略非法聚合条目（应为 TYPE:key,key）: {spec}")
    return out


def _load_aggregations_file(path: str) -> list[dict]:
    """从 JSON 读取聚合选择集（仪表盘「导出聚合」一键生成的格式）：
    [{"mtype", "events": ['memory_id:序号', ...]}, ...]；也接受
    {"aggregations": [...]} 包裹。免去手写 memory_id 列表——仪表盘圈出的
    选择集直接喂给 --aggregations-file。非法条目/缺文件打印提示并跳过。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠ 读取聚合文件失败（{e}）: {path}")
        return []
    if isinstance(data, dict):
        data = data.get("aggregations", [])
    if not isinstance(data, list):
        print(f"  ⚠ 聚合文件应为规格列表或含 aggregations 键的对象: {path}")
        return []
    out: list[dict] = []
    for spec in data:
        if not isinstance(spec, dict) or not spec.get("mtype") or not spec.get("events"):
            print(f"  ⚠ 忽略非法聚合条目: {spec}")
            continue
        out.append({"mtype": str(spec["mtype"]),
                    "events": [str(k) for k in spec["events"]]})
    print(f"  已从 {path} 读取 {len(out)} 条聚合规格（--aggregations-file）")
    return out


def _aggregation_specs(args) -> list[dict]:
    """合并 CLI --aggregations 与 --aggregations-file 的聚合规格（仪表盘导出
    JSON 免手写 memory_id 列表）。"""
    specs = _parse_aggregations(args.aggregations)
    if getattr(args, "aggregations_file", None):
        specs += _load_aggregations_file(args.aggregations_file)
    return specs


def _aggregation_for(agent: MemoryAgent, health: dict, spec: dict) -> dict:
    """委托 memagent.agent.aggregation_for（单一事实源：--aggregations 导出与
    仪表盘回放共用同一判定）。"""
    from memagent.agent import aggregation_for as _core

    return _core(agent, health, spec)


def _recompute_excluded(agent: MemoryAgent, keys: list[str],
                        recent_seconds: float) -> dict:
    """委托 memagent.agent.aggregation_recompute（单一事实源：resolved 聚合的
    剔除后证据包，--exclude-events 同链路）。"""
    from memagent.agent import aggregation_recompute as _core

    return _core(agent, keys, recent_seconds)


def _print_excluded(ex: dict) -> None:
    """排除事件后打印重算说明（含重判后的 health 结论提示）。"""
    exl = ex.get("excluded") or []
    if not exl:
        return
    print(f"⚠ 已排除 {len(exl)} 起事件（{', '.join(exl)}）→ 唤醒统计/漂移/"
          f"health 按剩余事件重算")


def _print_aggregations(ex: dict) -> None:
    """打印回放到 health.aggregations 的聚合结论（人工排查可回放）。"""
    aggs = (ex.get("health") or {}).get("aggregations") or []
    if not aggs:
        return
    print("== 聚合结论回放（health.aggregations）==")
    for a in aggs:
        sel = a["selected_n"]
        d = a["selected_dist"]
        rem_txt = (f"移除后剩余 {a['remaining_n']} 起中位 "
                   f"{a['remaining_median_ratio']:.3f} → {a['remaining_direction']}"
                   if a["remaining_median_ratio"] is not None
                   else f"移除后剩余 {a['remaining_n']} 起（观测不足）")
        print(f"  {a['mtype']}: 已选 {sel} 起（↑{d['up']} · ↓{d['down']} · ={d['flat']}）"
              f" · 干净段 {a['clean_direction']} · {rem_txt}\n"
              f"    → {a['verdict_text']}（{a['verdict']}）")
        rc = a.get("recomputed")
        if rc:
            rh = rc["health"]
            bt = rh["by_type"].get(a["mtype"], {})
            print(f"    → 已自动附带剔除后重算证据包（exclude "
                  f"{len(rc['excluded'])} 起）: consistency="
                  f"{bt.get('consistency')} · 唤醒 "
                  f"{bt.get('awakening', {}).get('direction')} · "
                  f"suggest={bt.get('suggest')} · "
                  f"warnings={len(rh.get('warnings', []))} · "
                  f"actions={len(rh.get('actions', []))} · "
                  f"summary={rh.get('summary')}")


def _parse_exclude_events(specs: list[str]) -> set:
    """解析 --exclude-events 'memory_id:index,...' → {(memory_id, 事件序号)}。
    非法条目（缺冒号/非数字序号）静默跳过（打印提示由调用方负责）。"""
    out: set = set()
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            print(f"  ⚠ 忽略非法排除条目（应为 memory_id:序号）: {spec}")
            continue
        mem_id, _, idx = spec.partition(":")
        try:
            out.add((mem_id, int(idx)))
        except ValueError:
            print(f"  ⚠ 忽略非法排除序号: {spec}")
    return out


def _exclude_compare(h_before: dict, h_after: dict) -> dict:
    """剔除前后对比块（health.exclude_compare）：每类型 before/after 的一致性、
    干净段/唤醒方向与建议并列，供 CI 直接读结论（--exclude-events /
    --exclude-clashes 重判后是否变一致）。before = 无排除基线，after = 重判后。"""
    def _row(h: dict) -> dict:
        rows = {}
        for t in ("skill", "semantic", "episodic"):
            b = h.get("by_type", {}).get(t) or {}
            rows[t] = {
                "consistency": b.get("consistency"),
                "clean_direction": (b.get("clean") or {}).get("direction"),
                "awakening_direction": (b.get("awakening") or {}).get("direction"),
                "suggest": b.get("suggest"),
            }
        return {
            "by_type": rows,
            "summary": h.get("summary"),
            "warnings_n": len(h.get("warnings", [])),
            "actions_n": len(h.get("actions", [])),
        }

    return {"before": _row(h_before), "after": _row(h_after)}


def _print_exclude_compare(ex: dict) -> None:
    """打印剔除前后对比（health.exclude_compare）：每类型 before → after。"""
    ec = (ex.get("health") or {}).get("exclude_compare")
    if not ec:
        return
    print("== 剔除前后对比（health.exclude_compare）==")
    for t in ("skill", "semantic", "episodic"):
        b, a = ec["before"]["by_type"][t], ec["after"]["by_type"][t]
        if b == a:
            continue
        print(f"  {t}: {b.get('consistency')} → {a.get('consistency')}"
              f"（干净段 {b.get('clean_direction')} → {a.get('clean_direction')} · "
              f"唤醒 {b.get('awakening_direction')} → {a.get('awakening_direction')} · "
              f"建议 {b.get('suggest')} → {a.get('suggest')}）")
    sb, sa = ec["before"]["summary"], ec["after"]["summary"]
    print(f"  summary: 一致 {sb.get('agree', 0)} → {sa.get('agree', 0)} · "
          f"冲突 {sb.get('conflict', 0)} → {sa.get('conflict', 0)} · "
          f"warnings {ec['before']['warnings_n']} → {ec['after']['warnings_n']}")


def _print_clashes(ex: dict) -> None:
    """打印 --exclude-clashes 自动剔除说明（dir ≠ clean 冲突成因 + 重判后状态）。"""
    exl = ex.get("excluded_clashes") or []
    if not exl:
        return
    h = ex["health"]
    s = h.get("summary", {})
    acts = [f"{a['mtype']} {a['suggest']}" for a in h.get("actions", [])]
    tail = f" · 行动 {', '.join(acts)}" if acts else ""
    print(f"⚠ 已按 dir ≠ clean 自动排除 {len(exl)} 起冲突成因事件"
          f"（{', '.join(exl)}）→ 重判后: 一致 {s.get('agree', 0)} · 冲突 "
          f"{s.get('conflict', 0)} · 单源 {s.get('one_sided', 0)} · 无信号 "
          f"{s.get('no_data', 0)}{tail}")


def export_signals(agent: MemoryAgent, base: str = "awakening_signals",
                   recent_seconds: float = 30 * 86400,
                   exclude_events: list[str] | None = None,
                   aggregations: list[dict] | None = None,
                   exclude_clashes: bool = False) -> dict:
    """把唤醒信号统计导出为 JSON + CSV + 事件明细（--sync --eval
    --export-signals）。

    与收工验证第 ⑦ 节同一数据源（awakening_signal_stats），并追加信号漂移
    时段对比（awakening_signal_periods：近 30 天 vs 更早的方向一致性）与
    **τ 学习器两路信号方向一致性**（tau_learner_health：干净段反推 vs 唤醒
    偏差代理）——信号导出与健康检查合表，供外部工具对信号做进一步分析。
    `exclude_events` = ['memory_id:序号', ...]（事件在 _awakening_events 过滤
    列表中的序号，仪表盘证据行即此序号）——排除事件后**重算唤醒统计/漂移/
    health**（冲突剔除假设检验：可疑事件排除后两路是否一致），事件明细与
    行号同步重排。`aggregations` = [{"mtype", "events": ['memory_id:序号', ...]},
    ...]——把仪表盘 Shift 多选聚合结论**回放为 health.aggregations**（事件子集
    + 全体/移除后中位与方向 + 判定，与仪表盘 JS 判定同闸门），CI 可直接读
    人工排查结论。写 {base}.json（完整统计 + 漂移对比 + 逐事件明细 + health
    健康检查）、{base}.csv（每类型一行，全字段自包含，含 τ 两路方向列）与
    {base}_events.csv（每条唤醒事件一行：ts / 类型 / dev / expected / 比值 /
    来源记忆 id，按时间排序，外部工具可自行按任意时间窗重切片）。返回
    {"json", "csv", "events_csv", "stats", "periods", "events", "health",
    "excluded"}。health 含 warnings（冲突告警）与 actions（行动清单：suggest
    ∈ {τ↓, τ↑, 需检查} 的类型 + 置信度 + 语义，与终端行动清单/CSV suggest_adjust
    列同源）——CI 按 warnings / actions 非空直接判定需处理。
    """
    from memagent.agent import awakening_signal_periods, awakening_signal_stats
    from memagent.visualize import _awakening_events

    exclude = _parse_exclude_events(exclude_events or [])
    stats = awakening_signal_stats(agent, exclude=exclude)
    periods = awakening_signal_periods(agent, recent_seconds=recent_seconds,
                                       exclude=exclude)
    health = tau_learner_health(agent, exclude=exclude)
    excluded_clashes: list[str] = []
    if exclude_clashes:   # 按 dir ≠ clean 自动剔除冲突成因（与仪表盘「选反向」同判定）
        from memagent.agent import clash_event_keys

        clash = clash_event_keys(agent, health, exclude=exclude)
        if clash:
            exclude |= clash
            excluded_clashes = sorted(f"{m}:{i}" for m, i in clash)
            stats = awakening_signal_stats(agent, exclude=exclude)
            periods = awakening_signal_periods(
                agent, recent_seconds=recent_seconds, exclude=exclude)
            health = tau_learner_health(agent, exclude=exclude)
    if exclude:   # 有实际排除 → 附剔除前后对比（before = 无排除基线）
        health["exclude_compare"] = _exclude_compare(
            tau_learner_health(agent), health)
    if aggregations:   # 仪表盘 Shift 多选聚合结论回放（与 JS 判定同闸门）
        aggs = []
        for spec in aggregations:
            agg = _aggregation_for(agent, health, spec)
            if agg["verdict"] == "resolved":   # 冲突成因圈定 → 一步附带剔除后证据包
                agg["recomputed"] = _recompute_excluded(agent, agg["events"],
                                                         recent_seconds)
            aggs.append(agg)
        health["aggregations"] = aggs
    now = round(agent._now(), 3)
    # 事件级明细：跨全部记忆收集（含来源记忆 id），按事件时刻排序
    events: list[dict] = []
    for mem in sorted(agent.store.all(), key=lambda m: m.id):
        for idx, ev in enumerate(_awakening_events(mem)):
            if (mem.id, idx) in exclude:   # 排除事件不进入明细/行号重排
                continue
            events.append({
                "memory_id": mem.id,
                "mtype": ev["mtype"],
                "ts": ev["ts"],
                "ts_relative_seconds": round(ev["ts"] - now, 1),
                "dev": ev["dev"],
                "expected": ev["expected"],
                "ratio": ev["ratio"],
                "dt": ev.get("dt"),       # 六元组日志才有；四元组为 None
                "n_cold": ev.get("n_cold"),
            })
    events.sort(key=lambda e: (e["ts"], e["memory_id"]))
    events_path = f"{base}_events.csv"
    health = _attach_action_events(health, events_path)
    payload = {
        "now": now,
        "recent_seconds": recent_seconds,
        "stats": stats,
        "periods": periods,
        "events": events,
        "health": health,
        "excluded": sorted(f"{k}:{v}" for k, v in exclude),
        "excluded_clashes": excluded_clashes,
    }
    json_path = f"{base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    def _tri(v: list | None) -> tuple:
        return tuple(v) if v else ("", "", "")

    csv_path = f"{base}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mtype", "events", "dev_min", "dev_med", "dev_max",
                    "exp_min", "exp_med", "exp_max", "ratio_med",
                    "up", "down", "flat", "dominant", "consistency",
                    "recent_events", "recent_dominant", "recent_consistency",
                    "earlier_events", "earlier_dominant", "earlier_consistency",
                    "drift_verdict", "direction_changed", "consistency_delta",
                    "clean_n", "clean_tau_est_s", "clean_cfg_tau_s",
                    "clean_direction", "awakening_direction", "tau_consistency",
                    "suggest_adjust", "suggest_confidence"])
        for t in ("skill", "semantic", "episodic"):
            s = stats[t]
            d = periods["by_type"][t]
            r, e = d["recent"], d["earlier"]
            h = health["by_type"][t]
            cl = h["clean"]
            w.writerow([
                t,
                s.get("events", 0),
                *_tri(s.get("dev")),
                *_tri(s.get("expected")),
                s.get("ratio_med", ""),
                s.get("up", 0), s.get("down", 0), s.get("flat", 0),
                s.get("dominant", ""), s.get("consistency", ""),
                r.get("events", 0), r.get("dominant", ""),
                r.get("consistency", ""),
                e.get("events", 0), e.get("dominant", ""),
                e.get("consistency", ""),
                d["verdict"], d["direction_changed"],
                d.get("consistency_delta", ""),
                cl.get("n", 0), cl.get("tau_est", ""),
                cl.get("cfg_tau", ""),
                cl.get("direction", ""),
                h["awakening"].get("direction", ""),
                h["consistency"],
                h["suggest"],
                h.get("confidence", "—"),
            ])

    # 事件级 CSV：每条唤醒事件一行（含来源记忆 id），按时间排序供任意窗口重切片
    with open(events_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["memory_id", "mtype", "ts", "ts_relative_seconds",
                    "dev", "expected", "ratio", "dt_seconds",
                    "retrievals_before"])
        for e in events:
            w.writerow([
                e["memory_id"], e["mtype"], round(e["ts"], 1),
                e["ts_relative_seconds"],
                e["dev"], e["expected"],
                e["ratio"] if e["ratio"] is not None else "",
                e["dt"] if e["dt"] is not None else "",
                e["n_cold"] if e["n_cold"] is not None else "",
            ])
    return {"json": json_path, "csv": csv_path, "events_csv": events_path,
            "stats": stats, "periods": periods, "events": events,
            "health": health, "excluded": sorted(
                f"{k}:{v}" for k, v in exclude),
            "excluded_clashes": excluded_clashes}



from memagent.agent import _TAU_DIR_CN  # 单一事实源：方向中文标签（agent.py）


def _tau_dir(tau_est: float, cfg_tau: float) -> str:
    """委托 memagent.agent._tau_dir（单一事实源）。"""
    from memagent.agent import _tau_dir as _core

    return _core(tau_est, cfg_tau)


def _ratio_dir(ratio: float) -> str:
    """委托 memagent.agent._ratio_dir（单一事实源）。"""
    from memagent.agent import _ratio_dir as _core

    return _core(ratio)


def _adjust_suggestion(h: dict) -> str:
    """委托 memagent.agent._adjust_suggestion（单一事实源）。"""
    from memagent.agent import _adjust_suggestion as _core

    return _core(h)


def tau_learner_health(agent: MemoryAgent, *args, **kwargs) -> dict:
    """τ 学习器健康检查：从真实记忆库扫描两路信号的**方向一致性**。

    委托 memagent.agent.tau_learner_health（单一事实源，导出合表 / 仪表盘画像列
    与收工验证共用），支持 exclude 排除集（{(memory_id, 事件序号)}——排除事件
    后重算唤醒源与冲突明细）。返回 {"by_type": {类型: {"clean": {"n", "tau_est",
    "cfg_tau", "direction"}, "awakening": {"n", "ratio_med", "direction"},
    "consistency", "suggest"}}, "summary": {"agree", "conflict",
    "one_sided", "no_data"}, "warnings", "actions"}。纯读取不改库。
    """
    from memagent.agent import tau_learner_health as _core

    return _core(agent, *args, **kwargs)


def eval_tau_learner(agent: MemoryAgent | None = None) -> dict:
    """τ 学习器健康检查（收工验证的一部分，随 --sync --eval 联动）。

    扫描真实记忆库的 fit_report（干净段反推）与 awakenings（唤醒偏差代理），
    报告两路信号的方向一致性——方向一致 = 学习器收敛方向明确（互相印证），
    冲突 = 需要检查，单源 = 无法交叉印证。纯读取，不修改记忆库。
    """
    print("== τ 学习器健康检查（两路信号方向一致性）==")
    if agent is None:
        print("  （未提供记忆库 agent——跳过）")
        return {}
    print(f"  学习器状态: tau_learning={'开' if agent.cfg.tau_learning else '关'} · "
          f"唤醒源={'开' if agent.cfg.tau_from_awakenings else '关'}")
    health = tau_learner_health(agent)
    for t in MemType:
        h = health["by_type"][t.value]
        cl, aw, cs = h["clean"], h["awakening"], h["consistency"]
        if cl["direction"] is None and aw["direction"] is None:
            print(f"   {t.value:<9} 无信号（干净段 {cl['n']} 条 / 唤醒 {aw['n']} 条）")
            continue
        if cl["direction"] is not None:
            print(f"   {t.value:<9} 干净段 {cl['n']} 条 实测τ≈{cl['tau_est']:.2f}s"
                  f" vs 配置 {cl['cfg_tau']:.2f}s → {_TAU_DIR_CN[cl['direction']]}")
        else:
            print(f"   {t.value:<9} 干净段不足（{cl['n']} 条 < "
                  f"{agent.cfg.tau_min_segments}）")
        if aw["direction"] is not None:
            print(f"           唤醒 {aw['n']} 条 ratio {aw['ratio_med']:.3f} "
                  f"→ {_TAU_DIR_CN[aw['direction']]}")
        else:
            print(f"           唤醒 无观测")
        if cs == "agree":
            if cl["direction"] == "flat":
                print(f"           ✔ 两路一致：都无明显偏差 → τ 已校准")
            else:
                print(f"           ✔ 两路一致：都指向「τ 配置"
                      f"{'偏大' if cl['direction'] == 'down' else '偏小'}」"
                      f" → 学习器收敛方向明确")
        elif cs == "conflict":
            print(f"           ✘ 两路冲突：干净段说「{_TAU_DIR_CN[cl['direction']]}」、"
                  f"唤醒说「{_TAU_DIR_CN[aw['direction']]}」"
                  f" → 检查观测污染 / 事件注入（学习器按置信度加权折中）")
        elif cs == "one_sided":
            print(f"           △ 单源：无法交叉印证"
                  f"（可 /recall 唤醒累积唤醒观测，或积累干净衰减段）")
    s = health["summary"]
    print(f"  摘要: 一致 {s['agree']} · 冲突 {s['conflict']} · 单源 {s['one_sided']} · "
          f"无信号 {s['no_data']}")
    if s["conflict"]:
        print("  ⚠ 存在两路信号冲突的类型——学习器仍会按置信度加权折中，"
              "但建议检查该类型的观测是否被污染（如检索与衰减混在一起）。")
    return health


def _snapshot_calibration(agent: MemoryAgent) -> dict:
    """各类型 τ（天）与 drift 因子的快照，供校准前后对比。"""
    out = {}
    for t in MemType:
        out[t.value] = {
            "tau_d": round(agent.cfg.tau_for(t) / 86400.0, 3),
            "drift": round(agent.cfg.reconsolidation_factor(t, "drift"), 3),
        }
    return out


def _learner_demo_agent() -> MemoryAgent:
    """受控合成 agent：配置信念 vs 已知真实参数失配 + 4 次唤醒观测。

    真实 τ=2 天 / 配置 3 天、真实 drift=3.5 / 信念 1.0——sleep() 末尾的
    learn_tau/learn_plasticity 应朝真实值方向校准。只造唤醒观测不预训练，
    让单次 sleep 的响应幅度一目了然。
    """
    clock = [0.0]
    cfg = AgentConfig(
        tau_by_type={MemType.EPISODIC: 3 * 86400.0},
        true_tau_by_type={MemType.EPISODIC: 2 * 86400.0},
        reconsolidation_by_type={MemType.EPISODIC: {"drift": 1.0, "importance": 1.0}},
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 3.5, "importance": 1.0}},
        tau_learning_rate=0.2,
        joint_awakening=True,
        awakening_plasticity_gain=0.5,
    )
    ag = MemoryAgent(cfg=cfg, now_fn=lambda: clock[0])
    m = ag.store.add("我昨天去吃了火锅", importance=0.3, mtype=MemType.EPISODIC)
    m.access_count = 2
    for _ in range(4):
        clock[0] += 1.1 * 3 * 86400.0
        m.demote_to_cold("火锅聚餐（已归档）")
        m = ag.recall(m.id[:6])
    return ag


def eval_learner_response(agent: MemoryAgent | None = None) -> dict:
    """学习器响应演示（收工验证的一部分，随 --sync --eval 联动）。

    扫描到高一致性信号（tau_learner_health 判定：某类型两路方向一致且非持平）
    时，用真实记忆库跑一次 sleep()——它末尾自动触发 learn_tau + learn_plasticity，
    对比校准前后各类型 τ 与 drift 因子，演示学习器如何响应信号；真实库信号不足
    （最常见：决策记忆库几乎不产生唤醒/干净段观测）时回退受控合成 agent，演示
    同一机制。sleep 的改动只发生在内存（--sync 已把沉淀结果落盘在前，收工验证
    不重新保存），不污染真实记忆库。返回 {"source": real|synthetic|none,
    "signals": [类型], "before"/"after": 各类型快照, "deltas": 有变化的类型,
    "sleep_report": 梦境报告}。
    """
    print("== 学习器响应演示（高一致性信号 → sleep() 校准）==")
    if agent is None:
        print("  （未提供记忆库 agent——跳过）")
        return {"source": "none"}
    health = tau_learner_health(agent)
    signals = [
        t.value for t in MemType
        if health["by_type"][t.value]["consistency"] == "agree"
        and health["by_type"][t.value]["clean"]["direction"] not in (None, "flat")
    ]
    if signals:
        print(f"  信号扫描: {'、'.join(signals)} 两路方向一致且非持平 → 高一致性信号")
        print("           用真实记忆库演示——sleep() 触发学习器校准（仅内存，不落盘）")
        demo, truth, source = agent, None, "real"
    else:
        print("  信号扫描: 无高一致性信号（两路一致且非持平的类型）"
              "→ 回退受控合成演示 agent")
        demo = _learner_demo_agent()
        truth = {MemType.EPISODIC.value: {
            "tau_d": round(demo.cfg.true_tau_by_type[MemType.EPISODIC] / 86400.0, 3),
            "drift": demo.cfg.true_reconsolidation_by_type[MemType.EPISODIC]["drift"],
        }}
        print(f"           （真实 τ={truth[MemType.EPISODIC.value]['tau_d']} 天 vs "
              f"配置信念 {demo.cfg.tau_for(MemType.EPISODIC) / 86400.0:.0f} 天 · "
              f"真实 drift={truth[MemType.EPISODIC.value]['drift']} vs 信念 "
              f"{demo.cfg.reconsolidation_factor(MemType.EPISODIC, 'drift'):.1f}）")
        source = "synthetic"
    before = _snapshot_calibration(demo)
    sleep_report = demo.sleep(duration=6 * 3600)
    after = _snapshot_calibration(demo)
    deltas = {t: {k: round(after[t][k] - before[t][k], 3) for k in before[t]}
              for t in before if before[t] != after[t]}
    # 真实库信号存在但学习器门控未达 → 无变化时回退合成演示，保证演示可见
    if source == "real" and not deltas:
        print("           真实库信号未达学习器门控（本次无参数变化）"
              "→ 回退受控合成演示 agent")
        demo = _learner_demo_agent()
        truth = {MemType.EPISODIC.value: {
            "tau_d": round(demo.cfg.true_tau_by_type[MemType.EPISODIC] / 86400.0, 3),
            "drift": demo.cfg.true_reconsolidation_by_type[MemType.EPISODIC]["drift"],
        }}
        before = _snapshot_calibration(demo)
        sleep_report = demo.sleep(duration=6 * 3600)
        after = _snapshot_calibration(demo)
        deltas = {t: {k: round(after[t][k] - before[t][k], 3) for k in before[t]}
                  for t in before if before[t] != after[t]}
        source = "synthetic"
    for t in MemType:
        b, a = before[t.value], after[t.value]
        if b == a:
            continue
        tr = (truth or {}).get(t.value)
        d = deltas[t.value]
        tau_note = (f" → 逼近真实 {tr['tau_d']} 天"
                    if tr and tr.get("tau_d") is not None else "")
        drift_note = (f" → 逼近真实 {tr['drift']}"
                      if tr and tr.get("drift") is not None else "")
        print(f"   {t.value:<9} τ {b['tau_d']:.3f} → {a['tau_d']:.3f} 天"
              f"（Δ{d['tau_d']:+.3f}{tau_note}）  ·  "
              f"drift {b['drift']:.3f} → {a['drift']:.3f}"
              f"（Δ{d['drift']:+.3f}{drift_note}）")
    if deltas:
        moved = [t for t in deltas]
        print(f"  校准完成: {'、'.join(moved)} 的参数已按信号方向调整"
              f"（sleep 内自动触发 learn_tau + learn_plasticity）")
    else:
        print("  （本次 sleep 无参数变化——信号未达学习器门控，继续积累观测）")
    return {"source": source, "signals": signals, "before": before,
            "after": after, "deltas": deltas, "sleep_report": sleep_report}


def eval_sync(agent: MemoryAgent, path: str) -> dict:
    """收工一步验证：AGENTS.md 加载评估 + 唤醒链路连续性检查 + τ 学习器健康检查
    + 学习器响应演示。

    返回 {"agents_md": 加载评估报告, "recall_chain": 唤醒链路结论,
    "tau_learner": τ 学习器健康报告, "learner_response": 学习器响应演示}。
    """
    md_report = eval_agents_md(agent, path)
    chain = eval_recall_chain(agent)
    tau = eval_tau_learner(agent)
    resp = eval_learner_response(agent)
    return {"agents_md": md_report, "recall_chain": chain, "tau_learner": tau,
            "learner_response": resp}


def _ask_llm(doc: str, question: str) -> str | None:
    """把文档作为上下文问 LLM（OpenAI 兼容，无 key 返回 None）。"""
    from memagent.llm import _default_post

    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    payload = {
        "model": os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": "你是质检员。阅读项目指令文件，用文件里的信息回答问题；文件没有相关信息就回答'文件未提及'。"},
            {"role": "user", "content": f"文件内容：\n{doc}\n\n问题：{question}"},
        ],
    }
    try:
        import json

        status, body = _default_post(
            f"{base_url}/chat/completions",
            {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            payload, 20.0,
        )
        if status != 200:
            return None
        return json.loads(body)["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def inject_into_agents_md(agent: MemoryAgent, path: str = "AGENTS.md",
                          topic: str | None = None, k: int = 5) -> str:
    """维护 AGENTS.md 顶部的动态记忆区块：文件不存在则创建；
    已存在 marker 则整体替换（不重复堆积）；否则在顶部插入。"""
    block = build_injection_md(agent, topic, k)
    section = f"{INJECT_MARKER_START}\n{block}\n{INJECT_MARKER_END}"
    if not os.path.exists(path):
        content = f"# 项目指令\n\n{section}\n"
    else:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if INJECT_MARKER_START in content and INJECT_MARKER_END in content:
            pre = content.split(INJECT_MARKER_START)[0]
            post = content.split(INJECT_MARKER_END, 1)[1]
            content = pre + section + post  # 替换旧区块
        else:
            content = section + "\n" + content  # 顶部插入
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def show(agent: MemoryAgent, topic: str | None = None) -> None:
    if topic:
        inject_block(agent, topic=topic, k=10)
    else:
        print("记忆层全部决策（按强度排序）：")
        for m, s in _ranked_decisions(agent, 20):
            print(f"  [{m.mtype.value:8}] 强度={s:.2f} 次数={m.access_count}  {m.content}")


def main(argv: list[str] | None = None) -> int:
    from memagent.cli import enable_utf8

    enable_utf8()  # 子进程/管道下强制 UTF-8 输出，避免 Windows GBK 解码崩溃
    parser = argparse.ArgumentParser(description="会话记忆：收工沉淀决策，开工自动注入")
    parser.add_argument("--start", action="store_true", help="开工：注入相关决策（默认模式）")
    parser.add_argument("--record", action="store_true", help="收工：从 git log 提炼 + --note 沉淀决策")
    parser.add_argument("--show", action="store_true", help="查看记忆层")
    parser.add_argument("--reset", action="store_true", help="清空记忆层")
    parser.add_argument("--topic", default=None, help="按主题检索（如 '再巩固' / '学习器'）")
    parser.add_argument("--since", default=None, help="git log 时间过滤（如 '2 hours ago'）")
    parser.add_argument("--note", action="append", default=[], metavar="文本", help="手动补充一条决策（可多次）")
    parser.add_argument("--limit", type=int, default=10, help="提炼的提交条数（默认 10）")
    parser.add_argument("--k", type=int, default=5, help="注入条数（默认 5）")
    parser.add_argument("--persist", default=DEFAULT_PERSIST, help="记忆持久化文件")
    parser.add_argument("--write-context", nargs="?", const="session_context.md", metavar="文件",
                        help="生成独立注入 prompt 文件（开工时供 agent 读取）")
    parser.add_argument("--inject-agents-md", nargs="?", const="AGENTS.md", metavar="文件",
                        help="把注入块维护到 AGENTS.md 顶部（自动替换旧区块）")
    parser.add_argument("--export-agents-md", nargs="?", const="__BOTH__", metavar="文件",
                        help="全量导出决策记忆为 AGENTS.md 风格文档；不带文件名时同时生成 AGENTS.md 与 CLAUDE.md（内容同步）")
    parser.add_argument("--eval-agents-md", nargs="?", const="AGENTS.md", metavar="文件",
                        help="实测全量加载效果：内容覆盖检查 + 可选 LLM 问答")
    parser.add_argument("--sync", action="store_true",
                        help="一键闭环：git log 提炼 → 沉淀决策 → 刷新 AGENTS.md 全量导出")
    parser.add_argument("--eval", action="store_true",
                        help="配合 --sync：完成后自动运行收工验证（AGENTS.md 加载评估 + 唤醒链路连续性检查）")
    parser.add_argument("--export-signals", nargs="?", const="awakening_signals",
                        metavar="基名", help="把唤醒信号统计导出为 JSON + CSV + 事件明细 CSV（配合 --sync --eval；也可单独运行）")
    parser.add_argument("--strict", action="store_true",
                        help="配合 --export-signals：退出码区分红灯——冲突类型（warnings 非空）→ 1 需排查；无冲突但存在 τ↓/τ↑ 行动项 → 2 需校准；皆无 → 0")
    parser.add_argument("--apply-suggestions", action="store_true",
                        help="配合 --export-signals：确认后按 suggest_adjust 行动项（τ↓/τ↑）批量跑 sleep() 让学习器实际校准并持久化")
    parser.add_argument("--yes", action="store_true",
                        help="配合 --apply-suggestions：跳过确认直接执行")
    parser.add_argument("--exclude-events", default=None, metavar="memory_id:序号,...",
                        help="配合 --export-signals：排除指定唤醒事件后重算统计/漂移/health（仪表盘证据行 Shift 多选可导出对应序号；排除后冲突可能消除）")
    parser.add_argument("--exclude-clashes", action="store_true",
                        help="配合 --export-signals：按 dir ≠ clean 自动排除冲突成因事件后重判 health 并导出（与仪表盘「选反向」同判定，免手写 memory_id）")
    parser.add_argument("--aggregations", action="append", default=[],
                        metavar="TYPE:key,key",
                        help="配合 --export-signals：把仪表盘 Shift 多选聚合结论回放为 health.aggregations（事件 key = memory_id:序号；可多次）")
    parser.add_argument("--aggregations-file", default=None, metavar="文件",
                        help="配合 --export-signals：从 JSON 读取聚合选择集（仪表盘「导出聚合」一键生成：[{\"mtype\", \"events\": [\"memory_id:序号\", ...]}, ...]，免手写）")
    args = parser.parse_args(argv)

    if args.reset:
        if os.path.exists(args.persist):
            os.remove(args.persist)
        print(f"已清空会话记忆（{args.persist}）")
        return 0

    agent = MemoryAgent(persist_path=args.persist)
    if args.record:
        commits = get_recent_commits(since=args.since, limit=args.limit)
        n = record(agent, commits, args.note)
        src = f"{len(commits)} 条提交 + {len(args.note)} 条补充"
        print(f"已沉淀 {n} 条决策（来源：{src}）→ {args.persist}")
        print("下次开工运行：python session_memory.py --start [--topic 关键词]")
        return 0
    if args.sync:
        r = sync(agent, since=args.since, limit=args.limit, notes=args.note)
        print("== 一键闭环完成 ==")
        print(f"① 从 git log 提炼 {r['commits']} 条提交")
        print(f"② 沉淀 {r['recorded']} 条决策（去重合并/测试效应强化）")
        print(f"③ 刷新全量导出 → {' + '.join(r['exports'])}（{r['total']} 条决策，内容同步）")
        if args.eval:
            eval_sync(agent, path=r["export"])
            if args.export_signals is not None:
                ex = export_signals(agent, args.export_signals,
                                    exclude_events=_split_exclude(args.exclude_events),
                                    aggregations=_aggregation_specs(args),
                                    exclude_clashes=args.exclude_clashes)
                _print_excluded(ex)
                _print_clashes(ex)
                _print_exclude_compare(ex)
                _print_aggregations(ex)
                print(f"已导出唤醒信号 → {ex['json']} + {ex['csv']} + "
                      f"{ex['events_csv']}（{_signals_print_suffix(ex)}）")
                _warn_conflict_types(ex)
                _print_adjust_actions(ex)
                if args.apply_suggestions:
                    apply_suggestions(agent, ex, yes=args.yes)
                return _strict_exit_code(ex, args.strict)
        return 0
    if args.export_agents_md is not None:
        if args.export_agents_md == "__BOTH__":
            exports = export_agents_md(agent, dual=True)
            n = len([m for m in agent.store.all() if m.kind != "turn"])
            print(f"已全量导出决策记忆 → {exports[0]} + {exports[1]}（{n} 条，内容同步）")
        else:
            p = export_agents_md(agent, path=args.export_agents_md)
            print(f"已全量导出决策记忆 → {p}（{len([m for m in agent.store.all() if m.kind != 'turn'])} 条）")
        return 0
    if args.eval_agents_md is not None:
        eval_agents_md(agent, path=args.eval_agents_md)
        return 0
    if args.write_context is not None:
        p = write_context_file(agent, path=args.write_context, topic=args.topic, k=args.k)
        print(f"已生成会话上下文 → {p}（开工时读取/粘贴给 agent）")
        return 0
    if args.inject_agents_md is not None:
        p = inject_into_agents_md(agent, path=args.inject_agents_md, topic=args.topic, k=args.k)
        print(f"已更新 {p} 顶部记忆区块（marker 内自动替换，不重复堆积）")
        return 0
    if args.show:
        show(agent, args.topic)
        return 0
    if args.export_signals is not None:
        ex = export_signals(agent, args.export_signals,
                            exclude_events=_split_exclude(args.exclude_events),
                            aggregations=_aggregation_specs(args),
                            exclude_clashes=args.exclude_clashes)
        _print_excluded(ex)
        _print_clashes(ex)
        _print_exclude_compare(ex)
        _print_aggregations(ex)
        print(f"已导出唤醒信号 → {ex['json']} + {ex['csv']} + {ex['events_csv']}"
              f"（{_signals_print_suffix(ex)}）")
        _warn_conflict_types(ex)
        _print_adjust_actions(ex)
        if args.apply_suggestions:
            apply_suggestions(agent, ex, yes=args.yes)
        return _strict_exit_code(ex, args.strict)
    # 默认 --start
    inject_block(agent, topic=args.topic, k=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
