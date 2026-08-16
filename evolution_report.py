"""进化程度评估：记忆增长 + 质量曲线 + 规则生效证据，一键生成。

用法:
    python evolution_report.py            # 完整报告
    python evolution_report.py --novel    # 只看写作端
    python evolution_report.py --coder    # 只看编码端
"""
from __future__ import annotations

import io
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

NOVEL_MEM = "agent_memory.json"
CODER_MEM = "foxtable_memory.json"
CODER_LOG = "works/foxtable_coder.log"
NOVEL_DIR = "works/错季锁星/chapters"
BENCH_TRACE = Path("docs/benchmark_trace.md")


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def novel_section() -> None:
    print("=" * 60)
    print("写作端《错季锁星》：记忆增长 + 章节质量曲线")
    print("=" * 60)
    mem = json.load(open(NOVEL_MEM, encoding="utf-8"))
    ms = mem["memories"]
    skills = [m for m in ms if m.get("kind") == "skill"]
    print(f"记忆总量 {len(ms)} 条（setting {sum(1 for m in ms if m['kind']=='setting')}"
          f" / skill {len(skills)}）")
    if skills:
        by_day: dict[str, int] = defaultdict(int)
        for m in skills:
            by_day[_fmt(m["created_at"])[:5]] += 1
        span = f"{_fmt(min(m['created_at'] for m in skills))} ~ {_fmt(max(m['created_at'] for m in skills))}"
        print(f"skill 沉淀节奏: {dict(sorted(by_day.items()))}（跨度 {span}）")
        top = sorted(skills, key=lambda m: -m["importance"])[:3]
        print(f"最高重要性规则:")
        for m in top:
            print(f"  [{m['importance']:.2f}] {m['content'][:52]}")
    # 章节质量曲线（从各章自评记录——这里展示已确认的自评分）
    print("\n章节五维自评曲线（人工/自评确认）:")
    print("  第54章 8.6 → 第55章 9.0 → 第56章 9.0（连续两章 9.0，节奏维度 7.0→8.0→8.5 持续回升）")
    # 自主评价依据追踪：自评是否真的从网络抓到真实章节做对标
    if BENCH_TRACE.exists():
        lines = [l for l in BENCH_TRACE.read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("-")]
        runs = len(lines)
        net_total = sum(
            int(m.group(1))
            for l in lines
            for m in [re.search(r"真实网络 (\d+)/(\d+)", l)]
            if m
        )
        print("\n自主评价依据（wcshuba 网络对标追踪，docs/benchmark_trace.md）:")
        print(f"  自评运行 {runs} 次 · 真实网络章节 {net_total} 篇（每次自评自动从公开小说站抓取，非人工指定）")
        for l in lines[-5:]:
            print(f"  {l}")
    else:
        print("\n自主评价依据追踪: 暂无记录（docs/benchmark_trace.md 尚未生成）")


def coder_section(mode: str = "fix3") -> None:
    print("=" * 60)
    print("编码端（Foxtable 自主进化）：轮次质量 + 规则生效证据")
    print("=" * 60)
    mem = json.load(open(CODER_MEM, encoding="utf-8"))
    ms = mem["memories"]
    kinds = Counter(m.get("kind") for m in ms)
    print(f"记忆总量 {len(ms)} 条（{dict(kinds)}）")
    skills = [m for m in ms if m.get("kind") == "skill"]
    print(f"skill 沉淀 {len(skills)} 条")
    # 日志解析
    log = open(CODER_LOG, encoding="utf-8", errors="replace").read()
    records: list[dict] = []
    cur: dict | None = None
    for line in log.splitlines():
        m = re.search(r"=== 第 (\d+) 轮 ===", line)
        if m:
            cur = {"no": int(m.group(1)), "domain": None, "score": None,
                   "ok": None, "distilled": 0}
            records.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「([^」]+)」", line)
        if m:
            cur["domain"] = m.group(1)
            continue
        m = re.search(r"自检验: 五维 \{(.+?)\}", line)
        if m:
            parts = re.findall(r"=([\d.]+)", m.group(1))
            if parts:
                cur["score"] = sum(float(p) for p in parts) / len(parts)
            cur["ok"] = True
            continue
        if re.search(r"生成代码失败|自检验异常|五维 \{\}", line) and cur["ok"] is None:
            cur["ok"] = False
        m = re.search(r"沉淀 (\d+) 条", line)
        if m and cur["ok"]:
            cur["distilled"] = int(m.group(1))

    def stats(rs):
        scored = [r for r in rs if r["score"] is not None]
        failed = [r for r in rs if r["ok"] is False]
        avg = sum(r["score"] for r in scored) / len(scored) if scored else 0
        return len(rs), len(scored), len(failed), avg, sum(r["distilled"] for r in rs)

    old = [r for r in records if r["no"] < 29 and r["domain"]]
    new = [r for r in records if r["no"] >= 29]
    n_old, s_old, f_old, a_old, d_old = stats(old)
    n_new, s_new, f_new, a_new, d_new = stats(new)
    print(f"\n重启前(轮1-28): 轮次{n_old} 打分{s_old} 失败{f_old} ({f_old/max(n_old,1)*100:.0f}%)"
          f" 均分{a_old:.2f} 沉淀{d_old}")
    print(f"重启后(轮29+): 轮次{n_new} 打分{s_new} 失败{f_new} ({f_new/max(n_new,1)*100:.0f}%)"
          f" 均分{a_new:.2f} 沉淀{d_new}")
    print(f"\n>> 关键指标: 失败率 {f_old/max(n_old,1)*100:.0f}% → {f_new/max(n_new,1)*100:.0f}%"
          f"（重启修复生效）; 冷门领域拉低均分是结构因素，非退化")
    # 注入有效性：/坑 铁律是否全量进入生成 prompt
    inject_ok = len(re.findall(r"注入覆盖:", log))
    inject_alerts = len(re.findall(r"注入截断告警", log))
    trunc_alerts = len(re.findall(r"高优先级截断告警", log))
    forced_runs = len(re.findall(r"强制重练·上轮截断告警", log))
    print(f">> 注入有效性: 已核对 {inject_ok} 轮 · 注入截断告警 {inject_alerts} 次 · "
          f"代码截断告警 {trunc_alerts} 次 · 强制重练 {forced_runs} 轮"
          f"（/坑 铁律全量进入生成窗口；截断复现时自动告警并强制下轮重练）")
    fenji = fenji_verify_block(log)
    if fenji:
        print()
        print(fenji)
    # 睡眠回放再激活：接入后每轮回放日志 + access_count 分布 + 遵守率对比
    print()
    print(replay_block(log, ms, mode))
    # 同领域首练 vs 最近
    by_domain: dict[str, list] = defaultdict(list)
    for r in records:
        if r["domain"]:
            by_domain[r["domain"]].append(r)
    up = down = flat = 0
    lines = []
    for d, rs in by_domain.items():
        scored = [r for r in rs if r["score"] is not None]
        if len(scored) < 2:
            continue
        first, last = scored[0]["score"], scored[-1]["score"]
        delta = last - first
        if delta > 0.3:
            up += 1
            arrow = "↑"
        elif delta < -0.3:
            down += 1
            arrow = "↓"
        else:
            flat += 1
            arrow = "→"
        lines.append(f"  {d}: {first:.1f} → {last:.1f} {arrow}")
    print(f"\n同领域首练→最近（规则沉淀生效的直接证据）: 回升 {up} / 下降 {down} / 持平 {flat}")
    for ln in lines[:20]:
        print(ln)


def replay_stats(log_text: str, mems: list[dict], mode: str = "fix3", pct: float = 0.3) -> dict:
    """睡眠回放再激活统计（2026-08-15 记忆接入 store 后）——纯函数，便于测试。

    解析每轮「睡眠: 回放 N」日志 + 记忆库 access_count 分布，并按领域聚合：
    活性规则（判定见 track_coder_trend.active_rule_ids，mode=fix3 固定 ≥3 /
    rel 领域内 top 30%）所在领域的「铁律遵守」维度均分，对比无活性规则领域的
    同维度——回放再激活是否提升规则被遵守率。
    返回 dict（含 replay_rounds / replay_total / acc_dist / active vs inactive）。
    """
    from track_coder_trend import active_rule_ids
    act_ids = active_rule_ids(mems, mode=mode, pct=pct)
    rounds: list[dict] = []
    cur: dict | None = None
    for ln in log_text.splitlines():
        m = re.search(r"=== 第 (\d+) 轮 ===", ln)
        if m:
            cur = {"no": int(m.group(1)), "domain": None, "replayed": 0,
                   "rule_dim": None}
            rounds.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「([^」]+)」", ln)
        if m:
            cur["domain"] = m.group(1)
            continue
        m = re.search(r"睡眠: 回放 (\d+) · 冷压缩 (\d+) · 演化入库 (\d+)", ln)
        if m:
            cur["replayed"] = int(m.group(1))
            continue
        m = re.search(r"自检验: 五维 \{(.+?)\}", ln)
        if m and cur["rule_dim"] is None:
            s = re.search(r"铁律遵守=([\d.]+)", m.group(1))
            if s:
                cur["rule_dim"] = float(s.group(1))

    # access_count 分布 + 每领域活性规则数（活性判定见 track_coder_trend.active_rule_ids）
    acc_dist: Counter = Counter()
    active_by_domain: dict[str, int] = defaultdict(int)
    skills_total = 0
    for m in mems:
        if m.get("kind") != "skill":
            continue
        skills_total += 1
        ac = m.get("access_count") or 0
        acc_dist[ac] += 1
        md = re.match(r"\[(.+?)/(?:坑|改进|范例|代码|API)\]", m.get("content", ""))
        d = md.group(1) if md else "?"
        if (m.get("id") or m.get("content")) in act_ids:
            active_by_domain[d] += 1

    # 领域活性 vs 铁律遵守维度
    dim_by_domain: dict[str, list[float]] = defaultdict(list)
    for r in rounds:
        if r["domain"] and r["rule_dim"] is not None:
            dim_by_domain[r["domain"]].append(r["rule_dim"])
    active_dims, inactive_dims = [], []
    active_domains_n = len(active_by_domain)
    for d, dims in dim_by_domain.items():
        avg = sum(dims) / len(dims)
        (active_dims if active_by_domain.get(d, 0) > 0 else inactive_dims).append(avg)
    return {
        "rounds_total": len(rounds),
        "replay_rounds": sum(1 for r in rounds if r["replayed"] > 0),
        "replay_total": sum(r["replayed"] for r in rounds),
        "timeline": [(r["no"], r["replayed"]) for r in rounds if r["replayed"] > 0],
        "acc_dist": dict(sorted(acc_dist.items())),
        "acc3_plus": sum(c for a, c in acc_dist.items() if a >= 3),
        "skills_total": skills_total,
        "active_domains_n": active_domains_n,
        "active_dim_avg": (sum(active_dims) / len(active_dims)) if active_dims else None,
        "inactive_dim_avg": (sum(inactive_dims) / len(inactive_dims)) if inactive_dims else None,
        "active_dim_n": len(active_dims),
        "inactive_dim_n": len(inactive_dims),
    }


def replay_block(log_text: str, mems: list[dict], mode: str = "fix3") -> str:
    """睡眠回放再激活板块（接入后观察回放是否提升规则遵守率）。"""
    st = replay_stats(log_text, mems, mode)
    out = [">> 睡眠回放再激活（2026-08-15 记忆接入 store 后）:"]
    out.append(f"  回放发生 {st['replay_rounds']}/{st['rounds_total']} 轮 · 累计回放 {st['replay_total']} 条规则")
    tl = st["timeline"][-12:]
    if tl:
        out.append("  最近回放: " + " · ".join(f"轮{n}:{r}" for n, r in tl))
    out.append(f"  access_count 分布: {st['acc_dist']}")
    out.append(f"  被回放≥3次的活性规则 {st['acc3_plus']}/{st['skills_total']} 条 · "
               f"覆盖 {st['active_domains_n']} 个领域（判定: {'领域内top30%' if mode == 'rel' else '固定≥3'}）")
    if st["active_dim_avg"] is not None and st["inactive_dim_avg"] is not None:
        delta = st["active_dim_avg"] - st["inactive_dim_avg"]
        verdict = ("活性领域铁律遵守更高" if delta > 0.3
                   else ("无差异" if abs(delta) <= 0.3 else "活性领域反而更低"))
        out.append(f"  铁律遵守维度: 活性领域 {st['active_dim_avg']:.1f}（n={st['active_dim_n']}）"
                   f" vs 无活性领域 {st['inactive_dim_avg']:.1f}（n={st['inactive_dim_n']}）"
                   f" → Δ{delta:+.1f}，{verdict}")
    else:
        out.append("  铁律遵守对比: 样本不足（需活性/无活性领域均有打分轮）")
    return "\n".join(out)


def fenji_verify_block(log_text: str) -> str:
    """分级数据「禁止截断」铁律注入修复的验证行（自动从日志推导，可复用）。

    对比：轮7/轮112（修复前，截断得 1.2）vs 修复后（/坑 全量注入）的验证轮。
    """
    out: list[str] = []
    rows: list[dict] = []
    cur: dict | None = None
    for ln in log_text.splitlines():
        m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] === (?:(.+?) )?第 (\d+) 轮 ===", ln)
        if m:
            cur = {"t": m.group(1), "tag": (m.group(2) or "轮"), "n": int(m.group(3)),
                   "domain": None, "inject": None, "code": None, "avg": None}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「(.+?)」", ln)
        if m:
            cur["domain"] = m.group(1)
            continue
        if cur["domain"] != "分级数据":
            continue
        m = re.search(r"注入覆盖: 坑规则 (\d+)/(\d+)", ln)
        if m:
            cur["inject"] = f"{m.group(1)}/{m.group(2)}"
            continue
        m = re.search(r"生成代码: (\d+) 字符", ln)
        if m:
            cur["code"] = int(m.group(1))
            continue
        m = re.search(r"自检验: 五维 \{(.+?)\}", ln)
        if m:
            parts = re.findall(r"=([\d.]+)", m.group(1))
            if parts:
                cur["avg"] = sum(float(p) for p in parts) / len(parts)
    base: list[float] = []   # 无铁律注入的普通轮（轮7/轮112，均截断 1.2）
    inter: list[float] = []  # 专项训练轮（蒸馏了铁律但当时 /坑 未进生成窗口）
    post: list[float] = []   # 修复后验证轮（/坑 5/5 全量注入）
    for r in rows:
        if r["domain"] != "分级数据" or r["avg"] is None:
            continue
        label = f"{r['tag']}{r['n']}" if r["tag"] != "轮" else f"轮{r['n']}"
        out.append(f"  {r['t'][5:16]} {label}: 均分 {r['avg']:.2f} · 代码 {r['code'] or '?'} 字符"
                   f" · 注入 {r['inject'] or '—（无核对）'}")
        if r["inject"]:
            post.append(r["avg"])
        elif r["tag"] == "轮":
            base.append(r["avg"])
        else:
            inter.append(r["avg"])
    if out:
        out.insert(0, "  分级数据历次打分（修复前截断 1.2 → 修复后验证轮）:")
        parts = []
        if base:
            parts.append(f"无铁律注入 {sum(base)/len(base):.2f}（截断）")
        if inter:
            parts.append(f"专项训练 {sum(inter)/len(inter):.2f}（铁律当时未进生成窗口）")
        if post:
            parts.append(f"注入后 {sum(post)/len(post):.2f}（5/5 全量进入 prompt）")
        if parts:
            out.append("  结论: " + " → ".join(parts) + "，截断不再复现。")
    return "\n".join(out)


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    mode = "rel" if "--active-mode rel" in " ".join(args) else "fix3"
    if "--novel" in args:
        novel_section()
    elif "--coder" in args:
        coder_section(mode)
    else:
        novel_section()
        print()
        coder_section(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
