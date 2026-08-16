"""Foxtable 编码 agent 质量曲线跟踪：多数据源合并，日志缺失也能重建历史趋势。

数据源优先级（同一轮号去重，主日志 > coder_stdout > 恢复文档 > cycle 文件）：
  1. works/foxtable_coder.log       —— 主日志（当前轮次，完整）
  2. works/coder_stdout.log         —— 进程 stdout 存档（轮 160~165，含分数）
  3. docs/log_recovery_20260815.md  —— 日志清空事故后的恢复文档（轮 145~165 分数 + 验证轮分数）
  4. works/foxtable/cycle_*.md      —— 每轮题目+代码存档（无分数；从文件头读轮号/领域/代码长度）
  5. docs/foxtable-domain-coverage.md —— 领域均分快照（给无分数轮次补「快照均分」参考列）

用法：
    python track_coder_trend.py            # 全部轮次（多源合并重建）
    python track_coder_trend.py --from 20  # 从第 20 轮起
    python track_coder_trend.py --csv      # 追加 trend.csv（含数据来源列）

依赖：纯标准库。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

DIMS = ["语法正确性", "API 规范性", "铁律遵守", "实战可用性", "最佳实践"]
ROOT = Path(__file__).resolve().parent
LOG = ROOT / "works" / "foxtable_coder.log"
STDOUT_LOG = ROOT / "works" / "coder_stdout.log"
RECOVERY_DOC = ROOT / "docs" / "log_recovery_20260815.md"
CYCLE_DIR = ROOT / "works" / "foxtable"
COVERAGE_DOC = ROOT / "docs" / "foxtable-domain-coverage.md"
FT_MEM = ROOT / "foxtable_memory.json"


def active_rule_ids(mems: list[dict], mode: str = "fix3", pct: float = 0.3) -> set[str]:
    """判定活性规则（被回放再激活的证据），返回记忆 id 集合。

    mode:
      fix3    - 固定阈值 access_count >= 3（现状；活跃簇膨胀到 12+ 后会饱和，
                失去对回放深度的区分度，但简单直观）
      rel     - 领域内相对排名：每领域取 access_count 前 pct（默认 30%）且 >2 的规则。
                对膨胀免疫——无论活跃簇涨到 12 还是 20，领域内 top 30% 始终保持区分。
                注意：**不能做全局相对排名**（88%+ 规则停留在 2，全局 top N% 会
                把阈值压到 2 导致 100% 饱和）。
    """
    from collections import defaultdict
    by_dom: dict[str, list[tuple[str, int]]] = defaultdict(list)  # domain -> [(key, acc)]
    for m in mems:
        if m.get("kind") != "skill":
            continue
        md = re.match(r"\[(.+?)/(?:坑|改进|范例|代码|API)\]", m.get("content", ""))
        if not md:
            continue
        key = m.get("id") or m.get("content")   # 无 id（测试数据）时回退 content
        by_dom[md.group(1)].append((key, m.get("access_count") or 0))
    out: set[str] = set()
    for _, items in by_dom.items():
        accs = [a for _, a in items]
        if mode == "rel":
            n = len(accs)
            ranked = sorted(accs, reverse=True)
            thr = ranked[max(0, int(n * pct) - 1)] if n else 2
            for key, a in items:
                if a > 2 and a >= thr:
                    out.add(key)
        else:
            for key, a in items:
                if a >= 3:
                    out.add(key)
    return out


def domain_activity_stats(mems: list[dict], mode: str = "fix3", pct: float = 0.3) -> dict[str, dict]:
    """按领域聚合规则活跃度（access_count 分布）→ {领域: {"active": n, "avg_acc": x}}。

    active = 活性规则数（见 active_rule_ids 的 mode 语义：fix3 固定 ≥3，
    rel 领域内 top 30%）；avg_acc = 该领域规则平均回放次数。
    skill 规则的领域从 content 的 `[领域/类型]` 前缀提取。
    """
    from collections import defaultdict
    act_ids = active_rule_ids(mems, mode=mode, pct=pct)
    agg: dict[str, dict] = defaultdict(lambda: {"active": 0, "accs": []})
    for m in mems:
        if m.get("kind") != "skill":
            continue
        md = re.match(r"\[(.+?)/(?:坑|改进|范例|代码|API)\]", m.get("content", ""))
        if not md:
            continue
        d = agg[md.group(1)]
        acc = m.get("access_count") or 0
        d["accs"].append(acc)
        if (m.get("id") or m.get("content")) in act_ids:
            d["active"] += 1
    return {d: {"active": v["active"],
               "avg_acc": (sum(v["accs"]) / len(v["accs"])) if v["accs"] else 0.0,
               "rules": len(v["accs"])}
            for d, v in agg.items()}

def domain_score_trend(cycles: list[dict]) -> dict[str, dict]:
    """每领域首练→最近均分 → {领域: {"first": x, "last": y, "delta": y-x}}。

    只用有分数的轮次（首练/最近各取一次），领域从轮记录解析。
    """
    from collections import defaultdict
    by: dict[str, list[float]] = defaultdict(list)
    for c in cycles:
        if c["scores"] and c.get("domain"):
            by[c["domain"]].append(sum(c["scores"].values()) / len(c["scores"]))
    out = {}
    for d, scores in by.items():
        if len(scores) < 2:
            continue
        first, last = scores[0], scores[-1]
        out[d] = {"first": first, "last": last, "delta": last - first}
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman 秩相关系数（纯标准库；样本 < 3 返回 0）。"""
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    denom = n * (n * n - 1) / 6.0
    return 1.0 - d2 / denom if denom else 0.0


def partial_spearman(xs: list[float], ys: list[float], zs: list[float]) -> float:
    """控制 z 后的 x-y 偏 Spearman 秩相关（一阶）。

    r_xy.z = (r_xy - r_xz·r_yz) / sqrt((1-r_xz²)(1-r_yz²))
    样本 < 3 返回 0。用于分离混杂变量（如练习时长/首练基数）。
    """
    if len(xs) < 3 or len(xs) != len(ys) or len(xs) != len(zs):
        return 0.0
    rxy = spearman(xs, ys)
    rxz = spearman(xs, zs)
    ryz = spearman(ys, zs)
    den = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    return (rxy - rxz * ryz) / den if den else 0.0


def domain_confound_analysis(log_text: str, mems: list[dict],
                             mode: str = "fix3") -> dict:
    """分离「练习时长」与「回放活性」混杂，重算活性×Δ 相关与偏相关。

    从日志解析每轮（排除截断/失败/≤0.5 抖动轮），每领域收集：
      active（活性规则数）· avg_acc · rules · n_rounds（练习轮数）·
      first_r（首练轮号）· first_s（首练分）· last_s（最近分）· delta（Δ均分）
    返回 {"rows": [...], "corr": {各零阶相关}, "partial": {各一阶偏相关}}。

    背景（2026-08-15）：track_coder_trend 关联表曾观察到活性×Δ 负相关，
    怀疑是「活性多的领域练得久→首练在无铁律早期→首练基数低→Δ 显得负」的
    结构性混杂。本函数用偏相关把练习时长（n_rounds/first_r）与首练基数
    （first_s）从活性×Δ 中剥离，判断负相关是否真实。
    """
    from collections import defaultdict
    by: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in _parse_valid_rounds(log_text):
        by[r["domain"]].append((r["no"], r["score"]))

    act = domain_activity_stats(mems, mode=mode)
    rows = []
    for d, seq in by.items():
        seq = sorted(seq)
        if len(seq) < 2:
            continue
        a = act.get(d, {"active": 0, "avg_acc": 0.0, "rules": 0})
        rows.append({"domain": d, "active": a["active"], "avg_acc": a["avg_acc"],
                     "rules": a.get("rules", 0), "n_rounds": len(seq),
                     "first_r": seq[0][0], "first_s": seq[0][1],
                     "last_r": seq[-1][0], "last_s": seq[-1][1],
                     "delta": seq[-1][1] - seq[0][1]})
    if len(rows) < 3:
        return {"rows": rows, "corr": {}, "partial": {}}
    active = [r["active"] for r in rows]
    delta = [r["delta"] for r in rows]
    n_rounds = [r["n_rounds"] for r in rows]
    first_r = [r["first_r"] for r in rows]
    first_s = [r["first_s"] for r in rows]
    avg_acc = [r["avg_acc"] for r in rows]
    rules = [r["rules"] for r in rows]
    corr = {
        "active_delta": spearman(active, delta),
        "active_rounds": spearman(active, n_rounds),
        "active_first_r": spearman(active, first_r),
        "rounds_delta": spearman(n_rounds, delta),
        "first_r_delta": spearman(first_r, delta),
        "active_avg_acc": spearman(active, avg_acc),
        "first_s_delta": spearman(first_s, delta),
        "first_s_active": spearman(first_s, active),
    }
    partial = {
        "active_delta|rounds": partial_spearman(active, delta, n_rounds),
        "active_delta|first_r": partial_spearman(active, delta, first_r),
        "active_delta|first_s": partial_spearman(active, delta, first_s),
        "active_delta|avg_acc": partial_spearman(active, delta, avg_acc),
        "active_delta|rules": partial_spearman(active, delta, rules),
    }
    return {"rows": rows, "corr": corr, "partial": partial}


def _parse_valid_rounds(log_text: str) -> list[dict]:
    """从日志解析每轮（排除截断/失败/≤0.5 抖动轮），返回 [{no, domain, score}]。"""
    from collections import defaultdict
    rounds: list[dict] = []
    cur: dict | None = None
    for ln in log_text.splitlines():
        m = re.search(r"=== 第 (\d+) 轮 ===", ln)
        if m:
            cur = {"no": int(m.group(1)), "domain": None, "score": None,
                   "trunc": False, "fail": False}
            rounds.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「([^」]+)」", ln)
        if m:
            cur["domain"] = m.group(1)
            continue
        m = re.search(r"自检验: 五维 \{(.+?)\}", ln)
        if m and cur["score"] is None:
            parts = re.findall(r"=([\d.]+)", m.group(1))
            if parts:
                cur["score"] = sum(float(p) for p in parts) / len(parts)
            continue
        if re.search(r"生成代码: \d+ 字符（截断", ln):
            cur["trunc"] = True
        if re.search(r"生成代码失败|自检验异常", ln):
            cur["fail"] = True
    out = []
    for r in rounds:
        if r["domain"] and not r["fail"] and not r["trunc"] \
                and r["score"] is not None and r["score"] > 0.5:
            out.append({"no": r["no"], "domain": r["domain"], "score": r["score"]})
    return out


def activity_timeseries(log_text: str, mems: list[dict], window: int = 20,
                        mode: str = "fix3") -> list[dict]:
    """活性×Δ 相关的时间切片：每 window 轮算一次 Spearman(活性, 窗口内表现)。

    每窗口（如轮 1-20, 21-40, ...）内，取该时段各领域的有效分数序列：
      w_mean  = 窗口内该领域均分（该时段表现水位）
      w_delta = 窗口内首轮→末轮 Δ（该时段涨跌）
    与该领域**当前**活性规则数做 Spearman，返回按窗口排序的序列：
      [{lo, hi, n_dom, r_mean, r_delta}]

    用途：观察相关性随回放扩窗（轮 ~214 起 replay-rounds 10）上线后是否
    从负转正/由弱转强——形成时间序列而非单点。

    注意（方法论限制）：活性用当前记忆快照回填历史窗口——早期窗口的领域
    规则后来被回放，会高估早期活性。趋势可比（相对变化有意义），绝对量不可。
    """
    from collections import defaultdict
    rounds = _parse_valid_rounds(log_text)
    act = domain_activity_stats(mems, mode=mode)
    if not rounds:
        return []
    lo_all = min(r["no"] for r in rounds)
    hi_all = max(r["no"] for r in rounds)
    series: list[dict] = []
    for lo in range(lo_all, hi_all + 1, window):
        hi = lo + window - 1
        by: dict[str, list[float]] = defaultdict(list)
        for r in rounds:
            if lo <= r["no"] <= hi:
                by[r["domain"]].append(r["score"])
        rows = []
        for d, scs in by.items():
            a = act.get(d, {"active": 0})
            if len(scs) >= 1:
                rows.append({"active": a["active"], "mean": sum(scs) / len(scs),
                             "delta": (scs[-1] - scs[0]) if len(scs) >= 2 else 0.0,
                             "n": len(scs)})
        if len(rows) < 3:
            continue
        r_mean = spearman([x["active"] for x in rows], [x["mean"] for x in rows])
        # 窗口内 Δ 只在有 ≥2 轮的领域上有意义
        delta_rows = [x for x in rows if x["n"] >= 2]
        r_delta = spearman([x["active"] for x in delta_rows],
                           [x["delta"] for x in delta_rows]) if len(delta_rows) >= 3 else 0.0
        series.append({"lo": lo, "hi": hi, "n_dom": len(rows),
                       "n_delta": len(delta_rows), "r_mean": r_mean, "r_delta": r_delta})
    return series


def print_timeseries_block(series: list[dict], window: int = 20) -> None:
    """打印活性×Δ 时间切片序列。"""
    if not series:
        print("\n=== 活性×Δ 时间切片 ===")
        print("  无数据")
        return
    print(f"\n=== 活性×Δ 相关时间切片（每 {window} 轮窗口，Spearman）===")
    print(f"{'窗口':<12}{'领域':>5}{'Δ领域':>6}{'r(活性,均分)':>14}{'r(活性,Δ)':>12}")
    for s in series:
        print(f"{s['lo']}-{s['hi']:<8}{s['n_dom']:>5}{s['n_delta']:>6}"
              f"{s['r_mean']:>+14.3f}{s['r_delta']:>+12.3f}")
    # 扩窗前 vs 后对比（replay-rounds 10 约轮 214 上线）
    before = [s for s in series if s["hi"] < 214]
    after = [s for s in series if s["lo"] >= 214]
    if before and after:
        def avg(ss, key):
            vals = [s[key] for s in ss if s[key] is not None]
            return sum(vals) / len(vals) if vals else None
        b_mean, a_mean = avg(before, "r_mean"), avg(after, "r_mean")
        b_delta, a_delta = avg(before, "r_delta"), avg(after, "r_delta")
        if b_mean is not None and a_mean is not None:
            print(f"  扩窗(轮214)前 r(活性,均分) 均值 {b_mean:+.3f} → 后 {a_mean:+.3f}"
                  f" (Δ{a_mean - b_mean:+.3f})")
        if b_delta is not None and a_delta is not None:
            print(f"  扩窗前 r(活性,Δ) 均值 {b_delta:+.3f} → 后 {a_delta:+.3f}"
                  f" (Δ{a_delta - b_delta:+.3f})")
    print("  (活性为当前快照回填，趋势可比；绝对量受回填高估影响)")


def print_confound_block(res: dict) -> None:
    """打印活性×Δ 偏相关分析板块（分离练习时长/首练基数混杂）。"""
    rows = res["rows"]
    if len(rows) < 3:
        print("\n=== 活性×Δ 偏相关（混杂分离）===")
        print("  样本不足（需 ≥3 领域各有 ≥2 轮有效分）")
        return
    c, p = res["corr"], res["partial"]
    print("\n=== 活性×Δ 偏相关：分离「练习时长」与「首练基数」混杂 ===")
    print(f"  样本 {len(rows)} 领域（≥2 轮有效分，排除截断/失败/抖动轮）")
    print(f"  零阶: r(活性,Δ)={c['active_delta']:+.3f} · r(活性,练习轮数)={c['active_rounds']:+.3f}"
          f" · r(练习轮数,Δ)={c['rounds_delta']:+.3f}")
    print(f"        r(首练分,Δ)={c['first_s_delta']:+.3f}（首练基数效应）"
          f" · r(首练轮号,Δ)={c['first_r_delta']:+.3f}")
    print(f"  一阶偏相关（控制混杂后）:")
    print(f"    控制练习轮数: r(活性,Δ|轮数)={p['active_delta|rounds']:+.3f}")
    print(f"    控制首练轮号: r(活性,Δ|首练轮)={p['active_delta|first_r']:+.3f}")
    print(f"    控制首练分:   r(活性,Δ|首练分)={p['active_delta|first_s']:+.3f}")
    verdict = []
    if abs(c["active_delta"]) > 0.3:
        verdict.append(f"原始相关 {c['active_delta']:+.2f}")
    others = [p["active_delta|rounds"], p["active_delta|first_s"]]
    if all(abs(v) < 0.15 for v in others) and abs(c["active_delta"]) > 0.3:
        verdict.append("控制混杂后归零 → 负相关是练习时长/首练基数的结构性伪影，非回放活性因果")
    elif abs(c["active_delta"]) <= 0.3:
        verdict.append("原始相关本就不显著（±0.3 内），无需混杂解释")
    else:
        verdict.append("控制混杂后仍显著 → 存在独立于练习时长的活性效应")
    print("  判定: " + "；".join(verdict))


def activity_correlation(mems: list[dict], cycles: list[dict], mode: str = "fix3",
                         pct: float = 0.3) -> dict:
    """规则活跃度 × 领域分数走势的相关性分析。

    返回 {"rows": [{domain, active, avg_acc, rules, first, last, delta}],
    "spearman": x, "active_n": n, "inactive_n": n,
    "active_delta_avg": x, "inactive_delta_avg": x}。
    active_delta_avg = 有活性规则领域的 Δ 均分均值 vs 无活性领域。
    mode/pct 透传给 domain_activity_stats（fix3 固定 ≥3 / rel 领域内 top pct）。
    """
    act = domain_activity_stats(mems, mode=mode, pct=pct)
    trend = domain_score_trend(cycles)
    rows = []
    for d, t in trend.items():
        a = act.get(d, {"active": 0, "avg_acc": 0.0})
        rows.append({"domain": d, "active": a["active"], "avg_acc": a["avg_acc"],
                     "rules": a.get("rules", 0), **t})
    if not rows:
        return {"rows": [], "spearman": 0.0, "active_n": 0, "inactive_n": 0,
                "active_delta_avg": None, "inactive_delta_avg": None}
    xs = [r["active"] for r in rows]
    ys = [r["delta"] for r in rows]
    active_deltas = [r["delta"] for r in rows if r["active"] > 0]
    inactive_deltas = [r["delta"] for r in rows if r["active"] == 0]
    return {
        "rows": sorted(rows, key=lambda r: -r["active"]),
        "spearman": spearman(xs, ys),
        "active_n": len(active_deltas),
        "inactive_n": len(inactive_deltas),
        "active_delta_avg": (sum(active_deltas) / len(active_deltas)) if active_deltas else None,
        "inactive_delta_avg": (sum(inactive_deltas) / len(inactive_deltas)) if inactive_deltas else None,
    }


def print_activity_block(act: dict) -> None:
    """打印规则活跃度 × 领域走势关联表。"""
    rows = act["rows"]
    if not rows:
        print("\n=== 规则活跃度 × 领域走势 ===")
        print("  无足够数据（需领域至少两轮有分）")
        return
    print("\n=== 规则活跃度 × 领域分数走势（access_count 回放 vs Δ均分）===")
    print(f"{'领域':<14}{'活性规则':>6}{'平均回放':>8}{'规则数':>6}{'首练':>6}{'最近':>6}{'Δ':>7}")
    for r in rows:
        print(f"{r['domain']:<14}{r['active']:>6}{r['avg_acc']:>8.1f}{r['rules']:>6}"
              f"{r['first']:>6.1f}{r['last']:>6.1f}{r['delta']:>+7.1f}")
    print(f"\nSpearman(活性规则数, Δ均分) = {act['spearman']:+.3f}")
    a_d = act["active_delta_avg"]
    i_d = act["inactive_delta_avg"]
    if a_d is not None and i_d is not None:
        print(f"有活性规则领域（n={act['active_n']}）Δ均分均值 {a_d:+.2f}"
              f" vs 无活性领域（n={act['inactive_n']}）{i_d:+.2f}"
              f" → 差 {a_d - i_d:+.2f}")
    elif a_d is not None:
        print(f"分组对比：全部 {act['active_n']} 个有分数领域均含活性规则（无对照组）")
    else:
        print(f"分组对比：活性 {act['active_n']} 领域 / 无活性 {act['inactive_n']} 领域（样本不足）")

HEAD_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] === 第 (\d+) 轮 ===")


def parse_log_file(path: Path) -> list[dict]:
    """解析标准日志格式（主日志 / coder_stdout 同格式）。"""
    cycles: list[dict] = []
    cur: dict | None = None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = HEAD_RE.search(ln)
        if m:
            cur = {"t": m.group(1), "n": int(m.group(2)), "domain": "", "code": 0,
                   "scores": {}, "distilled": 0, "fail": "", "src": path.name}
            cycles.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「(.+?)」", ln);  m and cur.__setitem__("domain", m.group(1))
        m = re.search(r"生成代码: (\d+) 字符", ln); m and cur.__setitem__("code", int(m.group(1)))
        m = re.search(r"生成代码失败: (.*)", ln);   m and cur.__setitem__("fail", m.group(1).strip())
        m = re.search(r"自检验异常: (.*)", ln);     m and cur.__setitem__("fail", m.group(1).strip())
        m = re.search(r"自检验: 五维 \{(.*?)\}", ln)
        if m:
            for dim in DIMS:
                s = re.search(dim + r"=(\d+\.?\d*)", m.group(1))
                if s:
                    cur["scores"][dim] = float(s.group(1))
        m = re.search(r"沉淀 (\d+) 条", ln); m and cur.__setitem__("distilled", int(m.group(1)))
    return cycles


def parse_recovery_doc() -> list[dict]:
    """从恢复文档解析轮 145~165 逐轮分数（五维）+ 验证轮分数。

    文档分节：『### 轮 145~165』表 = 逐轮（含验证轮旧代码/重跑），『### 专项名单受控验证』表是
    结论汇总（均分），不逐行重复出条目（信息已在逐轮/自然轮里）。
    """
    if not RECOVERY_DOC.exists():
        return []
    rows: list[dict] = []
    text = RECOVERY_DOC.read_text(encoding="utf-8", errors="replace")
    # 只解析『轮 145~165』小节（含标题行到下一个 ### 之间）
    m_sec = re.search(r"### 轮 145~165.*?\n(.*?)(?:\n### |\Z)", text, re.S)
    if not m_sec:
        return rows
    for ln in m_sec.group(1).splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", ln)
        if not m:
            continue
        n, dom, five, avg = int(m.group(1)), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        scores = {}
        # 用 findall 提取数字而非 split（兼容「0/0/0/0/0（219 字符截断）」这类带尾注的行）
        parts = re.findall(r"\d+(?:\.\d+)?", five)[:5]
        if len(parts) == 5:
            scores = {d: float(v) for d, v in zip(DIMS, parts)}
        rows.append({"t": "", "n": n, "domain": dom, "code": 0,
                     "scores": scores, "distilled": 0,
                     "fail": "" if scores else "无分数", "src": RECOVERY_DOC.name})
    return rows


def parse_cycle_files() -> list[dict]:
    """从 cycle_*.md 读每轮存在性/领域/代码长度（无分数）。验证轮文件头含「验证」标记。"""
    rows: list[dict] = []
    for p in sorted(CYCLE_DIR.glob("cycle_*.md")):
        m = re.match(r"cycle_(\d+)_(.+)\.md$", p.name)
        if not m:
            continue
        n = int(m.group(1))
        text = p.read_text(encoding="utf-8", errors="replace")
        mh = re.search(r"# 第(\d+)轮 · (.+)", text[:200])
        if mh:
            domain = mh.group(2).strip()
        else:
            mv = re.search(r"# (.+?)验证第(\d+)轮", text[:200])
            domain = f"{mv.group(1).strip()}（验证）" if mv else m.group(2).replace("_", " ")
        mcode = re.search(r"```vbnet\s*\n(.*?)\n```", text, re.S)
        code = len(mcode.group(1)) if mcode else 0
        rows.append({"t": "", "n": n, "domain": domain, "code": code,
                     "scores": {}, "distilled": 0, "fail": "", "src": p.name})
    return rows


def parse_coverage_snapshot() -> dict[str, float]:
    """解析 coverage 文档的领域均分快照 → {领域: 均分}。"""
    if not COVERAGE_DOC.exists():
        return {}
    snap: dict[str, float] = {}
    in_tbl = False
    for ln in COVERAGE_DOC.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("| 领域 | 练习 |"):
            in_tbl = True
            continue
        if in_tbl:
            m = re.match(r"\|\s*([^|]+?)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|", ln)
            if m:
                snap[m.group(1).strip()] = float(m.group(2))
    return snap


def merge_cycles() -> tuple[list[dict], list[dict]]:
    """多源合并：主日志 > coder_stdout > 恢复文档 > cycle 文件，按轮号去重。

    返回 (普通轮序列, 验证轮序列)。验证轮（domain 含「验证」，如轮155~159、受控验证轮）
    不占主轮号序列——主循环轮号 160 起续接，验证轮的 cycle 编号是独立的文件编号。
    """
    merged: dict[int, dict] = {}
    verify: dict[int, dict] = {}      # 按 cycle 编号合并（cycle 文件补代码长度，恢复文档补分数）

    def absorb(rows: list[dict], priority: int) -> None:
        for c in rows:
            if "验证" in c.get("domain", ""):
                v = verify.get(c["n"])
                if v is None or priority < v.get("_prio", 99):
                    cc = dict(c)
                    cc["_prio"] = priority
                    verify[c["n"]] = cc
                continue
            cur = merged.get(c["n"])
            if cur is None or priority < cur.get("_prio", 99):
                cc = dict(c)
                cc["_prio"] = priority
                merged[c["n"]] = cc

    absorb(parse_log_file(LOG), 0)                    # 主日志（当前轮）
    absorb(parse_log_file(STDOUT_LOG), 1)             # stdout 存档（轮160-165）
    absorb(parse_recovery_doc(), 2)                   # 恢复文档（轮145-165）
    absorb(parse_cycle_files(), 3)                    # cycle 文件（存在性/代码长度）

    for c in list(merged.values()) + list(verify.values()):
        c.pop("_prio", None)
    return sorted(merged.values(), key=lambda x: x["n"]), sorted(verify.values(), key=lambda x: x["n"])


def main() -> int:
    # Windows GBK 控制台打印中文需 UTF-8 包装；只在 CLI 入口做（pytest 导入不冲突）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Foxtable coder 质量曲线（多源重建）")
    ap.add_argument("--from", type=int, default=1, dest="from_n", help="从第 N 轮起")
    ap.add_argument("--csv", action="store_true", help="同时追加 trend.csv")
    ap.add_argument("--active-mode", choices=["fix3", "rel"], default="fix3",
                    help="active rule mode: fix3 or rel")
    ap.add_argument("--ts-window", type=int, default=20,
                    help="timeseries window size in rounds (default 20)")
    args = ap.parse_args()

    cycles, verify = merge_cycles()
    snap = parse_coverage_snapshot()
    cycles = [c for c in cycles if c["n"] >= args.from_n]
    if not cycles and not verify:
        print(f"第 {args.from_n} 轮之后没有数据")
        return 1

    def src_tag(c: dict) -> str:
        if c["src"] == STDOUT_LOG.name:
            return "stdout"
        if c["src"] == RECOVERY_DOC.name:
            return "恢复"
        if c["src"].endswith(".md") and c["src"].startswith("cycle_"):
            return "file"
        return "log"

    print("=== Foxtable 编码轮次趋势（多源重建）===")
    print(f"数据源: 主日志 · stdout存档 · 恢复文档 · cycle文件 · coverage快照（来源列标注）")
    print(f"{'轮':>3} {'领域':<11} {'代码':>5} {'语':>3} {'API':>3} {'铁':>3} {'战':>3} {'践':>3} {'均分':>5} {'沉淀':>3} {'快照':>5}  状态/来源")
    for c in cycles:
        sc = c["scores"]
        avg = sum(sc.values()) / len(sc) if sc else 0
        st = "✗" if c["fail"] else ("✓" if sc else "·")
        snap_v = ""
        if not sc:
            sv = snap.get(c["domain"])
            snap_v = f"{sv:.1f}" if sv is not None else ""
        print(f"{c['n']:>3} {c['domain']:<11} {c['code']:>5} "
              f"{sc.get('语法正确性', 0):>3.0f} {sc.get('API 规范性', 0):>3.0f} "
              f"{sc.get('铁律遵守', 0):>3.0f} {sc.get('实战可用性', 0):>3.0f} "
              f"{sc.get('最佳实践', 0):>3.0f} {avg:>5.1f} {c['distilled']:>3} "
              f"{snap_v:>5}  {st}（{src_tag(c)}）")

    if verify:
        print("--- 受控验证轮（不占主轮号）---")
        for c in verify:
            sc = c["scores"]
            avg = sum(sc.values()) / len(sc) if sc else 0
            st = "✗" if c["fail"] else ("✓" if sc else "·")
            print(f"{c['n']:>3} {c['domain']:<11} {c['code']:>5} "
                  f"{sc.get('语法正确性', 0):>3.0f} {sc.get('API 规范性', 0):>3.0f} "
                  f"{sc.get('铁律遵守', 0):>3.0f} {sc.get('实战可用性', 0):>3.0f} "
                  f"{sc.get('最佳实践', 0):>3.0f} {avg:>5.1f} {c['distilled']:>3} "
                  f"{'':>5}  {st}（{src_tag(c)}）")

    scored = [(c["n"], sum(c["scores"].values()) / 5) for c in cycles if c["scores"]]
    fails = sum(1 for c in cycles if c["fail"])
    dist = sum(c["distilled"] for c in cycles)
    no_score = sum(1 for c in cycles if not c["scores"])
    print()
    print(f"打分轮 {len(scored)}/{len(cycles)} | 打分轮均分 {sum(a for _, a in scored) / len(scored):.2f}"
          f" | 失败 {fails} 次 | 累计沉淀 {dist} 条 | 无分数轮 {no_score}（早期日志缺失，看 cycle 文件/快照列）")

    # 规则活跃度 × 领域走势（access_count 回放 vs Δ均分）
    mems = []
    if FT_MEM.exists():
        try:
            mems = json.loads(FT_MEM.read_text(encoding="utf-8")).get("memories", [])
        except Exception:
            mems = []
    act_res = activity_correlation(mems, cycles, mode=args.active_mode)
    print_activity_block(act_res)
    if args.active_mode == "rel":
        print("（活性判定: 领域内 top 30%，对 access_count 膨胀免疫）")
    # 活性×Δ 偏相关（分离练习时长/首练基数混杂）——从日志重新解析（排除截断/失败轮）
    if LOG.exists():
        try:
            log_text = LOG.read_text(encoding="utf-8", errors="replace")
            conf = domain_confound_analysis(log_text, mems, mode=args.active_mode)
            print_confound_block(conf)
            print_timeseries_block(activity_timeseries(log_text, mems, window=args.ts_window,
                                                       mode=args.active_mode), args.ts_window)
        except Exception:
            pass

    if args.csv:
        path = ROOT / "trend.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["cycle", "time", "domain", "code_chars", "avg_score", "distilled", "fail", "source"])
            for c in cycles + verify:
                sc = c["scores"]
                avg = sum(sc.values()) / len(sc) if sc else ""
                w.writerow([c["n"], c["t"], c["domain"], c["code"],
                            f"{avg:.1f}" if avg != "" else "", c["distilled"], c["fail"], src_tag(c)])
        print(f"CSV 已追加 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
