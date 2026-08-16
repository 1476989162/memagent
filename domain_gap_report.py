"""全领域「注入→产出」对照：规则入库了，分数到底有没有被拉起来？

分界：轮 114 起 `/坑` 铁律全量进入生成 prompt（注入排序修复上线，轮112 分级数据
1.2 分事故之后）。本脚本对每个领域做：
    - 修复前均分（轮 < 114 的普通轮） vs 修复后均分（轮 >= 114）
    - 该领域已沉淀的规则数（/坑 铁律 + /改进 + /范例）
判定「规则入库但分数没拉起」：有规则 + 修复后均分未超过修复前（<=+0.3）且
仍低于 7.0 → 列入下一批专项训练名单。

产出：
    docs/domain-injection-effect.md    全领域对照表 + 名单分析
    docs/next_focus_domains.json       下一批专项训练名单（机器可读，供专项脚本用）

用法:
    python domain_gap_report.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CODER_LOG = Path("works/foxtable_coder.log")
CODER_MEM = Path("foxtable_memory.json")
FIX_ROUND = 114          # 注入排序修复上线的第一轮（轮112 事故之后）
LIFT_EPS = 0.3           # 修复后均分 - 修复前均分 <= 0.3 视为「未拉起」
GOOD_SCORE = 7.0         # 未拉起且均分 < 7.0 才判定为「规则入库但分数没拉起」
OUT_MD = Path("docs/domain-injection-effect.md")
OUT_JSON = Path("docs/next_focus_domains.json")


def parse_rounds() -> list[dict]:
    """解析日志为轮次记录：no / tag / ts / domain / avg / distilled。"""
    rows: list[dict] = []
    cur: dict | None = None
    for ln in CODER_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] === (?:(.+?) )?第 (\d+) 轮 ===", ln)
        if m:
            cur = {"ts": m.group(1), "tag": m.group(2) or "轮",
                   "no": int(m.group(3)), "domain": None, "avg": None, "distilled": 0,
                   "truncated": False}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"抽题: 领域「(.+?)」", ln)
        if m:
            cur["domain"] = m.group(1)
            continue
        m = re.search(r"生成代码: \d+ 字符（(.+?)）→", ln)
        if m:
            cur["truncated"] = "截断" in m.group(1)
            continue
        m = re.search(r"自检验: 五维 \{(.+?)\}", ln)
        if m:
            parts = re.findall(r"=([\d.]+)", m.group(1))
            if parts:
                cur["avg"] = sum(float(p) for p in parts) / len(parts)
            continue
        m = re.search(r"沉淀 (\d+) 条", ln)
        if m:
            cur["distilled"] = int(m.group(1))
    return rows


def memory_rules_by_domain() -> dict[str, dict]:
    """按领域统计 skill 规则数：/坑 铁律、/改进、/范例。"""
    mem = json.loads(CODER_MEM.read_text(encoding="utf-8"))["memories"]
    out: dict[str, dict] = defaultdict(lambda: {"rules": 0, "pitfalls": 0, "improve": 0})
    for m in mem:
        if m.get("kind") != "skill":
            continue
        mm = re.match(r"^\[([^/\]]+)/([^/\]]+)\]", m.get("content", ""))
        if not mm:
            continue
        d, typ = mm.group(1), mm.group(2)
        out[d]["rules"] += 1
        if typ == "坑":
            out[d]["pitfalls"] += 1
        elif typ == "改进":
            out[d]["improve"] += 1
    return out


def analyze(rounds: list[dict], mem_rules: dict[str, dict]) -> tuple:
    """纯分析逻辑（可测）：按领域聚合修复前/后分数，判定并返回名单。

    返回 (rows_out, gaps, pending, first_after, special)。
    """
    before: dict[str, list[float]] = defaultdict(list)
    after: dict[str, list[float]] = defaultdict(list)
    special: dict[str, list[dict]] = defaultdict(list)
    for r in rounds:
        if r["avg"] is None or r["domain"] is None:
            continue
        if r["tag"] == "轮":
            (after if r["no"] >= FIX_ROUND else before)[r["domain"]].append(r["avg"])
        else:
            special[r["domain"]].append(r)

    domains = sorted(set(before) | set(after) | set(mem_rules))

    def stat(vals: list[float]) -> tuple[float, int] | None:
        return (sum(vals) / len(vals), len(vals)) if vals else None

    gaps: list[dict] = []      # 规则入库但分数没拉起
    pending: list[dict] = []   # 有规则但修复后还没被抽到（待验证）
    first_after: list[dict] = []  # 修复后才首练（无基线，暂无法判定）
    rows_out: list[dict] = []
    for d in domains:
        b = stat(before.get(d, []))
        a = stat(after.get(d, []))
        mr = mem_rules.get(d, {"rules": 0, "pitfalls": 0, "improve": 0})
        rules = mr["rules"]
        row = {"domain": d, "rules": rules, "pitfalls": mr["pitfalls"],
               "before": b, "after": a}
        rows_out.append(row)
        if not rules:
            row.setdefault("verdict", "无规则")
            continue
        if a is None:
            row["verdict"] = "待验证"
            pending.append(row)          # 有规则，修复后 0 个打分轮
            continue
        if b is None:
            row["verdict"] = "首练于修复后"
            first_after.append(row)      # 修复后才首练
            continue
        delta = a[0] - b[0]
        if delta > LIFT_EPS:
            row["verdict"] = "拉起"
        elif a[0] >= GOOD_SCORE:
            row["verdict"] = "持平(高分)"
        else:
            row["verdict"] = "未拉起"
            gaps.append(row)

    gaps.sort(key=lambda r: (r["after"][0], -r["rules"], r["before"][0] - r["after"][0]))
    pending.sort(key=lambda r: -r["rules"])
    return rows_out, gaps, pending, first_after, special


def verify_avg(domain: str, special: dict) -> float | None:
    """该领域受控验证轮（tag 含「验证」）的均分：取最新一轮（反映当前状态），
    排除截断轮（无效样本）。"""
    vs = [r for r in special.get(domain, [])
          if "验证" in r.get("tag", "") and r["avg"] is not None and not r["truncated"]]
    return vs[-1]["avg"] if vs else None


def main() -> int:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    rounds = parse_rounds()
    mem_rules = memory_rules_by_domain()
    rows_out, gaps, pending, first_after, special = analyze(rounds, mem_rules)

    # ---------- 控制台输出 ----------
    print("=" * 72)
    print(f"全领域「注入→产出」对照（修复分界: 轮 {FIX_ROUND}，/坑 铁律全量注入）")
    print("=" * 72)
    print(f"{'领域':<28}{'规则':>4}{'坑':>4}{'修复前':>10}{'修复后':>10}  判定")
    for r in sorted(rows_out, key=lambda x: -(x["after"][0] if x["after"] else -1)):
        d = r["domain"]
        b = f"{r['before'][0]:.1f}(n{r['before'][1]})" if r["before"] else "—"
        a = f"{r['after'][0]:.1f}(n{r['after'][1]})" if r["after"] else "—"
        print(f"{d:<28}{r['rules']:>4}{r['pitfalls']:>4}{b:>10}{a:>10}  {r['verdict']}")

    print("\n" + "=" * 72)
    print(f"下一批专项训练名单（规则入库但分数没拉起，{len(gaps)} 个）:")
    print("=" * 72)
    if not gaps:
        print("  （无——规则入库的领域修复后均分都拉起来了或维持高分）")
    for i, r in enumerate(gaps, 1):
        va = verify_avg(r["domain"], special)
        note = f" · 受控验证轮已拉到 {va:.1f}" if va else ""
        print(f"  {i}. {r['domain']}: {r['before'][0]:.1f} → {r['after'][0]:.1f}"
              f"（规则 {r['rules']} 条，含坑 {r['pitfalls']} 条{note}）")

    print(f"\n待验证（{len(pending)} 个：有规则但修复后尚未被抽到打分）:")
    for r in pending:
        print(f"  · {r['domain']}（规则 {r['rules']} 条）")

    # ---------- 文档产出 ----------
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    A = L.append
    A("# 全领域「注入→产出」对照\n")
    A(f"生成时间 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')} · "
      f"修复分界: 轮 {FIX_ROUND}（/坑 铁律全量进入生成 prompt 上线）\n")
    A("## 判定规则")
    A(f"- 「拉起」: 修复后均分 - 修复前均分 > {LIFT_EPS}；")
    A(f"- 「未拉起」: 有规则 + 修复后未超过修复前（≤+{LIFT_EPS}）且均分仍 < {GOOD_SCORE}；")
    A("- 「待验证」: 有规则但修复后 0 个打分轮（还没被抽到）；")
    A("- 「首练于修复后」: 修复后才第一次练，无修复前基线。\n")
    A("## 全领域对照表\n")
    A("| 领域 | 规则 | /坑铁律 | 修复前均分 | 修复后均分 | 判定 |")
    A("|---|---|---|---|---|---|")
    for r in sorted(rows_out, key=lambda x: -(x["after"][0] if x["after"] else -1)):
        d = r["domain"]
        b = f"{r['before'][0]:.1f}（n={r['before'][1]}）" if r["before"] else "—"
        a = f"{r['after'][0]:.1f}（n={r['after'][1]}）" if r["after"] else "—"
        A(f"| {d} | {r['rules']} | {r['pitfalls']} | {b} | {a} | {r['verdict']} |")
    A("")
    A("## 下一批专项训练名单\n")
    if gaps:
        for i, r in enumerate(gaps, 1):
            va = verify_avg(r["domain"], special)
            note = f"（受控验证轮已拉到 {va:.1f}）" if va else ""
            A(f"{i}. **{r['domain']}**: {r['before'][0]:.1f} → {r['after'][0]:.1f}"
              f"（已沉淀规则 {r['rules']} 条，含 /坑 {r['pitfalls']} 条）{note}")
    else:
        A("无。")
    A("")
    A("## 待验证领域\n")
    for r in pending:
        A(f"- {r['domain']}（规则 {r['rules']} 条，修复后未抽到）")
    if first_after:
        A("\n## 首练于修复后（无基线，暂不判定）\n")
        for r in first_after:
            A(f"- {r['domain']}: 修复后均分 {r['after'][0]:.1f}（n={r['after'][1]}）")
    # 专项/验证轮明细
    sp_domains = {d for d in special if special[d]}
    if sp_domains:
        A("\n## 专项/验证轮明细（人工干预轮，未计入前后对照）\n")
        for d in sorted(sp_domains):
            A(f"- **{d}**: " + "; ".join(
                f"{r['tag']}第{r['no']}轮 {r['avg']:.1f} 分（沉淀 {r['distilled']} 条）"
                for r in special[d]))
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    json.dump({"generated": __import__("datetime").datetime.now().isoformat(),
               "fix_round": FIX_ROUND,
               "gaps": [{"domain": r["domain"], "before": round(r["before"][0], 2),
                         "after": round(r["after"][0], 2),
                         "rules": r["rules"], "pitfalls": r["pitfalls"]} for r in gaps],
               "pending": [{"domain": r["domain"], "rules": r["rules"]} for r in pending]},
              OUT_JSON.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写入: {OUT_MD}\n已写入: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
