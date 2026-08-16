# -*- coding: utf-8 -*-
"""FoxTable 自进化净收益诊断（从 foxtable_coder.log 解析）。

日志结构：每轮 = "=== 第 N 轮 ===" + "抽题: 领域「X」" + "自检验: 五维 {...}"
从日志行顺序配对，而非时间戳（日志重建时时间戳会乱序）。

用法:
  python foxtable_evolution_diagnosis.py
  python foxtable_evolution_diagnosis.py --limit 200   # 只看最后 200 轮
"""
import re, json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
LOG  = ROOT / "works" / "foxtable_coder.log"

def parse_log():
    """返回 [(round_no, domain, scores_dict_or_None), ...]"""
    rounds = []
    cur_round = None
    cur_domain = None
    for ln in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r".*=== 第 (\d+) 轮 ===", ln)
        if m:
            cur_round = int(m.group(1))
            cur_domain = None
            continue
        m = re.search(r"抽题: 领域「(.+?)」", ln)
        if m:
            cur_domain = m.group(1)
            continue
        m = re.search(r"自检验: 五维 \{(.*?)\}", ln)
        if m and cur_round is not None:
            scores = {}
            for k, v in re.findall(r"([\u4e00-\u9fff\w\s]+?)\s*=\s*([\d.]+)", m.group(1)):
                scores[k.strip()] = float(v)
            rounds.append((cur_round, cur_domain or "未知", scores))
    return rounds

def main():
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx+1])

    rounds = parse_log()
    if limit:
        rounds = rounds[-limit:]

    # 去重轮号（同轮可能有多次自检验）
    seen = set()
    uniq = []
    for r in rounds:
        if r[0] not in seen:
            uniq.append(r)
            seen.add(r[0])
    rounds = uniq

    n_total = len(rounds)
    n_scores = sum(1 for _, _, s in rounds if s)
    n_zero = sum(1 for _, _, s in rounds if s and all(v == 0 for v in s.values()))

    print("=" * 70)
    print(f"  FoxTable 自进化净收益诊断  ({n_total} 轮，有评分 {n_scores} 轮，全 0 分 {n_zero} 轮)")
    print("=" * 70)

    # 全局趋势：前 1/3 vs 后 1/3
    scored = [r for r in rounds if r[2]]
    third = max(1, len(scored) // 3)
    first = scored[:third]
    last = scored[-third:]

    dims = ["语法正确性", "API 规范性", "铁律遵守", "实战可用性", "最佳实践"]
    print(f"\n【全局趋势 · 前{third}轮 vs 末{third}轮】")
    print(f"  {'维度':<12} {'前段均':>7} {'末段均':>7} {'Δ':>7} {'方向':>4}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")
    for dim in dims:
        a1 = sum(s.get(dim, 0) for _, _, s in first) / len(first)
        a2 = sum(s.get(dim, 0) for _, _, s in last) / len(last)
        d = a2 - a1
        arrow = "↑" if d > 0.3 else ("↓" if d < -0.3 else "→")
        print(f"  {dim:<12} {a1:>6.2f}  {a2:>6.2f}  {d:>+6.2f} {arrow:>4}")

    # 分领域
    domain_hist = defaultdict(list)
    for _, dom, s in rounds:
        if s: domain_hist[dom].append(s)

    print(f"\n【分领域 · 首 3 次 vs 末 3 次 · 铁律遵守】")
    print(f"  {'领域':<22} {'首均':>5} {'末均':>5} {'Δ':>6} {'轮数':>5} {'判定':>4}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*4}")
    res = []
    for dom, hist in domain_hist.items():
        if len(hist) < 2: continue
        af = sum(s.get("铁律遵守",0) for s in hist[:3]) / len(hist[:3])
        al = sum(s.get("铁律遵守",0) for s in hist[-3:]) / len(hist[-3:])
        d = al - af
        judge = "利" if d > 0.5 else ("弊" if d < -0.5 else "平")
        res.append((dom, af, al, d, len(hist), judge))
    res.sort(key=lambda x: x[3])
    for dom, af, al, d, n, judge in res:
        print(f"  {dom:<22} {af:>4.1f}  {al:>4.1f}  {d:>+5.1f}  {n:>4}  {judge:>4}")

    wins = sum(1 for r in res if r[5]=="利")
    losses = sum(1 for r in res if r[5]=="弊")
    flat = sum(1 for r in res if r[5]=="平")
    print(f"\n  判定汇总: 利 {wins} · 弊 {losses} · 平 {flat}")
    net = wins - losses
    if net > 0:   print(f"  净收益: +{net} ✅ 净正")
    elif net < 0: print(f"  净收益: {net} ⚠️ 净负")
    else:         print(f"  净收益: 0 → 持平")

    # 全部维度领域矩阵（简版）
    print(f"\n【全维度末段均分 Top 5 / Bottom 5 领域】")
    dom_avg = {}
    for dom, hist in domain_hist.items():
        if len(hist) < 3: continue
        tot = sum(s.get("铁律遵守",0) for s in hist[-5:]) / len(hist[-5:])
        dom_avg[dom] = tot
    sorted_dom = sorted(dom_avg.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5:")
    for dom, v in sorted_dom[:5]: print(f"    {dom:<22} {v:.2f}")
    print("  Bottom 5:")
    for dom, v in sorted_dom[-5:]: print(f"    {dom:<22} {v:.2f}")

    # 铁律沉淀量
    ft_mem = ROOT / "foxtable_memory.json"
    if ft_mem.exists():
        d = json.loads(ft_mem.read_text(encoding="utf-8"))
        mems = d.get("memories", [])
        ac = defaultdict(int)
        for m in mems: ac[m.get("access_count",0)] += 1
        active = sum(v for k,v in ac.items() if k>=3)
        print(f"\n【铁律沉淀】总量 {len(mems)} · 活性(access≥3) {active} · 僵尸(access=2) {ac.get(2,0)}")
        if ac.get(2,0) > len(mems)*0.6:
            print(f"  ⚠️ 僵尸规则 {ac.get(2,0)} 条占 {ac.get(2,0)/len(mems)*100:.0f}%——大量规则沉淀后从未被再使用，可能质量不佳")

    print(f"\n【建议】")
    if wins > losses + 3:
        print("  ✅ 自进化整体正向，继续跑")
    elif losses > wins + 3:
        print("  ⚠️ 自进化净负——建议:")
        print("    1. 引入规则重要性门限：只沉淀五维≥7 的改进")
        print("    2. 定期清理僵尸规则（access=2 且 >50 轮未用）")
        print("    3. 沉淀规则加「适用场景」字段，避免泛化规则干扰其他领域")
    else:
        print("  → 净收益接近 0，瓶颈不在进化方向")
        print(f"  → 有效轮 {n_scores}/{n_total}（{n_scores/n_total*100:.0f}%），LLM 空回复/全 0 分是主因")
        print("  → 5 次重试 + 8s 间隔已上线，重启后台后观察有效轮是否回升")

if __name__ == "__main__":
    main()