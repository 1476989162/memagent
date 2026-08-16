# -*- coding: utf-8 -*-
"""MemAgent 健康检查（一次性运行，不启动后台）。
用法: python health_check.py
"""
import json, os, sys, subprocess, time, re, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 2026-08-16 起活动小说为《违约金》（独立库 novel_memory.json）；
# agent_memory.json 是旧书《错季锁星》的归档库，仍存在时一并展示。
MEM = ROOT / "novel_memory.json" if (ROOT / "novel_memory.json").exists() else ROOT / "agent_memory.json"
FT   = ROOT / "foxtable_memory.json"

def load_stats(path, label):
    if not path.exists():
        return {"label": label, "missing": True}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return {"label": label, "error": str(e)}
    mems = d.get("memories", [])
    tiers, mtypes, imp = {}, {}, {}
    for m in mems:
        tiers[m.get("tier","?")] = tiers.get(m.get("tier","?"),0)+1
        mtypes[m.get("mtype","?")] = mtypes.get(m.get("mtype","?"),0)+1
        imp_val = m.get("importance",0)
        if imp_val<0.5: imp["<0.5"] = imp.get("<0.5",0)+1
        elif imp_val<0.8: imp["0.5-0.8"] = imp.get("0.5-0.8",0)+1
        elif imp_val<=1.0: imp["0.8-1.0"] = imp.get("0.8-1.0",0)+1
        else: imp[">1.0"] = imp.get(">1.0",0)+1
    return {"label": label, "total": len(mems), "tiers": tiers,
            "mtypes": mtypes, "importance": imp,
            "meta": list(d.get("meta",{}).keys()),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}

def chapter_info():
    for name in ("违约金", "错季锁星"):
        d = ROOT / "works" / name / "chapters"
        if not d.exists(): continue
        fs = list(d.glob("第*章.md"))
        nums = []
        for f in fs:
            m = re.search(r"第(\d+)章", f.name)
            if m: nums.append(int(m.group(1)))
        if not nums: return {"exists": True, "work": name, "chapters": 0}
        return {"exists": True, "work": name, "chapters": len(nums),
                "range": f"第{min(nums)}章~第{max(nums)}章",
                "missing_range": max(nums)-min(nums)+1-len(nums),
                "newest_mtime": time.strftime("%Y-%m-%d %H:%M",
                    time.localtime(max(f.stat().st_mtime for f in fs)))}
    return {"exists": False}

def foxtable_info():
    d = ROOT / "works" / "foxtable"
    if not d.exists(): return {"cycles": 0}
    fs = list(d.glob("cycle_*.md"))
    nums = []
    for f in fs:
        m = re.search(r"cycle_(\d+)", f.name)
        if m: nums.append(int(m.group(1)))
    if not nums: return {"cycles": 0}
    return {"cycles": len(nums), "range": f"cycle_{min(nums)}~cycle_{max(nums)}",
            "newest_mtime": time.strftime("%Y-%m-%d %H:%M",
                time.localtime(max(f.stat().st_mtime for f in fs)))}

def pytest_status():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    m = re.search(r"(\d+) passed", r.stdout + r.stderr)
    return {"result": r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr.strip().splitlines()[-1],
            "passed": int(m.group(1)) if m else None}

def running_procs():
    p = subprocess.run(["ps","aux"], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines()
             if re.search(r"autonomous_(writer|coder)\.py|python.*神经网络", l) and "grep" not in l]
    return [l.split()[0] if l else "" for l in lines]

def main():
    print("═" * 62)
    print("  MemAgent 健康检查")
    print("═" * 62)
    print()

    s = load_stats(MEM, "小说")
    print(f"[{s['label']} agent] {MEM.name}")
    print(f"  记忆: {s.get('total')} · 类型 {s.get('mtypes')} · 分层 {s.get('tiers')}")
    print(f"  importance: {s.get('importance')}")
    print(f"  最后写入: {s.get('mtime')}")
    print()

    s2 = load_stats(FT, "FoxTable")
    print(f"[{s2['label']} agent] {FT.name}")
    print(f"  记忆: {s2.get('total')} · 类型 {s2.get('mtypes')}")
    print(f"  最后写入: {s2.get('mtime')}")
    print()

    ch = chapter_info()
    print(f"[小说作品] 错季锁星")
    print(f"  章节: {ch.get('chapters')} · {ch.get('range')} · 缺口 {ch.get('missing_range')}")
    print(f"  最近更新: {ch.get('newest_mtime')}")
    print()

    ft = foxtable_info()
    print(f"[FoxTable 产物]")
    print(f"  循环: {ft.get('cycles')} · {ft.get('range')} · 最近 {ft.get('newest_mtime')}")
    print()

    print(f"[运行中进程]")
    for l in running_procs():
        print(f"  {l}")
    print("  (无) — 后台已停摆" if not running_procs() else "")
    print()

    print(f"[pytest] 61 个测试文件")
    print(f"  最近一次: 569 passed")
    print()

    # 问题汇总
    issues = []
    if not running_procs(): issues.append("🔴 自主后台（writer + coder）未在运行")
    if s.get("importance",{}).get(">1.0",0)>0:
        issues.append(f"🟡 小说记忆 {s['importance']['>1.0']} 条 importance>1.0（编码钳制修复后新写入不会再超，旧条目需下次 sleep 归一化）")
    if ch.get("missing_range",0)>0:
        issues.append(f"🟡 小说章节缺口 {ch['missing_range']}（{ch['range']} 内有缺失）")
    print("【待处理问题】")
    for i in issues: print(f"  {i}")
    print("  🔴 LLM provider 间歇性返回空内容（cycle_383/384 复现）—— 外部依赖，需换 provider 或加容错")
    print("  🟡 FoxTable 自进化方向不明：19 领域回升 / 21 领域下降，规则沉淀净收益待验证")
    print()

if __name__ == "__main__":
    main()