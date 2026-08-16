"""分级数据专项训练：先蒸馏该领域铁律入库，再连续练 3 轮看分数能否拉起来。

用法：
    python foxtable_focus.py            # 蒸馏铁律 + 练 3 轮
    python foxtable_focus.py --no-distill   # 只练不蒸馏（对照）
    python foxtable_focus.py --rounds 5     # 练 5 轮

说明：
    - 铁律写成 [分级数据/坑] skill 记忆（build_prompt 按领域注入生成端，
      build_critique_prompt 的 /坑 过滤注入审查端）
    - 复用 autonomous_coder 的 prompt/解析/沉淀/重试全套机制，只固定领域
    - 会获取实例锁，避免与后台进程并发写 foxtable_memory.json
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autonomous_coder as ac  # noqa: E402

DOMAIN = "分级数据"

# 该领域铁律：基于历史两轮实际失败（代码截断、InlineTreeSetting 不完整、递归无出口）
# 与 Foxtable 分级树 API 常识手工蒸馏，写成 /坑 条目供生成端与审查端双注入。
DISTILLED_RULES = [
    "[分级数据/坑] 递归展开BOM必须写终止条件：进入子行前先判 `dr.HasChild()` 或 `dr.GetChildren().Count = 0` 就返回；按父键(上级列)找子行时排除'上级=自身'的环，防止环形数据死循环。",
    "[分级数据/坑] 根节点行的上级列为空(Nothing/空串)——按上级列取值去 Find/FindRow 前必须先做 `IsNull` 或空串判断，否则 Nothing 参与比较会抛异常或匹配不到任何行。",
    "[分级数据/坑] ShowGridTree 前必须把 InlineTreeSetting 的 ParentCol/ChildCol/TreeCol 三项全部赋值（父列、子列、树显示列），缺任一项树形加载异常；ExpandGridTree(level) 的参数是展开层数。",
    "[分级数据/坑] 按层级筛选的标准做法：用 `Row.Level`（0=顶级）判断层级，把 Level>filterLevel 的行先收集进 List(Of DataRow) 再统一设 Visible=False；禁止在 For Each 遍历过程中直接改 Rows 集合或 Current，否则抛'集合已修改'异常。",
    "[分级数据/坑] 递归/完整函数必须一次写全：包含 Base Case 出口、递归调用、顶层入口三部分，输出禁止中途截断（历史两轮分级数据均因代码截断只得分 1.2）。",
]


def distill_rules(mems: list, now: float) -> int:
    added = 0
    for content in DISTILLED_RULES:
        if any("/坑" in m.get("content", "") and m["content"] == content for m in mems):
            continue
        mems.append({
            "id": str(uuid.uuid4()), "kind": "skill", "mtype": "skill",
            "content": content, "importance": 0.90, "access_count": 2,
            "last_access": now, "tier": "warm", "created_at": now,
            "history": [[now, 1.0, now, 2, 0.90]],
        })
        added += 1
    return added


def focused_round(agent, n: int, do_distill_ctx: bool) -> dict:
    """针对 DOMAIN 跑一轮：抽题→生成→自检→沉淀。返回 {scores, distilled, ok}。"""
    mems = ac.load_ft_memory()
    task = random.choice(ac.TASK_POOL[DOMAIN])
    ac.log(f"=== 分级数据专项 第 {n} 轮 ===")
    ac.log(f"抽题: 领域「{DOMAIN}」题目：{task[:50]}...（专项训练）")

    prompt = ac.build_prompt(mems, DOMAIN, task)
    reply = ac.respond_with_retry(agent.responder, prompt, timeout=120.0, min_len=50)
    code_match = re.search(r"```vbnet\s*\n(.*?)\n```", reply, re.S)
    code = code_match.group(1) if code_match else reply[:500]

    out_path = ac.WORK_DIR / f"cycle_{ac.next_cycle_number():03d}_{DOMAIN.replace('/','_')}.md"
    ac.atomic_write_text(
        out_path,
        f"# 分级数据专项第{n}轮 · {DOMAIN}\n\n## 题目\n{task}\n\n## 生成代码\n\n{reply}\n",
        overwrite=False,
    )
    ac.log(f"生成代码: {len(code)} 字符 → {out_path.name}")

    crit_prompt = ac.build_critique_prompt(DOMAIN, task, code, mems)
    crit_reply = ac.respond_with_retry(agent.responder, crit_prompt, timeout=60.0, min_len=30)
    parsed = ac.parse_scores(crit_reply)
    scores = parsed["scores"]
    s_str = ", ".join(f"{k}={v:.1f}" for k, v in scores.items()) or "无分数"
    ac.log(f"自检验: 五维 {{{s_str}}}")
    if parsed["overall"]:
        ac.log(f"  综合: {parsed['overall'][:100]}")

    n_saved = 0
    if parsed["improvements"]:
        n_saved = ac.persist_improvements(mems, DOMAIN, task, code, parsed["improvements"], scores)
        ac.log(f"  沉淀 {n_saved} 条")
        for imp in parsed["improvements"][:3]:
            ac.log(f"    → {imp[:80]}")
        ac.save_ft_memory(mems)
    else:
        ac.log("自检验: 未解析到改进建议，跳过沉淀")

    avg = sum(scores.values()) / len(scores) if scores else 0
    return {"scores": scores, "avg": avg, "distilled": n_saved, "ok": bool(scores)}


def main() -> int:
    ap = argparse.ArgumentParser(description="分级数据专项训练")
    ap.add_argument("--rounds", type=int, default=3, help="专项轮数（默认 3）")
    ap.add_argument("--no-distill", action="store_true", help="不蒸馏铁律（对照实验）")
    args = ap.parse_args()

    from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
    from memagent.memory import MemoryStore  # noqa: E402
    from memagent.responder import LLMResponder  # noqa: E402
    from memagent.cli import enable_utf8  # noqa: E402

    enable_utf8()
    lock = ac.FileLock(str(ac.FT_MEM_PATH) + ".autonomous.lock", timeout=0.0)
    try:
        lock.acquire()
    except ac.LockTimeoutError:
        print("已有自主进程在运行（持锁），无法安全写记忆库，退出。")
        return 2

    store = MemoryStore()
    responder = LLMResponder(persona="FoxTable 低代码开发专家")
    agent = MemoryAgent(store=store, responder=responder, cfg=AgentConfig(evolve_on_sleep=False))
    if agent.responder is None or not agent.responder.available:
        print("LLM 不可用")
        return 1

    now = time.time()
    mems = ac.load_ft_memory()
    n_rule = 0
    if not args.no_distill:
        n_rule = distill_rules(mems, now)
        ac.save_ft_memory(mems)
        print(f"蒸馏入库 {n_rule} 条 [分级数据/坑] 铁律（记忆库现有 {len(mems)} 条）")
        for r in DISTILLED_RULES[:3]:
            print("  ·", r[:66], "...")
    else:
        print("对照模式：不蒸馏铁律")

    print(f"\n开始 {args.rounds} 轮 {DOMAIN} 专项训练...")
    history = []
    for i in range(1, args.rounds + 1):
        try:
            r = focused_round(agent, i, not args.no_distill)
        except Exception as e:
            ac.log(f"专项第 {i} 轮异常: {e}")
            r = {"avg": 0, "ok": False, "distilled": 0}
        history.append(r)
        print(f"  第 {i} 轮: 均分 {r['avg']:.1f} | 沉淀 {r['distilled']} 条 | "
              f"{'✓' if r['ok'] else '✗'}")
        time.sleep(2)

    avg_ok = [r["avg"] for r in history if r["ok"]]
    print("\n=== 分级数据专项结果 ===")
    print(f"历史基线: 1.2 分（轮7，代码截断）")
    print(f"专项轨迹: {[f'{r['avg']:.1f}' if r['avg'] else '✗' for r in history]}")
    print(f"专项打分轮均分: {sum(avg_ok)/len(avg_ok):.2f} (n={len(avg_ok)})" if avg_ok else "全部失败")
    print(f"累计沉淀: {sum(r['distilled'] for r in history)} 条（含铁律 {n_rule} 条）")
    try:
        lock.release()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
