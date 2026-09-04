"""分级数据「禁止截断」铁律注入修复的受控验证轮。

对照实验：轮7 / 轮112 都因代码中途截断得 1.2 分（截断铁律当时被 [:15] 截掉，未进 prompt）。
本脚本用**同一道题**（在分级树中实现按层级筛选）在修复后的管线（/坑 全量注入）上重跑一轮，
核对 注入覆盖 是否 5/5，看分数是否被拉起来。

运行前提：后台 autonomous_coder 进程已停止（FileLock），结束后记得重启。
日志头部用「=== 分级数据验证 第 N 轮 ===」，与普通轮/专项轮区分，不污染趋势表。
"""
from __future__ import annotations

import argparse
import io
import random
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import autonomous_coder as ac

DOMAIN = "分级数据"
TASK = "在分级树中实现按层级筛选"  # 与轮7/轮112 完全相同，保证同难度对照


def truncation_heuristic(code: str) -> str:
    """粗略判断代码是否中途截断：只看最后一行是否以收尾结构结束。"""
    code = code.rstrip()
    if not code:
        return "空代码"
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    ok_endings = ("End Sub", "End Function", "End If", "End While", "End Select",
                  "End Try", "End Class", "End With", "End Using",
                  "Next", "Loop", "}", "Return", ")")
    if last.endswith(ok_endings):
        return "完整"
    # 截断的常见形态：最后一行断在语句中途
    if re.search(r"(Dim \w+ (As|=\s*$)|If \w+\.\w+\s*$|End$)", last):
        return "疑似截断"
    return "未以收尾结构结束（需人工确认）"


def verify_round(agent, n: int) -> dict:
    mems = ac.load_ft_memory()
    ac.log(f"=== 分级数据验证 第 {n} 轮 ===")
    ac.log(f"抽题: 领域「{DOMAIN}」题目：{TASK}...（受控验证·修复后管线）")

    prompt = ac.build_prompt(mems, DOMAIN, TASK)
    cov = ac.verify_injection(prompt, DOMAIN, mems)
    ac.log(f"注入覆盖: 坑规则 {cov['total']}/{cov['total']} 已注入" if cov["total"]
           else "注入覆盖: 坑规则 0/0 已注入")

    reply = ac.respond_with_retry(agent.responder, prompt, timeout=120.0, min_len=50)
    code_match = re.search(r"```vbnet\s*\n(.*?)\n```", reply, re.S)
    code = code_match.group(1) if code_match else reply[:500]
    status = truncation_heuristic(code)

    out_path = ac.WORK_DIR / f"cycle_{ac.next_cycle_number():03d}_{DOMAIN.replace('/','_')}.md"
    ac.atomic_write_text(
        out_path,
        f"# 分级数据验证第{n}轮 · {DOMAIN}\n\n## 题目\n{TASK}\n\n## 生成代码\n\n{reply}\n",
        overwrite=False,
    )
    ac.log(f"生成代码: {len(code)} 字符（{status}）→ {out_path.name}")

    crit_prompt = ac.build_critique_prompt(DOMAIN, TASK, code, mems)
    crit_reply = ac.respond_with_retry(agent.responder, crit_prompt, timeout=60.0, min_len=30)
    parsed = ac.parse_scores(crit_reply)
    scores = parsed["scores"]
    s_str = ", ".join(f"{k}={v:.1f}" for k, v in scores.items()) or "无分数"
    ac.log(f"自检验: 五维 {{{s_str}}}")
    if parsed["overall"]:
        ac.log(f"  综合: {parsed['overall'][:100]}")

    n_saved = 0
    if parsed["improvements"]:
        n_saved = ac.persist_improvements(mems, DOMAIN, TASK, code, parsed["improvements"], scores)
        ac.log(f"  沉淀 {n_saved} 条")
        for imp in parsed["improvements"][:3]:
            ac.log(f"    → {imp[:80]}")
        ac.save_ft_memory(mems)
    else:
        ac.log("自检验: 未解析到改进建议，跳过沉淀")

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n>>> 验证轮结果: 注入覆盖 {cov['total']}/{cov['total']}"
          f" · 代码 {len(code)} 字符（{status}）· 均分 {avg:.2f}（五维 {s_str}）")
    return {"scores": scores, "avg": avg, "code_len": len(code),
            "status": status, "distilled": n_saved, "inject": cov}


def main() -> int:
    ap = argparse.ArgumentParser(description="分级数据铁律注入修复验证")
    ap.add_argument("--rounds", type=int, default=1, help="验证轮数（默认 1）")
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
        print("后台自主进程仍在运行（持锁），请先在休息间隙停止它再跑验证。")
        return 2

    store = MemoryStore()
    responder = LLMResponder(persona="FoxTable 低代码开发专家")
    agent = MemoryAgent(store=store, responder=responder, cfg=AgentConfig(evolve_on_sleep=False))
    if agent.responder is None or not agent.responder.available:
        print("LLM 不可用")
        return 1

    for i in range(1, args.rounds + 1):
        try:
            verify_round(agent, i)
        except Exception as e:
            ac.log(f"验证第 {i} 轮异常: {e}")
            import traceback
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
