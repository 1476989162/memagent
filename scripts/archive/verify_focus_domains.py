"""专项训练名单领域受控验证：跑同一管线（注入修复后），核对注入覆盖与分数。

用途：docs/next_focus_domains.json 列出的「规则入库但分数没拉起」领域，等自然抽取
太久时，用本脚本在休息间隙做受控验证轮——同一题目池、同一条管线，只对比
注入后的表现（/坑 全量注入 + 窗口自适应 + 复犯升级），证据强度与自然轮一致。

日志头部用「=== <领域>验证 第 N 轮 ===」，与普通轮区分，不污染趋势表。
运行前提：后台 autonomous_coder 进程已停止（FileLock），结束后记得重启。

用法：
    python verify_focus_domains.py                      # 验证名单前 3 个领域（各 1 轮）
    python verify_focus_domains.py --domain JSON相关    # 指定领域
    python verify_focus_domains.py --rounds 2           # 每领域 2 轮
"""
from __future__ import annotations

import argparse
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import autonomous_coder as ac

GAPS_JSON = ac.Path(__file__).resolve().parent / "docs" / "next_focus_domains.json"


def verify_round(agent, domain: str, n: int) -> dict:
    mems = ac.load_ft_memory()
    task = __import__("random").choice(ac.TASK_POOL[domain])
    ac.log(f"=== {domain}验证 第 {n} 轮 ===")
    ac.log(f"抽题: 领域「{domain}」题目：{task[:50]}...（受控验证·注入修复后管线）")

    prompt = ac.build_prompt(mems, domain, task)
    cov = ac.verify_injection(prompt, domain, mems)
    ac.log(f"注入覆盖: 坑规则 {cov['total']}/{cov['total']} 已注入" if cov["total"]
           else "注入覆盖: 坑规则 0/0 已注入")

    reply = ac.respond_with_retry(agent.responder, prompt, timeout=120.0, min_len=50)
    code_match = re.search(r"```vbnet\s*\n(.*?)\n```", reply, re.S)
    code = code_match.group(1) if code_match else reply[:500]
    # 与 one_cycle 一致：代码块未闭合 = LLM 输出被截断（轮155 事故），
    # 块闭合后才对内容跑启发式
    status = ("截断（代码块未闭合）" if code_match is None
              else ac.truncation_heuristic(code))

    out_path = ac.WORK_DIR / f"cycle_{ac.next_cycle_number():03d}_{domain.replace('/','_')}.md"
    ac.atomic_write_text(
        out_path,
        f"# {domain}验证第{n}轮\n\n## 题目\n{task}\n\n## 生成代码\n\n{reply}\n",
        overwrite=False,
    )
    ac.log(f"生成代码: {len(code)} 字符（{status}）→ {out_path.name}")

    crit_prompt = ac.build_critique_prompt(domain, task, code, mems)
    crit_reply = ac.respond_with_retry(agent.responder, crit_prompt, timeout=60.0, min_len=30)
    parsed = ac.parse_scores(crit_reply)
    scores = parsed["scores"]
    s_str = ", ".join(f"{k}={v:.1f}" for k, v in scores.items()) or "无分数"
    ac.log(f"自检验: 五维 {{{s_str}}}")
    if parsed["overall"]:
        ac.log(f"  综合: {parsed['overall'][:100]}")

    if parsed["improvements"]:
        n_saved = ac.persist_improvements(mems, domain, task, code, parsed["improvements"], scores)
        ac.log(f"  沉淀 {n_saved} 条")
        for imp in parsed["improvements"][:3]:
            ac.log(f"    → {imp[:80]}")
        ac.save_ft_memory(mems)
    else:
        ac.log("自检验: 未解析到改进建议，跳过沉淀")

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n>>> [{domain}] 验证轮{n}: 注入 {cov['total']}/{cov['total']} · "
          f"代码 {len(code)} 字符（{status}）· 均分 {avg:.2f}（五维 {s_str}）")
    return {"domain": domain, "avg": avg, "status": status, "inject": cov}


def main() -> int:
    ap = argparse.ArgumentParser(description="专项名单领域受控验证")
    ap.add_argument("--domain", default=None, help="指定领域（默认读名单前 3 个）")
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    if args.domain:
        domains = [args.domain]
    elif GAPS_JSON.exists():
        gaps = __import__("json").loads(GAPS_JSON.read_text(encoding="utf-8"))["gaps"]
        domains = [g["domain"] for g in gaps[:3]]
    else:
        print(f"找不到 {GAPS_JSON}，请先运行 domain_gap_report.py")
        return 2

    from memagent.agent import AgentConfig, MemoryAgent
    from memagent.memory import MemoryStore
    from memagent.responder import LLMResponder
    from memagent.cli import enable_utf8

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

    for d in domains:
        for n in range(1, args.rounds + 1):
            try:
                verify_round(agent, d, n)
            except Exception as e:
                ac.log(f"{d}验证第 {n} 轮异常: {e}")
                import traceback
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
