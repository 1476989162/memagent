"""一次性脚本：对《错季锁星》第 55 章跑五维自评（对标），对标第 54 章 8.6 基线，并沉淀写作改进规则。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402

STORE_PATH = Path(__file__).resolve().parent / "agent_memory.json"
CH = Path("works/错季锁星/chapters/第55章.md")

# 上一章（第 54 章）自评基线，注入 prompt 让 LLM 校准
_BASELINE = (
    "上一章（第 54 章）五维自评基线：文风一致 9.0 / 节奏 7.0 / 伏笔回收 9.0 / "
    "露骨场景分寸 10.0 / 人物弧光 8.0，均分 8.6，最弱项是节奏（悬念靠说明性收尾落地，"
    "缺物理节拍）。本章（第 55 章）是活契与裴枕灯布局的收口章，请在同一标准下评分，"
    "并判断：收口是否落地为可见动作与代价，而不是解释性总结。"
)


def respond_with_retry(responder, prompt: str, *, timeout: float = 90.0,
                       attempts: int = 3, retry_delay: float = 8.0, **kw) -> str:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            reply = responder.respond(prompt, **kw, timeout=timeout)
            if reply and len(reply.strip()) >= 30:
                return reply
            last_err = RuntimeError("LLM 回复为空（可能只有 reasoning，没有最终 content）")
        except Exception as e:
            last_err = e
        if attempt < attempts:
            print(f"  LLM 回复为空/异常（第 {attempt}/{attempts} 次），"
                  f"{retry_delay}s 后重试：{str(last_err)[:60]}", flush=True)
            time.sleep(retry_delay)
    raise last_err or RuntimeError("LLM 回复为空")


def main() -> int:
    enable_utf8()
    store = MemoryStore(path=str(STORE_PATH))
    agent = MemoryAgent(store=store, persona="novelist",
                        cfg=AgentConfig(evolve_on_sleep=False))
    if agent.responder is None or not agent.responder.available:
        print("错误: LLM responder 不可用")
        return 1

    chapter_text = CH.read_text(encoding="utf-8")
    title = "错季锁星"
    sheet = agent.persona_sheet(limit=8)

    from memagent.critique import self_critique, persist_improvements
    import memagent.critique as mc
    import memagent.compat as cp

    # 让自评里的 LLM 调用走空回复重试
    _orig_call = cp.call_responder

    def _retry_call(responder, prompt, **kw):
        return respond_with_retry(responder, prompt, **kw)

    cp.call_responder = _retry_call

    # 向自评 prompt 注入上一章基线（不污染 critique 模块本体）
    _orig_build = mc._build_critique_prompt

    def _build_with_baseline(*a, **kw):
        prompt = _orig_build(*a, **kw)
        return prompt + f"\n\n## 上一章质量基线（用于校准）：\n{_BASELINE}\n"

    mc._build_critique_prompt = _build_with_baseline
    try:
        crit = self_critique(
            chapter_text=chapter_text,
            chapter_no=55,
            title=title,
            responder=agent.responder,
            persona_sheet=sheet,
            n_samples=3,
            timeout=90.0,
        )
    finally:
        cp.call_responder = _orig_call
        mc._build_critique_prompt = _orig_build

    if crit is None:
        print("自评未执行（LLM 不可用）")
        return 2

    print(f"对标: {crit.benchmark_note}")
    print("\n=== 五维评分 ===")
    for name, score in crit.scores.items():
        print(f"  {name}: {score:.1f}")
    if not crit.scores:
        print("  （未解析到分数）")

    print("\n=== 对标差距 ===")
    for g in crit.gaps:
        print(f"  - {g}")
    if not crit.gaps:
        print("  （未解析到差距）")

    print("\n=== 改进建议 ===")
    for i in crit.improvements:
        print(f"  • {i}")
    if not crit.improvements:
        print("  （未解析到改进建议）")

    print("\n=== 亮点 ===")
    for s in crit.strengths:
        print(f"  • {s}")
    if not crit.strengths:
        print("  （未解析到亮点）")

    if crit.overall:
        print(f"\n=== 综合评语 ===\n{crit.overall}")

    n = persist_improvements(agent, crit)
    print(f"\n沉淀写作改进规则: {n} 条")
    agent.save()
    print("记忆已保存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
