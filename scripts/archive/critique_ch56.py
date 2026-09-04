"""一次性脚本：对《错季锁星》第 56 章跑五维自评（对标），对标第 55 章 9.0 基线，并沉淀写作改进规则。

含格式漂移兜底：上次（ch55）LLM 未按「改进：」前缀输出导致 persist_improvements 解析到 0 条，
本次若再发生，自动从对标差距 + 综合评语蒸馏规则入库（与 distill_ch55_rules.py 同格式）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402
from memagent.critique import _sim  # noqa: E402

STORE_PATH = Path(__file__).resolve().parent / "agent_memory.json"
CH = Path("works/错季锁星/chapters/第56章.md")

# 上一章（第 55 章）自评基线，注入 prompt 让 LLM 校准
_BASELINE = (
    "上一章（第 55 章）五维自评基线：文风一致 9.0 / 节奏 8.0 / 伏笔回收 9.5 / "
    "露骨场景分寸 10.0 / 人物弧光 8.5，均分 9.0，最弱项是节奏（8.0，收口章动作落地但代价体感偏慢）。"
    "本章（第 56 章）是青铜门开启、门缝的光与典当旧账相见的开门章，请在同一标准下评分，"
    "并判断：开篇是否兑现了上一章结尾的代价、旧账相见是否落到可感的物理细节而非说明性陈述。"
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


# ---- 格式漂移兜底：从差距/评语蒸馏规则（与 distill_ch55_rules.py 同格式） ----
_DEFECT_RE = None  # 占位；下面用函数实现，避免引 coder 模块


def _distill_rules(crit) -> list[str]:
    """improvements 为空时，把对标差距 + 综合评语蒸馏成可执行规则。"""
    rules: list[str] = []
    # 差距通常已是"句式/镜头/情绪节奏/伏笔手法"层级的具体描述，直接可执行
    for gap in crit.gaps:
        gap = gap.strip()
        if not gap or len(gap) < 12:
            continue
        rules.append(f"写作改进：{gap}")
    if crit.overall:
        ov = crit.overall.strip()
        if ov and len(ov) >= 12:
            rules.append(f"写作改进：{ov}")
    return rules


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

    # 捕获原始回复，便于诊断分数解析失败
    _orig_parse = mc._parse_critique

    def _parse_and_dump(reply, chapter_no, title):
        Path("critique_ch56_raw.txt").write_text(reply, encoding="utf-8")
        return _orig_parse(reply, chapter_no, title)

    mc._parse_critique = _parse_and_dump

    # 向自评 prompt 注入上一章基线（不污染 critique 模块本体）
    _orig_build = mc._build_critique_prompt

    def _build_with_baseline(*a, **kw):
        prompt = _orig_build(*a, **kw)
        return prompt + f"\n\n## 上一章质量基线（用于校准）：\n{_BASELINE}\n"

    mc._build_critique_prompt = _build_with_baseline
    try:
        crit = self_critique(
            chapter_text=chapter_text,
            chapter_no=56,
            title=title,
            responder=agent.responder,
            persona_sheet=sheet,
            n_samples=3,
            timeout=90.0,
        )
    finally:
        cp.call_responder = _orig_call
        mc._build_critique_prompt = _orig_build
        mc._parse_critique = _orig_parse

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
        print("  （未解析到改进建议——触发兜底蒸馏）")

    print("\n=== 亮点 ===")
    for s in crit.strengths:
        print(f"  • {s}")
    if not crit.strengths:
        print("  （未解析到亮点）")

    if crit.overall:
        print(f"\n=== 综合评语 ===\n{crit.overall}")

    n = persist_improvements(agent, crit)

    # 格式漂移兜底：正常解析到 0 条时，从差距/评语蒸馏
    if n == 0:
        fallback = _distill_rules(crit)
        if fallback:
            existing = [m.content for m in store.all() if m.kind == "skill"]
            for content in fallback:
                if any(_sim(content, m) > 0.82 for m in existing):
                    print(f"  兜底跳过（近似重复）: {content[:28]}…")
                    continue
                agent.remember_skill(content, importance=0.7)
                existing.append(content)
                n += 1
                print(f"  兜底+入库: {content[:36]}…")
            agent.save()
            print("记忆已保存")

    print(f"\n沉淀写作改进规则: {n} 条")
    agent.save()
    print("记忆已保存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
