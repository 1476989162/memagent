"""把「跨章回声排查」沉淀为写作改进规则并入库（高 importance 确保进入前 12 注入）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402
from memagent.critique import _sim, writing_improvements  # noqa: E402

STORE_PATH = Path(__file__).resolve().parent / "agent_memory.json"

RULE = (
    "写作改进：跨章回声排查——动笔写下一章开头前，先回看上一章结尾最后 1-2 段的强画面钩子"
    "（画面/隐喻/意象/动作节拍）；若开头复读了其中的关键意象或语句（12 字以上重合即为复读），"
    "禁止原样搬用，把重复的意象转化为「已解决的决策」——如 ch52 把 ch51 结尾的"
    "「像是替某个想不起来的人敲了敲门」改写成「寄存的东西既已收回，这道门不必再去敲」，"
    "让信息向前推进而非原地复读。"
)


def main() -> int:
    enable_utf8()
    store = MemoryStore(path=str(STORE_PATH))
    agent = MemoryAgent(store=store, persona="novelist",
                        cfg=AgentConfig(evolve_on_sleep=False))

    existing = [m.content for m in store.all() if m.kind == "skill"]
    sims = sorted(((m, _sim(RULE, m)) for m in existing), key=lambda x: -x[1])
    print(f"当前 skill 记忆: {len(existing)} 条")
    print(f"最高相似度: {sims[0][1]:.2f} <- {sims[0][0][:40]}")
    if sims and sims[0][1] > 0.82:
        print("近似规则已存在，跳过入库")
        return 0

    agent.remember_skill(RULE, importance=1.3)
    agent.save()

    after = [m for m in store.all() if m.kind == "skill"]
    new = next(m for m in after if m.content == RULE)
    print(f"\n已入库: importance={new.importance:.3f}")
    print(f"skill 记忆现在 {len(after)} 条")

    # 验证进入 writing_improvements 前 12
    inj = writing_improvements(agent)
    idx = inj.find("跨章回声")
    print(f"writing_improvements 注入长度: {len(inj)} 字")
    print(f"规则出现在注入中: {idx >= 0}（位置 {idx}）" if idx >= 0 else "规则未出现在注入中!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
