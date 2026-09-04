"""一次性脚本：通过 memagent 正式管线续写《错季锁星》下一章。

- 读取 agent_memory.json 记忆库（人设/设定/伏笔）
- MemoryAgent(persona='novelist') → write_chapter()（注入人设档案+上一章结尾+剧情目标+写作规则）
- 空回复重试 / 写章级重试已并入 memagent/agent.py 正式实现（call_with_retry + write_chapter 内部重试）
- 写完后更新架构文档（characters/world/outline 增量）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402

STORE_PATH = Path(__file__).resolve().parent / "agent_memory.json"


def main() -> int:
    enable_utf8()
    store = MemoryStore(path=str(STORE_PATH))
    agent = MemoryAgent(store=store, persona="novelist",
                        cfg=AgentConfig(evolve_on_sleep=False))
    if agent.responder is None or not agent.responder.available:
        print("错误: LLM responder 不可用（未配置 OPENAI_API_KEY？）")
        return 1

    title = agent._work_title()
    print(f"作品名: 《{title}》", flush=True)
    sheet = agent.persona_sheet(limit=8)
    print("— 人设档案（top8）—")
    print(sheet or "（无）")
    print()

    # 空回复重试已并入管线（call_with_retry），无需再 monkey-patch
    w = agent.write_chapter()

    if not w.get("ok"):
        print(f"写作未完成: {w.get('reason')}")
        print(w)
        return 2

    print(f"完成: 《{w['title']}》第 {w['chapter']} 章「{w.get('chapter_title','')}」 "
          f"{w['words']} 字 → {w['path']}", flush=True)

    # 更新架构文档（人物/世界/大纲增量）
    try:
        from memagent.architecture import update_architecture
        chapter_text = Path(w["path"]).read_text(encoding="utf-8")
        work_dir = Path(w["path"]).parent.parent
        update_architecture(w["title"], chapter_text, w["chapter"], work_dir, agent)
        print("架构文档已更新", flush=True)
    except Exception as e:
        print(f"架构更新跳过: {e}")

    agent.save()
    print("记忆已保存（已连载章数已回写）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
