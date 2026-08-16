"""一次性脚本：通过 memagent 正式管线续写《错季锁星》下一章。

与 write_chapter_54.py 同一管线（MemoryAgent(persona='novelist') → write_chapter()），
差别：
- 本章剧情方向（用户指定）通过 patch next_chapter_goal 强制注入"剧情目标"，
  不写入记忆库、不产生污染；
- 空回复重试 / 写章级重试 / 短回复门槛已合并进 memagent/agent.py 正式实现
  （call_with_retry + write_chapter 内部重试循环），脚本不再需要 monkey-patch。
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

# 用户指定的本章剧情方向：抵达青铜门 + 活契/裴枕灯布局收口 + 防回声
DIRECTIVE = (
    "沈昭必须在第三场雨前抵达青铜门（崖底那扇掌心朝外的铜门）。"
    "让'活契'与裴枕灯的布局在本章收口：她以沈昭之名从第七口井换走的东西、"
    "枯手还契的代价、'日过七，门过九'的倒计时，在青铜门前汇合成一个可落地的结果。"
    "结尾用物理节拍收束（动作，不用说明性总结句），并严禁复读上一章结尾的句子或意象。"
)


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

    import memagent.architecture as arch

    # next_chapter_goal → 追加用户剧情方向（空回复重试已由管线 call_with_retry 兜底）
    _orig_ncg = arch.next_chapter_goal

    def _ncg_with_dir(agent_, title_, chapter_no, work_dir_):
        base = _orig_ncg(agent_, title_, chapter_no, work_dir_) or ""
        base = base.strip()
        return (base + " " + DIRECTIVE).strip()

    arch.next_chapter_goal = _ncg_with_dir
    try:
        # 写章级重试已并入 write_chapter（cfg.chapter_retries 次，失败不占号）
        w = agent.write_chapter()
    finally:
        arch.next_chapter_goal = _orig_ncg

    if not w or not w.get("ok"):
        print(f"写作未完成: {w.get('reason') if w else 'no-result'}")
        print(w)
        return 2

    print(f"完成: 《{w['title']}》第 {w['chapter']} 章「{w.get('chapter_title','')}」 "
          f"{w['words']} 字 → {w['path']}", flush=True)

    # 更新架构文档（人物/世界增量 + 伏笔入记忆）
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
