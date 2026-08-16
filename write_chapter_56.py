"""一次性脚本：通过 memagent 正式管线续写《错季锁星》下一章（第 56 章）。

管线同 write_chapter_55.py：
- MemoryAgent(persona='novelist') → write_chapter()（注入人设档案+上一章结尾+剧情目标+写作规则）
- 剧情方向（用户指定）通过 patch next_chapter_goal 强制注入，不污染记忆库
- 空回复重试 / 写章级重试 / 短回复门槛已在 memagent/agent.py 正式实现
- 写完更新架构文档（characters/world 增量 + 伏笔入记忆）
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

# 用户指定的本章剧情方向：青铜门开启 + 门缝的光与典当旧账相见 + 防回声
DIRECTIVE = (
    "沈昭在第三场雨前抵达崖底青铜门，本章必须让青铜门真正开启。"
    "开篇第一笔兑现上一章结尾的代价（补契时被抽走的两成淬气，霜线断在骨里的回声）。"
    "门开启时，门缝里漏出的光（不是风，是时辰）与沈昭典当的旧账相见："
    "他当掉的本命段——断魂崖梦境之核、少年离乡/初见裴枕灯/踏入沧澜旧都三截记忆缺口、"
    "被驯养在石室里的梦核、当票赎期空白的牙印——在光里逐一显形或回应。"
    "'日过七，门过九。第三场雨前'的倒计时在本章走到最后一步，雨滴节拍器给出明确张力。"
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
