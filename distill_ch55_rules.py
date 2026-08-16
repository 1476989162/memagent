"""蒸馏 ch55 自评的对标差距/综合评语为写作改进规则并入库。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402
from memagent.cli import enable_utf8  # noqa: E402
from memagent.critique import _sim  # noqa: E402

STORE_PATH = Path(__file__).resolve().parent / "agent_memory.json"

RULES = [
    "写作改进：冲击性信息必须过身体——角色听到重大真相时先给一个可见生理节拍（旧疤发烫、耳膜嗡鸣、呼吸乱一拍），再进入思考，信息冲击要过身体而不是只过脑子。",
    "写作改进：奇观瞬间先全景后特写——关键存在显现时拉开一个环境全景镜头（棺材在黑暗地底、锈纹顺砖缝爬向东北），再落到局部细节，压迫感靠景别落差制造，别只给局部特写。",
    "写作改进：代价即刻体感化——角色付出代价（如淬气被抽）时，代价的痛要在当页可见：让霜线不是退净，而是留下一截断在骨里的回声，别靠他人台词把代价感悬空。",
    "写作改进：环境节拍器绑定时限——用雨滴这类逐滴计数的节拍器制造'第几滴之前必须完成'的明确张力，让节拍与事件推进形成可感知的加速度，而不是只当背景音效。",
    "写作改进：收口章开篇第一笔兑现上一章结尾的代价——开门/抵达之后第一个动作就是承受代价的体感，不留喘息说明，让上一章的钩子在读者翻页时立刻咬一口。",
]


def main() -> int:
    enable_utf8()
    store = MemoryStore(path=str(STORE_PATH))
    agent = MemoryAgent(store=store, persona="novelist",
                        cfg=AgentConfig(evolve_on_sleep=False))

    existing = [m.content for m in store.all() if m.kind == "skill"]
    print(f"当前 skill 记忆: {len(existing)} 条")
    added = 0
    for content in RULES:
        if any(_sim(content, m) > 0.82 for m in existing):
            print(f"  跳过（近似重复）: {content[:28]}…")
            continue
        agent.remember_skill(content, importance=0.7)
        existing.append(content)
        added += 1
        print(f"  + 入库: {content[:32]}…")
    agent.save()
    after = len([m for m in store.all() if m.kind == "skill"])
    print(f"\n新增 {added} 条，skill 记忆现在 {after} 条，已保存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
