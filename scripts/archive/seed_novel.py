"""为《斩契》播种新记忆库 novel_memory.json（一次性脚本，可重复运行——幂等去重）。

背景（2026-08-16）：《错季锁星》114 章被读者（项目所有者）判"垃圾、不感兴趣"。
诊断：闭环只优化"自洽"（自评 8.33），从无读者信号；谜题通胀从不兑付；千字章
装不下剧情。同日都市版《违约金》试写 2 章后因合规风险废弃（现实题材易违规）。
《斩契》= 读者口味（杀伐果断 + 规则解谜必兑付）× 架空玄幻底色（合规安全）。
本脚本只种"设定锚点"（书名/主线/铁律/人物/兑付表），确保：
  - _work_title() 直接锚定《斩契》（不会触发 LLM 重新投票起名）；
  - persona_sheet() 前 8 条即写作宪法，每章 system prompt 都带着铁律。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memagent.agent import MemoryAgent  # noqa: E402
from memagent.memory import MemoryStore  # noqa: E402

STORE = Path(__file__).resolve().parent / "novel_memory.json"

# 顺序即优先级（importance 单列指定）；共 8 条 = persona_sheet 全量容量
SEEDS: list[tuple[str, float]] = [
    ("《斩契》：东方玄幻规则流长篇，纯架空世界（禁现实地名/机构/时政映射）。主角沈砚，游方斩契人，杀伐果断、认赌服输不内耗。", 0.99),
    ("《斩契》主线：沈砚天生赤纹之眼可见命契违约之痕，现实手段到尽头时携契入「履约境」对局清算违约者；每案 3-5 章完整兑付，线索对读者公平。", 0.96),
    ("写作铁律：单章 2000 字+；对话≤40%、静止倒设定连续≤3 段；悬置谜题≤3、每 3 章明写兑付≥1；爽点=主角主动行动带来可见局势变化。", 0.94),
    ("写作铁律：主角决定≤1 段，可以错不许拖；章末钩子必须具体（如「他要斩的契落款是他自己的名字」），禁止「巨大的秘密即将揭开」式空悬。", 0.94),
    ("履约境规则：进入即明示 3-5 条，读者与主角信息对称；清偿=收回被夺之物/公开罪行/强制履约/断修为，对方沾人命可依契取命但不铺陈酷刑。专有名词全书≤6（命契/赤纹/履约境/斩契人/契虫/司契）。", 0.92),
    ("人物表：老聋子（师父，装聋，瞒着沈砚旧契来历）；阿蘅（青槐镇药铺学徒，第一案引路人）；邱万山（卷一反派，灵矿东家）；苏未晚（契堂供奉，卷二登场）；司契（履约境执行者，空白面具，身份是终局主谜）。", 0.90),
    ("悬念兑付表：邱万山把命契押给了谁→第6章；沈砚被抹去的记忆→第12章；契堂批量收人契→第24章；司契身份与旧契债主→第38章全书终。不新增大纲外主谜题。", 0.90),
    ("《斩契》：已连载 0 章", 0.97),
]


def main() -> int:
    store = MemoryStore(path=str(STORE)) if STORE.exists() else MemoryStore()
    store.path = str(STORE)
    agent = MemoryAgent(store=store)
    existing = {m.content for m in agent.store.all() if m.kind == "setting"}
    added = 0
    for content, imp in SEEDS:
        if content in existing:
            continue
        agent.remember_setting(content, importance=imp)
        added += 1
    agent.save()
    total = sum(1 for m in agent.store.all() if m.kind == "setting")
    print(f"种子完成：新增 {added} 条 · 设定记忆共 {total} 条 → {STORE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
