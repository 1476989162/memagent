"""为《违约金》播种新记忆库 novel_memory.json（一次性脚本，可重复运行——幂等去重）。

背景（2026-08-16）：《错季锁星》114 章被读者（项目所有者）判"垃圾、不感兴趣"。
诊断：闭环只优化"自洽"（自评 8.33），从无读者信号；谜题通胀从不兑付；千字章
装不下剧情。新书按读者口味重建：杀伐果断 + 悬疑解谜必兑付 + 现实都市底色。
本脚本只种"设定锚点"（书名/主线/铁律/人物/兑付表），确保：
  - _work_title() 直接锚定《违约金》（不会触发 LLM 重新投票起名）；
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
    ("《违约金》：都市悬疑规则流长篇。主角江彻，29 岁，前经侦刑警转行不良资产清算人，杀伐果断、认赌服输不内耗。", 0.99),
    ("《违约金》主线：江彻能看见契约「赤字」，现实手段到尽头时进入「履约间」清算爆雷者；每个案件 3-5 章完整兑付，线索对读者公平。", 0.96),
    ("写作铁律：单章 2000 字+；对话≤40%、静止倒设定连续≤3 段；悬置谜题≤3、每 3 章明写兑付≥1；爽点=主角主动行动带来可见局势变化。", 0.94),
    ("写作铁律：主角决定≤1 段，可以错不许拖；章末钩子必须具体（如「他要查的账查到了自己名下」），禁止「巨大的秘密即将揭开」式空悬。", 0.94),
    ("履约间规则：进入即明示 3-5 条，读者与主角信息对称；清偿=清空隐匿资产/强制履约/公开真相等现实可执行方式，不直接取人性命。专有名词全书≤6（履约间/违约可视/赤字/清偿/白面/账契）。", 0.92),
    ("人物表：老周（周正庸，师父，瞒着江彻契约来历）；何小满（第一案引路人）；陈树（卷一反派）；苏晚棠（对家清算人，卷二登场）；白面（履约间代理人，身份是终局主谜）。", 0.90),
    ("悬念兑付表：启明星资金去向→第6章；江彻失去的三年→第12章；泰隆收购人契→第24章；白面身份与署名契约债主→第38章全书终。不新增大纲外主谜题。", 0.90),
    ("《违约金》：已连载 0 章", 0.97),
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
