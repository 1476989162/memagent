# -*- coding: utf-8 -*-
"""强制注入读者友好度硬规则——短、可执行、LLM 一眼就能当成约束。"""
import sys
sys.path.insert(0, r"E:\神经网络")
from memagent.agent import MemoryAgent, MemoryStore, AgentConfig

store = MemoryStore(path=r"E:\神经网络\agent_memory.json")
agent = MemoryAgent(store=store, persona="novelist", cfg=AgentConfig())

# 短、硬、可执行的规则——LLM 更容易识别为 skill
rules = [
    ("写作规则：新术语首次出现必须内嵌解释，格式『塔纹——他掌心自幼的九道裂痕』，不可裸用名词。", 0.99),
    ("写作规则：跨章旧设定出现时必须给 5-15 字提示，格式『锈脉——上次断魂崖那根贯穿他血脉的暗管』。", 0.99),
    ("写作规则：本章最多引入 3 个全新术语；第 4 个推后或合并解释。", 0.98),
    ("写作规则：新角色第一次出场必须给 3 个标签（外貌+身份+与主角关系）。", 0.98),
    ("写作规则：对话中禁止堆砌术语，术语后必须跟一句人话解释或角色反应。", 0.97),
    ("写作规则：新力量体系首次出现时给一段不超过 6 行的效果描写（痛觉/温度/颜色/声音至少两样）。", 0.97),
    ("写作规则：章节结尾必须能让读者 10 秒内复述『这一章主角做了什么、发生了什么』。", 0.96),
]

for rule, imp in rules:
    m = agent.remember_skill(rule, importance=imp)
    print(f"OK imp={m.importance:.2f} acc={m.access_count}  {rule[:50]}...")

store.save()
print("done")