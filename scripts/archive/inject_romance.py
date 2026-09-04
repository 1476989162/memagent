import sys
sys.path.insert(0, '.')
from memagent.agent import MemoryAgent, MemoryStore, AgentConfig

store = MemoryStore(path='agent_memory.json')
agent = MemoryAgent(store=store, persona='novelist', cfg=AgentConfig())

rule = (
    "【感情线推进指令】第21章已出现银白髓液凝成的女子指印悬于沈昭眉心之前（未具名女性角色）。"
    "后续写作要求："
    "1) 2-3章内给该女性角色命名并确立身份动机，让她从'指印'进化为有名字有立场的人物；"
    "2) 感情线推进须克制且自然——先靠共感/触觉/记忆碎片建立牵引（延续已有风格），再逐步升温到肢体接触与情感袒露；"
    "3) 在伏笔回收进入中盘后（第25-30章左右），安排至少一次明确的两人心灵交汇场景，可含亲吻/拥抱/肌肤相触，"
    "双方自愿均成年，服务于角色弧光与剧情推进，不为色而色；"
    "4) 感情戏写完后自评五维中'露骨场景分寸'必须>=8，且自评沉淀的改进规则须包含1条感情线推进相关技法；"
    "5) 铁律不变：禁未成年、禁非自愿、禁强迫胁迫迷奸剥削。"
)
agent.remember_skill(rule, importance=0.92)
agent.store.save()
skills = [m for m in agent.store.all() if m.kind == 'skill']
print(f'[OK] 感情线推进指令已入库，技能记忆总数: {len(skills)}')
print()
print('--- 最近 3 条 skill ---')
for s in sorted(skills, key=lambda x: x.timestamp, reverse=True)[:3]:
    print(f'  imp={s.importance:.2f}  {s.content[:50]}...')