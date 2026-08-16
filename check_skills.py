import sys, json
sys.path.insert(0, '.')
from memagent.agent import MemoryAgent, MemoryStore, AgentConfig

store = MemoryStore(path='agent_memory.json')
agent = MemoryAgent(store=store, persona='novelist', cfg=AgentConfig())
skills = [m for m in agent.store.all() if m.kind == 'skill']
print(f'技能记忆总数: {len(skills)}')
print()
# 检查跨源技法
cross = [s for s in skills if '跨源' in s.content]
print(f'跨源技法: {len(cross)} 条')
for s in sorted(skills, key=lambda x: -x.importance)[:5]:
    print(f'  imp={s.importance:.2f}  {s.content[:50]}...')