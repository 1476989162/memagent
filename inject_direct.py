import json, time
from pathlib import Path

path = Path("agent_memory.json")
d = json.loads(path.read_text(encoding="utf-8"))
mems = d["memories"]

# 清理旧的跨源/感情线
mems = [m for m in mems if not (m.get("kind") == "skill" and (m.get("content","").startswith("【跨源") or m.get("content","").startswith("【感情线")))]

now = time.time()
rules = [
    "【跨源技法·叙事结构】扫描 410 段仙侠范文：58% 心理先行->动作跟进，动作先行仅 16%。悬念揭露/内心转变/抉择用心理先行；战斗/逃跑/紧急场景用动作先行。按场景性质切换，不死用一种。",
    "【跨源技法·对话节制】410 段样本对话密度仅 0.13/千字，中位 0。情节靠叙述留白推进。能用动作/眼神/独白替代的对话删掉；每章对话占比不超 30%。",
    "【跨源技法·环境克制】样本环境密度均值 3.7/千字，中位 1。景物词极度克制。每千字环境词 3-5 个以内；1 个精准意象担环境+情绪+悬念三重功能，不叠 3 个通用意象。",
    "【跨源技法·隐喻密度】样本每千字 6.3 个如/似/若隐喻。关键意象(功法/情绪/转折)必配 1 个比喻；选具象+陌生组合(星钉锈迹像老人牙缝烟油)，禁如刀如剑如闪电烂俗比喻。",
    "【跨源技法·短句节拍】样本中位短句比例 0.41，最高 0.88。每 3-5 行叙述必有一句 8-12 字短促句；短句放段落开头或结尾收束，不放段落中间打断。",
    "【跨源技法·心理密度】样本心理密度中位 19.9/千字。心理描写是叙事主体。每章至少一段 50-100 字纯内心独白(无动作无对话)；独白用第二人称或自问自答代替第三人称叙述。",
    "【感情线推进指令】第21章已出现银白髓液凝成的女子指印悬于沈昭眉心之前(未具名女性角色)。后续要求：1) 2-3章内给该女性角色命名并确立身份动机；2) 感情线先靠共感/触觉/记忆碎片建立牵引，逐步升温到肢体接触与情感袒露；3) 中盘(第25-30章)安排至少一次两人心灵交汇场景，可含亲吻/拥抱/肌肤相触，双方自愿均成年，服务角色弧光；4) 自评五维中露骨场景分寸必须>=8；5) 铁律不变：禁未成年、禁非自愿、禁强迫胁迫迷奸剥削。",
]

import uuid
for content in rules:
    mem = {
        "id": str(uuid.uuid4()),
        "kind": "skill",
        "mtype": "skill",
        "content": content,
        "importance": 0.92,
        "access_count": 2,
        "last_access": now,
        "tier": "warm",
        "created_at": now,
        "history": [[now, 1.0, now, 2, 0.92]],
    }
    mems.append(mem)

d["memories"] = mems
path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

# 验证
mems2 = d["memories"]
skills = [m for m in mems2 if m.get("kind") == "skill"]
cross = [s for s in skills if s.get("content","").startswith("【跨源") or s.get("content","").startswith("【感情线")]
print(f'技能记忆: {len(skills)} 条')
print(f'跨源+感情线: {len(cross)} 条 (应=7)')
print()
for s in cross:
    print(f'  imp={s["importance"]:.2f} acc={s["access_count"]}  {s["content"][:50]}...')