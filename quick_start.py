"""memagent 全功能端到端演示：9 大机制一次跑通。

运行方式：
    python E:/神经网络/quick_start.py

演示流程（按人类成长逻辑）：
  1) 出生 → 出厂硬件初始化
  2) 第一次对话 → 记忆写入 + 情绪推断
  3) 生长步进 → 预测-验证
  4) 技能练习 → 熟练度曲线
  5) 长期目标设定 → 进展追踪
  6) 好奇驱动 → 自主探索
  7) 反事实推理 → 如果…会怎样
  8) 类比迁移 → 举一反三
  9) 社交学习 → 从其他 agent 学知识
 10) 睡眠巩固 → 记忆强化 + 冷压缩
 11) 自我模型 → 认知边界报告
"""

import sys, time
sys.path.insert(0, r"E:/神经网络")

from memagent.agent import MemoryAgent
from memagent.memory import MemoryStore, MemType, Tier
from memagent.emotion import Emotion
from memagent.cognition import Cognition

def hr(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def main():
    # ── 0. 初始化 ──
    store = MemoryStore()
    agent = MemoryAgent(store=store)
    hr("0. 出厂硬件")
    print(f"  TIME_SCALE = 1/86400 → 1 agent-秒 ≈ 1 人类-天")
    print(f"  情景 τ={agent.cfg.tau_for(MemType.EPISODIC):.1f}s, "
          f"语义 τ={agent.cfg.tau_for(MemType.SEMANTIC):.1f}s, "
          f"技能 τ={agent.cfg.tau_for(MemType.SKILL):.1f}s")
    print(f"  innate_bounds: {agent.cfg.innate_bounds}")

    # ── 1. 注册成长方向 ──
    hr("1. 设定成长方向（显式兴趣）")
    agent.set_growth_direction("音乐", 0.8, keywords=["音乐","钢琴","唱歌"])
    agent.set_growth_direction("编程", 0.6, keywords=["python","算法","代码"])
    print(f"  兴趣排名: {agent.interest.top(10)}")

    # ── 2. 第一次对话：记忆+情绪 ──
    hr("2. 第一次对话（记忆写入 + 情绪推断）")
    reply, hits = agent.respond("我今天弹钢琴弹错了，好沮丧")
    print(f"  回复: {reply[:80]}...")
    print(f"  命中: {len(hits)} 条")
    print(f"  current_emotion: {agent.current_emotion.label if agent.current_emotion else 'None'}")

    # ── 3. 多轮对话 + 生长 ──
    hr("3. 多轮对话 → 生长步进（预测-验证 + 模式提取）")
    dialog = [
        "我每天坚持练琴一小时",
        "今天练了一首新曲子，很有挑战性",
        "弹琴让我很快乐，想一直练下去",
        "练琴技术提升了",
        "我弹了一首很棒的曲子",
    ]
    for q in dialog:
        reply, hits = agent.respond(q)
    print(f"  对话轮数: {len(dialog)}")
    print(f"  生长步数: {agent.growth.growth_step_count}")
    print(f"  记录的模式数: {len([p for p in agent.growth.patterns if p.topic=='音乐'])}")

    # ── 4. 技能练习 ──
    hr("4. 技能发展（练习→熟练度）")
    agent.cognition.register_skill("钢琴演奏", "音乐")
    skill = agent.cognition.skills["钢琴演奏"]
    for i in range(20):
        success = i < 17  # 20 次中 17 次成功
        skill.record_practice(success, i)
    print(f"  练习 20 次 (成功 17 次)")
    print(f"  熟练度: {skill.mastery:.4f}")
    print(f"  成功: {skill.success_count}, 练习次数: {skill.practice_count}")

    # ── 5. 长期目标 ──
    hr("5. 设定长期目标 + 进展追踪")
    agent.cognition.set_goal("钢琴家", "mastery ≥ 0.9", "音乐", "钢琴演奏", 0.9)
    p = agent.cognition.update_goal_progress('钢琴家')
    print(f"  目标: 钢琴家（mastery ≥ 0.9）")
    print(f"  当前进度: {p['ratio']:.2%}")
    print(f"  当前 mastery: {p['mastery']:.4f}")
    print(f"  是否达成: {agent.cognition.goals['钢琴家'].achieved}")

    # ── 6. 好奇驱动探索 ──
    hr("6. 好奇驱动探索（自主提问→搜索→写入）")
    agent.growth.predict("音乐", "练习", "进步", confidence=0.15)  # 低置信→触发验证型问题
    explore_report = agent.curiosity.explore_loop(3)
    print(f"  探索 {explore_report['explore_count']} 个问题")
    print(f"  自己找到答案: {explore_report['discovered']}")
    print(f"  待问用户: {explore_report['unanswered']}")
    for d in explore_report.get("discovered_list", []):
        print(f"    → {d}")

    # ── 7. 反事实推理 ──
    hr("7. 反事实推理（如果当初不做 X 会怎样）")
    hits_cf = agent.retrieve("练习", k=3)
    if hits_cf:
        # 反事实推理：从已知模式出发，假设前件改变
        # GrowthEngine 通过 predict+observe 验证因果——这里展示预测列表
        preds = agent.growth.find_predictions(topic="音乐")
        print(f"  音乐领域现有预测: {len(preds)} 条")
        for p in preds[:3]:
            print(f"    如果{p.trigger}则{p.expected} (置信{p.confidence:.2f})")
        # 模拟反事实：假设"不练琴"→期望还是"技术提升"吗？
        cf = agent.growth.predict("音乐", "不练琴", "技术提升", confidence=0.05)
        print(f"  反事实预测: 如果{cf.trigger}则{cf.expected} (初始置信{cf.confidence:.2f})")
        # 用实际观察修正：不练琴不会进步
        agent.growth.observe("不练琴没有进步", topic="音乐")
        print(f"  反事实验证后: 置信={agent.growth.predictions[-1].confidence:.2f}")

    # ── 8. 类比迁移 ──
    hr("8. 类比迁移（举一反三）")
    # 建立两个领域的模式
    for _ in range(3):
        agent.growth.record_pattern("音乐", "练习", "技术提升")
    for _ in range(3):
        agent.growth.record_pattern("体育", "锻炼", "体能提升")
    learned = agent.analogy.auto_learn_analogy("音乐", "体育")
    print(f"  学到的类比: {len(learned)} 条")
    for l in learned:
        print(f"    {l['source_domain']} → {l['target_domain']}: 相似度 {l['similarity']:.2f}")
    suggestions = agent.analogy.suggest("体育", "体能提升")
    if suggestions:
        print(f"  迁移建议: {suggestions[0]['analogy']} (置信度 {suggestions[0]['confidence']:.2f})")

    # ── 9. 社交学习 ──
    hr("9. 多智能体社交学习")
    peer = MemoryAgent(store=MemoryStore())
    peer.graph.add_node("足球", labels={"运动"})
    peer.graph.add_node("篮球", labels={"运动"})
    peer.graph.add_edge("足球", "运动", "IS_A", 1.0)
    peer.graph.add_edge("篮球", "运动", "IS_A", 1.0)
    peer.remember("足球需要团队配合", importance=0.9)
    peer.remember("篮球也需要配合", importance=0.7)

    agent.social.register_peer("运动达人", peer)
    r_k = agent.social.share_knowledge("运动达人")
    r_m = agent.social.share_memory("运动达人")
    print(f"  从 peer 学到 {r_k['nodes_learned']} 个节点, {r_k['edges_learned']} 条边")
    print(f"  从 peer 学到 {r_m['memories_learned']} 条记忆")
    # 验证知识已写入
    friends = agent.graph.neighbors("足球")
    print(f"  足球的邻居: {friends}")

    # ── 10. 睡眠巩固 ──
    hr("10. 睡眠巩固（回放 + 降级 + 冷压缩）")
    # 先写入一批记忆让睡眠有内容可处理
    for _ in range(5):
        agent.remember("又练了一小时钢琴，继续进步", kind="fact")
    from memagent.memory import Tier
    hot = sum(1 for m in store.all() if m.tier==Tier.HOT)
    warm = sum(1 for m in store.all() if m.tier==Tier.WARM)
    cold = sum(1 for m in store.all() if m.tier==Tier.COLD)
    print(f"  睡眠前: Hot={hot}, Warm={warm}, Cold={cold}")
    sleep_report = agent.sleep()
    print(f"  睡眠报告: {sleep_report}")

    # ── 11. 自我模型 ──
    hr("11. 自我模型（元认知报告）")
    summary = agent.cognition.self_summary()
    print(f"  技能: {summary['skill_count']} 个")
    print(f"  目标: {summary['goal_count']} 个")
    for sk in summary['skills']:
        print(f"    {sk['name']}: mastery={sk['mastery']:.4f}, 练习={sk['practices']} 次, 成功率={sk['success_rate']:.2%}")

    # 确保认知边界有数据：显式注册技能 + 写入领域记忆
    agent.cognition.register_skill("钢琴演奏", "音乐")
    agent.remember("我已练过20次钢琴", kind="fact")
    agent.remember("我懂基本的乐理知识", kind="fact")
    agent.remember("我还不会读五线谱", kind="fact")

    hr("12. 认知边界报告")
    boundary = agent.cognition.knowledge_boundary(interest_getter=agent.interest.get)
    print(f"  主题 | 兴趣 | 技能水平 | 已知 | 未知")
    for b in boundary:
        print(f"  {b['topic']} | {b['interest']:.2f} | {b['skill_level']:.2f}")
        for k in b.get('known', []):
            print(f"    ✓ {k}")
        for u in b.get('unknown', []):
            print(f"    ? {u}")
    if not boundary:
        print("  (无可报告边界)")

    # ── 13. 最终状态 ──
    hr("13. 最终状态总览")
    print(f"  总记忆数: {sum(1 for _ in store.all())}")
    print(f"  知识图谱节点: {len(agent.graph.nodes)}")
    print(f"  兴趣排名: {agent.interest.top(10)}")
    print(f"  生长步数: {agent.growth.growth_step_count}")
    print(f"  记录的模式: {len(agent.growth.patterns)}")
    print(f"  社交互动: {agent.social.social_summary()}")

    print(f"\n{'='*60}")
    print("  演示完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    from memagent.memory import MemType, Tier
    main()