"""记忆再巩固测试：回忆后按重要程度微调原始记忆。"""

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.embedding import cosine_similarity, embed_text
from memagent.memory import MemType


def _agent(**kw) -> MemoryAgent:
    return MemoryAgent(cfg=AgentConfig(tau_seconds=30.0, **kw))


def test_low_importance_memory_drifts_toward_recall_context():
    agent = _agent()
    mem = agent.remember("我在楼下便利店买了瓶水", importance=0.1)
    qv = embed_text("楼下便利店的矿泉水")
    before = mem.embedding[:]
    agent.retrieve("楼下便利店的矿泉水", k=1)
    # 回忆后向量应更靠近回忆情境，且记了修订
    assert cosine_similarity(mem.embedding, qv) > cosine_similarity(before, qv)
    assert mem.revision_count == 1
    assert agent.last_reconsolidated == [mem.id]


def test_high_importance_memory_is_frozen():
    agent = _agent()
    mem = agent.remember("我的名字是小林", importance=0.95)
    before = mem.embedding[:]
    agent.retrieve("你是谁", k=1)
    assert mem.embedding == before       # 向量不变
    assert mem.revision_count == 0       # 无修订
    assert agent.last_reconsolidated == []


def test_importance_moves_with_relevance():
    agent = _agent(importance_drift=0.05)
    mem = agent.remember("我偶尔会去公园散步", importance=0.1)
    agent.retrieve("你去公园散步吗", k=1)   # 强相关 → 巩固
    assert mem.importance > 0.1


def test_importance_not_below_floor():
    agent = _agent(importance_drift=0.05)
    mem = agent.remember("一些很琐碎的细节", importance=0.06)
    # 弱相关（relevance 0.2）→ 去巩固，但被下限钳住
    agent._reconsolidate(mem, "无关话题", embed_text("无关话题"), relevance=0.2)
    assert mem.importance == agent.cfg.importance_floor


def test_reconsolidation_can_be_disabled():
    agent = _agent(reconsolidate=False)
    mem = agent.remember("我在楼下便利店买了瓶水", importance=0.1)
    before = mem.embedding[:]
    imp_before = mem.importance
    agent.retrieve("楼下便利店", k=1)
    assert mem.embedding == before
    assert mem.importance == imp_before
    assert mem.revision_count == 0


def test_content_updater_hook_edits_low_importance_only():
    def updater(mem, query, lability):
        return mem.content + "（回忆情境：" + query + "）"

    agent = _agent()
    agent.content_updater = updater
    low = agent.remember("我昨天买了面包", importance=0.1)
    high = agent.remember("我的名字是小林", importance=0.95)
    agent.retrieve("你昨天买了什么", k=1)
    agent.retrieve("你是谁", k=1)
    assert "回忆情境" in low.content          # 低重要性：内容被微调
    assert "回忆情境" not in high.content     # 高重要性：冻结
    assert high.revision_count == 0


def test_cumulative_drift_with_repeated_recall():
    agent = _agent(content_drift=0.15)
    mem = agent.remember("我在楼下便利店买了瓶水", importance=0.1)
    qv = embed_text("楼下便利店的矿泉水")
    agent.retrieve("楼下便利店的矿泉水", k=1)
    after_one = cosine_similarity(mem.embedding, qv)
    agent.retrieve("楼下便利店的矿泉水", k=1)
    after_two = cosine_similarity(mem.embedding, qv)
    # 多次回忆在可塑窗口内漂移应累积
    assert after_two > after_one
    assert mem.revision_count == 2


def test_episodic_drifts_more_than_skill():
    agent = _agent()
    skill = agent.remember("我在练习做饭", importance=0.1, mtype=MemType.SKILL)
    epi = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    skill_before, epi_before = skill.embedding[:], epi.embedding[:]
    for _ in range(3):
        agent.retrieve("练习做饭", k=1)
        agent.retrieve("昨天去吃了火锅", k=1)
    skill_drift = 1.0 - cosine_similarity(skill_before, skill.embedding)
    epi_drift = 1.0 - cosine_similarity(epi_before, epi.embedding)
    assert epi_drift > skill_drift        # 情景类更容易被情境改写
    assert skill_drift < 0.01             # 技能类回忆时高度稳定
    assert epi.revision_count >= 3        # 情景类每次都进入再巩固


def test_custom_per_type_factors():
    # 自定义：把情景类两个通道都设为 0 → 尽管可塑性高也完全不改写
    cfg = AgentConfig(
        tau_seconds=30.0,
        reconsolidation_by_type={MemType.EPISODIC: {"drift": 0.0, "importance": 0.0}},
    )
    agent = MemoryAgent(cfg=cfg)
    epi = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    before = epi.embedding[:]
    imp_before = epi.importance
    agent.retrieve("昨天吃的火锅", k=1)
    assert epi.embedding == before        # drift 因子 0 → 向量不变
    assert epi.importance == imp_before   # importance 因子 0 → 重要性不变
    assert epi.revision_count == 0


def test_semantic_uses_baseline_factor():
    agent = _agent()
    sem = agent.remember("北京是中国的首都", importance=0.1, mtype=MemType.SEMANTIC)
    before = sem.embedding[:]
    agent.retrieve("中国的首都", k=1)
    # 语义类因子 1.0 → 正常漂移（介于技能与情景之间）
    assert 1.0 - cosine_similarity(before, sem.embedding) > 0


def test_dedup_still_works_after_drift():
    agent = _agent()
    mem = agent.remember("我在楼下便利店买了瓶水", importance=0.1)
    agent.retrieve("楼下便利店的水", k=1)          # 向量被漂移
    dup = agent.remember("我在楼下便利店买了瓶水")  # 内容相同 → 仍应去重命中
    assert dup.id == mem.id
