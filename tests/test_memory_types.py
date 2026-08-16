"""记忆类型测试：自动识别 + 按类型分遗忘曲线（技能慢、情景快）。"""

import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType, Tier, classify_memory


def test_classify_episodic():
    assert classify_memory("我昨天去吃了火锅") is MemType.EPISODIC
    assert classify_memory("用户说：你好", kind="turn") is MemType.EPISODIC


def test_classify_skill():
    assert classify_memory("我在学习做饭") is MemType.SKILL
    assert classify_memory("我学会了弹钢琴") is MemType.SKILL


def test_classify_semantic():
    assert classify_memory("北京是中国的首都") is MemType.SEMANTIC
    assert classify_memory("我叫小林，住在杭州") is MemType.SEMANTIC  # 身份是稳定事实
    assert classify_memory("没有关键词的普通句子") is MemType.SEMANTIC  # 默认语义


def test_remember_auto_classifies():
    agent = MemoryAgent()
    assert agent.remember("我昨天去吃了火锅").mtype is MemType.EPISODIC
    assert agent.remember("我在学习做饭").mtype is MemType.SKILL
    assert agent.remember("北京是中国的首都").mtype is MemType.SEMANTIC


def test_skill_decays_slower_than_episodic():
    cfg = AgentConfig(
        tau_by_type={MemType.SKILL: 1000.0, MemType.SEMANTIC: 100.0, MemType.EPISODIC: 10.0},
    )
    agent = MemoryAgent(cfg=cfg)
    skill = agent.remember("练习", importance=0.1, mtype=MemType.SKILL)
    sem = agent.remember("事实", importance=0.1, mtype=MemType.SEMANTIC)
    epi = agent.remember("事件", importance=0.1, mtype=MemType.EPISODIC)
    past = time.time() - 30  # 三条记忆同样闲置 30 秒
    for m in (skill, sem, epi):
        m.last_access = past
    s_skill, s_sem, s_epi = agent._strength(skill), agent._strength(sem), agent._strength(epi)
    assert s_skill > s_sem > s_epi  # 技能最牢、情景最先淡
    assert s_epi <= 0.21  # 情景已衰减到接近强度下限


def test_cold_threshold_scales_with_type_tau():
    cfg = AgentConfig(
        tau_by_type={MemType.SKILL: 1000.0, MemType.EPISODIC: 10.0},
        cold_after_seconds=None,   # 用按类型推导的阈值
        cold_after_tau=2.0,
    )
    agent = MemoryAgent(cfg=cfg)
    skill = agent.remember("技能", importance=0.1, mtype=MemType.SKILL)
    epi = agent.remember("情景", importance=0.1, mtype=MemType.EPISODIC)
    past = time.time() - 30  # 30 秒无访问
    for m in (skill, epi):
        m.last_access = past
    agent.sleep()
    cold_ids = {m.id for m in agent.store.by_tier(Tier.COLD)}
    # 情景（τ=10s → 2τ=20s）已被压缩；技能（τ=1000s → 2000s）还留在 Warm
    assert epi.id in cold_ids
    assert skill.id not in cold_ids


def test_absolute_cold_after_still_works():
    cfg = AgentConfig(tau_by_type={MemType.SKILL: 1000.0}, cold_after_seconds=1.0)
    agent = MemoryAgent(cfg=cfg)
    m = agent.remember("技能", importance=0.1, mtype=MemType.SKILL)
    m.last_access = time.time() - 2
    agent.sleep()
    assert m.tier is Tier.COLD  # 绝对阈值优先于按类型推导


def test_mtype_persists_roundtrip(tmp_path):
    from memagent.memory import MemoryStore

    store = MemoryStore(path=str(tmp_path / "mem.json"))
    store.add("我昨天去吃了火锅")
    store.save()
    store2 = MemoryStore(path=str(tmp_path / "mem.json"))
    assert store2.all()[0].mtype is MemType.EPISODIC
