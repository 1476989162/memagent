"""核心行为测试：嵌入相似度、去重、升降级、衰减、睡眠巩固、唤醒。"""

import time

import pytest

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.embedding import cosine_similarity, embed_text
from memagent.memory import MemoryStore, Tier, estimate_importance


def test_embedding_chinese_similarity():
    a = embed_text("我喜欢爬山和咖啡")
    b = embed_text("我喜欢爬山，也爱喝咖啡")
    c = embed_text("我喜欢游泳和跑步")  # 只共享"我喜欢"，不共享"爬山"
    assert cosine_similarity(a, b) > cosine_similarity(a, c) > 0.0


def test_importance_heuristic():
    assert estimate_importance("我叫小林，住在杭州。") > estimate_importance("你好")


def test_remember_dedup():
    agent = MemoryAgent()
    m1 = agent.remember("用户说：我喜欢爬山")
    m2 = agent.remember("用户说：我喜欢爬山")
    assert m1.id == m2.id  # 几乎相同的记忆合并，不重复入库
    assert m1.access_count == 1  # 第二次写入触发了测试效应强化（+1）


def test_promote_to_hot_after_repeated_retrieval():
    cfg = AgentConfig(hot_after_access=3, k=1)
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我的名字是小林")
    for _ in range(4):
        agent.retrieve("你叫什么名字", k=1)
    hot = agent.store.by_tier(Tier.HOT)
    assert len(hot) == 1
    assert hot[0].access_count >= 3


def test_decay_strength_decreases_with_time():
    agent = MemoryAgent()
    mem = agent.remember("一个不太重要的细节")
    s0 = agent._strength(mem)
    mem.last_access = time.time() - agent.cfg.tau_seconds * 5  # 模拟 5 个 τ 之后
    s1 = agent._strength(mem)
    assert s1 < s0


def test_sleep_compresses_and_recall_awakens():
    cfg = AgentConfig(
        tau_seconds=30.0,
        cold_after_seconds=1.0,
        cold_max_access=2,
        k=1,
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("用户说：我今天去爬了玉皇山，天气不错")
    agent.remember("用户说：我今天又去爬了玉皇山，很开心")
    time.sleep(1.1)
    report = agent.sleep()
    assert report["cold_compressed"] >= 1
    cold = agent.store.by_tier(Tier.COLD)
    assert cold, "应有 Cold 摘要"
    assert cold[0].summary  # 摘要非空
    # 摘要应保留可检索要点（玉皇山）
    assert "玉皇山" in cold[0].summary

    revived = agent.recall(cold[0].id[:6])
    assert revived is not None
    assert revived.tier is Tier.WARM


def test_find_memories_substring():
    agent = MemoryAgent()
    agent.remember("我昨天去吃了火锅")
    agent.remember("我喜欢爬山和咖啡")
    hits = agent.find_memories("火锅")
    assert [h.content for h in hits] == ["我昨天去吃了火锅"]
    assert agent.find_memories("爬山")[0].content == "我喜欢爬山和咖啡"


def test_find_memories_multiple_words_and_case():
    agent = MemoryAgent()
    agent.remember("我昨天去吃了火锅")
    agent.remember("我喜欢用 Python 写代码")
    agent.remember("我喜欢咖啡")
    assert agent.find_memories("python")[0].content == "我喜欢用 Python 写代码"  # 不区分大小写
    assert agent.find_memories("喜欢 咖啡")[0].content == "我喜欢咖啡"            # 多词 = 同时包含
    assert agent.find_memories("喜欢 爬山") == []                               # AND 不满足


def test_find_memories_no_match_and_empty():
    assert MemoryAgent().find_memories("不存在的词") == []
    assert MemoryAgent().find_memories("   ") == []  # 空关键词


def test_find_memories_matches_cold_summary_and_originals():
    cfg = AgentConfig(tau_seconds=30.0, cold_after_seconds=1.0, cold_max_access=2, k=1)
    agent = MemoryAgent(cfg=cfg)
    agent.remember("用户说：我今天去爬了玉皇山")
    time.sleep(1.1)
    agent.sleep()
    cold = agent.store.by_tier(Tier.COLD)
    assert cold and "玉皇山" in cold[0].summary
    assert agent.find_memories("玉皇山")            # 摘要可搜
    assert agent.find_memories("爬了玉皇山")        # 原始内容（originals）可搜


def test_save_load_roundtrip(tmp_path):
    store = MemoryStore(path=str(tmp_path / "mem.json"))
    store.add("我叫小林")
    store.save()
    store2 = MemoryStore(path=str(tmp_path / "mem.json"))
    assert len(store2) == 1
    assert store2.all()[0].content == "我叫小林"
