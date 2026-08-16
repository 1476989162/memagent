"""类型迁移（情景语义化）测试：被反复检索的 episodic 固化为 semantic，低频反向淡化。"""

import json
import os
import tempfile
import time

from memagent.agent import AgentConfig, MemoryAgent
from memagent.memory import MemType, Tier


def _fast_cfg(**kw) -> AgentConfig:
    base = dict(
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
        cold_after_seconds=999999.0,  # 测试聚焦迁移，不触发压缩
        semanticization_tau_seconds=5.0,
        semanticize_threshold=3.0,
        desemanticize_threshold=0.8,
    )
    base.update(kw)
    return AgentConfig(**base)


def _retrieve_n(agent, query: str, n: int, gap: float = 0.05) -> None:
    for _ in range(n):
        agent.retrieve(query, k=1)
        time.sleep(gap)


def test_episodic_migrates_to_semantic_after_frequent_retrieval():
    agent = MemoryAgent(cfg=_fast_cfg())
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    assert m.mtype is MemType.EPISODIC
    _retrieve_n(agent, "昨天去看了场电影", 4)
    assert agent._semanticization_score(m) >= agent.cfg.semanticize_threshold
    rep = agent.sleep()
    assert m.mtype is MemType.SEMANTIC
    assert rep["migrations"] == 1
    assert m.migrations and m.migrations[-1][1] == "episodic→semantic"
    assert m.mtype_confidence is None  # 类型改由迁移决定，不再有分类置信度


def test_semantic_fades_back_to_episodic_when_unused():
    # replay=False：睡眠回放会在第二次 sleep 时重新激活记忆（access_count+1 + 采样），
    # 语义化评分被回放事件推高，"停止检索 → 淡化"的流程被干扰——本测试聚焦反向迁移，
    # 回放语义（被重放即巩固、不淡化）由 test_replay.py 单独验证
    agent = MemoryAgent(cfg=_fast_cfg(replay=False))
    m = agent.remember("北京是中国的首都", importance=0.1, mtype=MemType.SEMANTIC)
    _retrieve_n(agent, "中国的首都", 2)  # 曾经被使用 → 允许淡化
    rep = agent.sleep()
    assert m.mtype is MemType.SEMANTIC  # 刚检索过，评分仍高，先保持
    assert rep["migrations"] == 0
    time.sleep(6)  # 停止检索，评分衰减
    assert agent._semanticization_score(m) < agent.cfg.desemanticize_threshold
    rep2 = agent.sleep()
    assert m.mtype is MemType.EPISODIC
    assert rep2["migrations"] == 1
    assert m.migrations[-1][1] == "semantic→episodic"


def test_new_semantic_without_use_does_not_fade():
    """从未被使用过的 semantic 事实（access_count<2）不淡化为 episodic。"""
    agent = MemoryAgent(cfg=_fast_cfg())
    m = agent.remember("北京是中国的首都", importance=0.1, mtype=MemType.SEMANTIC)
    assert m.access_count == 0
    rep = agent.sleep()
    assert m.mtype is MemType.SEMANTIC  # 新存的事实不立即翻转
    assert rep["migrations"] == 0


def test_turn_memories_excluded():
    agent = MemoryAgent(cfg=_fast_cfg())
    m = agent.remember("用户说：我昨天去吃了火锅", kind="turn")
    assert m.mtype is MemType.EPISODIC
    _retrieve_n(agent, "昨天去吃了火锅", 4)
    rep = agent.sleep()
    assert m.mtype is MemType.EPISODIC  # 对话流水是瞬时记录，不迁移
    assert rep["migrations"] == 0


def test_cold_memories_excluded():
    agent = MemoryAgent(cfg=_fast_cfg())
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    m.demote_to_cold("电影相关记录", sources=[m])
    _retrieve_n(agent, "电影相关记录", 4)
    rep = agent.sleep()
    assert m.tier is Tier.COLD
    assert m.mtype is MemType.EPISODIC
    assert rep["migrations"] == 0


def test_semanticize_switch_disables_migration():
    agent = MemoryAgent(cfg=_fast_cfg(semanticize=False))
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 4)
    rep = agent.sleep()
    assert m.mtype is MemType.EPISODIC
    assert rep["migrations"] == 0


def test_migration_persists_across_roundtrip():
    agent = MemoryAgent(cfg=_fast_cfg())
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 4)
    agent.sleep()
    assert m.mtype is MemType.SEMANTIC
    d = m.to_dict()
    restored = type(m).from_dict(d)
    assert restored.mtype is MemType.SEMANTIC
    assert restored.migrations == m.migrations


def test_export_chain_carries_semanticization():
    agent = MemoryAgent(cfg=_fast_cfg())
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 4)
    agent.sleep()
    assert m.mtype is MemType.SEMANTIC
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "m")
        agent.plot_curves(base)
        with open(base + ".json", encoding="utf-8") as f:
            data = json.load(f)
        entry = next(x for x in data["memories"] if x["id"] == m.id)
        assert entry["mtype"] == "semantic"
        assert "semanticization_score" in entry
        assert entry["migrations"]
