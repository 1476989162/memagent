"""再巩固因子自适应测试：从修订日志估计实测可塑性，像学习器调 τ 一样自动调因子。"""

import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType


def _retrieve_n(agent, query: str, n: int, gap: float = 0.02) -> None:
    for _ in range(n):
        agent.retrieve(query, k=1)
        time.sleep(gap)


def test_learn_plasticity_moves_factors_toward_true():
    # 隐藏真实可塑性：drift 2.5 / importance 1.5，配置（信念）1.0/1.0
    cfg = AgentConfig(
        tau_seconds=30.0,
        reconsolidation_by_type={MemType.EPISODIC: {"drift": 1.0, "importance": 1.0}},
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 2.5, "importance": 1.5}},
        plasticity_min_events=3,
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 6)  # 6 个修订事件
    old = dict(agent.cfg.reconsolidation_by_type[MemType.EPISODIC])
    r = agent.learn_plasticity()
    assert r["updated"], r
    new = agent.cfg.reconsolidation_by_type[MemType.EPISODIC]
    assert old["drift"] < new["drift"] <= 2.5        # 向真实值移动但不过冲
    assert old["importance"] < new["importance"] <= 1.5
    assert agent.store.meta["learned_plasticity"][MemType.EPISODIC.value]["drift"] == new["drift"]


def test_revision_log_records_applied_factors():
    cfg = AgentConfig(
        tau_seconds=30.0,
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 2.5, "importance": 1.5}},
    )
    agent = MemoryAgent(cfg=cfg)
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    agent.retrieve("昨天去看了场电影", k=1)
    assert m.revisions and len(m.revisions[-1]) == 7
    _, _, _, _, mtype, f_drift, f_imp = m.revisions[-1]
    assert mtype == MemType.EPISODIC.value
    assert f_drift == 2.5
    assert f_imp == 1.5


def test_self_consistent_no_drift():
    # 无 true 因子 → 实际即模型 → 实测 == 配置 → 不应被误调
    cfg = AgentConfig(
        tau_seconds=30.0,
        reconsolidation_by_type={MemType.EPISODIC: {"drift": 1.0, "importance": 1.0}},
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 4)
    r = agent.learn_plasticity()
    assert r["updated"] == []
    assert any(s["reason"] == "偏差过小" for s in r["skipped"])


def test_skips_without_enough_events():
    agent = MemoryAgent()
    agent.remember("我昨天去看了场电影", importance=0.1)
    agent.retrieve("昨天去看了场电影", k=1)  # 只有 1 个事件
    r = agent.learn_plasticity()
    assert r["updated"] == []
    assert r["skipped"]


def test_learning_can_be_disabled():
    cfg = AgentConfig(
        tau_seconds=30.0,
        plasticity_learning=False,
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 2.5, "importance": 1.5}},
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 4)
    before = agent.cfg.reconsolidation_by_type[MemType.EPISODIC]["drift"]
    assert agent.learn_plasticity()["updated"] == []
    assert agent.cfg.reconsolidation_by_type[MemType.EPISODIC]["drift"] == before


def test_factor_clamped_to_bounds():
    cfg = AgentConfig(
        tau_seconds=30.0,
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 99.0, "importance": -99.0}},
        plasticity_min=0.0,
        plasticity_max=5.0,
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 4)
    agent.learn_plasticity()
    d = agent.cfg.reconsolidation_by_type[MemType.EPISODIC]
    assert d["drift"] <= 5.0
    assert d["importance"] >= 0.0


def test_learned_factors_persist_across_restart(tmp_path):
    cfg = AgentConfig(
        tau_seconds=30.0,
        reconsolidation_by_type={MemType.EPISODIC: {"drift": 1.0, "importance": 1.0}},
        true_reconsolidation_by_type={MemType.EPISODIC: {"drift": 2.5, "importance": 1.5}},
    )
    agent = MemoryAgent(cfg=cfg, persist_path=str(tmp_path / "m.json"))
    agent.remember("我昨天去看了场电影", importance=0.1)
    _retrieve_n(agent, "昨天去看了场电影", 6)
    agent.learn_plasticity()
    agent.store.save()
    learned = agent.cfg.reconsolidation_by_type[MemType.EPISODIC]["drift"]

    from memagent.memory import MemoryStore

    agent2 = MemoryAgent(store=MemoryStore(path=str(tmp_path / "m.json")))
    assert abs(agent2.cfg.reconsolidation_by_type[MemType.EPISODIC]["drift"] - learned) < 1e-9
    # 未学习过的通道/类型保持默认
    assert agent2.cfg.reconsolidation_by_type[MemType.SKILL]["drift"] == 0.15


def test_samples_not_misattributed_after_migration():
    """迁移后修订日志仍按事件发生时的类型归组，不会被误归属到新类型。"""
    cfg = AgentConfig(
        tau_seconds=30.0,
        reconsolidation_by_type={MemType.EPISODIC: {"drift": 2.5, "importance": 1.5}},
    )
    agent = MemoryAgent(cfg=cfg)
    m = agent.remember("我昨天去看了场电影", importance=0.1)
    agent.retrieve("昨天去看了场电影", k=1)   # episodic 事件
    m.mtype = MemType.SEMANTIC               # 之后迁移为语义
    agent.retrieve("昨天去看了场电影", k=1)   # semantic 事件
    samples = agent._plasticity_samples()
    assert len(samples[MemType.EPISODIC.value]["drift"]) == 1   # 仍记在 episodic
    assert len(samples[MemType.SEMANTIC.value]["drift"]) == 1
    assert m.revisions[-1][4] == MemType.SEMANTIC.value
