"""参数自适应学习器测试：按实测 τ 自动校准、门控、持久化、向后兼容。"""

import json
import time

from memagent import MemoryAgent
from memagent.agent import AgentConfig
from memagent.memory import MemType, MemoryStore


def _observe(agent, gap, n):
    for _ in range(n):
        agent._observe()
        time.sleep(gap)


def test_learn_tau_moves_toward_true_tau():
    cfg = AgentConfig(
        tau_by_type={MemType.EPISODIC: 6.0},
        true_tau_by_type={MemType.EPISODIC: 2.0},  # 真实遗忘快 3 倍
        innate_bounds={},  # 本测试用秒级 τ 验证 EMA 学习数学，与出厂日级边界无关
    )
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe(agent, 0.4, 6)
    old = agent.cfg.tau_for(MemType.EPISODIC)
    r = agent.learn_tau()
    new = agent.cfg.tau_for(MemType.EPISODIC)
    assert r["updated"], r
    assert 2.0 < new < old  # 向真实 τ 移动但不过冲
    assert agent.store.meta["learned_tau"][MemType.EPISODIC.value] == new


def test_learn_tau_skips_without_enough_data():
    agent = MemoryAgent()
    agent.remember("我叫小林")
    r = agent.learn_tau()
    assert r["updated"] == []
    assert r["skipped"]  # 观测不足/干净段不足


def test_learn_tau_disabled():
    cfg = AgentConfig(tau_learning=False)
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我叫小林")
    assert agent.learn_tau()["updated"] == []


def test_self_consistent_no_drift():
    # 无 true_tau → 现实即模型，实测 τ ≈ 配置 τ → 不应被误调
    cfg = AgentConfig(tau_by_type={MemType.EPISODIC: 5.0},
                      innate_bounds={})  # 秒级 τ 低于出厂日级下限，测试学习器本身
    agent = MemoryAgent(cfg=cfg)
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe(agent, 0.3, 6)
    old = agent.cfg.tau_for(MemType.EPISODIC)
    agent.learn_tau()
    new = agent.cfg.tau_for(MemType.EPISODIC)
    assert abs(new - old) / old < 0.15


def test_true_tau_fallback_matches_prediction_basis_with_emotion():
    """观测口径回归：非实验模式下回落 = 裸模型 τ，情绪标注不得带偏采样。

    此前回落到 _tau_for（多乘情绪/兴趣因子），带恐惧标签的记忆采样按
    τ×10+ 衰减而预测端用裸类型 τ——自洽环境也会产生幻影偏差。"""
    from memagent.emotion import Emotion

    agent = MemoryAgent(cfg=AgentConfig(tau_by_type={MemType.EPISODIC: 30.0},
                                        innate_bounds={}))
    m = agent.remember("我昨天去了趟图书馆", importance=0.1, mtype=MemType.EPISODIC,
                       emotion=Emotion(valence=-0.8, arousal=0.9,
                                       self_relevance=0.9, label="恐惧"))
    assert m.emotion is not None                     # 确认标注在位（否则测试空转）
    assert agent._true_tau_for(m) == agent.cfg.tau_for(MemType.EPISODIC)


def test_clean_segment_tau_est_exact_when_observation_is_model():
    """自洽环境 + 情绪标注：干净段反推的实测 τ 应精确回到配置值。

    观测与预测同源后，反推不再被情绪因子系统性抬高（此前 neutral ≈×1.10、
    恐惧 ×10+）。"""
    from memagent.visualize import fit_report

    clock = {"t": 1000.0}
    agent = MemoryAgent(
        cfg=AgentConfig(tau_by_type={MemType.EPISODIC: 30.0}, innate_bounds={}),
        now_fn=lambda: clock["t"],
    )
    m = agent.remember("我昨天去吃了火锅", importance=0.1, mtype=MemType.EPISODIC)
    agent._observe()
    clock["t"] += 6.0                                # 一个衰减段（远离触底）
    agent._observe()
    seg = fit_report(agent)["by_type"][MemType.EPISODIC.value]
    assert seg["clean"] >= 1 and seg["tau_est"] is not None
    assert abs(seg["tau_est"] - 30.0) / 30.0 < 0.05


def test_learned_tau_persists_across_restart(tmp_path):
    cfg = AgentConfig(
        tau_by_type={MemType.EPISODIC: 6.0},
        true_tau_by_type={MemType.EPISODIC: 2.0},
    )
    agent = MemoryAgent(cfg=cfg, persist_path=str(tmp_path / "m.json"))
    agent.remember("我昨天去吃了火锅", importance=0.1)
    _observe(agent, 0.4, 6)
    agent.learn_tau()
    agent.store.save()
    learned = agent.cfg.tau_for(MemType.EPISODIC)

    store2 = MemoryStore(path=str(tmp_path / "m.json"))
    agent2 = MemoryAgent(store=store2)  # 默认配置重启，学习开启 → 应用持久化 τ
    assert abs(agent2.cfg.tau_for(MemType.EPISODIC) - learned) < 0.5


def test_old_save_format_backward_compat(tmp_path):
    p = tmp_path / "m.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump([{"id": "abc", "content": "我叫小林", "tier": "warm"}], f, ensure_ascii=False)
    store = MemoryStore(path=str(p))
    assert len(store) == 1
    assert store.meta == {}  # 旧格式无 meta
    assert store.all()[0].content == "我叫小林"
