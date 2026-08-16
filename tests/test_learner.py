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
