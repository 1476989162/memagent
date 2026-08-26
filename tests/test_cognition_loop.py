# -*- coding: utf-8 -*-
"""认知闭环增量测试：REM 联想重组、扩散激活、检索诱导遗忘、舌尖现象。

对应人脑机制：
- REM 睡眠联想重组：远距弱相关记忆在睡眠中重新组合（洞察来源）
- 扩散激活：想起 A 沿联想网络带出相邻的 B（顺藤摸瓜）
- 检索诱导遗忘：成功提取抑制同主题竞争者（信噪比守门人）
- 舌尖现象：「知道记得但说不出」转为可用线索
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEMAGENT_TEST", "1")

from eval_retrieval import FIXED_NOW  # noqa: E402
from memagent.agent import AgentConfig, MemoryAgent  # noqa: E402
from memagent.instructions import build_tips  # noqa: E402


def _mk(**kw) -> MemoryAgent:
    base = dict(tau_seconds=30 * 24 * 3600, reconsolidate=False,
                hot_after_access=999)
    base.update(kw)
    return MemoryAgent(cfg=AgentConfig(**base), now_fn=lambda: FIXED_NOW)


def _add(agent, content):
    m = agent.store.add(content, importance=0.5, now=FIXED_NOW)
    m.last_access = FIXED_NOW
    return m


# ---------- 1) REM 联想重组 ----------

def test_sleep_creates_rem_links_for_midband_pairs():
    """活跃记忆与「相邻但不同」话题的记忆在睡眠后建立双向联想边。"""
    agent = _mk()
    a = _add(agent, "团队决定采用 React 重构前端页面")
    b = _add(agent, "前端组件库调研结论：React 生态最成熟")
    c = _add(agent, "今晚食堂的红烧肉做得不错")          # 无关噪声
    dup = _add(agent, "团队决定采用 React 重构前端页面!")  # 近重复，应被排除
    report = agent.sleep()
    links_a = agent.rem_links.get(a.id, {})
    assert b.id in links_a, f"中频对未建边: {report.get('rem_insights')}"
    assert dup.id not in links_a               # 近重复被 hi 上限/rank1 排除
    assert agent.rem_links[b.id][a.id] == links_a[b.id]   # 双向一致
    assert report["rem_insights"]


def test_rem_respects_max_edges_and_persistence():
    from memagent.memory import MemoryStore  # noqa: F401
    agent = _mk(rem_max_edges=1)
    for i in range(4):
        _add(agent, f"微服务架构决策{i}：服务网格与注册中心选型讨论")
    agent.sleep()
    total = sum(len(v) for v in agent.rem_links.values())
    assert total <= 2                          # 单边上限（双向存储计 2）
    # 持久化往返
    state = agent._persist_agent_state() or {}
    agent.store.meta["agent_state"] = state


# ---------- 2) 扩散激活 ----------

def test_spread_activation_brings_neighbor():
    """查询只命中 A 时，REM 邻居 B 以扩散标记进入结果。"""
    agent = _mk()
    a = _add(agent, "项目决定采用 PostgreSQL 作为主数据库")
    b = _add(agent, "连接池与超时重试参数调优实践记录")   # 与查询词汇不相交
    _add(agent, "今天中午吃了麻辣火锅")
    agent.rem_links[a.id] = {b.id: 0.6}
    agent.rem_links[b.id] = {a.id: 0.6}
    hits = agent.retrieve("数据库选型结论是什么", k=5)
    spread_hits = [h for h in hits if h.spread]
    assert spread_hits and spread_hits[0].memory.id == b.id
    assert spread_hits[0].relevance <= 0.6 * 0.5 + 1e-9   # 边权×系数封顶


def test_spread_disabled_keeps_old_behavior():
    agent = _mk(spread_activation=False)
    a = _add(agent, "项目决定采用 PostgreSQL 作为主数据库")
    b = _add(agent, "数据库连接池参数调优的实践经验记录")
    agent.rem_links[a.id] = {b.id: 0.6}
    hits = agent.retrieve("数据库选型结论是什么", k=5)
    assert not any(h.spread for h in hits)


# ---------- 3) 检索诱导遗忘 ----------

def test_rif_suppresses_competitor_once_with_cooldown():
    agent = _mk(k=2)
    x = _add(agent, "Redis 网关侧缓存部署方案：含哨兵高可用与持久化配置")
    y = _add(agent, "Redis 客户端超时重试参数的调优记录与压测数据")
    z = _add(agent, "Redis 大 key 拆分实践与内存监控告警阈值")
    r1 = agent.retrieve("网关 Redis 哨兵部署方案", k=2)
    hit_ids = {h.memory.id for h in r1}
    assert x.id in hit_ids
    competitors = [m for m in (y, z) if m.id not in hit_ids]
    assert competitors, "应有竞争者落在命中之外"
    for m in competitors:
        assert m.importance < 0.5                 # 被 ×0.98 抑制
    imp_snapshot = {m.id: m.importance for m in competitors}
    agent.retrieve("网关 Redis 哨兵部署方案", k=2)   # 冷却期内再检索
    for m in competitors:
        assert m.importance == imp_snapshot[m.id]  # 冷却：不再连续抑制


def test_rif_disabled_no_suppression():
    agent = _mk(retrieval_forgetting=False, k=2)
    x = _add(agent, "Redis 网关侧缓存部署方案：含哨兵高可用与持久化配置")
    z = _add(agent, "Redis 大 key 拆分实践与内存监控告警阈值")
    agent.retrieve("网关 Redis 哨兵部署方案", k=2)
    assert z.importance == 0.5


# ---------- 4) 舌尖现象 ----------

class _Hit:
    def __init__(self, memory, relevance):
        self.memory = memory
        self.relevance = relevance


def test_tip_of_tongue_band_extraction():
    agent = _mk()
    m = agent.store.add("我养了一只叫雪球的萨摩耶犬", importance=0.5,
                        now=FIXED_NOW)
    tips = build_tips("萨摩耶会掉毛吗", [_Hit(m, 0.18)])
    assert len(tips) == 1
    assert "萨摩耶" in tips[0]["hint"]
    assert tips[0]["id_prefix"] == m.id[:6]


def test_tips_ignore_confident_and_noise():
    agent = _mk()
    confident = agent.store.add("确认级记忆内容示例甲乙丙丁", importance=0.5,
                                now=FIXED_NOW)
    noise = agent.store.add("完全无关的记忆内容戊己庚辛", importance=0.5,
                            now=FIXED_NOW)
    hits = [_Hit(confident, 0.60), _Hit(noise, 0.05)]
    assert build_tips("确认级记忆", hits) == []


def test_tip_cue_falls_back_to_text_head():
    agent = _mk()
    m = agent.store.add("生僻主题记忆：斐波那契堆的摊还分析", importance=0.5,
                        now=FIXED_NOW)
    tips = build_tips("那个数据结构叫什么来着", [_Hit(m, 0.15)])
    assert tips and "斐波那契" in tips[0]["hint"]     # 回退到文本开头作线索
