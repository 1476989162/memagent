"""coder 记忆 ↔ MemoryAgent store 双向同步桥的单元测试。

背景（2026-08-15 排查）：agent.sleep() 的回放/冷压缩候选来自 store.all()，
但 coder 的记忆写入走 persist_improvements → 直接改 foxtable_memory.json，
从不经过 MemoryAgent 的 store → store 恒空 → 睡眠三项恒 0（「睡眠: 回放 0 · 冷压缩 0 · 演化入库 0」）。

本测试保护：
  - sync_ft_memory_to_store：dicts → store 的字段映射、幂等、事实字段更新；
  - sync_store_to_ft_memory：回放效果（access_count）写回 dicts，且不写回
    history / 冷压缩 / tier（保护注入系统）；
  - 端到端：真实 MemoryAgent.sleep() 对同步后的新规则真的回放（replayed_count > 0），
    且高 importance 规则不会被冷压缩（importance >= 0.85 全部超过 0.8 阈值）。
"""
import time

import autonomous_coder as ac
from memagent.agent import AgentConfig, MemoryAgent
from memagent.memory import MemoryStore


def _mems(n=3, importance=0.9, domain="Table"):
    now = time.time()
    return [{
        "id": f"mem{i}",
        "kind": "skill", "mtype": "skill",
        "content": f"[{domain}/改进] 题目「测试{i}」：规则内容{i}",
        "importance": importance, "access_count": 2,
        "last_access": now, "tier": "warm", "created_at": now,
        "history": [[now, 1.0, now, 2, importance]],
    } for i in range(n)]


# ---------- sync_ft_memory_to_store ----------

def test_sync_ft_to_store_maps_fields():
    store = MemoryStore()
    mems = _mems(2)
    added = ac.sync_ft_memory_to_store(store, mems)
    assert added == 2
    m = store.get("mem0")
    assert m is not None
    assert m.content == mems[0]["content"]
    assert m.importance == 0.9
    assert m.access_count == 2
    assert m.kind == "skill"
    assert len(store.all()) == 2


def test_sync_ft_to_store_idempotent():
    store = MemoryStore()
    mems = _mems(2)
    ac.sync_ft_memory_to_store(store, mems)
    added = ac.sync_ft_memory_to_store(store, mems)
    assert added == 0  # 已存在，不重复添加
    assert len(store.all()) == 2


def test_sync_ft_to_store_updates_facts_keeps_evolution():
    """事实字段（content/importance）以 dict 为准更新；
    演化状态（access_count 等）以 store 为准保留。"""
    store = MemoryStore()
    mems = _mems(1)
    ac.sync_ft_memory_to_store(store, mems)
    # store 侧回放 +1（sleep 效果）
    store.get("mem0").access_count = 3
    # dict 侧改了内容/importance（复犯升级等）
    mems[0]["content"] = "[Table/坑] 铁律新内容"
    mems[0]["importance"] = 1.2
    ac.sync_ft_memory_to_store(store, mems)
    m = store.get("mem0")
    assert m.content == "[Table/坑] 铁律新内容"  # 事实字段更新
    assert m.importance == 1.2
    assert m.access_count == 3  # 演化状态不被覆盖


# ---------- sync_store_to_ft_memory ----------

def test_sync_store_to_ft_writes_back_access_count():
    store = MemoryStore()
    mems = _mems(2)
    ac.sync_ft_memory_to_store(store, mems)
    store.get("mem0").access_count = 5
    changed = ac.sync_store_to_ft_memory(store, mems)
    assert changed["access_count"] == 1
    assert mems[0]["access_count"] == 5
    assert mems[1]["access_count"] == 2  # 未变化的条目不动


def test_sync_store_to_ft_does_not_write_history():
    """观测 history 不写回 JSON——避免每轮 sleep 的 _observe 无限膨胀记忆文件。"""
    store = MemoryStore()
    mems = _mems(1)
    ac.sync_ft_memory_to_store(store, mems)
    # 模拟 sleep 观测追加 history
    store.get("mem0").history.append([time.time(), 1.0, time.time(), 3, 0.9])
    ac.sync_store_to_ft_memory(store, mems)
    assert len(mems[0]["history"]) == 1  # 原 dict 的 history 未被追加


def test_sync_store_to_ft_ignores_removed_cold():
    """防御：store 侧被冷压缩合并移除的规则，dict 侧保留原文（不删不写回）。"""
    store = MemoryStore()
    mems = _mems(2)
    ac.sync_ft_memory_to_store(store, mems)
    # 模拟冷压缩：主记忆 demote_to_cold，其余从 store 移除
    store.get("mem1").demote_to_cold("摘要")
    store.remove("mem0")
    changed = ac.sync_store_to_ft_memory(store, mems)
    assert changed["access_count"] == 0
    assert mems[0]["access_count"] == 2  # 被移除的规则保留原文与计数
    assert mems[1]["access_count"] == 2  # 冷压缩不把 tier 写回


# ---------- 端到端：真实 agent.sleep() ----------

def _agent_with_store(mems):
    store = MemoryStore()
    ac.sync_ft_memory_to_store(store, mems)
    agent = MemoryAgent(store=store, responder=None,
                        cfg=AgentConfig(evolve_on_sleep=False))
    return agent


def test_sleep_replays_newly_synced_rules():
    """新规则 last_access = now 恰在 replay_window 内 → 回放真正命中（非 0）。"""
    mems = _mems(3)
    agent = _agent_with_store(mems)
    rep = agent.sleep()
    assert rep["replayed_count"] >= 3  # 刚同步的 3 条都被回放
    # 回放 = 再激活：access_count +1
    assert store_counts(agent.store) == 3


def store_counts(store):
    return sum(1 for m in store.all() if m.access_count >= 3)


def test_sleep_does_not_cold_compress_high_importance_rules():
    """coder 规则 importance 全 >= 0.85 > 0.8 阈值 → 冷压缩候选为 0。
    这是正确行为：高价值规则不该被压成摘要（保护注入系统）。"""
    mems = _mems(3, importance=0.9)
    agent = _agent_with_store(mems)
    rep = agent.sleep()
    assert rep["cold_compressed"] == 0
    assert len(agent.store.all()) == 3  # 全部保留，无合并移除


def test_full_cycle_bridge_roundtrip():
    """完整闭环：dicts → store → sleep → 写回 dicts → 变化计数正确。"""
    mems = _mems(2)
    store = MemoryStore()
    ac.sync_ft_memory_to_store(store, mems)
    agent = MemoryAgent(store=store, responder=None,
                        cfg=AgentConfig(evolve_on_sleep=False))
    agent.sleep()
    changed = ac.sync_store_to_ft_memory(store, mems)
    assert changed["access_count"] >= 2  # 两条都被回放写回
    assert all(m["access_count"] >= 3 for m in mems)


# ---------- 回放窗口扩展（replay_rounds）：旧教训周期性再激活 ----------

def _mems_with_ages(n_new=2, n_old=3):
    """n_new 条新沉淀（last_access=now）+ n_old 条旧教训（last_access 很久前）。"""
    now = time.time()
    mems = []
    for i in range(n_old):
        mems.append({
            "id": f"old{i}", "kind": "skill", "mtype": "skill",
            "content": f"[Table/改进] 题目「旧{i}」：旧教训{i}",
            "importance": 0.85, "access_count": 2,
            "last_access": now - 3600, "tier": "warm", "created_at": now - 3600,
            "history": [[now - 3600, 1.0, now - 3600, 2, 0.85]],
        })
    for i in range(n_new):
        mems.append({
            "id": f"new{i}", "kind": "skill", "mtype": "skill",
            "content": f"[Table/改进] 题目「新{i}」：新教训{i}",
            "importance": 0.85, "access_count": 2,
            "last_access": now, "tier": "warm", "created_at": now,
            "history": [[now, 1.0, now, 2, 0.85]],
        })
    return mems


def _small_window_cfg():
    """生产环境 TIME_SCALE=1/86400 → 默认窗口 1 秒；测试环境 TIME_SCALE=1
    窗口是 86400s，旧教训（1 小时前）也会进。显式用小窗口模拟生产语义。"""
    return AgentConfig(evolve_on_sleep=False, replay_window_seconds=1.0)


def test_replay_rounds_zero_keeps_current_semantics():
    """默认（0）：只有本轮新沉淀进回放窗口，旧教训不被再激活。"""
    mems = _mems_with_ages()
    store = MemoryStore()
    ac.sync_ft_memory_to_store(store, mems, replay_rounds=0)
    agent = MemoryAgent(store=store, responder=None, cfg=_small_window_cfg())
    rep = agent.sleep()
    assert rep["replayed_count"] == 2  # 只有 2 条新的被回放
    # 旧教训 last_access 未刷新，仍不在窗口
    assert store.get("old0").last_access < time.time() - 3000


def test_replay_rounds_expands_window_to_old_lessons():
    """replay_rounds>0：旧教训也被刷 last_access → 进回放窗口再激活。"""
    mems = _mems_with_ages(n_old=3)
    store = MemoryStore()
    ac.sync_ft_memory_to_store(store, mems, replay_rounds=5)
    agent = MemoryAgent(store=store, responder=None, cfg=_small_window_cfg())
    rep = agent.sleep()
    assert rep["replayed_count"] >= 5  # 3 旧 + 2 新全部回放
    # 旧教训 last_access 已被刷到 now（周期性再激活的机制）
    assert time.time() - store.get("old0").last_access < 60


def test_replay_rounds_window_respects_n():
    """窗口只覆盖最近 N 轮（约 3N 条），更旧的教训不进窗口。"""
    mems = _mems_with_ages(n_old=10, n_new=2)  # 10 旧 + 2 新 = 12 条
    store = MemoryStore()
    ac.sync_ft_memory_to_store(store, mems, replay_rounds=1)  # 窗口=最近1轮≈3条
    # 只有 last_access 最新的 3 条被刷
    now = time.time()
    refreshed = [m for m in store.all() if now - m.last_access < 60]
    assert len(refreshed) == 3
    assert "new0" in {m.id for m in refreshed}
    assert "old9" not in {m.id for m in refreshed}


def test_replay_rounds_writeback_accumulates_access_count():
    """扩窗后旧教训 access_count 持续积累（写回 JSON，可观测活性增长）。"""
    mems = _mems_with_ages(n_old=2, n_new=1)
    store = MemoryStore()
    for _ in range(2):  # 模拟两轮
        ac.sync_ft_memory_to_store(store, mems, replay_rounds=5)
        agent = MemoryAgent(store=store, responder=None, cfg=_small_window_cfg())
        agent.sleep()
        changed = ac.sync_store_to_ft_memory(store, mems)
        assert changed["access_count"] >= 3
    old0 = next(m for m in mems if m["id"] == "old0")
    assert old0["access_count"] >= 4  # 每轮 +1，两轮后 2→4
