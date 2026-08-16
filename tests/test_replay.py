"""睡眠回放测试：白天经历按时间顺序重放（再激活），睡眠中断时未回放的记忆次日模糊。

设计映射：锐波涟漪（sharp-wave ripple）——睡眠时海马体按时间顺序重放白天的
轨迹，每次重放 = 一次再激活（access_count +1 + 观测采样：强度微调、语义化评分
贡献）；完整睡眠重放全部候选，中断（sleep(duration)）时按预算只重放一部分，
未回放的候选次日"模糊"（importance × replay_fog_factor）。
"""

from memagent.agent import AgentConfig, MemoryAgent
from memagent.memory import MemType, Tier


def _mk(clock: list, **kw) -> MemoryAgent:
    base = dict(reconsolidate=False, replay_window_seconds=100.0)
    base.update(kw)
    return MemoryAgent(cfg=AgentConfig(**base), now_fn=lambda: clock[0])


def test_full_sleep_replays_all_active():
    """完整睡眠：窗口内活跃的记忆全部重放——access_count+1、观测采样+1、强度微调上升。"""
    clock = [1000.0]
    a = _mk(clock)
    m = a.store.add("我昨天去吃了火锅", importance=0.3)
    m.last_access = clock[0]
    clock[0] += 10
    s_before = a._strength(m)
    n_hist, acc = len(m.history), m.access_count
    r = a.sleep()
    assert r["replay_candidates"] == 1 and r["replayed_count"] == 1
    assert r["unreplayed_count"] == 0 and r["fogged"] == []
    assert r["replayed"] == ["我昨天去吃了火锅"]
    assert m.access_count == acc + 1              # 再激活（测试效应）
    assert len(m.history) == n_hist + 1           # 重放观测采样
    assert a._strength(m) > s_before              # 强度微调上升


def test_replay_chronological_order():
    """中断时按经历时间顺序重放：last_access 早的先重放，晚的未回放 → 模糊。"""
    clock = [1000.0]
    a = _mk(clock)
    m_early = a.store.add("早期经历", importance=0.3)
    m_early.last_access = clock[0]
    clock[0] += 10
    m_late = a.store.add("晚期经历", importance=0.3)
    m_late.last_access = clock[0]
    clock[0] += 5
    r = a.sleep(duration=1.0)                     # 预算 1 条
    assert r["replayed_count"] == 1
    assert r["replayed"] == ["早期经历"]          # 时间序：早的先重放
    assert m_early.access_count == 1 and m_late.access_count == 0
    assert r["unreplayed_count"] == 1
    assert r["fogged"][0]["content"].startswith("晚期经历")
    assert r["fogged"][0]["importance"] == 0.3 and r["fogged"][0]["fogged"] == 0.27
    assert m_late.importance == 0.27              # 0.3 × 0.9 次日模糊


def test_interrupted_sleep_budget_zero_fogs_all():
    """预算 0（几乎没睡着）：全部候选未回放 → 全部模糊。"""
    clock = [1000.0]
    a = _mk(clock)
    ms = []
    for text in ("经历甲", "经历乙"):
        m = a.store.add(text, importance=0.5)
        m.last_access = clock[0]
        ms.append(m)
        clock[0] += 1
    r = a.sleep(duration=0.5)                     # 预算 int(0.5×1) = 0
    assert r["replayed_count"] == 0 and r["unreplayed_count"] == 2
    assert all(m.importance == 0.45 for m in ms)  # 0.5 × 0.9
    assert all(m.access_count == 0 for m in ms)   # 未重放 → 无再激活


def test_replay_does_not_touch_last_access():
    """重放不改 last_access：否则每次睡眠都重置衰减时钟，记忆变得不朽。"""
    clock = [1000.0]
    a = _mk(clock)
    m = a.store.add("我昨天去吃了火锅", importance=0.3)
    m.last_access = clock[0]
    clock[0] += 10
    a.sleep()
    assert m.last_access == 1000.0                # 衰减时钟保持


def test_replay_excludes_cold_and_turn():
    """候选排除：Cold 已归档、turn 对话流水是瞬时记录——都不参与重放。"""
    clock = [1000.0]
    a = _mk(clock)
    cold = a.store.add("冷记忆", importance=0.3)
    cold.demote_to_cold("冷记忆")
    turn = a.store.add("用户说：你好", importance=0.1, kind="turn")
    fact = a.store.add("事实记忆", importance=0.3)
    for m in a.store.all():
        m.last_access = clock[0]
    r = a.sleep()
    assert r["replay_candidates"] == 1            # 只有 fact
    assert fact.access_count == 1
    assert cold.access_count == 0 and turn.access_count == 0


def test_replay_outside_window_not_candidate():
    """窗口外（久未活跃）的记忆不是"白天经历"，不重放。"""
    clock = [1000.0]
    a = _mk(clock)
    m = a.store.add("很久以前的经历", importance=0.3)
    m.last_access = clock[0] - 1000               # 窗口外（window=100s）
    clock[0] += 10
    r = a.sleep()
    assert r["replay_candidates"] == 0 and r["replayed_count"] == 0
    assert m.access_count == 0


def test_replay_switch_disables():
    """replay=False：行为与旧版一致——无重放、无采样、无模糊。"""
    clock = [1000.0]
    a = _mk(clock, replay=False)
    m = a.store.add("我昨天去吃了火锅", importance=0.3)
    m.last_access = clock[0]
    clock[0] += 10
    r = a.sleep()
    assert r["replayed_count"] == 0 and r["replay_candidates"] == 0
    assert m.access_count == 0 and m.importance == 0.3  # 无重放、无模糊（_observe 照常采样）


def test_replayed_memory_escapes_compression():
    """判别场景：access_count 恰等于 cold_max_access 的记忆本应被压缩，重放 +1 后
    超过上限 → 逃过压缩（被重放 = 更牢，次晨不易被遗忘）。"""
    clock = [1000.0]
    a = _mk(
        clock,
        tau_seconds=30.0,
        cold_after_seconds=1.0,
        cold_max_access=2,
    )
    m = a.store.add("我会被重放保住", importance=0.3)
    m.access_count = 2                            # 恰好 = cold_max_access
    m.last_access = clock[0]
    clock[0] += 2                                 # 闲置 2s > cold_after 1s
    r = a.sleep()
    assert r["replayed_count"] == 1
    assert m.access_count == 3                    # 重放 +1
    assert m.tier is not Tier.COLD                # 逃过本次压缩
    # 对照组：同状态但未参与重放（窗口外）→ 正常压缩
    clock[0] += 2
    m2 = a.store.add("对照记忆", importance=0.3)
    m2.access_count = 2
    m2.last_access = clock[0] - 1000              # 窗口外 → 不重放
    a.sleep()
    assert m2.tier is Tier.COLD


def test_replay_contributes_to_semanticization():
    """重放 = 使用事件：反复被重放的 episodic 记忆语义化评分累积，睡眠后迁移为 semantic
    （人脑：反复回放的情景经历固化为一般性知识）。"""
    clock = [1000.0]
    a = _mk(
        clock,
        tau_by_type={MemType.SKILL: 90.0, MemType.SEMANTIC: 30.0, MemType.EPISODIC: 8.0},
        semanticization_tau_seconds=10000.0,      # 评分衰减极慢 → 重放贡献可叠加
        semanticize_threshold=2.0,
    )
    m = a.store.add("我昨天去看了场电影", importance=0.1, mtype=MemType.EPISODIC)
    m.last_access = clock[0]
    for _ in range(4):
        clock[0] += 1
        a.sleep()                                 # 每次重放 1 次（首条重放无前序采样，
                                                  # 4 次睡眠 → 3 个被检测到的使用事件）
    assert m.access_count == 4
    assert a._semanticization_score(m) >= 2.0     # 重放推高语义化评分
    assert m.mtype is MemType.SEMANTIC            # 睡眠中发生迁移
