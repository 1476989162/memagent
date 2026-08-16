"""心游（spontaneous_recall）测试：无查询时按强度加权随机想起一条记忆。

设计映射：默认模式网络（DMN）——静息时大脑自发采样记忆库，不靠外部线索。
被想起的记忆获得再激活测试效应（touch + 采样）；强度加权让"越牢越容易被想起"、
触底记忆偶尔也会冒出来；随后它成为当晚睡眠回放的候选，形成
心游 → 想起 → 再激活 → 回放 → 更牢 的闭环。
"""

import random

from memagent.agent import AgentConfig, MemoryAgent
from memagent.memory import MemType, Tier


def _mk(clock: list, **kw) -> MemoryAgent:
    base = dict(reconsolidate=False, replay_window_seconds=100.0)
    base.update(kw)
    return MemoryAgent(cfg=AgentConfig(**base), now_fn=lambda: clock[0])


def test_spontaneous_recall_gives_testing_effect():
    """被想起的记忆获得再激活：次数 +1、时间刷新、观测采样 +1（无查询不触发再巩固）。"""
    clock = [1000.0]
    a = _mk(clock)
    m = a.store.add("我昨天去吃了火锅", importance=0.3)
    m.last_access = clock[0] - 500          # 已衰减（非触底）
    clock[0] += 1
    n_hist, acc, la = len(m.history), m.access_count, m.last_access
    picked = a.spontaneous_recall(rng=random.Random(1))
    assert picked is m                      # 唯一候选 → 必然想起它
    assert m.access_count == acc + 1        # 测试效应
    assert m.last_access == clock[0]        # 时间刷新
    assert len(m.history) == n_hist + 1     # 观测采样
    assert m.revision_count == 0            # 无查询上下文 → 不触发再巩固改写


def test_spontaneous_recall_strength_weighted():
    """强度加权：强记忆（≈0.88）被想起的频率显著高于触底记忆（0.2）。

    用"每全新 agent 采一次"统计——避免同 agent 内 touch 反馈的
    "越想起越牢"自增强（想起 → 刷新 → 更强 → 更易再想起）污染加权测量。
    """
    rng = random.Random(42)
    strong_picks = weak_picks = 0
    for _ in range(500):
        clock = [1000.0]
        a = _mk(clock)
        strong = a.store.add("我经常去爬山", importance=0.9, mtype=MemType.SKILL)
        strong.access_count = 5
        strong.last_access = clock[0]           # 新鲜 → 强度高
        weak = a.store.add("很久以前的琐事", importance=0.1, mtype=MemType.EPISODIC)
        weak.last_access = clock[0] - 10 * 24 * 3600   # 深衰 → 触底 0.2
        s_strong, s_weak = a._strength(strong), a._strength(weak)
        assert s_strong > 0.8 and s_weak == 0.2
        if a.spontaneous_recall(rng=rng) is strong:
            strong_picks += 1
        else:
            weak_picks += 1
    ratio = strong_picks / weak_picks
    # 理论 ≈0.88/0.2 = 4.4；500 次采样下 2× 阈值远离噪声（均匀 50/50 必失败）
    assert ratio > 2.0, f"强度加权失效: 强/弱 = {ratio:.2f}（{strong_picks}/{weak_picks}）"


def test_spontaneous_recall_excludes_cold():
    """Cold 深藏记忆不参与心游（需线索才能唤起）——永远不被想起、不受测试效应。"""
    clock = [1000.0]
    a = _mk(clock)
    cold = a.store.add("已归档的旧记忆", importance=0.3, now=clock[0])
    cold.last_access = clock[0]
    cold.demote_to_cold("已归档的旧记忆")
    warm = a.store.add("最近的记忆", importance=0.3, now=clock[0])
    warm.last_access = clock[0]
    rng = random.Random(7)
    for _ in range(200):
        a.spontaneous_recall(rng=rng)
    assert warm.access_count == 200          # 唯一非 Cold → 每次都被想起
    assert cold.access_count == 0            # Cold 永不被想起
    assert cold.last_access == 1000.0        # 时间也未刷新


def test_spontaneous_recall_empty_store():
    """空记忆库：无事可想起 → None。"""
    a = MemoryAgent(cfg=AgentConfig(reconsolidate=False))
    assert a.spontaneous_recall(rng=random.Random(0)) is None


def test_spontaneous_recall_feeds_replay():
    """闭环：被心游想起的记忆时间刷新 → 当晚成为睡眠回放候选（再激活叠加）。"""
    clock = [1000.0]
    a = _mk(clock)
    m = a.store.add("我今天突然想起的事", importance=0.3)
    m.last_access = clock[0] - 50            # 窗口内但已闲置 50s
    clock[0] += 1
    a.spontaneous_recall(rng=random.Random(3))   # 想起 → touch 到 now
    acc_before = m.access_count
    r = a.sleep()
    assert r["replayed_count"] == 1          # 活跃窗口内 → 当晚回放
    assert m.access_count == acc_before + 1  # 回放再激活叠加
