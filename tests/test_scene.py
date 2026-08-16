"""场景重建测试：检索命中时把相关记忆片段组合成连贯场景（片段组合回忆）。

设计映射：人脑回忆的是片段组合（场景重建）而非单条事实——把彼此相关的
记忆片段按经历顺序拼成连贯叙事。种子 = 检索命中；扩展 = 与种子共享主题词
（n-gram）且相似度达阈值的记忆（require_shared 门控过滤哈希碰撞噪声）；
时间窗保证同场景的时间连贯性；被纳入场景的片段获得再激活（测试效应）与
可选再巩固（重建会微调片段）。
"""

from memagent.agent import (
    AgentConfig,
    MemoryAgent,
    _fragment_relatedness,
    format_scene,
)
from memagent.memory import MemType, Tier


def _mk(clock: list, **kw) -> MemoryAgent:
    base = dict(reconsolidate=False)
    base.update(kw)
    return MemoryAgent(cfg=AgentConfig(**base), now_fn=lambda: clock[0])


def _seed_scene(clock: list) -> tuple:
    """三条同场景片段（共享「西湖」主题词）+ 一条无关记忆，返回 (agent, 记忆们)。"""
    a = _mk(clock)
    m1 = a.store.add("我在西湖边散步", importance=0.5, now=clock[0]); clock[0] += 10
    m2 = a.store.add("西湖边的风很舒服", importance=0.4, now=clock[0]); clock[0] += 10
    m3 = a.store.add("我们在西湖边散步后吃了晚饭", importance=0.4, now=clock[0]); clock[0] += 10
    unrelated = a.store.add("项目后端用 FastAPI 和 SQLite", importance=0.6, now=clock[0])
    return a, (m1, m2, m3, unrelated)


def test_compose_scene_groups_related_and_excludes_unrelated():
    """同主题片段（共享「西湖」）拼进场景，无关记忆（无共享 n-gram）被排除。"""
    clock = [1000.0]
    a, (m1, m2, m3, unrelated) = _seed_scene(clock)
    scene = a.compose_scene("西湖那天", k=1)  # k=1 → 只有 m1 是种子，m2/m3 走扩展
    assert scene is not None and scene.count == 3
    ids = {f.memory.id for f in scene.fragments}
    assert {m1.id, m2.id, m3.id} <= ids
    assert unrelated.id not in ids            # 无关片段不进场景
    assert all(f.memory.id != unrelated.id for f in scene.fragments)


def test_compose_scene_chronological_narrative():
    """片段按经历顺序（created_at）排序，叙事用时序连接词拼接。"""
    clock = [1000.0]
    a, (m1, m2, m3, _unrelated) = _seed_scene(clock)
    scene = a.compose_scene("西湖那天", k=1)
    order = [f.memory.id for f in scene.fragments]
    assert order == [m1.id, m2.id, m3.id]     # 经历顺序
    assert [f.role for f in scene.fragments] == ["开头", "中间", "结尾"]
    assert "先是「我在西湖边散步」" in scene.narrative
    assert "接着「西湖边的风很舒服」" in scene.narrative
    assert "最后「我们在西湖边散步后吃了晚饭」" in scene.narrative
    assert scene.title                        # 标题取自最强相关片段
    assert 0.0 <= scene.coherence <= 1.0      # 连贯度度量
    assert scene.strength > 0                 # 整体强度


def test_compose_scene_single_fragment_returns_none():
    """场景需要 ≥2 个片段——单条命中只是记忆，不是场景。"""
    clock = [1000.0]
    a = _mk(clock)
    a.store.add("我昨天去吃了火锅", importance=0.3, now=clock[0])
    assert a.compose_scene("火锅") is None
    assert a.last_scene is None


def test_compose_scene_similarity_threshold():
    """scene_similarity 门控：阈值抬高后弱相关片段（0.42）被排除。"""
    clock = [1000.0]
    a, (_m1, m2, _m3, _unrelated) = _seed_scene(clock)
    scene = a.compose_scene("西湖那天", k=1)
    assert any(f.memory.id == m2.id for f in scene.fragments)
    a2, (_p, _q, _r, _u) = _seed_scene([2000.0])
    scene2 = a2.compose_scene("西湖那天", k=1)
    assert scene2 is not None
    # 抬高阈值到 0.5：m2（相关度 0.418）被排除，只剩 m1+m3（0.58）
    a2.cfg.scene_similarity = 0.5
    scene3 = a2.compose_scene("西湖那天", k=1)
    assert scene3 is not None
    assert all(f.memory.id != _q.id for f in scene3.fragments)


def test_compose_scene_time_window():
    """时间窗：出生时间距种子超过 scene_time_window 的片段不属于同一场景。

    far 与种子同主题（相关度 0.33 ≥ 阈值，会通过相似度门控）且最近访问过
    （强度高）——唯一被排除的原因就是时间窗（created_at 在 5 年前）。"""
    clock = [1000.0]
    a = _mk(clock)
    m1 = a.store.add("我在西湖边散步", importance=0.5, now=clock[0]); clock[0] += 10
    m2 = a.store.add("西湖边的风很舒服", importance=0.4, now=clock[0]); clock[0] += 10
    far = a.store.add("西湖边又修了新路", importance=0.4,
                      now=clock[0] - 5 * 365 * 24 * 3600)  # 出生 5 年前
    far.last_access = clock[0]                # 最近访问过 → 强度高（排除与强度无关）
    assert _fragment_relatedness(m1, far, require_shared=True) >= 0.2  # 相似度门控本会放行
    scene = a.compose_scene("我在西湖边散步", k=1)  # 精确命中 m1 → 种子锚定 m1 时刻
    assert scene is not None and scene.count == 2
    assert any(f.memory.id == m2.id for f in scene.fragments)   # 窗口内片段入场景
    assert all(f.memory.id != far.id for f in scene.fragments)  # 跨年片段被时间窗排除


def test_compose_scene_cold_fragment():
    """Cold 摘要片段：via_summary=True，叙事用摘要文本（索引向量一致）。"""
    clock = [1000.0]
    a = _mk(clock)
    cold = a.store.add("我在西湖边散步", importance=0.5, now=clock[0])
    cold.demote_to_cold("西湖散步（已归档）")
    clock[0] += 10
    warm = a.store.add("西湖散步很舒服", importance=0.4, now=clock[0])
    scene = a.compose_scene("西湖", k=1)
    assert scene is not None and scene.count == 2
    cold_frag = next(f for f in scene.fragments if f.via_summary)
    assert cold_frag.text == "西湖散步（已归档）"      # 摘要文本进场景
    warm_frag = next(f for f in scene.fragments if not f.via_summary)
    assert warm_frag.text == "西湖散步很舒服"          # Warm 用完整内容


def test_compose_scene_testing_effect_and_reconsolidation():
    """扩展片段获得再激活（touch+采样）；scene_reconsolidates=True 时再巩固（修订+1）。

    测量目标「西湖边的风很舒服」排在检索第 3 位——retrieve 只 touch/再巩固
    hits[:2]，所以它的 touch/修订只能来自场景扩展，干净可测。"""
    clock = [1000.0]
    a = _mk(clock)
    a.store.add("我在西湖边散步", importance=0.5, now=clock[0]); clock[0] += 10
    a.store.add("我们在西湖边散步后吃了晚饭", importance=0.4, now=clock[0]); clock[0] += 10
    m2 = a.store.add("西湖边的风很舒服", importance=0.4, now=clock[0]); clock[0] += 10
    n_hist, acc = len(m2.history), m2.access_count
    a.compose_scene("西湖散步", k=1)
    assert m2.access_count == acc + 1         # 扩展 touch 测试效应
    assert len(m2.history) == n_hist + 1      # 观测采样
    # 再巩固开启（agent 级 reconsolidate 也要开）：扩展片段修订 +1
    a2 = _mk([5000.0], reconsolidate=True)
    a2.store.add("我在西湖边散步", importance=0.3, now=5000.0)
    a2.store.add("我们在西湖边散步后吃了晚饭", importance=0.3, now=5010.0)
    p2 = a2.store.add("西湖边的风很舒服", importance=0.3, now=5020.0)
    a2.compose_scene("西湖散步", k=1)
    assert p2.revision_count == 1
    # 场景级再巩固关闭：修订不变（即使 agent 级再巩固开启）
    a3 = _mk([9000.0], scene_reconsolidates=False, reconsolidate=True)
    a3.store.add("我在西湖边散步", importance=0.3, now=9000.0)
    a3.store.add("我们在西湖边散步后吃了晚饭", importance=0.3, now=9010.0)
    q2 = a3.store.add("西湖边的风很舒服", importance=0.3, now=9020.0)
    a3.compose_scene("西湖散步", k=1)
    assert q2.revision_count == 0


def test_compose_scene_max_fragments():
    """scene_max_fragments 上限：种子优先，扩展按相关度截断。"""
    clock = [1000.0]
    a = _mk(clock, scene_max_fragments=2)
    m1 = a.store.add("我在西湖边散步", importance=0.5, now=clock[0]); clock[0] += 10
    m2 = a.store.add("西湖边的风很舒服", importance=0.4, now=clock[0]); clock[0] += 10
    m3 = a.store.add("我们在西湖边散步后吃了晚饭", importance=0.4, now=clock[0]); clock[0] += 10
    scene = a.compose_scene("西湖", k=1)
    assert scene is not None and scene.count == 2


def test_compose_scene_config_off():
    """scene_reconstruction=False → 不重建、无副作用。"""
    clock = [1000.0]
    a = _mk(clock, scene_reconstruction=False)
    m1 = a.store.add("我在西湖边散步", importance=0.5, now=clock[0]); clock[0] += 10
    m2 = a.store.add("西湖边的风很舒服", importance=0.4, now=clock[0]); clock[0] += 10
    acc = m2.access_count
    assert a.compose_scene("西湖") is None
    assert m2.access_count == acc             # 扩展片段未被 touch


def test_respond_shows_scene_in_template_reply():
    """respond() 命中同场景片段时，模板回复展示重建出的连贯场景。"""
    clock = [1000.0]
    a = _mk(clock)
    a.store.add("我在西湖边散步", importance=0.5, now=clock[0]); clock[0] += 10
    a.store.add("西湖边的风很舒服", importance=0.4, now=clock[0]); clock[0] += 10
    reply, hits = a.respond("西湖那天")
    assert "我记得一段连贯的场景" in reply
    assert "先是「我在西湖边散步」" in reply   # 叙事以场景呈现
    assert a.last_scene is not None and a.last_scene.count == 2
    assert hits                                # 检索链路不受影响


def test_respond_single_memory_keeps_old_reply():
    """单条命中（构不成场景）→ 回复保持原样（我记得：…）。"""
    clock = [1000.0]
    a = _mk(clock)
    a.store.add("我昨天去吃了火锅", importance=0.3, now=clock[0])
    reply, _hits = a.respond("我昨天吃了什么")
    assert "我记得：" in reply
    assert "我记得一段连贯的场景" not in reply
    assert "我昨天去吃了火锅" in reply


def test_format_scene_render():
    """format_scene：中文展示含标题/叙事/时序角色。"""
    clock = [1000.0]
    a, _mems = _seed_scene(clock)
    scene = a.compose_scene("西湖那天", k=1)
    out = format_scene(scene)
    assert "场景 · " in out and "重建叙事" in out
    assert "[开头]" in out and "[结尾]" in out
